"""豆瓣短评爬虫（课后作业任务一、任务二）。

设计约定：
- 账号密码只由使用者在**有头浏览器**里亲自输入，不写进代码、不传给 Agent。
- 登录成功后把 Playwright 的 storage_state 存到项目内，后续脚本复用该登录态。
- 短评按 ``start`` 翻页，页间 sleep，避免给对方服务器造成压力。

用法（在项目根目录、已安装依赖的虚拟环境里执行）：

    python scripts/douban_crawler.py login     # 有头浏览器登录并保存登录态
    python scripts/douban_crawler.py check     # 验收：复用登录态访问短评页不再跳登录
    python scripts/douban_crawler.py crawl     # 翻页抓短评并写出 jsonl

可选参数见 ``--help``，例如 ``--headed``（有头抓取，便于人工过验证码）、
``--delay 3``（页间等待秒数）、``--min-count 200``（不足则报错）。
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path
from typing import Any

from playwright.sync_api import BrowserContext, Page, sync_playwright

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "douban"
STATE_PATH = DATA_DIR / ".storage_state.json"

DEFAULT_SUBJECT_ID = "37450627"  # 《痴迷》Obsession (2025)
DEFAULT_MOVIE_NAME = "痴迷"
PAGE_SIZE = 20
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# 在页面里一次性取出结构化的短评字段，避免逐个 DOM 节点来回通信。
EXTRACT_JS = """
() => [...document.querySelectorAll('.comment-item')].map((el) => {
  const rating = el.querySelector('[class*="allstar"]');
  const short = el.querySelector('.comment-content .short')
             || el.querySelector('.comment-content');
  const time = el.querySelector('.comment-time');
  return {
    cid: el.getAttribute('data-cid') || '',
    text: (short ? short.textContent : '').trim(),
    rating_class: rating ? rating.className : '',
    rating_title: rating ? (rating.getAttribute('title') || '') : '',
    time: time ? (time.getAttribute('title') || time.textContent.trim()) : '',
  };
})
"""

RATING_TITLES = {
    "很差": 1,
    "较差": 2,
    "还行": 3,
    "推荐": 4,
    "力荐": 5,
}


def comments_url(subject_id: str, start: int) -> str:
    return (
        f"https://movie.douban.com/subject/{subject_id}/comments"
        f"?start={start}&limit={PAGE_SIZE}&status=P&sort=new_score"
    )


def score_from_item(item: dict[str, Any]) -> int | None:
    """allstar30 → 3 分；没有星级（未打分）的短评返回 None。"""
    match = re.search(r"allstar(\d)0", item.get("rating_class", ""))
    if match:
        return int(match.group(1))
    return RATING_TITLES.get(item.get("rating_title", "").strip())


def new_context(pw: Any, headed: bool, use_state: bool) -> tuple[Any, BrowserContext]:
    """启动浏览器；有登录态就注入，没有也能抓（豆瓣对未登录用户有页数上限）。"""
    browser = pw.chromium.launch(headless=not headed)
    kwargs: dict[str, Any] = {
        "user_agent": USER_AGENT,
        "viewport": {"width": 1280, "height": 860},
        "locale": "zh-CN",
    }
    if use_state and STATE_PATH.exists():
        kwargs["storage_state"] = str(STATE_PATH)
        print(f"复用登录态：{STATE_PATH}")
    else:
        print("提示：未找到登录态，以未登录身份抓取（可访问的页数有限）。")
    context = browser.new_context(**kwargs)
    return browser, context


def looks_blocked(page: Page) -> bool:
    """识别反爬验证页 / 权限页 / 登录墙。"""
    if "sec.douban.com" in page.url:
        return True
    title = page.title()
    return any(word in title for word in ("没有访问权限", "登录", "异常", "验证码"))


def wait_for_login(context: BrowserContext, timeout_s: int) -> bool:
    """轮询 cookie 判断登录是否完成；dbcl2 是豆瓣的登录态 cookie。"""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        names = {c["name"] for c in context.cookies()}
        if "dbcl2" in names:
            return True
        time.sleep(1)
    return False


def cmd_login(args: argparse.Namespace) -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=False)
        context = browser.new_context(
            user_agent=USER_AGENT,
            viewport={"width": 1280, "height": 860},
            locale="zh-CN",
        )
        page = context.new_page()
        page.goto("https://www.douban.com/", wait_until="domcontentloaded")
        print("\n>>> 浏览器已打开，请在页面里完成豆瓣登录（扫码 / 账号密码均可）。")
        print(">>> 账号密码不要写进代码，也不要发给 Agent。")
        print(f">>> 最多等待 {args.login_timeout} 秒，登录成功后会自动保存登录态。\n")

        if not wait_for_login(context, args.login_timeout):
            print("登录超时：没有检测到登录态 cookie，未保存任何凭据。")
            browser.close()
            return 1

        context.storage_state(path=str(STATE_PATH))
        print(f"登录态已保存：{STATE_PATH}（仅本地，已被 .gitignore 排除）")

        # 同一浏览器会话里立刻验收：访问短评页不再跳登录。
        page.goto(
            comments_url(args.subject_id, 0),
            wait_until="domcontentloaded",
            timeout=30000,
        )
        page.wait_for_timeout(1500)
        count = page.locator(".comment-item").count()
        blocked = looks_blocked(page)
        print(f"验收访问：title={page.title()!r}｜短评条数={count}｜被拦截={blocked}")
        browser.close()
    return 0 if count > 0 and not blocked else 1


def cmd_check(args: argparse.Namespace) -> int:
    if not STATE_PATH.exists():
        print(f"没有登录态文件 {STATE_PATH}，请先执行 login。")
        return 1
    with sync_playwright() as pw:
        browser, context = new_context(pw, args.headed, use_state=True)
        page = context.new_page()
        page.goto(
            comments_url(args.subject_id, 0),
            wait_until="domcontentloaded",
            timeout=30000,
        )
        page.wait_for_timeout(1500)
        count = page.locator(".comment-item").count()
        blocked = looks_blocked(page)
        has_login_cookie = "dbcl2" in {c["name"] for c in context.cookies()}
        print(
            f"复用登录态访问：url={page.url}\n"
            f"title={page.title()!r}｜短评条数={count}｜被拦截={blocked}\n"
            f"登录态 cookie(dbcl2)：{has_login_cookie}"
        )
        browser.close()
    return 0 if count > 0 and not blocked else 1


def cmd_crawl(args: argparse.Namespace) -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / f"{args.movie_name}.jsonl"
    raw_path = DATA_DIR / f"{args.movie_name}_raw.jsonl"

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    blocked_pages = 0

    with sync_playwright() as pw:
        browser, context = new_context(pw, args.headed, use_state=True)
        page = context.new_page()

        for page_no in range(args.max_pages):
            start = page_no * PAGE_SIZE
            try:
                page.goto(
                    comments_url(args.subject_id, start),
                    wait_until="domcontentloaded",
                    timeout=30000,
                )
                page.wait_for_timeout(1200)
            except Exception as exc:
                print(f"  start={start} 打开失败：{exc}")
                break

            if looks_blocked(page):
                blocked_pages += 1
                print(
                    f"  start={start} 触发访问限制（title={page.title()!r}）。"
                    "可加 --headed 人工过验证后重试，或先执行 login。"
                )
                if blocked_pages >= 2:
                    break
                time.sleep(args.delay * 2)
                continue

            items: list[dict[str, Any]] = page.evaluate(EXTRACT_JS)
            added = 0
            for item in items:
                score = score_from_item(item)
                text = re.sub(r"\s+", " ", item.get("text", "")).strip()
                if score is None or not text:
                    continue  # 没打星或空文本不进训练集
                cid = item.get("cid") or f"{start}-{added}"
                if cid in seen:
                    continue
                seen.add(cid)
                rows.append(
                    {
                        "text": text,
                        "score": score,
                        "time": item.get("time", ""),
                        "cid": cid,
                    }
                )
                added += 1

            progress = (
                f"  start={start:3d}｜抓到 {len(items):2d} 条"
                f"｜新增 {added:2d} 条｜累计 {len(rows)}"
            )
            print(progress)
            if len(items) == 0:
                print("  本页没有短评，已到末页。")
                break
            time.sleep(args.delay + random.uniform(0, 1.0))

        browser.close()

    if not rows:
        print("没有抓到任何带星级的短评，未写出文件。")
        return 1

    # 主文件严格两列，供 data.py 直接读入；raw 保留抓取时间与来源 id 便于抽查。
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(
                    {"text": row["text"], "score": row["score"]}, ensure_ascii=False
                )
                + "\n"
            )
    with raw_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    scores = [r["score"] for r in rows]
    dist = {s: scores.count(s) for s in range(1, 6)}
    print(f"\n写出：{out_path}（{len(rows)} 条，两列 text/score）")
    print(f"原始：{raw_path}（含 time / cid）")
    print(f"星级分布：{dist}")
    if len(rows) < args.min_count:
        print(f"不足 {args.min_count} 条：请确认已登录，或增大 --max-pages 后重试。")
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="豆瓣短评：登录、验收、抓取")
    parser.add_argument("command", choices=["login", "check", "crawl"])
    parser.add_argument(
        "--subject-id", default=DEFAULT_SUBJECT_ID, help="豆瓣电影 subject id"
    )
    parser.add_argument(
        "--movie-name", default=DEFAULT_MOVIE_NAME, help="输出文件名用的电影名"
    )
    parser.add_argument("--delay", type=float, default=3.0, help="页间等待秒数")
    parser.add_argument("--max-pages", type=int, default=25, help="最多翻多少页")
    parser.add_argument("--min-count", type=int, default=200, help="少于该条数则判失败")
    parser.add_argument(
        "--login-timeout", type=int, default=300, help="等待手动登录的秒数"
    )
    parser.add_argument(
        "--headed", action="store_true", help="check/crawl 也用有头浏览器"
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "login":
        return cmd_login(args)
    if args.command == "check":
        return cmd_check(args)
    return cmd_crawl(args)


if __name__ == "__main__":
    raise SystemExit(main())
