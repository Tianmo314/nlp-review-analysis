"""中文影评词云（课后作业任务五）。

jieba 分词 → 停用词过滤 → wordcloud 出图，**强制指定中文字体**（否则满屏方框）。
按评分分成好评 / 差评两组各画一张，返回可直接塞进 <img src> 的 base64 PNG。

命令行也能单独出图，便于离线检查：

    python wordcloud_utils.py --data data/douban/痴迷.jsonl --out artifacts/wordcloud
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

import jieba
import numpy as np
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap
from wordcloud import WordCloud

from data import match_column, read_table

ROOT = Path(__file__).resolve().parent
STOPWORDS_FILE = ROOT / "assets" / "stopwords.txt"
FONT_ENV = "WORDCLOUD_FONT"

# Windows / macOS / Linux 常见中文字体；也可用 WORDCLOUD_FONT 环境变量覆盖。
# 雅黑优先（字重适中、观感更现代），缺失时退回黑体。
FONT_CANDIDATES: tuple[str, ...] = (
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/simsun.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
)

# 影片专名，避免被 jieba 切碎成「痴 / 迷」「许 / 愿 / 柳」。
DOMAIN_WORDS: tuple[str, ...] = (
    "痴迷",
    "心愿柳",
    "许愿柳",
    "妮基",
    "贝尔",
    "库里·巴克",
    "印达·纳瓦雷特",
    "迈克尔·约翰斯顿",
    "恐怖片",
    "惊悚片",
    "心理恐怖",
    "jump scare",
    "细思极恐",
    "后劲",
    "占有欲",
    "两性关系",
    "男主",
    "女主",
    "男二",
    "女二",
    "结局",
    "反转",
    "铺垫",
    "节奏",
    "氛围",
    "配乐",
    "演技",
    "剧情",
    "镜头",
    "观感",
    "烂尾",
    "高开低走",
    "意难平",
)

SCORE_ALIASES: tuple[str, ...] = ("score", "y_true", "label", "rating")
_NO_LETTER = re.compile(r"^[\s\d\W_]+$")


def resolve_font() -> str:
    """返回可用的中文字体文件路径；找不到就明确报错，绝不退回到默认字体。"""
    env = os.environ.get(FONT_ENV)
    candidates = ([env] if env else []) + list(FONT_CANDIDATES)
    candidates += [str(p) for p in (ROOT / "assets").glob("*.tt*")]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(Path(candidate))
    raise FileNotFoundError(
        "找不到中文字体，词云会显示成方框。请设置环境变量 WORDCLOUD_FONT "
        "指向一个 .ttf/.ttc 中文字体（Windows 常用 C:/Windows/Fonts/simhei.ttf）。"
    )


def load_stopwords() -> set[str]:
    """读停用词表；# 开头是注释，空行忽略。"""
    words: set[str] = set()
    for line in STOPWORDS_FILE.read_text(encoding="utf-8").splitlines():
        word = line.strip()
        if word and not word.startswith("#"):
            words.add(word)
    return words


def cut_words(texts: list[str], stopwords: set[str]) -> list[str]:
    """jieba 精确模式切词，过滤单字、数字标点、停用词。"""
    for word in DOMAIN_WORDS:
        jieba.add_word(word)
    kept: list[str] = []
    for text in texts:
        for word in jieba.lcut(text):
            word = word.strip().lower()
            if len(word) < 2 or word in stopwords:
                continue
            if _NO_LETTER.match(word):  # 纯数字 / 标点 / 空白
                continue
            if not any("\u4e00" <= ch <= "\u9fff" or ch.isalpha() for ch in word):
                continue
            kept.append(word)
    return kept


def _trim_colormap(name: str, low: float = 0.45) -> LinearSegmentedColormap:
    """截掉色板过浅的一端：浅色词落在米白底上会发灰，看着脏。"""
    base = colormaps[name]
    samples = base(np.linspace(low, 1.0, 128))
    return LinearSegmentedColormap.from_list(f"{name}_trim", samples)


def render_image(freq: dict[str, int], colormap: str, font_path: str) -> str:
    """用词频生成 PNG，返回 data:image/png;base64,…。"""
    if not freq:
        return ""
    cloud = WordCloud(
        font_path=font_path,
        width=920,
        height=480,
        background_color="#fcfbf8",
        colormap=_trim_colormap(colormap),
        max_words=85,
        # 0.45 让高频词不再一家独大，整张图的字号过渡更平缓
        relative_scaling=0.45,
        min_font_size=12,
        max_font_size=86,
        prefer_horizontal=0.95,
        margin=3,
        scale=2,
        collocations=False,
        random_state=42,
    )
    cloud.generate_from_frequencies(freq)
    buffer = io.BytesIO()
    cloud.to_image().save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def load_reviews(path: Path) -> list[tuple[str, float]]:
    """从本地表格里取 (文本, 评分)，兼容 text/score 与 text/y_true 两种列名。"""
    frame = read_table(path)
    text_col = match_column(frame, "text", "文本")
    score_col = next(
        (
            c
            for alias in SCORE_ALIASES
            for c in frame.columns
            if str(c).lower() == alias
        ),
        None,
    )
    if score_col is None:
        raise KeyError(
            f"{path} 里找不到评分列，期望 {list(SCORE_ALIASES)} 之一；"
            f"当前列：{[str(c) for c in frame.columns]}"
        )
    rows: list[tuple[str, float]] = []
    for text, score in zip(frame[text_col], frame[score_col], strict=True):
        try:
            value = float(score)
        except (TypeError, ValueError):
            continue
        cleaned = str(text).strip()
        if cleaned:
            rows.append((cleaned, value))
    return rows


def build_payload(
    path: Path,
    good_min: float = 4.0,
    bad_max: float = 2.0,
    min_bad: int = 30,
) -> dict[str, Any]:
    """按评分分组统计词频并出两张图；差评样本太少时自动放宽到 3 星。"""
    reviews = load_reviews(path)
    good_texts = [t for t, s in reviews if s >= good_min]
    bad_texts = [t for t, s in reviews if s <= bad_max]
    relaxed = False
    if len(bad_texts) < min_bad:
        bad_texts = [t for t, s in reviews if s <= 3.0]
        relaxed = True

    stopwords = load_stopwords()
    font_path = resolve_font()

    def group(texts: list[str], colormap: str) -> dict[str, Any]:
        counter = Counter(cut_words(texts, stopwords))
        top = counter.most_common(120)
        return {
            "n_reviews": len(texts),
            "n_tokens": sum(counter.values()),
            "top_words": [{"word": w, "count": c} for w, c in top[:15]],
            "image": render_image(dict(top), colormap, font_path),
        }

    return {
        "available": True,
        "source": str(path),
        "source_name": path.name,
        "n_reviews": len(reviews),
        "font": font_path,
        "font_name": Path(font_path).name,
        "stopwords": len(stopwords),
        "thresholds": {
            "good": f"≥ {good_min:g} 星",
            "bad": f"≤ {3 if relaxed else bad_max:g} 星",
            "relaxed": relaxed,
        },
        "good": group(good_texts, "GnBu"),
        "bad": group(bad_texts, "OrRd"),
    }


def resolve_review_path(config_path: str = "configs/douban.yaml") -> Path:
    """优先用配置里的本地数据，回退到训练产出的测试集预测。"""
    from config import load_config

    cfg = load_config(config_path)
    if cfg.data.source == "local" and cfg.data.path:
        candidate = Path(cfg.data.path)
        candidate = candidate if candidate.is_absolute() else cfg.root / candidate
        if candidate.exists():
            return candidate
    fallback = cfg.artifact_dir / "test_predictions.csv"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        f"找不到影评数据：{cfg.data.path} 与 {fallback} 都不存在。请先跑爬虫和训练。"
    )


def _write_files(payload: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for key in ("good", "bad"):
        image = payload[key]["image"]
        if not image:
            continue
        raw = base64.b64decode(image.split(",", 1)[1])
        target = out_dir / f"wordcloud_{key}.png"
        target.write_bytes(raw)
        print(f"  {target}（{payload[key]['n_reviews']} 条评论）")
    summary = {k: v for k, v in payload.items() if k not in ("good", "bad")} | {
        "good": {k: v for k, v in payload["good"].items() if k != "image"},
        "bad": {k: v for k, v in payload["bad"].items() if k != "image"},
    }
    (out_dir / "wordcloud_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="生成好评 / 差评中文词云")
    parser.add_argument("--data", type=Path, default=None, help="影评 jsonl/csv")
    parser.add_argument("--config", default="configs/douban.yaml")
    parser.add_argument("--out", type=Path, default=Path("artifacts/wordcloud"))
    parser.add_argument("--good-min", type=float, default=4.0)
    parser.add_argument("--bad-max", type=float, default=2.0)
    args = parser.parse_args()

    path = args.data or resolve_review_path(args.config)
    payload = build_payload(path, args.good_min, args.bad_max)
    print(
        f"数据源：{path}｜{payload['n_reviews']} 条｜字体：{payload['font']}\n"
        f"停用词：{payload['stopwords']} 个｜分组：{payload['thresholds']}"
    )

    def top_line(group: dict[str, Any]) -> str:
        return str([(w["word"], w["count"]) for w in group["top_words"][:10]])

    print(f"好评 Top10：{top_line(payload['good'])}")
    print(f"差评 Top10：{top_line(payload['bad'])}")
    _write_files(payload, args.out)


if __name__ == "__main__":
    main()
