# 文本向量 + 浅层模型流水线（分类 / 回归）

用 sentence-transformers 提取句向量，接浅层模型完成文本分类（LinearSVC / LogisticRegression）或数值回归（Ridge / LinearSVR）。
数据集、Embedding 模型、任务类型全部由 `configs/*.yaml` 决定，仓库本身不绑定某个模型、数据集或任务。

## 1. 环境准备

使用 conda，Python 3.11。下面以环境名 `torch-env` 为例：

```bash
conda create -n torch-env python=3.11 -y
conda activate torch-env

# 先按显卡安装对应的 torch（示例为 CUDA 12.4；纯 CPU 机器跳过 --index-url）
pip install torch --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
```

## 2. 跑通分类（默认配置）

```bash
python main.py prepare                                # 下载数据集与 embedding 模型到项目内
python main.py train -c configs/classification.yaml   # 编码、调参、训练、测试集评估
python main.py predict -c configs/classification.yaml --text "这家酒店位置很好，服务也很周到。"
python webapp.py -c configs/classification.yaml       # 网页工作台，http://127.0.0.1:7860
```

训练打印 Accuracy、Macro-F1、分类报告和混淆矩阵。参考结果：ChnSentiCorp 测试集 Accuracy 0.8867、
Macro-F1 0.8866（`bge-small-zh-v1.5` + `LinearSVC`，C=1.0）。
设备默认 `auto`（cuda → mps → cpu），可用 `--device` 显式指定。

## 3. 跑通回归

豆瓣长评 → 评分（1-5）的示例配置。数据不随仓库分发，先放到 `data/niulai/clean.jsonl`：

```bash
python main.py train -c configs/regression.yaml   # 报告 R²、RMSE、MAE、Pearson、Spearman
python webapp.py -c configs/regression.yaml
```

## 4. 数据与模型放在哪里

两者都落在仓库目录内，不写 `~/.cache/huggingface`：

```
data/<数据集名>/...     # Hugging Face 数据集快照
models/<模型名>/...     # Embedding 权重
```

`prepare` 自动下载（仓库同时提供 bin 与 safetensors 权重时只下后者），也可以手动：

```bash
hf download --repo-type dataset lansinuote/ChnSentiCorp --local-dir data/ChnSentiCorp
hf download BAAI/bge-small-zh-v1.5 --local-dir models/bge-small-zh-v1.5
```

国内网络可先 `export HF_ENDPOINT=https://hf-mirror.com`。`data/`、`models/`、`artifacts/` 均在 `.gitignore` 中，换机器重新下载即可。

## 5. 换成自己的数据与模型

本地数据支持 `csv / tsv / parquet / jsonl`（每行一条）与 `json`（记录数组），至少一列文本、一列标签，列名大小写不敏感：

```yaml
# configs/my.yaml —— 分类；回归把 task 改为 regression，label_column 换成数值列
task: classification
data:
  source: local
  path: data/my_reviews.jsonl
  text_column: text
  label_column: label
  label_names: ["负向", "正向"]   # 可选，顺序对应标签 0、1
```

三种划分方式按优先级，满足其一即可：

1. 目录内有 `train / validation / test` 文件（`train-00000-of-00001.parquet` 这类前缀也认，前缀映射可用 `data.splits` 改）
2. 单文件里有划分列时写 `data.split_column`（取值 train / training / validation / valid / val / dev / test，三类需各出现至少一次）
3. 都没有时自动按 8:1:1 随机切分，分类任务尽量分层

长评论设置 `embedding.chunk_chars: 380` 按块编码再平均（`null` 表示整篇编码，超长截断）。
换 Embedding 模型改 `embedding.model` 一行：仓库名自动下载到 `models/`，或直接指向本地目录，维度自动读取。

## 6. 配置项速查

| 字段 | 说明 |
| --- | --- |
| `task` | `classification` 或 `regression` |
| `data.source` | `huggingface` 或 `local` |
| `data.name` / `data.path` | HF 数据集仓库名 / 本地文件或目录（相对仓库根目录） |
| `data.text_column` / `data.label_column` | 文本列、标签列，大小写不敏感 |
| `data.split_column` | 单文件按列切分；train/validation/test 三类值需各出现至少一次 |
| `data.splits` | 逻辑划分 → 文件名前缀，默认 `train/validation/test` |
| `data.max_samples` | 每个划分最多取前多少条，调试用 |
| `data.label_names` | 分类别名显示名，顺序对应 0..N-1；回归忽略 |
| `embedding.model` | 模型仓库名或本地目录 |
| `embedding.batch_size` | 编码批大小，显存不足时调小 |
| `embedding.chunk_chars` | 长文本切块字数，`null` 为整篇 |
| `model.name` | 分类 `linear_svc` / `logistic_regression`；回归 `ridge` / `linear_svr` |
| `model.param_grid` | 调参网格，留空用内置默认值 |
| `model.scoring` | 调参指标，留空分类用 `f1_macro`、回归用 `neg_root_mean_squared_error` |
| `model.refit_full` | `true` 时评估后用全量数据重训交付模型（指标仍来自留出 test） |
| `runtime.device` | `auto` / `cuda` / `mps` / `cpu` |
| `runtime.seed` | 划分与模型随机种子 |
| `output.dir` | 产物目录，默认 `artifacts` |

## 7. 命令

```bash
python main.py prepare                                  # 只下载数据与模型
python main.py train -c configs/classification.yaml     # 训练 + 评估
python main.py train -c configs/classification.yaml --limit 300   # 每个划分取前 300 条，快速验证
python main.py train -c configs/classification.yaml --recompute   # 忽略向量缓存重新编码
python main.py predict -c configs/classification.yaml --text "..."          # 单条预测
python main.py predict -c configs/classification.yaml --file data/x.jsonl --out preds.csv  # 批量
python main.py chat -c configs/classification.yaml      # 交互预测
python main.py embed -c configs/classification.yaml "..."          # 查看句向量
python webapp.py -c configs/classification.yaml         # 网页工作台（另有 --port / --no-browser / --device）
```

回归任务把 `-c` 换成 `configs/regression.yaml`。

## 8. 产物

| 文件 | 内容 |
| --- | --- |
| `artifacts/model.joblib` | 模型（`refit_full: true` 时为全量重训版）、embedding 模型名与维度、最佳参数、标签名、指标 |
| `artifacts/embeddings.npz` | 三个划分的向量与标签（带「embedding 配置 + 数据内容摘要」签名，配置或数据变了自动重算） |
| `artifacts/metrics.json` | 测试集指标、数据来源、划分规模、最佳参数 |
| `artifacts/test_predictions.csv` | 测试集逐条预测（原文、真实值、预测值、decision function） |
| `artifacts/config.yaml` | 训练时使用的配置副本，便于复现 |
| `artifacts/space_tsne.npz` | 网页工作台的 t-SNE 投影缓存 |

## 9. 网页工作台

交互预测、句向量热力图、测试集 t-SNE 散点（可点选）、最近邻样本、模型输出分布、测试指标。
分类显示类别与 decision function 标尺；多于两类的多分类自动降级为按「预测正确 / 错误」着色；
回归显示预测评分，散点按真实标签高低着色。
`refit_full: true` 时，工作台的判定来自全量重训后的交付模型（对测试集属 in-sample 口径），顶部指标徽章仍来自留出测试集。
默认只监听 `127.0.0.1`（可用 `--host` 改绑）。

## 10. 常见问题

1. **cuda / MPS 报错**：cuda 提示「检测不到 GPU」说明装的是 CPU 版 torch，按官网命令重装；MPS 报错可先用 `--device cpu` 绕过。
2. **显存不足**：调小 `embedding.batch_size`；长文本再调小 `embedding.chunk_chars`。
3. **下载慢或超时**：设 `HF_ENDPOINT=https://hf-mirror.com`，或用 `hf download` 手动下载。
4. **指标和之前对不上**：向量缓存按「embedding 配置 + 数据内容摘要」复用，改模型、改数据、改种子会自动重算；若 embedding 权重内容变了但仓库名没变，加 `--recompute`。
5. **标签是字符串**：无需改代码，自动按字典序编码；用 `data.label_names` 指定显示名。

## 11. 开发与来源

```bash
ruff format . && ruff check .
mypy main.py config.py data.py encoder.py modeling.py webapp.py
```

- 示例数据集：[`lansinuote/ChnSentiCorp`](https://huggingface.co/datasets/lansinuote/ChnSentiCorp)（中文酒店评论情感）
- 示例模型：[`BAAI/bge-small-zh-v1.5`](https://huggingface.co/BAAI/bge-small-zh-v1.5)

仓库不附带数据与权重；公开分发或商用前请自行核对各自的使用条款。

## 12. 课后作业：豆瓣影评 → 评分回归 → 词云工作台

本次在参考项目上做增量开发，选片《痴迷》（Obsession, 2025，豆瓣 subject `37450627`）。
新增/改动：

| 文件 | 作用 |
| --- | --- |
| `scripts/douban_crawler.py` | 有头浏览器登录、保存与复用登录态、按 `start` 翻页抓短评 |
| `configs/douban.yaml` | 回归配置：`source: local` 指向影评 jsonl，长评设 `chunk_chars` |
| `wordcloud_utils.py` | jieba 分词 + 停用词 + wordcloud 出图（也能独立用 CLI 导 PNG） |
| `assets/stopwords.txt` | 中文停用词表，保留「恐怖 / 剧情 / 结局」这类有信息量的实词 |
| `webapp.py` / `templates/index.html` | 新增 `/api/wordcloud` 与工作台「影评词云」面板 |

新增依赖已写入 `requirements.txt`：`playwright jieba wordcloud pillow`。
首次使用需装浏览器二进制：`python -m playwright install chromium`。

### 12.1 任务一：登录豆瓣并接管浏览器

```bash
python scripts/douban_crawler.py login     # 弹出有头 Chrome，由使用者本人登录
python scripts/douban_crawler.py check     # 复用登录态访问短评页，确认不再跳登录
```

账号密码只在浏览器里输入，不写进代码、不交给 Agent；登录态落在
`data/douban/.storage_state.json`（仅本地，`.gitignore` 已排除）。
`login` 检测到 `dbcl2` cookie 后会在**同一浏览器会话**里打开短评页完成验收；
`check` 额外打印该 cookie 是否存在。

### 12.2 任务二：抓短评

```bash
python scripts/douban_crawler.py crawl --delay 3 --max-pages 25
```

产出 `data/douban/痴迷.jsonl`（严格两列 `text` / `score`，可被 `data.py` 直接读入），
另存 `data/douban/痴迷_raw.jsonl` 保留 `time` / `cid` 便于抽查。
星级由 `allstar10~50` 换算 1~5；未打分与空文本跳过；按 `cid` 去重；页间随机 sleep 3~4 秒。
`data/*` 不入库，与仓库「数据不随仓库分发」的约定一致。

### 12.3 任务三：下载 Embedding 模型

```bash
# 国内网络先设镜像（PowerShell: $env:HF_ENDPOINT="https://hf-mirror.com"）
export HF_ENDPOINT=https://hf-mirror.com
hf download BAAI/bge-small-zh-v1.5 --local-dir models/bge-small-zh-v1.5
python main.py embed -c configs/douban.yaml "测试"
```

也可直接 `python main.py prepare -c configs/douban.yaml`，由 `encoder.py` 下载到 `models/`。
验收：输出指定维度的单位向量，L2 范数 ≈ 1。

### 12.4 任务四：训练回归

```bash
python main.py train -c configs/douban.yaml
python webapp.py -c configs/douban.yaml
```

`artifacts/douban/metrics.json` 记录 R²、RMSE、MAE、Pearson、Spearman 与最佳参数。

### 12.5 任务五：词云面板（中文分词）

```bash
python wordcloud_utils.py --data data/douban/痴迷.jsonl --out artifacts/wordcloud
python webapp.py -c configs/douban.yaml   # 点右上角「影评词云」
```

jieba 精确模式切词，加入影片专名自定义词典，过滤单字/数字/标点/停用词；
wordcloud **强制指定中文字体**（`WORDCLOUD_FONT` 可覆盖；Windows 优先微软雅黑 `msyh.ttc`，缺失回退黑体 `simhei.ttf`），
否则满屏方框。排版上固定 95% 横排、`relative_scaling=0.45` 让字号梯度更平缓、并裁掉色板最浅的一段，
避免高频词一家独大、浅色词在米白底上发灰。默认好评 `score >= 4`、差评 `score <= 2`，差评样本不足 30 条时自动放宽到 3 星并在面板上标注。
接口 `GET /api/wordcloud` 返回两张 base64 PNG 与 Top 词，`?refresh=1` 可强制重画。

### 12.6 本次运行记录（2026-11-09）

| 项目 | 结果 |
| --- | --- |
| 登录态验收 | 短评页返回 20 条、未被拦截；`dbcl2` 登录 cookie 存在 |
| 抓取 | **394 条**（≥200）；星级分布 1:26 / 2:41 / 3:102 / 4:158 / 5:67 |
| 数据抽查 | 固定种子抽 10 条：文本通顺、星级与语义相符、无空文本；2 条文本重复但 cid 不同，已按 cid 去重 |
| Embedding | `bge-small-zh-v1.5` 本地目录，512 维；`python main.py embed "测试"` → L2 范数 **1.000000** |
| 划分 | train 315 / validation 39 / test 40（8:1:1 随机切分） |
| 最佳参数 | `Ridge(alpha=10)`（验证集 RMSE 0.9091） |
| 测试集指标 | **R² 0.0909｜RMSE 1.2001｜MAE 0.9805｜Pearson 0.5382｜Spearman 0.5322** |
| 词云 | 好评 225 条 / 差评 67 条；jieba + 419 个停用词；字体 `msyh.ttc`；浏览器面板实测正常（1840×960） |

**一句话评价**：模型能大致排出评分高低（Pearson 0.54、Spearman 0.53），但绝对分值平均差近 1 星（MAE 0.98）、
R² 只有 0.09，属于「排序方向可用、精确打分不可用」的弱模型。瓶颈主要在数据：394 条样本偏少、评分集中在 3~4 星
（3 星 102 条、4 星 158 条），且短评多为几十字的口语化吐槽，语义信号本身有限；换更长影评或更大样本会更有意义。

复现顺序：

```bash
python scripts/douban_crawler.py login
python scripts/douban_crawler.py crawl --delay 3 --max-pages 25
python main.py embed "测试"
python main.py train -c configs/douban.yaml
python wordcloud_utils.py --data data/douban/痴迷.jsonl --out artifacts/wordcloud
python webapp.py -c configs/douban.yaml
```

