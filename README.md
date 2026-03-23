# Unsplash_Ollama：从原始元数据到五分类数据集的完整流程说明

本仓库用于把 Unsplash 原始元数据清洗成可用的五分类数据集，核心目标是：

- **先过滤掉 art / illustration / poster 等非目标样本**。
- **把剩余样本归到 5 个目标类**：`城市、建筑` / `室内` / `自然` / `静物` / `人像`。
- **通过“规则 + LLM(Ollama) +（可选）图像复审”分阶段提升质量**。

---

## 1. 方法总览（你要的“方法 + 处理流程”）

整套方法不是一步到位，而是“**多阶段漏斗**”：

1. **结构化主表构建**：从 `photos.csv*` 中抽取可用字段，得到统一主表（manifest）。
2. **硬过滤**：基于 `keywords.csv*` 的关键词，先剔除明确艺术类样本。
3. **规则预分类**：用严格词典和打分策略先吃掉“高确定性”样本；把模糊样本送给 LLM。
4. **Ollama 复核**：仅对模糊样本调用模型，输出结构化标签与置信度，并支持断点续跑。
5. **结果合并与分流**：合并 rule 与 Ollama 结果，低置信度进入 `need_review`。
6. **（可选扩展）下载图片 + 美学筛选 + CLIP 复审 + 最终打包**。

你可以把它理解为：

- **阶段 1（文本元数据主流程）**：先把“能稳定判定的”做好。
- **阶段 2（视觉增强流程）**：再用图像模型做质量提升和再筛选。

---

## 2. 环境准备

### 2.1 安装 Python 依赖

```bash
pip install -r requirements.txt
```

### 2.2 启动 Ollama 并拉取模型
```bash
ollama pull gemma3
```

> 如果你使用别的本地模型，请在 `config.py` 修改 `OLLAMA_MODEL`。

### 2.3 配置数据目录（必须）

编辑 `config.py`：

```python
DATA_ROOT = Path(r"YOUR_UNSPLASH_DATASET_DIR").resolve()
```

例如：

```python
DATA_ROOT = Path(r"F:/unsplash-research-dataset-full-latest").resolve()
```

> `WORK_DIR` 会自动指向 `DATA_ROOT/work`。

---

## 3. 输入与输出

### 3.1 原始输入（必需）

放在 `DATA_ROOT` 下：

- `photos.csv*`
- `keywords.csv*`

### 3.2 主流程核心输出（阶段 1）

默认位于 `DATA_ROOT/work`：

- `manifest.parquet`：主表
- `photos_no_art.parquet`：过滤 art 后主表
- `removed_art.parquet`：被 art 规则过滤的样本
- `preclassified.parquet`：规则打分结果（含直判样本）
- `need_llm.parquet`：待 Ollama 样本
- `ollama_results.jsonl` / `ollama_results.parquet`：模型输出
- `classified.parquet`：合并后的分类结果
- `need_review.parquet`：低置信度 / 异常样本
- `category_stats.csv`：类别统计

---

## 4. 标准主流程（推荐先跑这个）

你可以“逐步执行”，也可以“一键执行”。

### 4.1 逐步执行

```bash
python 01_build_manifest.py
python 02_filter_art.py
python 03_rule_preclassify.py
python 04_ollama_classify.py
python 05_merge_and_review.py
```

### 4.2 一键执行

```bash
python run_all.py
```

---

## 5. 每一步到底在做什么（详细版）

### Step 1：`01_build_manifest.py`（主表构建）

**做什么：**

- 读取 `photos.csv*`。
- 抽取必要字段（如描述、地理、相机、下载统计、AI 描述等）。
- 输出 `manifest.parquet`（zstd 压缩）。

**作用：**

- 把原始 CSV 变成后续处理友好的单一结构化入口。

---

### Step 2：`02_filter_art.py`（硬过滤 art）

**做什么：**

- 读取 `keywords.csv*`。
- 用 `config.py` 中 `ART_KEYWORDS`（如 `art`, `painting`, `illustration` 等）提取 art 样本 ID。
- 将主表分成：
  - `photos_no_art.parquet`（保留）
  - `removed_art.parquet`（剔除）

**作用：**

- 在最前面拦截掉明显不符合目标范围的数据，降低后续模型成本。

---

### Step 3：`03_rule_preclassify.py`（严格规则预分类）

**做什么：**

- 拼接 `text_for_cls`（description、ai_description、location、landmark 等）。
- 使用更保守词典（`STRICT_VOCAB`）对五类打分。
- 对证据不足、冲突高、疑似图形/抽象/文档类样本做拦截（`HARD_REJECT_TERMS`）。
- 将结果拆分为：
  - `category_source=rule` 的直判样本（高置信）
  - `needs_llm=True` 的待 LLM 样本（写入 `need_llm.parquet`）

**这一步的核心思想：**

- 规则不是为了“尽量多判”，而是为了“把明显样本快速吃掉，并且把风险样本拦出来”。

---

### Step 4：`04_ollama_classify.py`（仅处理模糊样本）

**做什么：**

- 只读取 `need_llm.parquet`。
- 使用 JSON Schema 约束输出结构：
  - `is_target`
  - `label`
  - `confidence`
  - `reason`
- 对明显不可靠元数据先做本地拒绝（减少无效请求）。
- 支持并发、重试、断点续跑（通过 `ollama_results.jsonl`）。

**作用：**

- 用 LLM 处理规则难以稳定判断的“灰区样本”。

---

### Step 5：`05_merge_and_review.py`（合并与待复核分流）

**做什么：**

- 合并规则结果和 Ollama 结果。
- 对 `needs_llm=True` 的样本回填 Ollama 标签与置信度。
- 若 Ollama 失败，则 fallback 到规则 top1（低置信）。
- 低于 `REVIEW_CONFIDENCE_THRESHOLD` 的样本打 `review_flag=True`，写入 `need_review.parquet`。
- 输出类别统计 `category_stats.csv`。

**作用：**

- 得到可直接消费的分类结果，同时保留人工复核入口。

---

## 6. 关键配置怎么调（实战建议）

集中在 `config.py`：

- `MAX_WORKERS` / `REQUEST_TIMEOUT` / `MAX_RETRIES`：控制 Ollama 并发与稳健性。
- `REVIEW_CONFIDENCE_THRESHOLD`：控制进入人工复核的严格度。
- `ART_KEYWORDS`：控制 art 硬过滤覆盖范围。
- `LABELS`：固定五类标签集合。

建议流程：

1. 先抽样（5k~10k）跑全流程。
2. 查看 `need_review.parquet` 中的误差类型。
3. 再微调阈值、关键词词典和并发参数。
4. 最后跑全量。

---

## 7. 可选扩展流程（图像侧增强）

如果你希望在文本流程基础上继续提质，可以走扩展链路：

```bash
python 06_download_pictures.py
python 07_aesthetic_score.py
python 08_aesthetic_pass.py
python 09_review_with_clip.py
python 10_second_pass_clip_review.py --input <manifest_clip_review.parquet>
python 11_build_final_dataset.py --input <manifest_final_selected.parquet> --output-root <final_dataset_dir>
```

### 扩展流程含义

- `06`：下载图片并把下载状态回写 metadata。
- `07`：CLIP 特征 + LAION aesthetic predictor 做审美分。
- `08`：按分位数阈值过滤低审美样本。
- `09`：用 zero-shot CLIP 对已有标签做第一轮复审（keep / relabel / review）。
- `10`：对第一轮 `review/error` 做第二轮复审，进一步减少人工量。
- `11`：按最终标签整理目录并导出最终 manifest、photo metadata、review log。

---

## 8. 常见问题（FAQ）

### Q1：为什么有些样本被“拒绝”而不是硬分到五类？
因为流程是“保守优先”。当元数据不足或冲突高时，拒绝比误标更安全。

### Q2：为什么人像容易和城市/室内混淆？
有些文本只出现 `person/people`，但主体可能是街景或室内空间。规则里已做“弱人像抑制”。

### Q3：为什么我跑得慢？
优先检查：

- Ollama 模型是否过大
- `MAX_WORKERS` 是否超过机器承载
- `REQUEST_TIMEOUT` 是否过低导致频繁重试

### Q4：如何保证可复现？
固定：

- 数据目录版本
- `config.py` 参数
- Ollama 模型名与版本
- 执行脚本顺序

---

## 9. 推荐你直接照着走的最小落地方案

1. 改 `config.py` 的 `DATA_ROOT` 和 `OLLAMA_MODEL`。
2. `python run_all.py` 跑完主流程。
3. 抽查 `classified.parquet` + `need_review.parquet`。
4. 如果误差可接受，再进入扩展流程（06~11）。

这样可以先用最小成本拿到一版可用数据集，再逐步提纯。