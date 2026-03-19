# Unsplash + Ollama 清洗与五分类流水线

## 目标
1. 从 `keywords.csv*` 中筛掉明显属于 art / artwork / painting 等艺术品类关键词的图片  
2. 将剩余图片按 5 类分类：
   - 城市、建筑
   - 室内
   - 自然
   - 静物
   - 人像

## 输入文件
- `photos.csv*`
- `keywords.csv*`

## 输出文件
- `work/manifest.parquet`
- `work/photos_no_art.parquet`
- `work/removed_art.parquet`
- `work/preclassified.parquet`
- `work/need_llm.parquet`
- `work/ollama_results.parquet`
- `work/classified.parquet`
- `work/need_review.parquet`
- `work/category_stats.csv`

## 使用方法

### 1. 安装 Python 依赖
```bash
pip install -r requirements.txt
```

### 2. 安装并启动 Ollama
先确认你已经安装并启动 Ollama，然后拉一个你机器里可用的指令模型，例如：
```bash
ollama pull gemma3
```

### 3. 修改 `config.py`
把：
```python
DATA_ROOT = Path(r"YOUR_UNSPLASH_DATASET_DIR").resolve()
```
改成你自己的数据集目录，例如：
```python
DATA_ROOT = Path(r"F:/unsplash-research-dataset-full-latest").resolve()
```

如果你 pull 的模型不是 `gemma3`，也要改这里：
```python
OLLAMA_MODEL = "gemma3"
```

### 4. 依次运行
```bash
python 01_build_manifest.py
python 02_filter_art.py
python 03_rule_preclassify.py
python 04_ollama_classify.py
python 05_merge_and_review.py
```

或者一键运行：
```bash
python run_all.py
```

## 各脚本说明

### `01_build_manifest.py`
- 合并 `photos.csv*`
- 聚合 `keywords.csv*`
- 生成主表 `manifest.parquet`

### `02_filter_art.py`
- 用明确 art 关键词过滤图片
- 保留非 art 主表
- 导出被删样本

### `03_rule_preclassify.py`
- 构造 `text_for_cls`
- 用规则先吃掉明显样本
- 把模糊样本打到 `need_llm.parquet`

### `04_ollama_classify.py`
- 只对 `need_llm.parquet` 调用 Ollama
- 使用 JSON Schema 强制结构化输出
- 支持断点续跑，结果先写入 `ollama_results.jsonl`

### `05_merge_and_review.py`
- 合并 rule 与 Ollama 结果
- 输出最终结果和待复核样本

## 建议的实际工作流
1. 先抽 5k-10k 样本试跑
2. 检查 `need_review.parquet`
3. 调整 `CATEGORY_VOCAB` 和阈值
4. 再跑全量

## 注意
- 这套方案第一版是“基于文本元数据的分类”，不是直接看图分类
- 后续若要进一步提精度，可以对低置信度样本再加 VLM 二次审核
