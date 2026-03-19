import pandas as pd

from config import (
    CLASSIFIED_FILE,
    NEED_LLM_FILE,
    NEED_REVIEW_FILE,
    OLLAMA_RESULTS_FILE,
    PRECLASSIFIED_FILE,
    REVIEW_CONFIDENCE_THRESHOLD,
    STATS_FILE,
)
from utils import ensure_exists

def main() -> None:
    ensure_exists(PRECLASSIFIED_FILE, "请先运行 03_rule_preclassify.py")
    ensure_exists(OLLAMA_RESULTS_FILE, "请先运行 04_ollama_classify.py")

    pre_df = pd.read_parquet(PRECLASSIFIED_FILE)
    ollama_df = pd.read_parquet(OLLAMA_RESULTS_FILE)

    if len(ollama_df) == 0:
        merged = pre_df.copy()
    else:
        merged = pre_df.merge(
            ollama_df[
                [
                    "photo_id",
                    "ollama_label",
                    "ollama_confidence",
                    "ollama_reason",
                    "ollama_ok",
                    "ollama_error",
                ]
            ],
            on="photo_id",
            how="left",
        )

    # 规则直判样本保留 rule 结果；待 LLM 样本用 Ollama 结果回填
    use_ollama = merged["needs_llm"] == True

    merged.loc[use_ollama, "category"] = merged.loc[use_ollama, "ollama_label"]
    merged.loc[use_ollama, "category_confidence"] = merged.loc[use_ollama, "ollama_confidence"]
    merged.loc[use_ollama, "category_source"] = "ollama"

    # Ollama 失败样本保底：仍用规则 top1 兜底，但打上低置信度 review
    failed = use_ollama & merged["category"].isna()
    merged.loc[failed, "category"] = merged.loc[failed, "rule_top1_label"]
    merged.loc[failed, "category_confidence"] = 0.2
    merged.loc[failed, "category_source"] = "fallback_rule"

    merged["review_flag"] = merged["category_confidence"].fillna(0) < REVIEW_CONFIDENCE_THRESHOLD
    merged.to_parquet(CLASSIFIED_FILE, index=False)

    need_review = merged[merged["review_flag"] == True].copy()
    need_review.to_parquet(NEED_REVIEW_FILE, index=False)

    stats = (
        merged.groupby(["category", "category_source"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["count", "category"], ascending=[False, True])
    )
    stats.to_csv(STATS_FILE, index=False, encoding="utf-8-sig")

    print(f"[OK] 最终分类文件：{CLASSIFIED_FILE}")
    print(f"[OK] 待人工复核文件：{NEED_REVIEW_FILE}")
    print(f"[OK] 统计文件：{STATS_FILE}")
    print("\n分类分布：")
    print(stats.to_string(index=False))

if __name__ == "__main__":
    main()
