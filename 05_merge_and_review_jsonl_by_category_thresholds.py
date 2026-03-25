import pandas as pd

from config import (
    CLASSIFIED_FILE,
    NEED_REVIEW_FILE,
    OLLAMA_RESULTS_JSONL,
    PRECLASSIFIED_FILE,
    REVIEW_CONFIDENCE_THRESHOLD,
    STATS_FILE,
)
from utils import ensure_exists


OLLAMA_KEEP_COLS = [
    "photo_id",
    "ollama_label",
    "ollama_confidence",
    "ollama_reason",
    "ollama_ok",
    "ollama_error",
    "seq",
]

# =========================
# 类别差异化阈值配置
# =========================
DEFAULT_RULE_MIN_CONFIDENCE = float(REVIEW_CONFIDENCE_THRESHOLD)

CATEGORY_RULE_MIN_CONFIDENCE = {
    "自然": 0.98,
    "城市、建筑": 0.95,
    "人像": 0.8,
    "室内": 0.7,
    "静物": 0.8,
}

DEFAULT_OLLAMA_MIN_CONFIDENCE = 0.75

CATEGORY_OLLAMA_MIN_CONFIDENCE = {
    "自然": 0.9,
    "城市、建筑": 0.85,
    "人像": 0.60,
    "室内": 0.60,
    "静物": 0.50,
}

CLASSIFIED_ALL_FILE = CLASSIFIED_FILE.with_name("classified_all.parquet")
REJECTED_FILE = CLASSIFIED_FILE.with_name("rejected_or_low_conf.parquet")
ALL_STATS_FILE = STATS_FILE.with_name("category_stats_all.csv")


def safe_to_csv(df: pd.DataFrame, path):
    try:
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return path
    except PermissionError:
        from datetime import datetime
        alt_file = path.with_name(
            f"{path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{path.suffix}"
        )
        df.to_csv(alt_file, index=False, encoding="utf-8-sig")
        print(f"[WARN] 目标文件被占用，已改存到: {alt_file}")
        return alt_file


def load_ollama_results_from_jsonl() -> pd.DataFrame:
    ensure_exists(OLLAMA_RESULTS_JSONL, "请先运行 04_ollama_classify_fast.py，至少生成 ollama_results.jsonl")

    if OLLAMA_RESULTS_JSONL.stat().st_size == 0:
        return pd.DataFrame(columns=OLLAMA_KEEP_COLS)

    ollama_df = pd.read_json(OLLAMA_RESULTS_JSONL, lines=True)
    if len(ollama_df) == 0:
        return pd.DataFrame(columns=OLLAMA_KEEP_COLS)

    for col in OLLAMA_KEEP_COLS:
        if col not in ollama_df.columns:
            ollama_df[col] = pd.NA

    ollama_df = ollama_df[OLLAMA_KEEP_COLS].copy()
    ollama_df["photo_id"] = ollama_df["photo_id"].astype(str)
    ollama_df["ollama_confidence"] = pd.to_numeric(ollama_df["ollama_confidence"], errors="coerce")
    ollama_df["ollama_ok"] = ollama_df["ollama_ok"].astype("boolean")
    ollama_df["seq"] = pd.to_numeric(ollama_df["seq"], errors="coerce")
    ollama_df = ollama_df.sort_values("seq", na_position="last")
    ollama_df = ollama_df.drop_duplicates(subset=["photo_id"], keep="last")
    return ollama_df


def get_rule_threshold(category) -> float:
    if pd.isna(category):
        return DEFAULT_RULE_MIN_CONFIDENCE
    return float(CATEGORY_RULE_MIN_CONFIDENCE.get(str(category), DEFAULT_RULE_MIN_CONFIDENCE))


def get_ollama_threshold(category) -> float:
    if pd.isna(category):
        return DEFAULT_OLLAMA_MIN_CONFIDENCE
    return float(CATEGORY_OLLAMA_MIN_CONFIDENCE.get(str(category), DEFAULT_OLLAMA_MIN_CONFIDENCE))


def main() -> None:
    ensure_exists(PRECLASSIFIED_FILE, "请先运行 03_rule_preclassify.py")

    pre_df = pd.read_parquet(PRECLASSIFIED_FILE).copy()
    pre_df["photo_id"] = pre_df["photo_id"].astype(str)

    ollama_df = load_ollama_results_from_jsonl()

    if len(ollama_df) == 0:
        merged = pre_df.copy()
        for col in [
            "ollama_label",
            "ollama_confidence",
            "ollama_reason",
            "ollama_ok",
            "ollama_error",
        ]:
            if col not in merged.columns:
                merged[col] = pd.NA
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

    use_ollama = merged["needs_llm"] == True
    merged["ollama_confidence"] = pd.to_numeric(merged["ollama_confidence"], errors="coerce")
    merged["category_confidence"] = pd.to_numeric(merged["category_confidence"], errors="coerce")

    merged.loc[use_ollama, "category"] = merged.loc[use_ollama, "ollama_label"]
    merged.loc[use_ollama, "category_confidence"] = merged.loc[use_ollama, "ollama_confidence"]
    merged.loc[use_ollama, "category_source"] = "ollama"

    failed = use_ollama & merged["category"].isna()
    merged.loc[failed, "category"] = merged.loc[failed, "rule_top1_label"]
    merged.loc[failed, "category_confidence"] = 0.2
    merged.loc[failed, "category_source"] = "fallback_rule"

    merged["rule_min_conf_threshold"] = merged["category"].map(get_rule_threshold)
    merged["ollama_min_conf_threshold"] = merged["category"].map(get_ollama_threshold)

    merged["effective_min_conf_threshold"] = DEFAULT_RULE_MIN_CONFIDENCE
    merged.loc[merged["category_source"] == "rule", "effective_min_conf_threshold"] = merged.loc[
        merged["category_source"] == "rule", "rule_min_conf_threshold"
    ]
    merged.loc[merged["category_source"] == "ollama", "effective_min_conf_threshold"] = merged.loc[
        merged["category_source"] == "ollama", "ollama_min_conf_threshold"
    ]
    merged.loc[merged["category_source"] == "fallback_rule", "effective_min_conf_threshold"] = 1.0

    merged["review_flag"] = (
        pd.to_numeric(merged["category_confidence"], errors="coerce").fillna(0)
        < pd.to_numeric(merged["effective_min_conf_threshold"], errors="coerce").fillna(1.0)
    )

    merged["reject_reason"] = pd.NA
    merged.loc[merged["category"].isna(), "reject_reason"] = "missing_category"
    merged.loc[merged["category_source"] == "fallback_rule", "reject_reason"] = "fallback_rule"

    merged.loc[
        (merged["category_source"] == "ollama")
        & (
            merged["ollama_confidence"].fillna(0)
            < pd.to_numeric(merged["ollama_min_conf_threshold"], errors="coerce").fillna(DEFAULT_OLLAMA_MIN_CONFIDENCE)
        ),
        "reject_reason",
    ] = "low_ollama_confidence"

    merged.loc[
        (merged["category_source"] == "rule")
        & (
            merged["category_confidence"].fillna(0)
            < pd.to_numeric(merged["rule_min_conf_threshold"], errors="coerce").fillna(DEFAULT_RULE_MIN_CONFIDENCE)
        ),
        "reject_reason",
    ] = "low_rule_confidence"

    final_keep = (
        merged["category"].notna()
        & (merged["review_flag"] == False)
        & (merged["category_source"] != "fallback_rule")
        & (
            (
                (merged["category_source"] == "rule")
                & (
                    merged["category_confidence"].fillna(0)
                    >= pd.to_numeric(merged["rule_min_conf_threshold"], errors="coerce").fillna(DEFAULT_RULE_MIN_CONFIDENCE)
                )
            )
            | (
                (merged["category_source"] == "ollama")
                & (
                    merged["ollama_confidence"].fillna(0)
                    >= pd.to_numeric(merged["ollama_min_conf_threshold"], errors="coerce").fillna(DEFAULT_OLLAMA_MIN_CONFIDENCE)
                )
            )
        )
    )

    final_df = merged[final_keep].copy()
    rejected_df = merged[~final_keep].copy()

    final_df.to_parquet(CLASSIFIED_FILE, index=False)
    merged.to_parquet(CLASSIFIED_ALL_FILE, index=False)
    rejected_df.to_parquet(NEED_REVIEW_FILE, index=False)
    rejected_df.to_parquet(REJECTED_FILE, index=False)

    stats_all = (
        merged.groupby(["category", "category_source"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["count", "category"], ascending=[False, True])
    )

    stats_final = (
        final_df.groupby(["category", "category_source"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["count", "category"], ascending=[False, True])
    )

    reject_stats = (
        rejected_df.groupby(["category", "category_source", "reject_reason"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["count", "category"], ascending=[False, True])
    )

    all_stats_path = safe_to_csv(stats_all, ALL_STATS_FILE)
    final_stats_path = safe_to_csv(stats_final, STATS_FILE)
    reject_stats_path = safe_to_csv(
        reject_stats,
        STATS_FILE.with_name("category_reject_stats.csv"),
    )

    print(f"[INFO] Ollama 结果来源：{OLLAMA_RESULTS_JSONL}")
    print(f"[INFO] 已读取 Ollama 唯一 photo_id 数：{len(ollama_df)}")
    print("[INFO] 分类别 Ollama 阈值：")
    for k, v in CATEGORY_OLLAMA_MIN_CONFIDENCE.items():
        print(f"  - {k}: {v:.2f}")
    print()
    print("[INFO] 分类别 Rule 阈值：")
    for k, v in CATEGORY_RULE_MIN_CONFIDENCE.items():
        print(f"  - {k}: {v:.2f}")

    print(f"\n[OK] 全量合并结果：{CLASSIFIED_ALL_FILE}")
    print(f"[OK] 最终过滤后分类文件：{CLASSIFIED_FILE}")
    print(f"[OK] 被剔除/待复核文件：{NEED_REVIEW_FILE}")
    print(f"[OK] 全量统计：{all_stats_path}")
    print(f"[OK] 最终统计：{final_stats_path}")
    print(f"[OK] 剔除原因统计：{reject_stats_path}")
    print()
    print(f"[SUMMARY] preclassified 总数: {len(pre_df)}")
    print(f"[SUMMARY] final classified 总数: {len(final_df)}")
    print(f"[SUMMARY] rejected / review 总数: {len(rejected_df)}")

    if len(stats_final) > 0:
        print("\n最终保留样本分布：")
        print(stats_final.to_string(index=False))


if __name__ == "__main__":
    main()
