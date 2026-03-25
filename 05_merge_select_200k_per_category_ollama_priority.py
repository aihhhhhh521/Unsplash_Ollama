#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
05_merge_select_200k_per_category_ollama_priority.py

目标：
1. 保留当前 05_merge_and_review_jsonl_by_category_thresholds.py 中的 Rule / Ollama 阈值逻辑不变。
2. 先完成全量 merge，并得到“合格候选池（eligible）”。
3. 在 eligible 中做二次筛选：
   - 目标类别固定为 5 类：自然 / 城市、建筑 / 人像 / 室内 / 静物
   - 每个类别单独保留约 40k（上限 40k）
   - 宁缺毋滥：某类不足 40k 时，不跨类回填
   - 同一类别内：Ollama 优先，其次按置信度高优先，再按 downloads / views 做弱排序
4. 输出：
   - classified.parquet                最终选中的样本（理论上最多约 200k）
   - classified_all.parquet            全量 merge + eligibility + 选中状态审计表
   - need_review.parquet               未入选样本（含不达阈值 + 达阈值但因类别配额未入选）
   - rejected_or_low_conf.parquet      同上，便于兼容旧链路
   - category_stats.csv                最终选中分布
   - category_stats_all.csv            全量 merge 后分布
   - category_reject_stats.csv         未入选原因统计
   - category_selection_plan.csv       类别配额执行情况
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
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


# =========================
# 目标规模与类别配置
# =========================
TARGET_PER_CATEGORY = 40_000
TARGET_CATEGORIES = ["自然", "城市、建筑", "人像", "室内", "静物"]


# =========================
# 阈值配置：保持与当前 05 一致
# =========================
DEFAULT_RULE_MIN_CONFIDENCE = float(REVIEW_CONFIDENCE_THRESHOLD)
CATEGORY_RULE_MIN_CONFIDENCE = {
    "自然": 0.98,
    "城市、建筑": 0.95,
    "人像": 0.80,
    "室内": 0.70,
    "静物": 0.80,
}

DEFAULT_OLLAMA_MIN_CONFIDENCE = 0.75
CATEGORY_OLLAMA_MIN_CONFIDENCE = {
    "自然": 0.90,
    "城市、建筑": 0.85,
    "人像": 0.60,
    "室内": 0.60,
    "静物": 0.50,
}


OLLAMA_KEEP_COLS = [
    "photo_id",
    "ollama_label",
    "ollama_confidence",
    "ollama_reason",
    "ollama_ok",
    "ollama_error",
    "seq",
]

CLASSIFIED_ALL_FILE = CLASSIFIED_FILE.with_name("classified_all.parquet")
REJECTED_FILE = CLASSIFIED_FILE.with_name("rejected_or_low_conf.parquet")
ALL_STATS_FILE = STATS_FILE.with_name("category_stats_all.csv")
REJECT_STATS_FILE = STATS_FILE.with_name("category_reject_stats.csv")
SELECTION_PLAN_FILE = STATS_FILE.with_name("category_selection_plan.csv")


def safe_to_csv(df: pd.DataFrame, path: Path) -> Path:
    try:
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return path
    except PermissionError:
        alt = path.with_name(f"{path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{path.suffix}")
        df.to_csv(alt, index=False, encoding="utf-8-sig")
        print(f"[WARN] 目标文件被占用，已改存到: {alt}")
        return alt


def get_rule_threshold(category) -> float:
    if pd.isna(category):
        return DEFAULT_RULE_MIN_CONFIDENCE
    return float(CATEGORY_RULE_MIN_CONFIDENCE.get(str(category), DEFAULT_RULE_MIN_CONFIDENCE))


def get_ollama_threshold(category) -> float:
    if pd.isna(category):
        return DEFAULT_OLLAMA_MIN_CONFIDENCE
    return float(CATEGORY_OLLAMA_MIN_CONFIDENCE.get(str(category), DEFAULT_OLLAMA_MIN_CONFIDENCE))


def load_ollama_results_from_jsonl() -> pd.DataFrame:
    ensure_exists(OLLAMA_RESULTS_JSONL, "请先运行 04_ollama_classify.py，至少生成 ollama_results.jsonl")

    if OLLAMA_RESULTS_JSONL.stat().st_size == 0:
        return pd.DataFrame(columns=OLLAMA_KEEP_COLS)

    records = []
    with OLLAMA_RESULTS_JSONL.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            obj["_line_no"] = line_no
            records.append(obj)

    if not records:
        return pd.DataFrame(columns=OLLAMA_KEEP_COLS)

    ollama_df = pd.DataFrame(records)
    for col in OLLAMA_KEEP_COLS:
        if col not in ollama_df.columns:
            ollama_df[col] = pd.NA

    ollama_df = ollama_df[[*OLLAMA_KEEP_COLS, "_line_no"]].copy()
    ollama_df["photo_id"] = ollama_df["photo_id"].astype(str)
    ollama_df["ollama_confidence"] = pd.to_numeric(ollama_df["ollama_confidence"], errors="coerce")
    ollama_df["ollama_ok"] = ollama_df["ollama_ok"].astype("boolean")
    ollama_df["seq"] = pd.to_numeric(ollama_df["seq"], errors="coerce")

    # 不强依赖 seq；直接按 JSONL 中最后一次出现为准
    ollama_df = ollama_df.sort_values("_line_no")
    ollama_df = ollama_df.drop_duplicates(subset=["photo_id"], keep="last")
    return ollama_df.drop(columns=["_line_no"])


def build_merged_table(pre_df: pd.DataFrame, ollama_df: pd.DataFrame) -> pd.DataFrame:
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

    merged["ollama_confidence"] = pd.to_numeric(merged.get("ollama_confidence"), errors="coerce")
    merged["category_confidence"] = pd.to_numeric(merged.get("category_confidence"), errors="coerce")
    merged["stats_downloads"] = pd.to_numeric(merged.get("stats_downloads"), errors="coerce")
    merged["stats_views"] = pd.to_numeric(merged.get("stats_views"), errors="coerce")

    use_ollama = merged["needs_llm"] == True

    # 默认保留 03 的 rule 结果
    merged["category_final"] = merged.get("category")
    merged["category_confidence_final"] = merged.get("category_confidence")
    merged["category_source_final"] = merged.get("category_source")

    # needs_llm 样本：有有效 ollama 返回则严格优先使用 ollama
    has_ollama_label = merged["ollama_label"].notna() & (merged["ollama_label"].astype(str).str.strip() != "")
    use_ollama_effective = use_ollama & has_ollama_label
    merged.loc[use_ollama_effective, "category_final"] = merged.loc[use_ollama_effective, "ollama_label"]
    merged.loc[use_ollama_effective, "category_confidence_final"] = merged.loc[use_ollama_effective, "ollama_confidence"]
    merged.loc[use_ollama_effective, "category_source_final"] = "ollama"

    # needs_llm 但没有有效 ollama 标签：回退仅用于审计，不进入最终保留
    fallback_mask = use_ollama & ~has_ollama_label
    merged.loc[fallback_mask, "category_final"] = merged.loc[fallback_mask, "rule_top1_label"]
    merged.loc[fallback_mask, "category_confidence_final"] = 0.20
    merged.loc[fallback_mask, "category_source_final"] = "fallback_rule"

    return merged


def mark_eligibility(merged: pd.DataFrame) -> pd.DataFrame:
    merged = merged.copy()

    merged["rule_min_conf_threshold"] = merged["category_final"].map(get_rule_threshold)
    merged["ollama_min_conf_threshold"] = merged["category_final"].map(get_ollama_threshold)

    merged["effective_min_conf_threshold"] = DEFAULT_RULE_MIN_CONFIDENCE
    rule_mask = merged["category_source_final"] == "rule"
    ollama_mask = merged["category_source_final"] == "ollama"
    fallback_mask = merged["category_source_final"] == "fallback_rule"

    merged.loc[rule_mask, "effective_min_conf_threshold"] = merged.loc[rule_mask, "rule_min_conf_threshold"]
    merged.loc[ollama_mask, "effective_min_conf_threshold"] = merged.loc[ollama_mask, "ollama_min_conf_threshold"]
    merged.loc[fallback_mask, "effective_min_conf_threshold"] = 1.0

    merged["reject_reason"] = pd.NA
    merged.loc[merged["category_final"].isna(), "reject_reason"] = "missing_category"
    merged.loc[fallback_mask, "reject_reason"] = "fallback_rule"

    low_ollama = ollama_mask & (
        merged["ollama_confidence"].fillna(0)
        < pd.to_numeric(merged["ollama_min_conf_threshold"], errors="coerce").fillna(DEFAULT_OLLAMA_MIN_CONFIDENCE)
    )
    merged.loc[low_ollama, "reject_reason"] = "low_ollama_confidence"

    low_rule = rule_mask & (
        merged["category_confidence_final"].fillna(0)
        < pd.to_numeric(merged["rule_min_conf_threshold"], errors="coerce").fillna(DEFAULT_RULE_MIN_CONFIDENCE)
    )
    merged.loc[low_rule, "reject_reason"] = "low_rule_confidence"

    merged["eligible"] = False
    merged.loc[
        rule_mask
        & merged["category_final"].notna()
        & ~low_rule,
        "eligible",
    ] = True
    merged.loc[
        ollama_mask
        & merged["category_final"].notna()
        & ~low_ollama,
        "eligible",
    ] = True

    # 仅保留目标 5 类；其他类别不纳入最终 40k/类配额
    non_target_mask = merged["category_final"].notna() & ~merged["category_final"].astype(str).isin(TARGET_CATEGORIES)
    merged.loc[non_target_mask, "eligible"] = False
    merged.loc[non_target_mask & merged["reject_reason"].isna(), "reject_reason"] = "non_target_category"

    merged["effective_confidence"] = np.where(
        merged["category_source_final"].eq("ollama"),
        merged["ollama_confidence"],
        merged["category_confidence_final"],
    )
    merged["effective_confidence"] = pd.to_numeric(merged["effective_confidence"], errors="coerce")

    merged["source_priority"] = np.where(merged["category_source_final"].eq("ollama"), 1, 0)
    merged["selection_status"] = np.where(merged["eligible"], "eligible", "rejected")

    return merged


def sort_for_priority(df: pd.DataFrame, by_category: bool) -> pd.DataFrame:
    sort_cols = []
    ascending = []
    if by_category:
        sort_cols.append("category_final")
        ascending.append(True)

    sort_cols.extend([
        "source_priority",      # Ollama 优先
        "effective_confidence", # 置信度高优先
        "stats_downloads",      # 次级弱排序
        "stats_views",
        "photo_id",
    ])
    ascending.extend([False, False, False, False, True])

    return df.sort_values(sort_cols, ascending=ascending, kind="mergesort")


def select_topk_per_category(eligible_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eligible_df = eligible_df.copy()
    if len(eligible_df) == 0:
        empty_plan = pd.DataFrame(
            columns=[
                "category",
                "available",
                "target_quota",
                "selected",
                "shortfall",
            ]
        )
        return eligible_df, eligible_df, empty_plan

    eligible_sorted = sort_for_priority(eligible_df, by_category=True)
    selected_parts = []
    leftover_parts = []
    plan_rows = []

    for cat in TARGET_CATEGORIES:
        cat_df = eligible_sorted[eligible_sorted["category_final"] == cat].copy()
        available = int(len(cat_df))
        take_n = min(available, TARGET_PER_CATEGORY)
        shortfall = max(TARGET_PER_CATEGORY - available, 0)

        chosen = cat_df.iloc[:take_n].copy()
        if len(chosen) > 0:
            chosen["selection_stage"] = "category_quota"
            chosen["selected_final"] = True
            chosen["rank_in_selected_category"] = np.arange(1, len(chosen) + 1)
            selected_parts.append(chosen)

        leftover = cat_df.iloc[take_n:].copy()
        if len(leftover) > 0:
            leftover["selection_stage"] = "not_selected_by_category_quota"
            leftover["selected_final"] = False
            leftover_parts.append(leftover)

        plan_rows.append(
            {
                "category": cat,
                "available": available,
                "target_quota": int(TARGET_PER_CATEGORY),
                "selected": int(take_n),
                "shortfall": int(shortfall),
            }
        )

    selected_df = pd.concat(selected_parts, ignore_index=True) if selected_parts else eligible_df.iloc[0:0].copy()
    leftover_df = pd.concat(leftover_parts, ignore_index=True) if leftover_parts else eligible_df.iloc[0:0].copy()
    plan_df = pd.DataFrame(plan_rows)

    if len(selected_df) > 0:
        selected_df = sort_for_priority(selected_df, by_category=True).copy()

    if len(leftover_df) > 0:
        leftover_df = sort_for_priority(leftover_df, by_category=True).copy()

    return selected_df, leftover_df, plan_df


def build_stats(merged: pd.DataFrame, final_df: pd.DataFrame, rejected_df: pd.DataFrame):
    stats_all = (
        merged.groupby(["category_final", "category_source_final", "eligible"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["count", "category_final"], ascending=[False, True])
    )

    stats_final = (
        final_df.groupby(["category_final", "category_source_final", "selection_stage"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["count", "category_final"], ascending=[False, True])
    )

    reject_stats = (
        rejected_df.groupby(["category_final", "category_source_final", "reject_reason", "selection_stage"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["count", "category_final"], ascending=[False, True])
    )

    return stats_all, stats_final, reject_stats


def main() -> None:
    ensure_exists(PRECLASSIFIED_FILE, "请先运行 03_rule_preclassify_enriched.py")

    pre_df = pd.read_parquet(PRECLASSIFIED_FILE).copy()
    pre_df["photo_id"] = pre_df["photo_id"].astype(str)

    ollama_df = load_ollama_results_from_jsonl()
    merged = build_merged_table(pre_df, ollama_df)
    merged = mark_eligibility(merged)

    eligible_df = merged[merged["eligible"] == True].copy()
    selected_df, leftover_df, plan_df = select_topk_per_category(eligible_df)

    selected_ids = set(selected_df["photo_id"].astype(str)) if len(selected_df) > 0 else set()

    merged["selected_final"] = merged["photo_id"].astype(str).isin(selected_ids)
    merged.loc[merged["selected_final"], "selection_status"] = "selected"

    # 已达阈值但超出该类别 40k 配额的样本
    eligible_not_selected_mask = (merged["eligible"] == True) & (merged["selected_final"] == False)
    merged.loc[eligible_not_selected_mask & merged["reject_reason"].isna(), "reject_reason"] = "not_selected_by_category_quota"
    merged.loc[eligible_not_selected_mask, "selection_status"] = "eligible_but_not_selected"
    merged.loc[eligible_not_selected_mask, "selection_stage"] = "not_selected_by_category_quota"

    if len(selected_df) > 0:
        stage_map = selected_df.set_index("photo_id")["selection_stage"].to_dict()
        rank_map = selected_df.set_index("photo_id")["rank_in_selected_category"].to_dict()
        merged.loc[merged["selected_final"], "selection_stage"] = merged.loc[
            merged["selected_final"], "photo_id"
        ].map(stage_map)
        merged.loc[merged["selected_final"], "rank_in_selected_category"] = merged.loc[
            merged["selected_final"], "photo_id"
        ].map(rank_map)

    final_df = merged[merged["selected_final"] == True].copy()
    rejected_df = merged[merged["selected_final"] == False].copy()

    final_df = sort_for_priority(final_df, by_category=True)
    rejected_df = sort_for_priority(rejected_df, by_category=True)

    final_df.to_parquet(CLASSIFIED_FILE, index=False)
    merged.to_parquet(CLASSIFIED_ALL_FILE, index=False)
    rejected_df.to_parquet(NEED_REVIEW_FILE, index=False)
    rejected_df.to_parquet(REJECTED_FILE, index=False)

    stats_all, stats_final, reject_stats = build_stats(merged, final_df, rejected_df)

    all_stats_path = safe_to_csv(stats_all, ALL_STATS_FILE)
    final_stats_path = safe_to_csv(stats_final, STATS_FILE)
    reject_stats_path = safe_to_csv(reject_stats, REJECT_STATS_FILE)
    plan_path = safe_to_csv(plan_df, SELECTION_PLAN_FILE)

    print(f"[INFO] Ollama 结果来源：{OLLAMA_RESULTS_JSONL}")
    print(f"[INFO] 已读取 Ollama 唯一 photo_id 数：{len(ollama_df)}")
    print(f"[INFO] 每类别目标样本数：{TARGET_PER_CATEGORY}")
    print(f"[INFO] 目标类别：{', '.join(TARGET_CATEGORIES)}")
    print(f"[INFO] 理论最大总样本数：{TARGET_PER_CATEGORY * len(TARGET_CATEGORIES)}")
    print("[INFO] Ollama 阈值保持不变：")
    for k, v in CATEGORY_OLLAMA_MIN_CONFIDENCE.items():
        print(f"  - {k}: {v:.2f}")
    print("[INFO] Rule 阈值：")
    for k, v in CATEGORY_RULE_MIN_CONFIDENCE.items():
        print(f"  - {k}: {v:.2f}")

    print(f"\n[OK] 全量合并审计表：{CLASSIFIED_ALL_FILE}")
    print(f"[OK] 最终结果（每类最多 40k）：{CLASSIFIED_FILE}")
    print(f"[OK] 未入选/待复核：{NEED_REVIEW_FILE}")
    print(f"[OK] 全量统计：{all_stats_path}")
    print(f"[OK] 最终统计：{final_stats_path}")
    print(f"[OK] 未入选原因统计：{reject_stats_path}")
    print(f"[OK] 选样计划：{plan_path}")

    print()
    print(f"[SUMMARY] preclassified 总数: {len(pre_df)}")
    print(f"[SUMMARY] eligible 总数: {len(eligible_df)}")
    print(f"[SUMMARY] final selected 总数: {len(final_df)}")
    print(f"[SUMMARY] not selected / review 总数: {len(rejected_df)}")

    if len(plan_df) > 0:
        print("\n各类别配额执行情况：")
        print(plan_df.to_string(index=False))

    if len(stats_final) > 0:
        print("\n最终保留样本分布：")
        print(stats_final.to_string(index=False))


if __name__ == "__main__":
    main()
