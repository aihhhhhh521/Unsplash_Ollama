#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
inspect_preclassified_by_photoid.py

用途：
- 输入一个 photo_id
- 从 preclassified.parquet 中查找该样本
- 输出规则判断时会用到的 description / keywords / location / landmark 等信息
- 同时打印 text_for_cls、rule 结果，便于排查为什么被分到某个类别
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.dataset as ds


DEFAULT_INPUT = Path(r"D:/PyProjects/Dataset/unsplash-research-dataset-full-latest/work/preclassified.parquet")


def safe_str(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def to_builtin(x: Any):
    """把 pandas / numpy / Arrow 标量安全转成 json 可序列化的 Python 内建类型。"""
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass

    # pandas/numpy 标量常见都有 item()
    if hasattr(x, "item"):
        try:
            return x.item()
        except Exception:
            pass

    if isinstance(x, dict):
        return {str(k): to_builtin(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [to_builtin(v) for v in x]

    # 再兜底一次：尽量保留原值，json.dumps 时 default=str
    return x


def split_pipe_text(text: Any) -> list[str]:
    s = safe_str(text)
    if not s:
        return []
    parts = [p.strip() for p in s.split("|")]
    out = []
    seen = set()
    for p in parts:
        if not p:
            continue
        low = p.lower()
        if low in seen:
            continue
        seen.add(low)
        out.append(p)
    return out


def load_one_row(input_file: Path, photo_id: str) -> pd.Series:
    if not input_file.exists():
        raise FileNotFoundError(f"找不到输入文件：{input_file}")

    dataset = ds.dataset(str(input_file), format="parquet")
    table = dataset.to_table(filter=(ds.field("photo_id") == str(photo_id)))
    df = table.to_pandas()

    if df.empty:
        raise KeyError(f"在 {input_file} 中未找到 photo_id={photo_id}")

    if len(df) > 1:
        df = df.drop_duplicates(subset=["photo_id"], keep="first")

    return df.iloc[0]


def collect_payload(row: pd.Series) -> dict[str, Any]:
    payload = {
        "photo_id": safe_str(row.get("photo_id")),
        "category": safe_str(row.get("category")),
        "category_confidence": to_builtin(row.get("category_confidence")),
        "category_source": safe_str(row.get("category_source")),
        "needs_llm": to_builtin(row.get("needs_llm")),
        "rule_gate_pass": to_builtin(row.get("rule_gate_pass")),
        "rule_reject_reason": safe_str(row.get("rule_reject_reason")),
        "rule_top1_label": safe_str(row.get("rule_top1_label")),
        "rule_top1_score": to_builtin(row.get("rule_top1_score")),
        "rule_top2_score": to_builtin(row.get("rule_top2_score")),
        "rule_margin": to_builtin(row.get("rule_margin")),
        "photo_description": safe_str(row.get("photo_description")),
        "ai_description": safe_str(row.get("ai_description")),
        "photo_location_name": safe_str(row.get("photo_location_name")),
        "photo_location_city": safe_str(row.get("photo_location_city")),
        "photo_location_country": safe_str(row.get("photo_location_country")),
        "ai_primary_landmark_name": safe_str(row.get("ai_primary_landmark_name")),
        "keyword_count": to_builtin(row.get("keyword_count")),
        "keyword_user_count": to_builtin(row.get("keyword_user_count")),
        "keyword_ai_max_conf": to_builtin(row.get("keyword_ai_max_conf")),
        "keywords_user_joined": safe_str(row.get("keywords_user_joined")),
        "keywords_highconf_joined": safe_str(row.get("keywords_highconf_joined")),
        "keywords_joined": safe_str(row.get("keywords_joined")),
        "conv_keywords_joined": safe_str(row.get("conv_keywords_joined")),
        "text_for_cls": safe_str(row.get("text_for_cls")),
        "rule_scores_json": safe_str(row.get("rule_scores_json")),
        "rule_candidate_labels_json": safe_str(row.get("rule_candidate_labels_json")),
        "rule_matched_terms_json": safe_str(row.get("rule_matched_terms_json")),
    }
    return payload


def try_parse_json(text: str):
    text = safe_str(text)
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return text


def print_section(title: str, content: Any = None):
    print(f"\n===== {title} =====")
    if content is None:
        return
    if isinstance(content, list):
        if not content:
            print("(空)")
        else:
            for i, x in enumerate(content, start=1):
                print(f"{i:02d}. {to_builtin(x)}")
        return
    if isinstance(content, dict):
        if not content:
            print("(空)")
        else:
            print(json.dumps(to_builtin(content), ensure_ascii=False, indent=2, default=str))
        return
    text = safe_str(content)
    print(text if text else "(空)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("photo_id", help="要查询的 photo_id")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="preclassified.parquet 路径")
    parser.add_argument("--json", action="store_true", help="以 JSON 形式输出")
    args = parser.parse_args()

    row = load_one_row(args.input, str(args.photo_id))
    payload = collect_payload(row)

    if args.json:
        payload["keywords_user_list"] = split_pipe_text(payload["keywords_user_joined"])
        payload["keywords_highconf_list"] = split_pipe_text(payload["keywords_highconf_joined"])
        payload["keywords_all_list"] = split_pipe_text(payload["keywords_joined"])
        payload["conv_keywords_list"] = split_pipe_text(payload["conv_keywords_joined"])
        payload["rule_scores"] = try_parse_json(payload["rule_scores_json"])
        payload["rule_candidate_labels"] = try_parse_json(payload["rule_candidate_labels_json"])
        payload["rule_matched_terms"] = try_parse_json(payload["rule_matched_terms_json"])
        print(json.dumps(to_builtin(payload), ensure_ascii=False, indent=2, default=str))
        return

    print(f"[INFO] input_file = {args.input}")
    print(f"[INFO] photo_id = {payload['photo_id']}")

    print_section("基本判定结果", {
        "category": payload["category"],
        "category_confidence": payload["category_confidence"],
        "category_source": payload["category_source"],
        "needs_llm": payload["needs_llm"],
        "rule_gate_pass": payload["rule_gate_pass"],
        "rule_reject_reason": payload["rule_reject_reason"],
        "rule_top1_label": payload["rule_top1_label"],
        "rule_top1_score": payload["rule_top1_score"],
        "rule_top2_score": payload["rule_top2_score"],
        "rule_margin": payload["rule_margin"],
    })

    print_section("描述字段 photo_description", payload["photo_description"])
    print_section("描述字段 ai_description", payload["ai_description"])

    print_section("关键词 user", split_pipe_text(payload["keywords_user_joined"]))
    print_section("关键词 highconf", split_pipe_text(payload["keywords_highconf_joined"]))
    print_section("关键词 all", split_pipe_text(payload["keywords_joined"]))
    print_section("关键词 conv", split_pipe_text(payload["conv_keywords_joined"]))

    print_section("位置与地标", {
        "photo_location_name": payload["photo_location_name"],
        "photo_location_city": payload["photo_location_city"],
        "photo_location_country": payload["photo_location_country"],
        "ai_primary_landmark_name": payload["ai_primary_landmark_name"],
    })

    print_section("关键词统计", {
        "keyword_count": payload["keyword_count"],
        "keyword_user_count": payload["keyword_user_count"],
        "keyword_ai_max_conf": payload["keyword_ai_max_conf"],
    })

    print_section("text_for_cls（规则实际拼接文本）", payload["text_for_cls"])
    print_section("rule_scores_json", try_parse_json(payload["rule_scores_json"]))
    print_section("rule_candidate_labels_json", try_parse_json(payload["rule_candidate_labels_json"]))
    print_section("rule_matched_terms_json", try_parse_json(payload["rule_matched_terms_json"]))


if __name__ == "__main__":
    main()
