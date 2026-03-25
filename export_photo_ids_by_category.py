#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="筛选 classified.parquet 的指定类别，并将 photo_id 保存为 txt。"
    )
    parser.add_argument("--input", required=True, help="classified.parquet 文件路径")
    parser.add_argument(
        "--category",
        nargs="+",
        required=True,
        help='要筛选的类别，可传一个或多个，例如："自然" "人像"',
    )
    parser.add_argument("--output", default=None, help="输出 txt 文件路径，单类别时推荐使用")
    parser.add_argument("--output-dir", default=None, help="输出目录，多类别时会分别生成 txt")
    parser.add_argument("--category-column", default="category", help='类别列名，默认 "category"')
    parser.add_argument("--id-column", default="photo_id", help='ID 列名，默认 "photo_id"')
    parser.add_argument("--drop-duplicates", action="store_true", help="导出前对 photo_id 去重")
    parser.add_argument("--sort-ids", action="store_true", help="导出前对 photo_id 排序")
    return parser.parse_args()


def sanitize_filename(name: str) -> str:
    bad = '<>:"/\\|?*'
    s = "".join("_" if ch in bad else ch for ch in str(name)).strip()
    return s or "unknown"


def export_one(
    df: pd.DataFrame,
    category: str,
    category_column: str,
    id_column: str,
    output_path: Path,
    drop_duplicates: bool = False,
    sort_ids: bool = False,
) -> int:
    sub = df[df[category_column] == category].copy()
    ids = sub[id_column].dropna().astype(str)

    if drop_duplicates:
        ids = ids.drop_duplicates()

    if sort_ids:
        ids = ids.sort_values(kind="stable")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(ids.tolist()), encoding="utf-8")
    return len(ids)


def main() -> int:
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] 输入文件不存在：{input_path}", file=sys.stderr)
        return 1

    df = pd.read_parquet(input_path)

    if args.category_column not in df.columns:
        print(f'[ERROR] 未找到类别列：{args.category_column}\n可用列：{list(df.columns)}', file=sys.stderr)
        return 1

    if args.id_column not in df.columns:
        print(f'[ERROR] 未找到 ID 列：{args.id_column}\n可用列：{list(df.columns)}', file=sys.stderr)
        return 1

    categories = args.category

    if len(categories) == 1:
        category = categories[0]
        if args.output:
            out_path = Path(args.output)
        else:
            out_dir = Path(args.output_dir) if args.output_dir else input_path.parent
            out_path = out_dir / f"{sanitize_filename(category)}_photo_ids.txt"

        count = export_one(
            df=df,
            category=category,
            category_column=args.category_column,
            id_column=args.id_column,
            output_path=out_path,
            drop_duplicates=args.drop_duplicates,
            sort_ids=args.sort_ids,
        )
        print(f"[OK] 类别：{category}")
        print(f"[OK] 数量：{count}")
        print(f"[OK] 已保存到：{out_path}")
        return 0

    out_dir = Path(args.output_dir) if args.output_dir else input_path.parent / "photo_id_lists"
    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for category in categories:
        out_path = out_dir / f"{sanitize_filename(category)}_photo_ids.txt"
        count = export_one(
            df=df,
            category=category,
            category_column=args.category_column,
            id_column=args.id_column,
            output_path=out_path,
            drop_duplicates=args.drop_duplicates,
            sort_ids=args.sort_ids,
        )
        total += count
        print(f"[OK] 类别：{category} | 数量：{count} | 文件：{out_path}")

    print(f"[DONE] 共导出 {len(categories)} 个类别，累计 {total} 条 photo_id。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
