#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="预览 parquet 文件内容")
    parser.add_argument("--input", required=True, help="parquet 文件路径")
    parser.add_argument("--head", type=int, default=10, help="预览前几行，默认 10")
    parser.add_argument(
        "--columns",
        nargs="*",
        default=None,
        help="只预览指定列，例如 --columns photo_id category",
    )
    parser.add_argument(
        "--show-schema",
        action="store_true",
        help="显示 parquet schema 和列类型",
    )
    parser.add_argument(
        "--show-stats",
        action="store_true",
        help="显示基本统计信息（行数、列数、文件大小）",
    )
    return parser.parse_args()


def sizeof_fmt(num: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(num) < 1024.0:
            return f"{num:.2f} {unit}"
        num /= 1024.0
    return f"{num:.2f} PB"


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)

    if not input_path.exists():
        print(f"[ERROR] 文件不存在：{input_path}", file=sys.stderr)
        return 1

    try:
        pf = pq.ParquetFile(input_path)
    except Exception as e:
        print(f"[ERROR] 无法读取 parquet metadata：{e}", file=sys.stderr)
        return 1

    if args.show_stats:
        print("=== Basic Stats ===")
        print(f"path       : {input_path}")
        print(f"file size  : {sizeof_fmt(input_path.stat().st_size)}")
        print(f"rows       : {pf.metadata.num_rows}")
        print(f"columns    : {pf.metadata.num_columns}")
        print(f"row groups : {pf.metadata.num_row_groups}")
        print()

    if args.show_schema:
        print("=== Schema ===")
        schema = pf.schema_arrow
        for name, dtype in zip(schema.names, schema.types):
            print(f"{name}: {dtype}")
        print()

    try:
        if args.columns:
            df = pd.read_parquet(input_path, columns=args.columns)
        else:
            df = pd.read_parquet(input_path)
    except Exception as e:
        print(f"[ERROR] 读取 parquet 数据失败：{e}", file=sys.stderr)
        return 1

    print("=== DataFrame Info ===")
    print(f"shape: {df.shape}")
    print("columns:")
    for c in df.columns:
        print(f"  - {c} ({df[c].dtype})")
    print()

    print(f"=== Head({args.head}) ===")
    with pd.option_context(
        "display.max_columns", None,
        "display.width", 200,
        "display.max_colwidth", 120,
    ):
        print(df.head(args.head).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
