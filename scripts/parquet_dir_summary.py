#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pandas as pd


def sizeof_fmt(num: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(num) < 1024.0:
            return f"{num:.2f} {unit}"
        num /= 1024.0
    return f"{num:.2f} PB"


def summarize_parquet(file_path: Path) -> dict[str, Any]:
    pf = pq.ParquetFile(file_path)
    meta = pf.metadata
    schema = pf.schema_arrow

    col_names = schema.names
    col_types = {field.name: str(field.type) for field in schema}

    return {
        "file_name": file_path.name,
        "file_path": str(file_path.resolve()),
        "file_size_bytes": file_path.stat().st_size,
        "file_size_human": sizeof_fmt(file_path.stat().st_size),
        "num_rows": meta.num_rows,
        "num_columns": len(col_names),
        "num_row_groups": meta.num_row_groups,
        "columns": col_names,
        "column_types": col_types,
        "created_by": meta.created_by,
        "format_version": meta.format_version,
    }


def find_parquet_files(root: Path, recursive: bool) -> list[Path]:
    pattern = "**/*.parquet" if recursive else "*.parquet"
    return sorted(root.glob(pattern))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan a directory and summarize all Parquet files."
    )
    parser.add_argument("input_dir", help="Directory that contains parquet files")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search subdirectories",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="",
        help="Optional path to save a flat CSV summary",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Optional path to save the full JSON summary",
    )
    parser.add_argument(
        "--show-columns",
        action="store_true",
        help="Print column names and types for each parquet file",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"[ERROR] Invalid directory: {input_dir}")

    parquet_files = find_parquet_files(input_dir, recursive=args.recursive)
    if not parquet_files:
        raise SystemExit(f"[INFO] No parquet files found in: {input_dir}")

    summaries: list[dict[str, Any]] = []
    for path in parquet_files:
        try:
            info = summarize_parquet(path)
            summaries.append(info)
        except Exception as e:
            summaries.append(
                {
                    "file_name": path.name,
                    "file_path": str(path.resolve()),
                    "error": str(e),
                }
            )

    print(f"Found {len(summaries)} parquet file(s) in: {input_dir.resolve()}\n")
    for item in summaries:
        print(f"File: {item['file_name']}")
        print(f"  Path: {item['file_path']}")
        if "error" in item:
            print(f"  Error: {item['error']}\n")
            continue
        print(f"  Rows: {item['num_rows']}")
        print(f"  Columns: {item['num_columns']}")
        print(f"  Row groups: {item['num_row_groups']}")
        print(f"  Size: {item['file_size_human']} ({item['file_size_bytes']} bytes)")
        print(f"  Format version: {item['format_version']}")
        print(f"  Created by: {item['created_by']}")
        if args.show_columns:
            print("  Schema:")
            for k, v in item["column_types"].items():
                print(f"    - {k}: {v}")
        print()

    if args.output_csv:
        flat_rows = []
        for item in summaries:
            base = {
                "file_name": item.get("file_name"),
                "file_path": item.get("file_path"),
                "file_size_bytes": item.get("file_size_bytes"),
                "file_size_human": item.get("file_size_human"),
                "num_rows": item.get("num_rows"),
                "num_columns": item.get("num_columns"),
                "num_row_groups": item.get("num_row_groups"),
                "format_version": item.get("format_version"),
                "created_by": item.get("created_by"),
                "columns": ", ".join(item.get("columns", [])) if item.get("columns") else "",
                "error": item.get("error", ""),
            }
            flat_rows.append(base)
        pd.DataFrame(flat_rows).to_csv(args.output_csv, index=False, encoding="utf-8-sig")
        print(f"[OK] CSV summary saved to: {Path(args.output_csv).resolve()}")

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] JSON summary saved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
