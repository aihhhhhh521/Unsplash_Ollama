#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import duckdb

from config import DATA_ROOT, PHOTOS_NO_ART_FILE, WORK_DIR

TEMP_DIR = WORK_DIR / "_duckdb_tmp"
DUCKDB_MEMORY_LIMIT = "8GB"
DUCKDB_THREADS = 1
DUCKDB_MAX_TEMP_DIRECTORY_SIZE = "500GiB"

KEYWORDS_AGG_FILE = WORK_DIR / "keywords_agg.parquet"
CONVERSIONS_AGG_FILE = WORK_DIR / "conversions_agg.parquet"
PHOTOS_NO_ART_ENRICHED_FILE = WORK_DIR / "photos_no_art_enriched.parquet"

KW_PARTIAL_DIR = WORK_DIR / "kw_partials"
CV_PARTIAL_DIR = WORK_DIR / "cv_partials"


def init_duckdb() -> duckdb.DuckDBPyConnection:
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    db_path = WORK_DIR / "resume_from_partials.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute(f"SET temp_directory='{TEMP_DIR.as_posix()}';")
    con.execute(f"SET memory_limit='{DUCKDB_MEMORY_LIMIT}';")
    con.execute(f"SET threads={DUCKDB_THREADS};")
    con.execute("SET preserve_insertion_order=false;")
    con.execute(f"SET max_temp_directory_size='{DUCKDB_MAX_TEMP_DIRECTORY_SIZE}';")
    return con


def finalize_keywords_agg_resume(con: duckdb.DuckDBPyConnection) -> None:
    kw_files = sorted(KW_PARTIAL_DIR.glob("kw_partial_*.parquet"))
    if not kw_files:
        raise FileNotFoundError("未找到 kw_partials/*.parquet，请先运行 01b_aggregate_side_tables_lowmem.py 的第一步")

    partial_glob = f"{KW_PARTIAL_DIR.as_posix()}/*.parquet"
    print(f"[INFO] 复用 keyword partial 文件数: {len(kw_files)}")

    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE kw_dedup AS
    SELECT
        photo_id,
        kw,
        max(ai_conf) AS ai_conf,
        max(user_suggested) AS user_suggested
    FROM read_parquet('{partial_glob}')
    GROUP BY 1, 2;
    """)

    con.execute(f"""
    COPY (
        SELECT
            photo_id,
            count(*) AS keyword_count,
            sum(user_suggested) AS keyword_user_count,
            max(ai_conf) AS keyword_ai_max_conf,
            string_agg(kw, ' | ' ORDER BY user_suggested DESC, ai_conf DESC, kw) AS keywords_joined,
            string_agg(kw, ' | ' ORDER BY ai_conf DESC, kw)
                FILTER (WHERE user_suggested = 1 OR ai_conf >= 55) AS keywords_highconf_joined,
            string_agg(kw, ' | ' ORDER BY kw)
                FILTER (WHERE user_suggested = 1) AS keywords_user_joined
        FROM kw_dedup
        GROUP BY 1
    )
    TO '{KEYWORDS_AGG_FILE.as_posix()}'
    (FORMAT PARQUET, COMPRESSION ZSTD);
    """)

    print(f"[OK] 已生成 {KEYWORDS_AGG_FILE}")


def ensure_cv_partial_dir() -> None:
    CV_PARTIAL_DIR.mkdir(parents=True, exist_ok=True)


def build_conversions_partials_if_missing(con: duckdb.DuckDBPyConnection) -> None:
    existing = sorted(CV_PARTIAL_DIR.glob("cv_partial_*.parquet"))
    if existing:
        print(f"[INFO] 检测到已有 conversion partials，直接复用：{len(existing)} 个")
        return

    conversion_files = sorted(DATA_ROOT.glob("conversions.csv*"))
    if not conversion_files:
        raise FileNotFoundError("未找到任何 conversions.csv* 文件")

    ensure_cv_partial_dir()
    print(f"[INFO] matched conversion files: {len(conversion_files)}")

    for i, fp in enumerate(conversion_files):
        out = CV_PARTIAL_DIR / f"cv_partial_{i:03d}.parquet"
        print(f"[CV {i+1}/{len(conversion_files)}] {fp.name}")

        con.execute(f"""
        COPY (
            SELECT
                CAST(photo_id AS VARCHAR) AS photo_id,
                regexp_replace(lower(trim(CAST(keyword AS VARCHAR))), '\\s+', ' ', 'g') AS kw,
                count(*) AS cnt
            FROM read_csv_auto(
                '{fp.as_posix()}',
                union_by_name=true,
                ignore_errors=true
            )
            WHERE photo_id IS NOT NULL
              AND keyword IS NOT NULL
              AND length(trim(CAST(keyword AS VARCHAR))) BETWEEN 3 AND 80
              AND lower(trim(CAST(keyword AS VARCHAR))) NOT IN ('', 'null', 'none', 'n/a')
              AND regexp_matches(lower(trim(CAST(keyword AS VARCHAR))), '.*[a-zA-Z一-龥].*')
            GROUP BY 1, 2
        )
        TO '{out.as_posix()}'
        (FORMAT PARQUET, COMPRESSION ZSTD);
        """)

    print(f"[OK] 已生成 {len(conversion_files)} 个 conversion partial 文件")


def finalize_conversions_agg(con: duckdb.DuckDBPyConnection) -> None:
    cv_files = sorted(CV_PARTIAL_DIR.glob("cv_partial_*.parquet"))
    if not cv_files:
        raise FileNotFoundError("未找到 cv_partials/*.parquet")

    partial_glob = f"{CV_PARTIAL_DIR.as_posix()}/*.parquet"
    print(f"[INFO] 汇总 conversion partial 文件数: {len(cv_files)}")

    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE cv_counts AS
    SELECT
        photo_id,
        kw,
        sum(cnt) AS cnt
    FROM read_parquet('{partial_glob}')
    GROUP BY 1, 2;
    """)

    con.execute(f"""
    COPY (
        SELECT
            photo_id,
            sum(cnt) AS conv_event_count,
            count(*) AS conv_unique_keyword_count,
            string_agg(kw, ' | ' ORDER BY cnt DESC, kw) AS conv_keywords_joined,
            arg_max(kw, cnt) AS conv_top_keyword
        FROM cv_counts
        GROUP BY 1
    )
    TO '{CONVERSIONS_AGG_FILE.as_posix()}'
    (FORMAT PARQUET, COMPRESSION ZSTD);
    """)

    print(f"[OK] 已生成 {CONVERSIONS_AGG_FILE}")


def build_enriched_main(con: duckdb.DuckDBPyConnection) -> None:
    if not PHOTOS_NO_ART_FILE.exists():
        raise FileNotFoundError("未找到 photos_no_art.parquet，请先运行 02_filter_art.py")
    if not KEYWORDS_AGG_FILE.exists():
        raise FileNotFoundError("未找到 keywords_agg.parquet")
    if not CONVERSIONS_AGG_FILE.exists():
        raise FileNotFoundError("未找到 conversions_agg.parquet")

    con.execute(f"""
    COPY (
        SELECT
            p.*,
            k.keyword_count,
            k.keyword_user_count,
            k.keyword_ai_max_conf,
            k.keywords_joined,
            k.keywords_highconf_joined,
            k.keywords_user_joined,
            c.conv_event_count,
            c.conv_unique_keyword_count,
            c.conv_keywords_joined,
            c.conv_top_keyword
        FROM read_parquet('{PHOTOS_NO_ART_FILE.as_posix()}') p
        LEFT JOIN read_parquet('{KEYWORDS_AGG_FILE.as_posix()}') k
            ON CAST(p.photo_id AS VARCHAR) = k.photo_id
        LEFT JOIN read_parquet('{CONVERSIONS_AGG_FILE.as_posix()}') c
            ON CAST(p.photo_id AS VARCHAR) = c.photo_id
    )
    TO '{PHOTOS_NO_ART_ENRICHED_FILE.as_posix()}'
    (FORMAT PARQUET, COMPRESSION ZSTD);
    """)

    print(f"[OK] 已生成 {PHOTOS_NO_ART_ENRICHED_FILE}")


def main() -> None:
    print(f"[INFO] DATA_ROOT = {DATA_ROOT}")
    print(f"[INFO] PHOTOS_NO_ART_FILE = {PHOTOS_NO_ART_FILE}")
    print(f"[INFO] TEMP_DIR = {TEMP_DIR}")
    print(f"[INFO] KEYWORDS_AGG_FILE = {KEYWORDS_AGG_FILE}")
    print(f"[INFO] CONVERSIONS_AGG_FILE = {CONVERSIONS_AGG_FILE}")
    print(f"[INFO] PHOTOS_NO_ART_ENRICHED_FILE = {PHOTOS_NO_ART_ENRICHED_FILE}")

    con = init_duckdb()

    print("[1/4] 从现有 kw_partials 汇总 keywords_agg.parquet ...")
    finalize_keywords_agg_resume(con)

    print("[2/4] 生成 / 复用 conversion partials ...")
    build_conversions_partials_if_missing(con)

    print("[3/4] 汇总 conversions_agg.parquet ...")
    finalize_conversions_agg(con)

    print("[4/4] 合并回主表 ...")
    build_enriched_main(con)

    stats = con.execute(f"""
    SELECT
        (SELECT COUNT(*) FROM read_parquet('{PHOTOS_NO_ART_FILE.as_posix()}')) AS photos_no_art_rows,
        (SELECT COUNT(*) FROM read_parquet('{KEYWORDS_AGG_FILE.as_posix()}')) AS keyword_rows,
        (SELECT COUNT(*) FROM read_parquet('{CONVERSIONS_AGG_FILE.as_posix()}')) AS conversion_rows,
        (SELECT COUNT(*) FROM read_parquet('{PHOTOS_NO_ART_ENRICHED_FILE.as_posix()}')) AS enriched_rows
    """).fetchdf()

    print(stats.to_string(index=False))
    con.close()


if __name__ == "__main__":
    main()
