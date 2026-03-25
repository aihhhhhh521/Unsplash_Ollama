#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
01b_aggregate_side_tables_lowmem.py

低内存 / 低临时盘占用版本：
1. keywords.csv* / conversions.csv* 按分片逐个处理
2. 先生成每个分片的 partial parquet
3. 再把 partial parquet 汇总成最终聚合表
4. temp_directory 放到 WORK_DIR 所在盘，避免把 C 盘写满

输出：
- work/keywords_agg.parquet
- work/conversions_agg.parquet
- work/photos_no_art_enriched.parquet
"""

from __future__ import annotations

from pathlib import Path
import duckdb

from config import DATA_ROOT, PHOTOS_NO_ART_FILE, WORK_DIR

# ===== 可调参数 =====
TEMP_DIR = WORK_DIR / "_duckdb_tmp"          # 改到 D 盘 / work 目录所在盘
DUCKDB_MEMORY_LIMIT = "4GB"                  # 先保守一点
DUCKDB_THREADS = 2                           # 线程数调低，减少内存和临时盘压力
DUCKDB_MAX_TEMP_DIRECTORY_SIZE = "500GiB"    # 若目标盘空间足够，可再调大

KEYWORDS_AGG_FILE = WORK_DIR / "keywords_agg.parquet"
CONVERSIONS_AGG_FILE = WORK_DIR / "conversions_agg.parquet"
PHOTOS_NO_ART_ENRICHED_FILE = WORK_DIR / "photos_no_art_enriched.parquet"

KW_PARTIAL_DIR = WORK_DIR / "kw_partials"
CV_PARTIAL_DIR = WORK_DIR / "cv_partials"


def init_duckdb() -> duckdb.DuckDBPyConnection:
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    db_path = WORK_DIR / "aggregate_side_tables_lowmem.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute(f"SET temp_directory='{TEMP_DIR.as_posix()}';")
    con.execute(f"SET memory_limit='{DUCKDB_MEMORY_LIMIT}';")
    con.execute(f"SET threads={DUCKDB_THREADS};")
    con.execute("SET preserve_insertion_order=false;")
    con.execute(f"SET max_temp_directory_size='{DUCKDB_MAX_TEMP_DIRECTORY_SIZE}';")
    return con


def clear_partial_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for p in path.glob("*.parquet"):
        p.unlink()


def build_keywords_partials(con: duckdb.DuckDBPyConnection) -> None:
    keyword_files = sorted(DATA_ROOT.glob("keywords.csv*"))
    clear_partial_dir(KW_PARTIAL_DIR)

    if not keyword_files:
        raise FileNotFoundError("未找到任何 keywords.csv* 文件")

    print(f"[INFO] matched keyword files: {len(keyword_files)}")
    for i, fp in enumerate(keyword_files):
        out = KW_PARTIAL_DIR / f"kw_partial_{i:03d}.parquet"
        print(f"[KW {i+1}/{len(keyword_files)}] {fp.name}")

        con.execute(f"""
        COPY (
            SELECT
                CAST(photo_id AS VARCHAR) AS photo_id,
                regexp_replace(lower(trim(CAST(keyword AS VARCHAR))), '\\s+', ' ', 'g') AS kw,
                max(
                    greatest(
                        coalesce(try_cast(ai_service_1_confidence AS DOUBLE), -1),
                        coalesce(try_cast(ai_service_2_confidence AS DOUBLE), -1),
                        0
                    )
                ) AS ai_conf,
                max(
                    CASE
                        WHEN lower(trim(CAST(suggested_by_user AS VARCHAR))) IN ('t','true','1','yes','y')
                        THEN 1 ELSE 0
                    END
                ) AS user_suggested
            FROM read_csv_auto(
                '{fp.as_posix()}',
                union_by_name=true,
                ignore_errors=true
            )
            WHERE photo_id IS NOT NULL
              AND keyword IS NOT NULL
              AND length(trim(CAST(keyword AS VARCHAR))) BETWEEN 2 AND 80
              AND lower(trim(CAST(keyword AS VARCHAR))) NOT IN ('', 'null', 'none', 'n/a')
            GROUP BY 1, 2
        )
        TO '{out.as_posix()}'
        (FORMAT PARQUET, COMPRESSION ZSTD);
        """)


def finalize_keywords_agg(con: duckdb.DuckDBPyConnection) -> None:
    partial_glob = f"{KW_PARTIAL_DIR.as_posix()}/*.parquet"

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

    con.execute("""
    CREATE OR REPLACE TEMP TABLE kw_all_top AS
    SELECT photo_id, kw, ai_conf, user_suggested
    FROM (
        SELECT
            *,
            row_number() OVER (
                PARTITION BY photo_id
                ORDER BY user_suggested DESC, ai_conf DESC, kw
            ) AS rn
        FROM kw_dedup
    )
    WHERE rn <= 32;
    """)

    con.execute("""
    CREATE OR REPLACE TEMP TABLE kw_highconf_top AS
    SELECT photo_id, kw, ai_conf
    FROM (
        SELECT
            *,
            row_number() OVER (
                PARTITION BY photo_id
                ORDER BY ai_conf DESC, kw
            ) AS rn
        FROM kw_dedup
        WHERE user_suggested = 1 OR ai_conf >= 55
    )
    WHERE rn <= 24;
    """)

    con.execute("""
    CREATE OR REPLACE TEMP TABLE kw_user_top AS
    SELECT photo_id, kw
    FROM (
        SELECT
            *,
            row_number() OVER (
                PARTITION BY photo_id
                ORDER BY kw
            ) AS rn
        FROM kw_dedup
        WHERE user_suggested = 1
    )
    WHERE rn <= 24;
    """)

    con.execute(f"""
    COPY (
        WITH kw_all_agg AS (
            SELECT
                photo_id,
                count(*) AS keyword_count,
                max(ai_conf) AS keyword_ai_max_conf,
                sum(user_suggested) AS keyword_user_count,
                string_agg(kw, ' | ' ORDER BY user_suggested DESC, ai_conf DESC, kw) AS keywords_joined
            FROM kw_all_top
            GROUP BY 1
        ),
        kw_highconf_agg AS (
            SELECT
                photo_id,
                string_agg(kw, ' | ' ORDER BY ai_conf DESC, kw) AS keywords_highconf_joined
            FROM kw_highconf_top
            GROUP BY 1
        ),
        kw_user_agg AS (
            SELECT
                photo_id,
                string_agg(kw, ' | ' ORDER BY kw) AS keywords_user_joined
            FROM kw_user_top
            GROUP BY 1
        )
        SELECT
            a.photo_id,
            a.keyword_count,
            a.keyword_user_count,
            a.keyword_ai_max_conf,
            a.keywords_joined,
            h.keywords_highconf_joined,
            u.keywords_user_joined
        FROM kw_all_agg a
        LEFT JOIN kw_highconf_agg h USING (photo_id)
        LEFT JOIN kw_user_agg u USING (photo_id)
    )
    TO '{KEYWORDS_AGG_FILE.as_posix()}'
    (FORMAT PARQUET, COMPRESSION ZSTD);
    """)


def build_conversions_partials(con: duckdb.DuckDBPyConnection) -> None:
    conversion_files = sorted(DATA_ROOT.glob("conversions.csv*"))
    clear_partial_dir(CV_PARTIAL_DIR)

    if not conversion_files:
        raise FileNotFoundError("未找到任何 conversions.csv* 文件")

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


def finalize_conversions_agg(con: duckdb.DuckDBPyConnection) -> None:
    partial_glob = f"{CV_PARTIAL_DIR.as_posix()}/*.parquet"

    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE cv_counts AS
    SELECT
        photo_id,
        kw,
        sum(cnt) AS cnt
    FROM read_parquet('{partial_glob}')
    GROUP BY 1, 2;
    """)

    con.execute("""
    CREATE OR REPLACE TEMP TABLE cv_top AS
    SELECT photo_id, kw, cnt
    FROM (
        SELECT
            *,
            row_number() OVER (
                PARTITION BY photo_id
                ORDER BY cnt DESC, kw
            ) AS rn
        FROM cv_counts
    )
    WHERE rn <= 12;
    """)

    con.execute(f"""
    COPY (
        SELECT
            photo_id,
            sum(cnt) AS conv_event_count,
            count(*) AS conv_unique_keyword_count,
            string_agg(kw, ' | ' ORDER BY cnt DESC, kw) AS conv_keywords_joined,
            first(kw ORDER BY cnt DESC, kw) AS conv_top_keyword
        FROM cv_top
        GROUP BY 1
    )
    TO '{CONVERSIONS_AGG_FILE.as_posix()}'
    (FORMAT PARQUET, COMPRESSION ZSTD);
    """)


def build_enriched_main(con: duckdb.DuckDBPyConnection) -> None:
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


def main() -> None:
    if not PHOTOS_NO_ART_FILE.exists():
        raise FileNotFoundError("请先运行 02_filter_art.py，确保 photos_no_art.parquet 已生成")

    print(f"[INFO] DATA_ROOT = {DATA_ROOT}")
    print(f"[INFO] PHOTOS_NO_ART_FILE = {PHOTOS_NO_ART_FILE}")
    print(f"[INFO] TEMP_DIR = {TEMP_DIR}")
    print(f"[INFO] KEYWORDS_AGG_FILE = {KEYWORDS_AGG_FILE}")
    print(f"[INFO] CONVERSIONS_AGG_FILE = {CONVERSIONS_AGG_FILE}")
    print(f"[INFO] PHOTOS_NO_ART_ENRICHED_FILE = {PHOTOS_NO_ART_ENRICHED_FILE}")

    con = init_duckdb()

    print("[1/5] keywords 分片聚合 ...")
    build_keywords_partials(con)

    print("[2/5] 汇总 keywords partials ...")
    finalize_keywords_agg(con)

    print("[3/5] conversions 分片聚合 ...")
    build_conversions_partials(con)

    print("[4/5] 汇总 conversions partials ...")
    finalize_conversions_agg(con)

    print("[5/5] 合并回主表 ...")
    build_enriched_main(con)

    stats = con.execute(f"""
    SELECT
        (SELECT COUNT(*) FROM read_parquet('{PHOTOS_NO_ART_FILE.as_posix()}')) AS photos_no_art_rows,
        (SELECT COUNT(*) FROM read_parquet('{KEYWORDS_AGG_FILE.as_posix()}')) AS keyword_rows,
        (SELECT COUNT(*) FROM read_parquet('{CONVERSIONS_AGG_FILE.as_posix()}')) AS conversion_rows,
        (SELECT COUNT(*) FROM read_parquet('{PHOTOS_NO_ART_ENRICHED_FILE.as_posix()}')) AS enriched_rows
    """).fetchdf()

    print(stats.to_string(index=False))
    print(f"[OK] {KEYWORDS_AGG_FILE}")
    print(f"[OK] {CONVERSIONS_AGG_FILE}")
    print(f"[OK] {PHOTOS_NO_ART_ENRICHED_FILE}")

    con.close()


if __name__ == "__main__":
    main()
