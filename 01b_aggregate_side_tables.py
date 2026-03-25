#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
01b_aggregate_side_tables.py

作用：
1. 读取 02 产出的 photos_no_art.parquet
2. 聚合 keywords.csv* -> 一图一行
3. 聚合 conversions.csv* -> 一图一行
4. merge 回主表，生成 photos_no_art_enriched.parquet

说明：
- keywords 是强辅助特征，会真正用于 03/04 分类
- conversions 是弱辅助特征，只做轻权重补充
"""

from __future__ import annotations

from pathlib import Path
import duckdb

from config import DATA_ROOT, PHOTOS_NO_ART_FILE, WORK_DIR

TEMP_DIR = Path(r"D:/duckdb_tmp")
DUCKDB_MEMORY_LIMIT = "6GB"
DUCKDB_THREADS = 4

KEYWORDS_AGG_FILE = WORK_DIR / "keywords_agg.parquet"
CONVERSIONS_AGG_FILE = WORK_DIR / "conversions_agg.parquet"
PHOTOS_NO_ART_ENRICHED_FILE = WORK_DIR / "photos_no_art_enriched.parquet"


def create_empty_keywords_agg(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(f"""
    COPY (
        SELECT
            CAST(NULL AS VARCHAR) AS photo_id,
            CAST(NULL AS BIGINT) AS keyword_count,
            CAST(NULL AS BIGINT) AS keyword_user_count,
            CAST(NULL AS DOUBLE) AS keyword_ai_max_conf,
            CAST(NULL AS VARCHAR) AS keywords_joined,
            CAST(NULL AS VARCHAR) AS keywords_highconf_joined,
            CAST(NULL AS VARCHAR) AS keywords_user_joined
        WHERE FALSE
    )
    TO '{KEYWORDS_AGG_FILE.as_posix()}'
    (FORMAT PARQUET, COMPRESSION ZSTD);
    """)


def create_empty_conversions_agg(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(f"""
    COPY (
        SELECT
            CAST(NULL AS VARCHAR) AS photo_id,
            CAST(NULL AS BIGINT) AS conv_event_count,
            CAST(NULL AS BIGINT) AS conv_unique_keyword_count,
            CAST(NULL AS VARCHAR) AS conv_keywords_joined,
            CAST(NULL AS VARCHAR) AS conv_top_keyword
        WHERE FALSE
    )
    TO '{CONVERSIONS_AGG_FILE.as_posix()}'
    (FORMAT PARQUET, COMPRESSION ZSTD);
    """)


def build_keywords_agg(con: duckdb.DuckDBPyConnection) -> None:
    keywords_glob = f"{DATA_ROOT.as_posix()}/keywords.csv*"
    if not any(DATA_ROOT.glob("keywords.csv*")):
        print("[WARN] 未找到 keywords.csv*，将生成空 keywords_agg.parquet")
        create_empty_keywords_agg(con)
        return

    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE kw_src AS
    SELECT
        CAST(photo_id AS VARCHAR) AS photo_id,
        lower(trim(CAST(keyword AS VARCHAR))) AS kw_raw,
        COALESCE(TRY_CAST(ai_service_1_confidence AS DOUBLE), -1) AS conf1,
        COALESCE(TRY_CAST(ai_service_2_confidence AS DOUBLE), -1) AS conf2,
        CASE
            WHEN lower(trim(CAST(suggested_by_user AS VARCHAR))) IN ('t','true','1','yes','y') THEN 1
            ELSE 0
        END AS user_suggested
    FROM read_csv_auto(
        '{keywords_glob}',
        union_by_name=true,
        ignore_errors=true
    )
    WHERE photo_id IS NOT NULL
      AND keyword IS NOT NULL;
    """)

    con.execute("""
    CREATE OR REPLACE TEMP TABLE kw_clean AS
    SELECT
        photo_id,
        regexp_replace(kw_raw, '\\s+', ' ', 'g') AS kw,
        greatest(conf1, conf2, 0) AS ai_conf,
        user_suggested
    FROM kw_src
    WHERE kw_raw IS NOT NULL
      AND length(trim(kw_raw)) BETWEEN 2 AND 80
      AND kw_raw NOT IN ('', 'null', 'none', 'n/a');
    """)

    con.execute("""
    CREATE OR REPLACE TEMP TABLE kw_dedup AS
    SELECT
        photo_id,
        kw,
        max(ai_conf) AS ai_conf,
        max(user_suggested) AS user_suggested
    FROM kw_clean
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

    con.execute("""
    CREATE OR REPLACE TEMP TABLE kw_all_agg AS
    SELECT
        photo_id,
        count(*) AS keyword_count,
        max(ai_conf) AS keyword_ai_max_conf,
        sum(user_suggested) AS keyword_user_count,
        string_agg(kw, ' | ' ORDER BY user_suggested DESC, ai_conf DESC, kw) AS keywords_joined
    FROM kw_all_top
    GROUP BY 1;
    """)

    con.execute("""
    CREATE OR REPLACE TEMP TABLE kw_highconf_agg AS
    SELECT
        photo_id,
        string_agg(kw, ' | ' ORDER BY ai_conf DESC, kw) AS keywords_highconf_joined
    FROM kw_highconf_top
    GROUP BY 1;
    """)

    con.execute("""
    CREATE OR REPLACE TEMP TABLE kw_user_agg AS
    SELECT
        photo_id,
        string_agg(kw, ' | ' ORDER BY kw) AS keywords_user_joined
    FROM kw_user_top
    GROUP BY 1;
    """)

    con.execute(f"""
    COPY (
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


def build_conversions_agg(con: duckdb.DuckDBPyConnection) -> None:
    conversions_glob = f"{DATA_ROOT.as_posix()}/conversions.csv*"
    if not any(DATA_ROOT.glob("conversions.csv*")):
        print("[WARN] 未找到 conversions.csv*，将生成空 conversions_agg.parquet")
        create_empty_conversions_agg(con)
        return

    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE cv_src AS
    SELECT
        CAST(photo_id AS VARCHAR) AS photo_id,
        lower(trim(CAST(keyword AS VARCHAR))) AS kw_raw,
        lower(trim(CAST(conversion_type AS VARCHAR))) AS conversion_type
    FROM read_csv_auto(
        '{conversions_glob}',
        union_by_name=true,
        ignore_errors=true
    )
    WHERE photo_id IS NOT NULL
      AND keyword IS NOT NULL;
    """)

    con.execute("""
    CREATE OR REPLACE TEMP TABLE cv_clean AS
    SELECT
        photo_id,
        regexp_replace(kw_raw, '\\s+', ' ', 'g') AS kw
    FROM cv_src
    WHERE kw_raw IS NOT NULL
      AND length(trim(kw_raw)) BETWEEN 3 AND 80
      AND kw_raw NOT IN ('', 'null', 'none', 'n/a')
      AND regexp_matches(kw_raw, '.*[a-zA-Z一-龥].*');
    """)

    con.execute("""
    CREATE OR REPLACE TEMP TABLE cv_counts AS
    SELECT
        photo_id,
        kw,
        count(*) AS cnt
    FROM cv_clean
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


def main() -> None:
    if not PHOTOS_NO_ART_FILE.exists():
        raise FileNotFoundError("请先运行 02_filter_art.py，确保 photos_no_art.parquet 已生成")

    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    db_path = WORK_DIR / "aggregate_side_tables.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute(f"SET temp_directory='{TEMP_DIR.as_posix()}';")
    con.execute(f"SET memory_limit='{DUCKDB_MEMORY_LIMIT}';")
    con.execute(f"SET threads={DUCKDB_THREADS};")
    con.execute("SET preserve_insertion_order=false;")

    print(f"[INFO] DATA_ROOT = {DATA_ROOT}")
    print(f"[INFO] PHOTOS_NO_ART_FILE = {PHOTOS_NO_ART_FILE}")
    print(f"[INFO] KEYWORDS_AGG_FILE = {KEYWORDS_AGG_FILE}")
    print(f"[INFO] CONVERSIONS_AGG_FILE = {CONVERSIONS_AGG_FILE}")
    print(f"[INFO] PHOTOS_NO_ART_ENRICHED_FILE = {PHOTOS_NO_ART_ENRICHED_FILE}")

    print("[1/3] 聚合 keywords.csv* ...")
    build_keywords_agg(con)

    print("[2/3] 聚合 conversions.csv* ...")
    build_conversions_agg(con)

    print("[3/3] 合并 side tables -> photos_no_art_enriched.parquet ...")
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

    stats = con.execute(f"""
    SELECT
        (SELECT COUNT(*) FROM read_parquet('{PHOTOS_NO_ART_FILE.as_posix()}')) AS photos_no_art_rows,
        (SELECT COUNT(*) FROM read_parquet('{KEYWORDS_AGG_FILE.as_posix()}')) AS keyword_rows,
        (SELECT COUNT(*) FROM read_parquet('{CONVERSIONS_AGG_FILE.as_posix()}')) AS conversion_rows,
        (SELECT COUNT(*) FROM read_parquet('{PHOTOS_NO_ART_ENRICHED_FILE.as_posix()}')) AS enriched_rows
    """).fetchdf()

    print(stats.to_string(index=False))
    print(f"[OK] keywords 聚合表：{KEYWORDS_AGG_FILE}")
    print(f"[OK] conversions 聚合表：{CONVERSIONS_AGG_FILE}")
    print(f"[OK] enriched 主表：{PHOTOS_NO_ART_ENRICHED_FILE}")

    con.close()


if __name__ == "__main__":
    main()
