#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
01c_resume_keywords_only_sharded.py

用途：
1. 只复用已生成的 kw_partials/*.parquet
2. 不再处理 conversions.csv*
3. 用“按 photo_id 哈希分桶”的方式汇总 keywords_agg.parquet，显著降低内存峰值
4. 只把 keywords 聚合结果 merge 回 photos_no_art.parquet，生成 photos_no_art_enriched.parquet

适用场景：
- 01b / 01c 在“汇总 kw_partials -> keywords_agg.parquet”这一步 OOM
- 16G 内存机器
"""

from __future__ import annotations

from pathlib import Path
import duckdb

from config import PHOTOS_NO_ART_FILE, WORK_DIR

# =========================
# 可调参数
# =========================
TEMP_DIR = WORK_DIR / "_duckdb_tmp"
DUCKDB_MEMORY_LIMIT = "4GB"     # 16G 机器建议先 4GB；若仍 OOM，可改 3GB
DUCKDB_THREADS = 1              # 保持 1 最稳
DUCKDB_MAX_TEMP_DIRECTORY_SIZE = "500GiB"

# 分桶数：越大越省内存，但越慢
KW_NUM_SHARDS = 128             # 16G 机器建议 128；若仍 OOM，可改 256

# 每张图最多保留多少个关键词进入最终 string_agg
# 数量越小越省内存，同时对后续 rule/LLM 也更干净
KW_TOP_LIMIT_ALL = 32
KW_TOP_LIMIT_HIGHCONF = 24
KW_TOP_LIMIT_USER = 24

KEYWORDS_AGG_FILE = WORK_DIR / "keywords_agg.parquet"
PHOTOS_NO_ART_ENRICHED_FILE = WORK_DIR / "photos_no_art_enriched.parquet"

KW_PARTIAL_DIR = WORK_DIR / "kw_partials"
KW_SHARD_DIR = WORK_DIR / "kw_agg_shards"


def init_duckdb() -> duckdb.DuckDBPyConnection:
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    KW_SHARD_DIR.mkdir(parents=True, exist_ok=True)

    db_path = WORK_DIR / "resume_keywords_only_sharded.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute(f"SET temp_directory='{TEMP_DIR.as_posix()}';")
    con.execute(f"SET memory_limit='{DUCKDB_MEMORY_LIMIT}';")
    con.execute(f"SET threads={DUCKDB_THREADS};")
    con.execute("SET preserve_insertion_order=false;")
    con.execute(f"SET max_temp_directory_size='{DUCKDB_MAX_TEMP_DIRECTORY_SIZE}';")
    return con


def clear_dir(path: Path, pattern: str = "*.parquet") -> None:
    path.mkdir(parents=True, exist_ok=True)
    for p in path.glob(pattern):
        p.unlink()


def finalize_keywords_agg_sharded(con: duckdb.DuckDBPyConnection) -> None:
    kw_files = sorted(KW_PARTIAL_DIR.glob("kw_partial_*.parquet"))
    if not kw_files:
        raise FileNotFoundError("未找到 kw_partials/*.parquet，请先至少完成关键词 partial 生成")

    partial_glob = f"{KW_PARTIAL_DIR.as_posix()}/*.parquet"
    clear_dir(KW_SHARD_DIR)

    print(f"[INFO] 复用 keyword partial 文件数: {len(kw_files)}")
    print(f"[INFO] keywords 分桶数: {KW_NUM_SHARDS}")
    print(f"[INFO] DUCKDB_MEMORY_LIMIT = {DUCKDB_MEMORY_LIMIT}")
    print(f"[INFO] KW_TOP_LIMIT_ALL = {KW_TOP_LIMIT_ALL}")

    for shard_id in range(KW_NUM_SHARDS):
        shard_file = KW_SHARD_DIR / f"kw_agg_shard_{shard_id:03d}.parquet"
        print(f"[KW-SHARD {shard_id+1}/{KW_NUM_SHARDS}]")

        # 这一桶只处理 photo_id 哈希命中的子集，显著降低内存峰值
        # 同时在桶内先做 kw 去重，再做 top-N 限制，避免 string_agg 太长
        con.execute(f"""
        COPY (
            WITH kw_dedup AS (
                SELECT
                    photo_id,
                    kw,
                    max(ai_conf) AS ai_conf,
                    max(user_suggested) AS user_suggested
                FROM read_parquet('{partial_glob}')
                WHERE abs(hash(photo_id)) % {KW_NUM_SHARDS} = {shard_id}
                GROUP BY 1, 2
            ),
            kw_ranked AS (
                SELECT
                    *,
                    row_number() OVER (
                        PARTITION BY photo_id
                        ORDER BY user_suggested DESC, ai_conf DESC, kw
                    ) AS rn_all,
                    row_number() OVER (
                        PARTITION BY photo_id
                        ORDER BY ai_conf DESC, kw
                    ) AS rn_highconf,
                    row_number() OVER (
                        PARTITION BY photo_id
                        ORDER BY kw
                    ) AS rn_user
                FROM kw_dedup
            )
            SELECT
                photo_id,
                count(*) FILTER (WHERE rn_all <= {KW_TOP_LIMIT_ALL}) AS keyword_count,
                sum(user_suggested) FILTER (WHERE rn_all <= {KW_TOP_LIMIT_ALL}) AS keyword_user_count,
                max(ai_conf) AS keyword_ai_max_conf,
                string_agg(kw, ' | ' ORDER BY user_suggested DESC, ai_conf DESC, kw)
                    FILTER (WHERE rn_all <= {KW_TOP_LIMIT_ALL}) AS keywords_joined,
                string_agg(kw, ' | ' ORDER BY ai_conf DESC, kw)
                    FILTER (
                        WHERE (user_suggested = 1 OR ai_conf >= 55)
                          AND rn_highconf <= {KW_TOP_LIMIT_HIGHCONF}
                    ) AS keywords_highconf_joined,
                string_agg(kw, ' | ' ORDER BY kw)
                    FILTER (
                        WHERE user_suggested = 1
                          AND rn_user <= {KW_TOP_LIMIT_USER}
                    ) AS keywords_user_joined
            FROM kw_ranked
            GROUP BY 1
        )
        TO '{shard_file.as_posix()}'
        (FORMAT PARQUET, COMPRESSION ZSTD);
        """)

    shard_glob = f"{KW_SHARD_DIR.as_posix()}/*.parquet"
    con.execute(f"""
    COPY (
        SELECT * FROM read_parquet('{shard_glob}')
    )
    TO '{KEYWORDS_AGG_FILE.as_posix()}'
    (FORMAT PARQUET, COMPRESSION ZSTD);
    """)

    print(f"[OK] 已生成 {KEYWORDS_AGG_FILE}")


def build_keywords_only_enriched_main(con: duckdb.DuckDBPyConnection) -> None:
    if not PHOTOS_NO_ART_FILE.exists():
        raise FileNotFoundError("未找到 photos_no_art.parquet，请先运行 02_filter_art.py")
    if not KEYWORDS_AGG_FILE.exists():
        raise FileNotFoundError("未找到 keywords_agg.parquet")

    con.execute(f"""
    COPY (
        SELECT
            p.*,
            k.keyword_count,
            k.keyword_user_count,
            k.keyword_ai_max_conf,
            k.keywords_joined,
            k.keywords_highconf_joined,
            k.keywords_user_joined
        FROM read_parquet('{PHOTOS_NO_ART_FILE.as_posix()}') p
        LEFT JOIN read_parquet('{KEYWORDS_AGG_FILE.as_posix()}') k
            ON CAST(p.photo_id AS VARCHAR) = k.photo_id
    )
    TO '{PHOTOS_NO_ART_ENRICHED_FILE.as_posix()}'
    (FORMAT PARQUET, COMPRESSION ZSTD);
    """)

    print(f"[OK] 已生成 {PHOTOS_NO_ART_ENRICHED_FILE}")


def main() -> None:
    print(f"[INFO] PHOTOS_NO_ART_FILE = {PHOTOS_NO_ART_FILE}")
    print(f"[INFO] TEMP_DIR = {TEMP_DIR}")
    print(f"[INFO] KEYWORDS_AGG_FILE = {KEYWORDS_AGG_FILE}")
    print(f"[INFO] PHOTOS_NO_ART_ENRICHED_FILE = {PHOTOS_NO_ART_ENRICHED_FILE}")
    print(f"[INFO] KW_PARTIAL_DIR = {KW_PARTIAL_DIR}")
    print(f"[INFO] KW_SHARD_DIR = {KW_SHARD_DIR}")

    con = init_duckdb()

    print("[1/2] 从现有 kw_partials 分桶汇总 keywords_agg.parquet ...")
    finalize_keywords_agg_sharded(con)

    print("[2/2] 仅用 keywords 合并回主表 ...")
    build_keywords_only_enriched_main(con)

    stats = con.execute(f"""
    SELECT
        (SELECT COUNT(*) FROM read_parquet('{PHOTOS_NO_ART_FILE.as_posix()}')) AS photos_no_art_rows,
        (SELECT COUNT(*) FROM read_parquet('{KEYWORDS_AGG_FILE.as_posix()}')) AS keyword_rows,
        (SELECT COUNT(*) FROM read_parquet('{PHOTOS_NO_ART_ENRICHED_FILE.as_posix()}')) AS enriched_rows
    """).fetchdf()

    print(stats.to_string(index=False))
    con.close()


if __name__ == "__main__":
    main()
