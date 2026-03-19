from pathlib import Path
import duckdb

from config import ART_KEYWORDS, DATA_ROOT, MANIFEST_FILE, PHOTOS_NO_ART_FILE, REMOVED_ART_FILE

TEMP_DIR = Path(r"C:/duckdb_tmp")
DUCKDB_MEMORY_LIMIT = "4GB"
DUCKDB_THREADS = 2

def main() -> None:
    if not MANIFEST_FILE.exists():
        raise FileNotFoundError("请先运行 01_build_manifest.py")

    keywords_glob = f"{DATA_ROOT.as_posix()}/keywords.csv*"
    if not any(DATA_ROOT.glob("keywords.csv*")):
        raise FileNotFoundError(f"未找到 keywords.csv*，请检查目录：{DATA_ROOT}")

    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute(f"SET temp_directory='{TEMP_DIR.as_posix()}';")
    con.execute(f"SET memory_limit='{DUCKDB_MEMORY_LIMIT}';")
    con.execute(f"SET threads={DUCKDB_THREADS};")
    con.execute("SET preserve_insertion_order=false;")

    art_sql = ",".join([f"'{x}'" for x in ART_KEYWORDS])

    print("[1/3] 从 keywords.csv* 提取 art photo_id ...")
    con.execute(f"""
    CREATE OR REPLACE TABLE art_ids AS
    SELECT DISTINCT
        CAST(photo_id AS VARCHAR) AS photo_id
    FROM read_csv_auto(
        '{keywords_glob}',
        union_by_name=true,
        ignore_errors=true
    )
    WHERE photo_id IS NOT NULL
      AND keyword IS NOT NULL
      AND lower(trim(keyword)) IN ({art_sql});
    """)

    art_count = con.execute("SELECT COUNT(*) FROM art_ids").fetchone()[0]
    print(f"[INFO] art photo_id 数量：{art_count}")

    print("[2/3] 导出非 art 图片 ...")
    con.execute(f"""
    COPY (
        SELECT m.*
        FROM read_parquet('{MANIFEST_FILE.as_posix()}') m
        LEFT JOIN art_ids a
        ON CAST(m.photo_id AS VARCHAR) = a.photo_id
        WHERE a.photo_id IS NULL
    )
    TO '{PHOTOS_NO_ART_FILE.as_posix()}'
    (FORMAT PARQUET, COMPRESSION ZSTD);
    """)

    print("[3/3] 导出被过滤的 art 图片 ...")
    con.execute(f"""
    COPY (
        SELECT m.*
        FROM read_parquet('{MANIFEST_FILE.as_posix()}') m
        INNER JOIN art_ids a
        ON CAST(m.photo_id AS VARCHAR) = a.photo_id
    )
    TO '{REMOVED_ART_FILE.as_posix()}'
    (FORMAT PARQUET, COMPRESSION ZSTD);
    """)

    stats = con.execute(f"""
    SELECT
        (SELECT COUNT(*) FROM read_parquet('{MANIFEST_FILE.as_posix()}')) AS total_rows,
        (SELECT COUNT(*) FROM read_parquet('{PHOTOS_NO_ART_FILE.as_posix()}')) AS kept_rows,
        (SELECT COUNT(*) FROM read_parquet('{REMOVED_ART_FILE.as_posix()}')) AS removed_rows
    """).fetchdf()

    print(stats.to_string(index=False))
    print(f"[OK] 非 art 主表：{PHOTOS_NO_ART_FILE}")
    print(f"[OK] 被过滤样本：{REMOVED_ART_FILE}")

if __name__ == "__main__":
    main()