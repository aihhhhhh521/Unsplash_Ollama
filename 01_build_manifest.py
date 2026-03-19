from pathlib import Path
import duckdb

from config import DATA_ROOT, MANIFEST_FILE
from utils import ensure_exists

def main() -> None:
    photos_glob = f"{DATA_ROOT.as_posix()}/photos.csv*"
    keywords_glob = f"{DATA_ROOT.as_posix()}/keywords.csv*"

    # 粗检查：至少要有 photos.csv000 和 keywords.csv000 之类的分片文件
    has_photos = any(DATA_ROOT.glob("photos.csv*"))
    has_keywords = any(DATA_ROOT.glob("keywords.csv*"))
    if not has_photos:
        raise FileNotFoundError(f"未找到 photos.csv*，请检查目录：{DATA_ROOT}")
    if not has_keywords:
        raise FileNotFoundError(f"未找到 keywords.csv*，请检查目录：{DATA_ROOT}")

    con = duckdb.connect()

    # 读取主表
    con.execute(f'''
    CREATE OR REPLACE TABLE photos AS
    SELECT *
    FROM read_csv_auto('{photos_glob}', union_by_name=true, ignore_errors=true);
    ''')

    # 读取关键词表并做清洗
    con.execute(f'''
    CREATE OR REPLACE TABLE keywords_raw AS
    SELECT
        CAST(photo_id AS VARCHAR) AS photo_id,
        lower(trim(keyword)) AS keyword
    FROM read_csv_auto('{keywords_glob}', union_by_name=true, ignore_errors=true)
    WHERE photo_id IS NOT NULL
      AND keyword IS NOT NULL
      AND trim(keyword) <> '';
    ''')

    # 聚合到照片级别
    con.execute('''
    CREATE OR REPLACE TABLE kw_agg AS
    SELECT
        photo_id,
        string_agg(DISTINCT keyword, ' | ' ORDER BY keyword) AS keywords_text,
        count(DISTINCT keyword) AS keyword_count
    FROM keywords_raw
    GROUP BY photo_id;
    ''')

    # 合并成 manifest
    con.execute(f'''
    COPY (
        SELECT
            p.*,
            coalesce(k.keywords_text, '') AS keywords_text,
            coalesce(k.keyword_count, 0) AS keyword_count
        FROM photos p
        LEFT JOIN kw_agg k
        ON CAST(p.photo_id AS VARCHAR) = k.photo_id
    ) TO '{MANIFEST_FILE.as_posix()}'
    (FORMAT PARQUET, COMPRESSION ZSTD);
    ''')

    print(f"[OK] manifest 已生成：{MANIFEST_FILE}")

if __name__ == "__main__":
    main()
