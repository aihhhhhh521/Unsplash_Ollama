import duckdb

from config import ART_KEYWORDS, MANIFEST_FILE, PHOTOS_NO_ART_FILE, REMOVED_ART_FILE
from utils import ensure_exists

def main() -> None:
    ensure_exists(MANIFEST_FILE, "请先运行 01_build_manifest.py")

    # 用 |keyword| 形式做边界匹配，避免把 partial substring 误判成 art 关键词
    escaped = [x.replace(" ", "\\ ") for x in ART_KEYWORDS]
    pattern = "|".join([x.replace("\\ ", " ") for x in escaped])
    regex = rf'\|(?:{pattern})\|'

    con = duckdb.connect()

    con.execute(f'''
    COPY (
        SELECT *
        FROM read_parquet('{MANIFEST_FILE.as_posix()}')
        WHERE NOT regexp_matches(
            '|' || lower(coalesce(keywords_text, '')) || '|',
            '{regex}'
        )
    ) TO '{PHOTOS_NO_ART_FILE.as_posix()}'
    (FORMAT PARQUET, COMPRESSION ZSTD);
    ''')

    con.execute(f'''
    COPY (
        SELECT *
        FROM read_parquet('{MANIFEST_FILE.as_posix()}')
        WHERE regexp_matches(
            '|' || lower(coalesce(keywords_text, '')) || '|',
            '{regex}'
        )
    ) TO '{REMOVED_ART_FILE.as_posix()}'
    (FORMAT PARQUET, COMPRESSION ZSTD);
    ''')

    stats = con.execute(f'''
    SELECT
        (SELECT COUNT(*) FROM read_parquet('{MANIFEST_FILE.as_posix()}')) AS total_rows,
        (SELECT COUNT(*) FROM read_parquet('{PHOTOS_NO_ART_FILE.as_posix()}')) AS kept_rows,
        (SELECT COUNT(*) FROM read_parquet('{REMOVED_ART_FILE.as_posix()}')) AS removed_rows
    ''').fetchdf()

    print(stats.to_string(index=False))
    print(f"[OK] 非 art 主表：{PHOTOS_NO_ART_FILE}")
    print(f"[OK] 被过滤样本：{REMOVED_ART_FILE}")

if __name__ == "__main__":
    main()
