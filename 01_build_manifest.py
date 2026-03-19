from pathlib import Path
import duckdb

from config import DATA_ROOT, MANIFEST_FILE, WORK_DIR

# 这里可以放到空间大的盘，但这一步已经不重了
TEMP_DIR = Path(r"C:/duckdb_tmp")

DUCKDB_MEMORY_LIMIT = "4GB"
DUCKDB_THREADS = 2

def main() -> None:
    photos_glob = f"{DATA_ROOT.as_posix()}/photos.csv*"

    if not any(DATA_ROOT.glob("photos.csv*")):
        raise FileNotFoundError(f"未找到 photos.csv*，请检查目录：{DATA_ROOT}")

    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    db_path = WORK_DIR / "build_manifest.duckdb"
    con = duckdb.connect(str(db_path))

    con.execute(f"SET temp_directory='{TEMP_DIR.as_posix()}';")
    con.execute(f"SET memory_limit='{DUCKDB_MEMORY_LIMIT}';")
    con.execute(f"SET threads={DUCKDB_THREADS};")
    con.execute("SET preserve_insertion_order=false;")

    print(f"[INFO] DATA_ROOT = {DATA_ROOT}")
    print(f"[INFO] TEMP_DIR = {TEMP_DIR}")
    print(f"[INFO] DB_PATH = {db_path}")
    print(f"[INFO] MANIFEST_FILE = {MANIFEST_FILE}")

    print("[1/2] 读取 photos.csv* 并抽取必要字段 ...")
    con.execute(f"""
    COPY (
        SELECT
            CAST(photo_id AS VARCHAR) AS photo_id,
            photo_url,
            photo_image_url,
            photo_submitted_at,
            photo_featured,
            photo_width,
            photo_height,
            photo_aspect_ratio,
            photo_description,
            photographer_username,
            photographer_first_name,
            photographer_last_name,
            exif_camera_make,
            exif_camera_model,
            exif_iso,
            exif_aperture_value,
            exif_focal_length,
            exif_exposure_time,
            photo_location_name,
            photo_location_latitude,
            photo_location_longitude,
            photo_location_country,
            photo_location_city,
            stats_views,
            stats_downloads,
            ai_description,
            ai_primary_landmark_name,
            ai_primary_landmark_latitude,
            ai_primary_landmark_longitude,
            ai_primary_landmark_confidence,
            blur_hash
        FROM read_csv_auto(
            '{photos_glob}',
            union_by_name=true,
            ignore_errors=true
        )
        WHERE photo_id IS NOT NULL
    )
    TO '{MANIFEST_FILE.as_posix()}'
    (FORMAT PARQUET, COMPRESSION ZSTD);
    """)

    total = con.execute(f"""
        SELECT COUNT(*) FROM read_parquet('{MANIFEST_FILE.as_posix()}')
    """).fetchone()[0]

    print("[2/2] 完成")
    print(f"[OK] manifest 已生成：{MANIFEST_FILE}")
    print(f"[INFO] 行数：{total}")

    con.close()

if __name__ == "__main__":
    main()