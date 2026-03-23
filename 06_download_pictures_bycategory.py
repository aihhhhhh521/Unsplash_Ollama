from __future__ import annotations

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import mimetypes
import re

import pandas as pd
import requests

from config import CLASSIFIED_FILE

# =========================
# 输出目录：按最终分类结果落盘
# =========================
OUT_ROOT = Path(r"D:/PyProjects/Dataset/unsplash-research-dataset-full-latest/work/dataset")
IMG_ROOT = OUT_ROOT / "images"
BY_CATEGORY_ROOT = IMG_ROOT / "by_category"
META_ROOT = OUT_ROOT / "metadata"

BY_CATEGORY_ROOT.mkdir(parents=True, exist_ok=True)
META_ROOT.mkdir(parents=True, exist_ok=True)

# =========================
# 输出文件
# =========================
PHOTO_METADATA_PARQUET = META_ROOT / "photo_metadata.parquet"
PHOTO_METADATA_CSV = META_ROOT / "photo_metadata.csv"
DATASET_METADATA_PARQUET = META_ROOT / "dataset_metadata.parquet"
DATASET_METADATA_CSV = META_ROOT / "dataset_metadata.csv"
DOWNLOADED_METADATA_PARQUET = META_ROOT / "dataset_metadata_downloaded.parquet"
DOWNLOADED_METADATA_CSV = META_ROOT / "dataset_metadata_downloaded.csv"
DOWNLOAD_FAILURES_PARQUET = META_ROOT / "download_failures.parquet"
DOWNLOAD_FAILURES_CSV = META_ROOT / "download_failures.csv"
DOWNLOAD_STATS_CSV = META_ROOT / "download_stats.csv"
CATEGORY_DOWNLOAD_STATS_CSV = META_ROOT / "category_download_stats.csv"
DOWNLOAD_RESULTS_JSONL = META_ROOT / "download_results_by_category.jsonl"

# =========================
# 参数
# =========================
MAX_WORKERS = 16
REQUEST_TIMEOUT = 60
RETRY = 3
CHUNK_SIZE = 1024 * 256
FLUSH_EVERY = 200
LIMIT_ROWS = None

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36"
)

PHOTO_INFO_COLS = [
    "photo_id",
    "photo_url",
    "photo_image_url",
    "photo_submitted_at",
    "photo_featured",
    "photo_width",
    "photo_height",
    "photo_aspect_ratio",
    "photo_description",
    "photographer_username",
    "photographer_first_name",
    "photographer_last_name",
    "exif_camera_make",
    "exif_camera_model",
    "exif_iso",
    "exif_aperture_value",
    "exif_focal_length",
    "exif_exposure_time",
    "photo_location_name",
    "photo_location_latitude",
    "photo_location_longitude",
    "photo_location_country",
    "photo_location_city",
    "stats_views",
    "stats_downloads",
    "ai_description",
    "ai_primary_landmark_name",
    "ai_primary_landmark_latitude",
    "ai_primary_landmark_longitude",
    "ai_primary_landmark_confidence",
    "blur_hash",
]

CLASSIFICATION_COLS = [
    "category",
    "category_confidence",
    "category_source",
    "review_flag",
    "ollama_confidence",
    "ollama_label",
    "ollama_reason",
    "ollama_ok",
    "ollama_error",
    "needs_llm",
    "rule_top1_label",
    "rule_top1_score",
    "rule_margin",
    "rule_gate_pass",
    "rule_reject_reason",
]

DOWNLOAD_COLS = [
    "download_ok",
    "download_error",
    "download_http_status",
    "local_image_relpath",
    "local_image_path",
    "file_size",
    "image_ext",
    "downloaded_at",
    "download_attempts",
    "from_cache",
]


def safe_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()



def safe_category_name(x: str) -> str:
    text = safe_str(x)
    if not text:
        return "未分类"
    text = re.sub(r'[\\/:*?"<>|]+', "_", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text



def guess_ext(resp, url: str) -> str:
    ctype = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
    if ctype:
        ext = mimetypes.guess_extension(ctype)
        if ext:
            return ".jpg" if ext == ".jpe" else ext
    url_no_query = url.lower().split("?")[0]
    for ext in [".jpg", ".jpeg", ".png", ".webp"]:
        if url_no_query.endswith(ext):
            return ext
    return ".jpg"



def build_category_relpath(photo_id: str, category: str, ext: str) -> Path:
    cat = safe_category_name(category)
    pid = safe_str(photo_id)
    return Path("by_category") / cat / f"{pid}{ext}"



def append_jsonl(records: list[dict], path: Path) -> None:
    if not records:
        return
    with path.open("a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")



def load_existing_download_results(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=["photo_id", *DOWNLOAD_COLS])

    df = pd.read_json(path, lines=True)
    if df.empty:
        return pd.DataFrame(columns=["photo_id", *DOWNLOAD_COLS])

    if "downloaded_at" in df.columns:
        df = df.sort_values("downloaded_at")
    df["photo_id"] = df["photo_id"].astype(str)
    df = df.drop_duplicates(subset=["photo_id"], keep="last").reset_index(drop=True)
    return df



def is_existing_success(row: pd.Series) -> bool:
    if not bool(row.get("download_ok", False)):
        return False
    relpath = safe_str(row.get("local_image_relpath", ""))
    if not relpath:
        return False
    return (IMG_ROOT / relpath).exists()



def download_one(row: dict) -> dict:
    photo_id = safe_str(row["photo_id"])
    url = safe_str(row["photo_image_url"])
    category = safe_str(row["category"])

    headers = {"User-Agent": USER_AGENT}
    last_err = ""
    last_status = None

    for attempt in range(1, RETRY + 1):
        try:
            with requests.get(url, stream=True, timeout=REQUEST_TIMEOUT, headers=headers) as resp:
                last_status = resp.status_code
                resp.raise_for_status()
                ext = guess_ext(resp, url)
                relpath = build_category_relpath(photo_id, category, ext)
                out_path = IMG_ROOT / relpath
                out_path.parent.mkdir(parents=True, exist_ok=True)

                with out_path.open("wb") as f:
                    for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)

            return {
                "photo_id": photo_id,
                "download_ok": True,
                "download_error": "",
                "download_http_status": last_status,
                "local_image_relpath": str(relpath).replace("\\", "/"),
                "local_image_path": str(out_path),
                "file_size": out_path.stat().st_size if out_path.exists() else None,
                "image_ext": ext,
                "downloaded_at": datetime.now().isoformat(timespec="seconds"),
                "download_attempts": attempt,
                "from_cache": False,
            }
        except Exception as e:
            last_err = str(e)

    return {
        "photo_id": photo_id,
        "download_ok": False,
        "download_error": last_err[:1000],
        "download_http_status": last_status,
        "local_image_relpath": "",
        "local_image_path": "",
        "file_size": None,
        "image_ext": "",
        "downloaded_at": datetime.now().isoformat(timespec="seconds"),
        "download_attempts": RETRY,
        "from_cache": False,
    }



def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    preferred = []
    for col in PHOTO_INFO_COLS + CLASSIFICATION_COLS + DOWNLOAD_COLS:
        if col in df.columns and col not in preferred:
            preferred.append(col)
    remaining = [c for c in df.columns if c not in preferred]
    return df[preferred + remaining]



def main() -> None:
    if not CLASSIFIED_FILE.exists():
        raise FileNotFoundError(f"未找到最终分类文件: {CLASSIFIED_FILE}")

    df = pd.read_parquet(CLASSIFIED_FILE).copy()
    print(f"[INFO] 读取 classified: {len(df)} 行")

    df = df[df["photo_image_url"].notna()].copy()
    df = df[df["category"].notna()].copy()
    df["photo_id"] = df["photo_id"].astype(str)
    df["category"] = df["category"].map(safe_category_name)
    df = df.drop_duplicates(subset=["photo_id"], keep="first").reset_index(drop=True)
    if LIMIT_ROWS is not None:
        df = df.head(LIMIT_ROWS).copy()

    print(f"[INFO] 最终下载候选池: {len(df)}")

    # 纯图片元数据：不含下载状态字段
    photo_meta_cols = [c for c in (PHOTO_INFO_COLS + CLASSIFICATION_COLS) if c in df.columns]
    photo_meta = df[photo_meta_cols].copy()
    photo_meta = reorder_columns(photo_meta)
    photo_meta.to_parquet(PHOTO_METADATA_PARQUET, index=False)
    photo_meta.to_csv(PHOTO_METADATA_CSV, index=False, encoding="utf-8-sig")

    existing_results = load_existing_download_results(DOWNLOAD_RESULTS_JSONL)
    existing_map = {str(row["photo_id"]): row for _, row in existing_results.iterrows()} if not existing_results.empty else {}

    todo_rows = []
    cached_records = []
    for row in df[["photo_id", "photo_image_url", "category"]].to_dict("records"):
        pid = str(row["photo_id"])
        old = existing_map.get(pid)
        if old is not None and is_existing_success(old):
            rec = old.to_dict()
            rec["from_cache"] = True
            cached_records.append(rec)
        else:
            todo_rows.append(row)

    print(f"[INFO] 已存在可复用下载结果: {len(cached_records)} 张")
    print(f"[INFO] 本次需下载: {len(todo_rows)} 张")

    if cached_records:
        append_jsonl(cached_records, DOWNLOAD_RESULTS_JSONL)

    pending_records: list[dict] = []
    finished = 0

    if len(todo_rows) > 0:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = [ex.submit(download_one, row) for row in todo_rows]
            for fut in as_completed(futures):
                rec = fut.result()
                pending_records.append(rec)
                finished += 1
                if finished % FLUSH_EVERY == 0:
                    append_jsonl(pending_records, DOWNLOAD_RESULTS_JSONL)
                    pending_records = []
                    print(f"[INFO] 已完成 {finished}/{len(todo_rows)}")

        append_jsonl(pending_records, DOWNLOAD_RESULTS_JSONL)

    dl_df = load_existing_download_results(DOWNLOAD_RESULTS_JSONL)
    manifest = df.merge(dl_df, on="photo_id", how="left", suffixes=("", "_dl"))

    for col, default_val in {
        "download_ok": False,
        "download_error": "",
        "download_http_status": None,
        "local_image_relpath": "",
        "local_image_path": "",
        "file_size": None,
        "image_ext": "",
        "downloaded_at": "",
        "download_attempts": 0,
        "from_cache": False,
    }.items():
        if col not in manifest.columns:
            manifest[col] = default_val
        else:
            manifest[col] = manifest[col].fillna(default_val)

    manifest = reorder_columns(manifest)

    manifest_downloaded = manifest[manifest["download_ok"] == True].copy()
    failures = manifest[manifest["download_ok"] != True].copy()

    manifest.to_parquet(DATASET_METADATA_PARQUET, index=False)
    manifest.to_csv(DATASET_METADATA_CSV, index=False, encoding="utf-8-sig")

    manifest_downloaded.to_parquet(DOWNLOADED_METADATA_PARQUET, index=False)
    manifest_downloaded.to_csv(DOWNLOADED_METADATA_CSV, index=False, encoding="utf-8-sig")

    failures.to_parquet(DOWNLOAD_FAILURES_PARQUET, index=False)
    failures.to_csv(DOWNLOAD_FAILURES_CSV, index=False, encoding="utf-8-sig")

    stats_rows = [
        {"metric": "classified_final_total", "value": int(len(df))},
        {"metric": "downloaded_ok", "value": int((manifest["download_ok"] == True).sum())},
        {"metric": "download_failed", "value": int((manifest["download_ok"] != True).sum())},
    ]
    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(DOWNLOAD_STATS_CSV, index=False, encoding="utf-8-sig")

    cat_stats = (
        manifest.assign(download_ok_bool=manifest["download_ok"].astype(bool))
        .groupby(["category", "download_ok_bool"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["category", "download_ok_bool"], ascending=[True, False])
    )
    cat_stats["download_status"] = cat_stats["download_ok_bool"].map({True: "downloaded_ok", False: "download_failed"})
    cat_stats = cat_stats[["category", "download_status", "count"]]
    cat_stats.to_csv(CATEGORY_DOWNLOAD_STATS_CSV, index=False, encoding="utf-8-sig")

    print(f"[OK] 图片 metadata（不含下载状态）: {PHOTO_METADATA_PARQUET}")
    print(f"[OK] 数据集 metadata（含本地路径与下载状态）: {DATASET_METADATA_PARQUET}")
    print(f"[OK] 已下载样本 metadata: {DOWNLOADED_METADATA_PARQUET}")
    print(f"[OK] 下载失败清单: {DOWNLOAD_FAILURES_PARQUET}")
    print(f"[OK] 下载统计: {DOWNLOAD_STATS_CSV}")
    print(f"[OK] 分类下载统计: {CATEGORY_DOWNLOAD_STATS_CSV}")
    print(f"[OK] 图片目录（按分类）: {BY_CATEGORY_ROOT}")


if __name__ == "__main__":
    main()
