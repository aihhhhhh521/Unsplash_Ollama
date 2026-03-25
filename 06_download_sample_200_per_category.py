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
# 输入
# =========================
INPUT_FILE = CLASSIFIED_FILE

# =========================
# 采样参数
# =========================
SAMPLE_PER_CATEGORY = 200
RANDOM_SEED = 20260325
TARGET_CATEGORIES = ["自然", "城市、建筑", "人像", "室内", "静物"]

# =========================
# 输出目录
# =========================
OUT_ROOT = Path(r"D:/PyProjects/Dataset/unsplash-research-dataset-full-latest/work/sample_200_per_category")
IMG_ROOT = OUT_ROOT / "images"
BY_CATEGORY_ROOT = IMG_ROOT / "by_category"
META_ROOT = OUT_ROOT / "metadata"

BY_CATEGORY_ROOT.mkdir(parents=True, exist_ok=True)
META_ROOT.mkdir(parents=True, exist_ok=True)

# =========================
# 输出文件
# =========================
SAMPLED_PARQUET = META_ROOT / "sampled_200_per_category.parquet"
SAMPLED_CSV = META_ROOT / "sampled_200_per_category.csv"
DOWNLOAD_RESULTS_JSONL = META_ROOT / "download_results_sampled.jsonl"
DOWNLOADED_PARQUET = META_ROOT / "sampled_downloaded.parquet"
DOWNLOADED_CSV = META_ROOT / "sampled_downloaded.csv"
FAILURES_PARQUET = META_ROOT / "sampled_download_failures.parquet"
FAILURES_CSV = META_ROOT / "sampled_download_failures.csv"
STATS_CSV = META_ROOT / "sample_download_stats.csv"

# =========================
# 下载参数
# =========================
MAX_WORKERS = 16
REQUEST_TIMEOUT = 60
RETRY = 3
CHUNK_SIZE = 1024 * 256
FLUSH_EVERY = 100

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
    "category_final",
    "category_confidence_final",
    "category_source_final",
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


def choose_category_column(df: pd.DataFrame) -> str:
    for col in ["category", "category_final"]:
        if col in df.columns:
            return col
    raise KeyError("classified.parquet 中找不到 category 或 category_final 列。")


def sample_rows(df: pd.DataFrame, category_col: str) -> pd.DataFrame:
    df = df.copy()
    df["photo_id"] = df["photo_id"].astype(str)
    df = df.drop_duplicates(subset=["photo_id"], keep="first")

    sample_parts = []
    stats_rows = []

    for cat in TARGET_CATEGORIES:
        cat_df = df[df[category_col].astype(str) == cat].copy()
        available = len(cat_df)
        take_n = min(SAMPLE_PER_CATEGORY, available)

        if take_n > 0:
            chosen = cat_df.sample(n=take_n, random_state=RANDOM_SEED).copy()
            chosen["sample_category"] = cat
            chosen["sample_quota"] = SAMPLE_PER_CATEGORY
            chosen["sample_random_seed"] = RANDOM_SEED
            sample_parts.append(chosen)

        stats_rows.append(
            {
                "category": cat,
                "available": available,
                "target_sample": SAMPLE_PER_CATEGORY,
                "sampled": take_n,
                "shortfall": max(0, SAMPLE_PER_CATEGORY - take_n),
            }
        )

    sampled_df = pd.concat(sample_parts, ignore_index=True) if sample_parts else df.iloc[0:0].copy()
    stats_df = pd.DataFrame(stats_rows)

    if len(sampled_df) > 0:
        sampled_df = sampled_df.sort_values([category_col, "photo_id"], ascending=[True, True], kind="mergesort")

    return sampled_df, stats_df


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
    for col in (
        PHOTO_INFO_COLS
        + CLASSIFICATION_COLS
        + ["sample_category", "sample_quota", "sample_random_seed"]
        + DOWNLOAD_COLS
    ):
        if col in df.columns and col not in preferred:
            preferred.append(col)

    rest = [c for c in df.columns if c not in preferred]
    return df[preferred + rest]


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"找不到输入文件：{INPUT_FILE}")

    df = pd.read_parquet(INPUT_FILE).copy()
    if "photo_id" not in df.columns or "photo_image_url" not in df.columns:
        raise KeyError("classified.parquet 至少需要包含 photo_id 和 photo_image_url 列。")

    category_col = choose_category_column(df)
    if category_col != "category":
        df["category"] = df[category_col]

    sampled_df, stats_df = sample_rows(df, "category")

    sampled_df.to_parquet(SAMPLED_PARQUET, index=False)
    sampled_df.to_csv(SAMPLED_CSV, index=False, encoding="utf-8-sig")
    stats_df.to_csv(STATS_CSV, index=False, encoding="utf-8-sig")

    if len(sampled_df) == 0:
        print("[WARN] 没有采样到任何记录。")
        return

    existing_df = load_existing_download_results(DOWNLOAD_RESULTS_JSONL)
    existing_map = {}
    if len(existing_df) > 0:
        existing_map = {str(r["photo_id"]): r for _, r in existing_df.iterrows()}

    todo_records = []
    cached_results = []

    for _, row in sampled_df.iterrows():
        pid = safe_str(row["photo_id"])
        old = existing_map.get(pid)
        if old is not None and is_existing_success(old):
            cached_results.append(
                {
                    "photo_id": pid,
                    "download_ok": True,
                    "download_error": "",
                    "download_http_status": old.get("download_http_status"),
                    "local_image_relpath": old.get("local_image_relpath", ""),
                    "local_image_path": old.get("local_image_path", ""),
                    "file_size": old.get("file_size"),
                    "image_ext": old.get("image_ext", ""),
                    "downloaded_at": datetime.now().isoformat(timespec="seconds"),
                    "download_attempts": 0,
                    "from_cache": True,
                }
            )
        else:
            todo_records.append(row.to_dict())

    new_results = []
    buffer = []

    if todo_records:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = [ex.submit(download_one, rec) for rec in todo_records]
            for i, fut in enumerate(as_completed(futures), start=1):
                result = fut.result()
                new_results.append(result)
                buffer.append(result)

                if len(buffer) >= FLUSH_EVERY:
                    append_jsonl(buffer, DOWNLOAD_RESULTS_JSONL)
                    buffer.clear()

                if i % 50 == 0:
                    print(f"[INFO] 已完成 {i}/{len(todo_records)} 张下载")

    if buffer:
        append_jsonl(buffer, DOWNLOAD_RESULTS_JSONL)

    all_results = pd.DataFrame(cached_results + new_results)
    if len(all_results) == 0:
        all_results = pd.DataFrame(columns=["photo_id", *DOWNLOAD_COLS])

    sampled_df["photo_id"] = sampled_df["photo_id"].astype(str)
    all_results["photo_id"] = all_results["photo_id"].astype(str)

    final_df = sampled_df.merge(all_results, on="photo_id", how="left")
    final_df = reorder_columns(final_df)

    ok_df = final_df[final_df["download_ok"] == True].copy()
    fail_df = final_df[final_df["download_ok"] != True].copy()

    final_df.to_parquet(DOWNLOADED_PARQUET, index=False)
    final_df.to_csv(DOWNLOADED_CSV, index=False, encoding="utf-8-sig")
    fail_df.to_parquet(FAILURES_PARQUET, index=False)
    fail_df.to_csv(FAILURES_CSV, index=False, encoding="utf-8-sig")

    print(f"[OK] 输入分类结果：{INPUT_FILE}")
    print(f"[OK] 采样清单：{SAMPLED_PARQUET}")
    print(f"[OK] 采样统计：{STATS_CSV}")
    print(f"[OK] 下载完成清单：{DOWNLOADED_PARQUET}")
    print(f"[OK] 下载失败清单：{FAILURES_PARQUET}")
    print(f"[SUMMARY] 采样总数: {len(sampled_df)}")
    print(f"[SUMMARY] 下载成功: {len(ok_df)}")
    print(f"[SUMMARY] 下载失败: {len(fail_df)}")
    print("\n各类别采样情况：")
    print(stats_df.to_string(index=False))


if __name__ == "__main__":
    main()
