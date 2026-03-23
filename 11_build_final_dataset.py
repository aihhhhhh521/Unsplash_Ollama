from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd
from tqdm import tqdm

PHOTO_METADATA_PREFERRED = [
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

LABELS = ["城市、建筑", "室内", "自然", "静物", "人像"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build final dataset folder from final selected parquet.")
    p.add_argument("--input", required=True, help="Path to manifest_final_selected.parquet")
    p.add_argument("--output-root", required=True, help="Root path of final dataset folder")
    p.add_argument("--copy-mode", choices=["copy", "move"], default="copy")
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def safe_label(label: str) -> str:
    return str(label).replace("/", "_").replace("\\", "_").strip()


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(in_path)
    out_root = Path(args.output_root)
    img_root = out_root / "images"
    meta_root = out_root / "metadata"
    img_root.mkdir(parents=True, exist_ok=True)
    meta_root.mkdir(parents=True, exist_ok=True)

    for lb in LABELS:
        (img_root / safe_label(lb)).mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(in_path).copy()
    if args.limit:
        df = df.head(args.limit).copy()

    required = ["photo_id", "local_image_path", "final_label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df = df[df["final_label"].notna()].copy()
    df = df[df["local_image_path"].notna()].copy()
    df["photo_id"] = df["photo_id"].astype(str)
    df["final_label"] = df["final_label"].astype(str)
    df["path_exists"] = df["local_image_path"].map(lambda p: Path(str(p)).exists())
    df = df[df["path_exists"] == True].copy().reset_index(drop=True)

    relpaths = []
    statuses = []
    errors = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="organizing final dataset"):
        src = Path(str(row["local_image_path"]))
        label = safe_label(row["final_label"])
        ext = src.suffix if src.suffix else ".jpg"
        dst = img_root / label / f"{row['photo_id']}{ext}"
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if args.copy_mode == "copy":
                shutil.copy2(src, dst)
            else:
                shutil.move(src, dst)
            relpaths.append(str(dst.relative_to(out_root)).replace("\\", "/"))
            statuses.append(True)
            errors.append("")
        except Exception as e:
            relpaths.append("")
            statuses.append(False)
            errors.append(str(e)[:500])

    df["final_image_relpath"] = relpaths
    df["final_image_ok"] = statuses
    df["final_image_error"] = errors

    final_manifest = df[df["final_image_ok"] == True].copy().reset_index(drop=True)
    review_log = df[df["final_image_ok"] == False].copy().reset_index(drop=True)

    # Export final manifest
    final_manifest_path = meta_root / "final_manifest.parquet"
    final_manifest_csv = meta_root / "final_manifest.csv"
    final_manifest.to_parquet(final_manifest_path, index=False)
    final_manifest.to_csv(final_manifest_csv, index=False, encoding="utf-8-sig")

    # Export photo metadata (original photo fields + final label/path)
    photo_cols = [c for c in PHOTO_METADATA_PREFERRED if c in final_manifest.columns]
    extra_cols = [c for c in ["final_label", "final_source", "final_image_relpath"] if c in final_manifest.columns]
    photo_meta = final_manifest[photo_cols + extra_cols].copy()
    photo_meta_path = meta_root / "photo_metadata.parquet"
    photo_meta_csv = meta_root / "photo_metadata.csv"
    photo_meta.to_parquet(photo_meta_path, index=False)
    photo_meta.to_csv(photo_meta_csv, index=False, encoding="utf-8-sig")

    # Export review log for failures and unresolved info traceability
    review_cols = [c for c in final_manifest.columns if c.startswith("clip_") or c.startswith("review_") or c.startswith("aesthetic_")]
    review_cols = [c for c in review_cols if c in df.columns]
    base_review_cols = [c for c in ["photo_id", "category", "final_label", "final_source", "local_image_path", "final_image_relpath", "final_image_ok", "final_image_error"] if c in df.columns]
    review_export_cols = list(dict.fromkeys(base_review_cols + review_cols))
    review_log_path = meta_root / "review_log.parquet"
    review_log_csv = meta_root / "review_log.csv"
    df[review_export_cols].to_parquet(review_log_path, index=False)
    df[review_export_cols].to_csv(review_log_csv, index=False, encoding="utf-8-sig")

    stats = final_manifest.groupby(["final_label", "final_source"], dropna=False).size().reset_index(name="count").sort_values("count", ascending=False)
    stats_path = meta_root / "category_stats.csv"
    stats.to_csv(stats_path, index=False, encoding="utf-8-sig")

    readme = meta_root / "README.txt"
    readme.write_text(
        "Final dataset structure:\n"
        "- images/<五类>/photo_id.ext : downloaded images organized by final label\n"
        "- metadata/final_manifest.* : full table with original metadata + review results + final paths\n"
        "- metadata/photo_metadata.* : original photo metadata fields from photos.csv plus final label/path\n"
        "- metadata/review_log.* : complete review/export log\n"
        "- metadata/category_stats.csv : final label counts\n",
        encoding="utf-8",
    )

    print(f"[OK] final manifest: {final_manifest_path}")
    print(f"[OK] photo metadata: {photo_meta_path}")
    print(f"[OK] review log: {review_log_path}")
    print(f"[OK] stats: {stats_path}")
    print(f"[SUMMARY] final selected images: {len(final_manifest)}")


if __name__ == "__main__":
    main()
