from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd
import torch
from PIL import Image, ImageFile
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification

ImageFile.LOAD_TRUNCATED_IMAGES = True

ALL_LABELS = ["城市、建筑", "室内", "自然", "静物", "人像"]
LABEL_TO_TEXT = {
    "城市、建筑": "This is a photo of urban architecture or a city scene.",
    "室内": "This is a photo of an indoor room or interior space.",
    "自然": "This is a photo of nature, landscape, plants, or animals.",
    "静物": "This is a photo of objects, products, food, or still life.",
    "人像": "This is a portrait photo of a person as the main subject.",
}
TEXT_TO_LABEL = {v: k for k, v in LABEL_TO_TEXT.items()}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Second-pass CLIP review for first-pass review/error samples.")
    p.add_argument("--input", required=True, help="Path to manifest_clip_review.parquet from first-pass CLIP review")
    p.add_argument("--output-dir", default=None, help="Directory to save round2 outputs; default is input parent")
    p.add_argument("--checkpoint", default="openai/clip-vit-large-patch14")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit", type=int, default=None, help="Optional row limit for testing")
    p.add_argument("--keep-score-min", type=float, default=0.56)
    p.add_argument("--keep-margin-min", type=float, default=0.10)
    p.add_argument("--relabel-score-min", type=float, default=0.60)
    p.add_argument("--relabel-margin-min", type=float, default=0.12)
    p.add_argument("--drop-score-max", type=float, default=0.45)
    p.add_argument("--drop-margin-max", type=float, default=0.05)
    return p.parse_args()


def unique_valid_labels(*values: object) -> List[str]:
    out: List[str] = []
    for v in values:
        if v is None or pd.isna(v):
            continue
        s = str(v).strip()
        if not s or s not in ALL_LABELS:
            continue
        if s not in out:
            out.append(s)
    return out


def get_candidates(row: pd.Series) -> List[str]:
    # Restrict the second pass to the most plausible classes when available.
    cands = unique_valid_labels(
        row.get("category"),
        row.get("clip_top1_label"),
        row.get("clip_top2_label"),
    )
    # If first pass labels are missing / broken, fall back to all 5 labels.
    return cands if len(cands) >= 2 else ALL_LABELS.copy()


def decide(old_category: str, top1_label: str, top1_score: float, margin: float,
           keep_score_min: float, keep_margin_min: float,
           relabel_score_min: float, relabel_margin_min: float,
           drop_score_max: float, drop_margin_max: float) -> tuple[str, str | None]:
    if top1_label == old_category and top1_score >= keep_score_min and margin >= keep_margin_min:
        return "keep_old", old_category
    if top1_label != old_category and top1_score >= relabel_score_min and margin >= relabel_margin_min:
        return "relabel_top1", top1_label
    if top1_score <= drop_score_max and margin <= drop_margin_max:
        return "drop", None
    return "still_review", None


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(in_path)
    out_dir = Path(args.output_dir) if args.output_dir else in_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    round2_path = out_dir / "manifest_clip_review_round2.parquet"
    round2_csv = out_dir / "manifest_clip_review_round2.csv"
    final_path = out_dir / "manifest_final_selected.parquet"
    final_csv = out_dir / "manifest_final_selected.csv"
    remain_path = out_dir / "manifest_round2_need_manual.parquet"
    remain_csv = out_dir / "manifest_round2_need_manual.csv"
    stats_csv = out_dir / "manifest_final_selected_stats.csv"

    df = pd.read_parquet(in_path).copy()
    if args.limit:
        df = df.head(args.limit).copy()

    required = ["photo_id", "local_image_path", "category", "review_action"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df["photo_id"] = df["photo_id"].astype(str)
    df = df[df["local_image_path"].notna()].copy()
    df["path_exists"] = df["local_image_path"].map(lambda p: Path(str(p)).exists())
    df = df[df["path_exists"] == True].copy().reset_index(drop=True)

    first_keep = df[df["review_action"].isin(["keep", "relabel"])].copy()
    to_review = df[df["review_action"].isin(["review", "error"])].copy().reset_index(drop=True)

    print(f"[INFO] first-pass keep/relabel rows: {len(first_keep)}")
    print(f"[INFO] second-pass rows to review: {len(to_review)}")

    if len(to_review) == 0:
        final_df = first_keep.copy()
        final_df["final_label"] = final_df.apply(
            lambda r: r["category"] if r["review_action"] == "keep" else r.get("final_label_after_clip_review", r.get("clip_top1_label")),
            axis=1,
        )
        final_df["final_source"] = final_df["review_action"].map({"keep": "clip_keep_round1", "relabel": "clip_relabel_round1"})
        final_df.to_parquet(final_path, index=False)
        final_df.to_csv(final_csv, index=False, encoding="utf-8-sig")
        print(f"[OK] saved final selected parquet: {final_path}")
        return

    processor = AutoProcessor.from_pretrained(args.checkpoint)
    model = AutoModelForZeroShotImageClassification.from_pretrained(args.checkpoint).to(args.device).eval()

    rows = []
    with torch.no_grad():
        for _, row in tqdm(to_review.iterrows(), total=len(to_review), desc="second-pass clip review"):
            photo_id = str(row["photo_id"])
            old_category = str(row["category"])
            image_path = str(row["local_image_path"])
            candidate_labels = get_candidates(row)
            candidate_texts = [LABEL_TO_TEXT[x] for x in candidate_labels]
            try:
                image = Image.open(image_path).convert("RGB")
                inputs = processor(images=image, text=candidate_texts, return_tensors="pt", padding=True).to(args.device)
                outputs = model(**inputs)
                probs = outputs.logits_per_image[0].softmax(dim=-1).cpu()
                k = min(2, len(candidate_texts))
                topk = torch.topk(probs, k=k)
                top1_idx = int(topk.indices[0])
                top1_text = candidate_texts[top1_idx]
                top1_label = TEXT_TO_LABEL[top1_text]
                top1_score = float(topk.values[0])
                if k >= 2:
                    top2_idx = int(topk.indices[1])
                    top2_text = candidate_texts[top2_idx]
                    top2_label = TEXT_TO_LABEL[top2_text]
                    top2_score = float(topk.values[1])
                else:
                    top2_label = None
                    top2_score = 0.0
                margin = top1_score - top2_score
                second_action, second_final = decide(
                    old_category, top1_label, top1_score, margin,
                    args.keep_score_min, args.keep_margin_min,
                    args.relabel_score_min, args.relabel_margin_min,
                    args.drop_score_max, args.drop_margin_max,
                )
                rows.append({
                    "photo_id": photo_id,
                    "clip_review2_ok": True,
                    "clip_review2_error": "",
                    "clip_review2_candidates": "|".join(candidate_labels),
                    "clip_review2_top1_label": top1_label,
                    "clip_review2_top2_label": top2_label,
                    "clip_review2_top1_score": top1_score,
                    "clip_review2_top2_score": top2_score,
                    "clip_review2_margin": margin,
                    "clip_review2_action": second_action,
                    "clip_review2_final_label": second_final,
                })
            except Exception as e:
                rows.append({
                    "photo_id": photo_id,
                    "clip_review2_ok": False,
                    "clip_review2_error": str(e)[:500],
                    "clip_review2_candidates": "|".join(candidate_labels),
                    "clip_review2_top1_label": None,
                    "clip_review2_top2_label": None,
                    "clip_review2_top1_score": None,
                    "clip_review2_top2_score": None,
                    "clip_review2_margin": None,
                    "clip_review2_action": "error",
                    "clip_review2_final_label": None,
                })

    review2_df = pd.DataFrame(rows)
    round2_df = to_review.merge(review2_df, on="photo_id", how="left")
    round2_df.to_parquet(round2_path, index=False)
    round2_df.to_csv(round2_csv, index=False, encoding="utf-8-sig")

    # first-pass accepted rows
    keep_round1 = first_keep.copy()
    keep_round1["final_label"] = keep_round1.apply(
        lambda r: r["category"] if r["review_action"] == "keep" else (r.get("final_label_after_clip_review") if pd.notna(r.get("final_label_after_clip_review")) else r.get("clip_top1_label")),
        axis=1,
    )
    keep_round1["final_source"] = keep_round1["review_action"].map({"keep": "clip_keep_round1", "relabel": "clip_relabel_round1"})

    # second-pass accepted rows
    keep_round2 = round2_df[round2_df["clip_review2_action"].isin(["keep_old", "relabel_top1"])].copy()
    keep_round2["final_label"] = keep_round2["clip_review2_final_label"]
    keep_round2["final_source"] = keep_round2["clip_review2_action"].map({"keep_old": "clip_keep_round2", "relabel_top1": "clip_relabel_round2"})

    final_df = pd.concat([keep_round1, keep_round2], ignore_index=True)
    final_df = final_df[final_df["final_label"].notna()].copy()
    final_df = final_df.drop_duplicates(subset=["photo_id"], keep="last").reset_index(drop=True)

    remain_df = round2_df[~round2_df["clip_review2_action"].isin(["keep_old", "relabel_top1"])].copy()
    remain_df.to_parquet(remain_path, index=False)
    remain_df.to_csv(remain_csv, index=False, encoding="utf-8-sig")

    final_df.to_parquet(final_path, index=False)
    final_df.to_csv(final_csv, index=False, encoding="utf-8-sig")

    stats = final_df.groupby(["final_label", "final_source"], dropna=False).size().reset_index(name="count").sort_values("count", ascending=False)
    stats.to_csv(stats_csv, index=False, encoding="utf-8-sig")

    print(f"[OK] saved round2 review parquet: {round2_path}")
    print(f"[OK] saved final selected parquet: {final_path}")
    print(f"[OK] saved remaining manual-review parquet: {remain_path}")
    print(f"[SUMMARY] final selected rows: {len(final_df)}")
    print(f"[SUMMARY] still need manual review rows: {len(remain_df)}")


if __name__ == "__main__":
    main()
