from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageFile
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification

ImageFile.LOAD_TRUNCATED_IMAGES = True

# =========================
# 路径
# =========================
IN_PARQUET = Path(
    r"D:\PyProjects\Dataset\unsplash-research-dataset-full-latest\work\dataset\metadata\manifest_aesthetic_keep.parquet"
)
OUT_PARQUET = IN_PARQUET.parent / "manifest_clip_review.parquet"
OUT_CSV = IN_PARQUET.parent / "manifest_clip_review.csv"

# =========================
# 模型
# =========================
CHECKPOINT = "openai/clip-vit-large-patch14"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# 五类文本提示
# =========================
CANDIDATE_TEXTS = [
    "This is a photo of urban architecture or a city scene.",
    "This is a photo of an indoor room or interior space.",
    "This is a photo of nature, landscape, plants, or animals.",
    "This is a photo of objects, products, food, or still life.",
    "This is a portrait photo of a person as the main subject.",
]

TEXT_TO_LABEL = {
    "This is a photo of urban architecture or a city scene.": "城市、建筑",
    "This is a photo of an indoor room or interior space.": "室内",
    "This is a photo of nature, landscape, plants, or animals.": "自然",
    "This is a photo of objects, products, food, or still life.": "静物",
    "This is a portrait photo of a person as the main subject.": "人像",
}

# =========================
# 阈值（起始版，后面可调）
# =========================
KEEP_SCORE_MIN = 0.55
KEEP_MARGIN_MIN = 0.15
RELABEL_SCORE_MIN = 0.60
RELABEL_MARGIN_MIN = 0.20


def main():
    if not IN_PARQUET.exists():
        raise FileNotFoundError(f"未找到输入文件: {IN_PARQUET}")

    df = pd.read_parquet(IN_PARQUET).copy()
    print(f"[INFO] loaded rows: {len(df)}")

    # 只保留图片路径存在的样本
    df = df[df["local_image_path"].notna()].copy()
    df["path_exists"] = df["local_image_path"].map(lambda p: Path(str(p)).exists())
    df = df[df["path_exists"] == True].copy().reset_index(drop=True)

    # 要有旧标签，CLIP 才能做“复查”
    df = df[df["category"].notna()].copy()
    df["photo_id"] = df["photo_id"].astype(str)
    df["category"] = df["category"].astype(str)

    print(f"[INFO] valid rows for review: {len(df)}")

    processor = AutoProcessor.from_pretrained(CHECKPOINT)
    model = AutoModelForZeroShotImageClassification.from_pretrained(CHECKPOINT).to(DEVICE).eval()

    results = []

    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), desc="clip review"):
            photo_id = row["photo_id"]
            old_category = row["category"]
            img_path = str(row["local_image_path"])

            try:
                image = Image.open(img_path).convert("RGB")

                inputs = processor(
                    images=image,
                    text=CANDIDATE_TEXTS,
                    return_tensors="pt",
                    padding=True,
                ).to(DEVICE)

                outputs = model(**inputs)
                probs = outputs.logits_per_image[0].softmax(dim=-1).cpu()

                top2 = torch.topk(probs, k=2)
                top1_idx = int(top2.indices[0])
                top2_idx = int(top2.indices[1])

                top1_text = CANDIDATE_TEXTS[top1_idx]
                top2_text = CANDIDATE_TEXTS[top2_idx]

                top1_label = TEXT_TO_LABEL[top1_text]
                top2_label = TEXT_TO_LABEL[top2_text]

                top1_score = float(top2.values[0])
                top2_score = float(top2.values[1])
                margin = top1_score - top2_score

                # 复查决策
                if top1_label == old_category and top1_score >= KEEP_SCORE_MIN and margin >= KEEP_MARGIN_MIN:
                    review_action = "keep"
                    final_label = old_category
                elif top1_label != old_category and top1_score >= RELABEL_SCORE_MIN and margin >= RELABEL_MARGIN_MIN:
                    review_action = "relabel"
                    final_label = top1_label
                else:
                    review_action = "review"
                    final_label = None

                results.append({
                    "photo_id": photo_id,
                    "clip_review_ok": True,
                    "clip_review_error": "",
                    "old_category": old_category,
                    "clip_top1_label": top1_label,
                    "clip_top2_label": top2_label,
                    "clip_top1_score": top1_score,
                    "clip_top2_score": top2_score,
                    "clip_margin": margin,
                    "review_action": review_action,
                    "final_label_after_clip_review": final_label,
                })

            except Exception as e:
                results.append({
                    "photo_id": photo_id,
                    "clip_review_ok": False,
                    "clip_review_error": str(e)[:500],
                    "old_category": old_category,
                    "clip_top1_label": None,
                    "clip_top2_label": None,
                    "clip_top1_score": None,
                    "clip_top2_score": None,
                    "clip_margin": None,
                    "review_action": "error",
                    "final_label_after_clip_review": None,
                })

    res_df = pd.DataFrame(results)
    out_df = df.merge(res_df, on=["photo_id", "old_category"], how="left")

    if "path_exists" in out_df.columns:
        out_df = out_df.drop(columns=["path_exists"])

    out_df.to_parquet(OUT_PARQUET, index=False)
    out_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print(f"[OK] saved parquet: {OUT_PARQUET}")
    print(f"[OK] saved csv: {OUT_CSV}")


if __name__ == "__main__":
    main()