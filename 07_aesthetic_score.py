from __future__ import annotations

import os
from pathlib import Path
from urllib.request import urlretrieve

import torch
import torch.nn as nn
import pandas as pd
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import open_clip

ImageFile.LOAD_TRUNCATED_IMAGES = True

# =========================
# 路径配置
# =========================
MANIFEST = Path(r"D:\PyProjects\Dataset\unsplash-research-dataset-full-latest\work\dataset\metadata\dataset_metadata_downloaded.parquet")
OUT_PARQUET = MANIFEST.parent / "manifest_aesthetic.parquet"
OUT_CSV = MANIFEST.parent / "manifest_aesthetic.csv"
AESTHETIC_ONLY_PARQUET = MANIFEST.parent / "aesthetic_scores.parquet"
AESTHETIC_ONLY_CSV = MANIFEST.parent / "aesthetic_scores.csv"

# =========================
# 速度优先配置
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 256 if DEVICE == "cuda" else 32
NUM_WORKERS = 8
USE_FP16 = True if DEVICE == "cuda" else False

# 速度优先：ViT-B-32
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"
AESTHETIC_CLIP_MODEL = "vit_b_32"   # 对应 512 维线性头

# 若要更稳但更慢，可改成：
# CLIP_MODEL_NAME = "ViT-L-14"
# CLIP_PRETRAINED = "laion2b_s32b_b82k"
# AESTHETIC_CLIP_MODEL = "vit_l_14"


def get_aesthetic_model(clip_model: str = "vit_b_32") -> nn.Module:
    """
    按 LAION README 方式自动下载并加载 aesthetic predictor 线性头
    """
    cache_folder = Path.home() / ".cache" / "emb_reader"
    cache_folder.mkdir(parents=True, exist_ok=True)

    weight_path = cache_folder / f"sa_0_4_{clip_model}_linear.pth"
    if not weight_path.exists():
        url = f"https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_{clip_model}_linear.pth?raw=true"
        print(f"[INFO] downloading aesthetic weights to {weight_path}")
        urlretrieve(url, weight_path)

    if clip_model == "vit_b_32":
        model = nn.Linear(512, 1)
    elif clip_model == "vit_l_14":
        model = nn.Linear(768, 1)
    else:
        raise ValueError("clip_model must be 'vit_b_32' or 'vit_l_14'")

    state = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


class ImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, preprocess):
        self.df = df.reset_index(drop=True)
        self.preprocess = preprocess

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        photo_id = str(row["photo_id"])
        img_path = str(row["local_image_path"])

        try:
            img = Image.open(img_path).convert("RGB")
            img = self.preprocess(img)
            return {
                "photo_id": photo_id,
                "image": img,
                "ok": True,
                "error": ""
            }
        except Exception as e:
            # 用零张量占位，后面会根据 ok=False 忽略
            return {
                "photo_id": photo_id,
                "image": torch.zeros(3, 224, 224),
                "ok": False,
                "error": str(e)[:500]
            }


def collate_fn(batch):
    photo_ids = [x["photo_id"] for x in batch]
    images = torch.stack([x["image"] for x in batch], dim=0)
    oks = [x["ok"] for x in batch]
    errors = [x["error"] for x in batch]
    return photo_ids, images, oks, errors


def main():
    if not MANIFEST.exists():
        raise FileNotFoundError(f"未找到输入 parquet: {MANIFEST}")

    df = pd.read_parquet(MANIFEST).copy()
    print(f"[INFO] loaded rows: {len(df)}")

    # 只处理下载成功且本地路径存在的图片
    df = df[df["download_ok"] == True].copy()
    df = df[df["local_image_path"].notna()].copy()
    df["photo_id"] = df["photo_id"].astype(str)
    df = df.drop_duplicates(subset=["photo_id"], keep="first").reset_index(drop=True)

    # 再次校验路径存在
    df["path_exists"] = df["local_image_path"].map(lambda p: Path(str(p)).exists())
    df = df[df["path_exists"] == True].copy().reset_index(drop=True)
    print(f"[INFO] valid images: {len(df)}")

    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME,
        pretrained=CLIP_PRETRAINED,
        device=DEVICE
    )
    model.eval()

    aesthetic_head = get_aesthetic_model(AESTHETIC_CLIP_MODEL).to(DEVICE).eval()

    dataset = ImageDataset(df[["photo_id", "local_image_path"]], preprocess)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
        collate_fn=collate_fn,
        persistent_workers=(NUM_WORKERS > 0),
    )

    results = []

    use_amp = DEVICE == "cuda" and USE_FP16
    autocast_ctx = torch.cuda.amp.autocast if use_amp else torch.cpu.amp.autocast

    with torch.no_grad():
        for photo_ids, images, oks, errors in tqdm(loader, desc="aesthetic scoring"):
            images = images.to(DEVICE, non_blocking=True)

            if use_amp:
                with torch.cuda.amp.autocast():
                    feats = model.encode_image(images)
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                    scores = aesthetic_head(feats).squeeze(-1)
            else:
                feats = model.encode_image(images)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                scores = aesthetic_head(feats).squeeze(-1)

            scores = scores.float().cpu().tolist()

            for pid, ok, err, score in zip(photo_ids, oks, errors, scores):
                if ok:
                    results.append({
                        "photo_id": pid,
                        "aesthetic_ok": True,
                        "aesthetic_error": "",
                        "aesthetic_score": float(score),
                    })
                else:
                    results.append({
                        "photo_id": pid,
                        "aesthetic_ok": False,
                        "aesthetic_error": err,
                        "aesthetic_score": None,
                    })

    score_df = pd.DataFrame(results)

    out_df = df.merge(score_df, on="photo_id", how="left")
    if "path_exists" in out_df.columns:
        out_df = out_df.drop(columns=["path_exists"])

    # 排序方便人工检查
    out_df = out_df.sort_values(
        by=["aesthetic_ok", "aesthetic_score"],
        ascending=[False, False],
        na_position="last"
    ).reset_index(drop=True)

    score_df.to_parquet(AESTHETIC_ONLY_PARQUET, index=False)
    score_df.to_csv(AESTHETIC_ONLY_CSV, index=False, encoding="utf-8-sig")

    out_df.to_parquet(OUT_PARQUET, index=False)
    out_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print(f"[OK] aesthetic only parquet: {AESTHETIC_ONLY_PARQUET}")
    print(f"[OK] aesthetic only csv: {AESTHETIC_ONLY_CSV}")
    print(f"[OK] merged manifest parquet: {OUT_PARQUET}")
    print(f"[OK] merged manifest csv: {OUT_CSV}")


if __name__ == "__main__":
    main()