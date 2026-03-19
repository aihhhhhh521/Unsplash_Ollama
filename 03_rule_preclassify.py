from collections import defaultdict

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from config import (
    CATEGORY_VOCAB,
    LABELS,
    PARQUET_BATCH_SIZE,
    PHOTOS_NO_ART_FILE,
    PRECLASSIFIED_FILE,
    NEED_LLM_FILE,
    RULE_DIRECT_MIN_MARGIN,
    RULE_DIRECT_MIN_SCORE,
    RULE_WEIGHTS,
    TEXT_MAX_CHARS,
)
from utils import ensure_exists, norm_text, safe_str, truncate_text, json_dumps


def build_text_for_cls(row: pd.Series) -> str:
    parts = []
    fields = [
        ("photo_description", row.get("photo_description")),
        ("ai_description", row.get("ai_description")),
        ("location_name", row.get("photo_location_name")),
        ("location_city", row.get("photo_location_city")),
        ("location_country", row.get("photo_location_country")),
        ("landmark", row.get("ai_primary_landmark_name")),
    ]
    for k, v in fields:
        sv = safe_str(v).strip()
        if sv:
            parts.append(f"{k}: {sv}")
    text = "\n".join(parts)
    return truncate_text(text, TEXT_MAX_CHARS)


def score_one(row: pd.Series) -> dict:
    full_text = norm_text(build_text_for_cls(row))
    scores = defaultdict(int)

    for label, groups in CATEGORY_VOCAB.items():
        strong = groups["strong"]
        weak = groups["weak"]

        for term in strong:
            t = term.lower()
            if t in full_text:
                scores[label] += RULE_WEIGHTS["text_strong"]

        for term in weak:
            t = term.lower()
            if t in full_text:
                scores[label] += RULE_WEIGHTS["text_weak"]

    # 人像做一点抑制，避免只因为出现 people/man/woman 就压过场景类
    if scores["人像"] > 0 and "portrait" not in full_text and "selfie" not in full_text and "face" not in full_text:
        scores["人像"] = max(0, scores["人像"] - 2)

    ranked = sorted(((k, scores.get(k, 0)) for k in LABELS), key=lambda x: x[1], reverse=True)
    top_label, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0
    margin = top_score - second_score

    direct = top_score >= RULE_DIRECT_MIN_SCORE and margin >= RULE_DIRECT_MIN_MARGIN
    confidence = min(0.96, 0.50 + top_score * 0.05 + margin * 0.03) if direct else None

    return {
        "text_for_cls": build_text_for_cls(row),
        "rule_top1_label": top_label,
        "rule_top1_score": top_score,
        "rule_top2_score": second_score,
        "rule_margin": margin,
        "rule_scores_json": json_dumps({k: scores.get(k, 0) for k in LABELS}),
        "needs_llm": not direct,
        "category": top_label if direct else None,
        "category_confidence": round(confidence, 3) if confidence is not None else None,
        "category_source": "rule" if direct else None,
    }


def normalize_batch_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    关键函数：
    把每个 batch 的 dtype 固定住，避免 ParquetWriter 因 schema 漂移报错。
    """

    # ---------- 整数列 ----------
    int_cols = [
        "photo_width",
        "photo_height",
        "stats_views",
        "stats_downloads",
        "rule_top1_score",
        "rule_top2_score",
        "rule_margin",
    ]
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    # ---------- 浮点列 ----------
    float_cols = [
        "photo_aspect_ratio",
        "exif_iso",
        "photo_location_latitude",
        "photo_location_longitude",
        "ai_primary_landmark_latitude",
        "ai_primary_landmark_longitude",
        "ai_primary_landmark_confidence",
        "category_confidence",
    ]
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    # ---------- 布尔列 ----------
    bool_cols = [
        "photo_featured",
        "needs_llm",
    ]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype("boolean")

    # ---------- 字符串列 ----------
    string_cols = [
        "photo_id",
        "photo_url",
        "photo_image_url",
        "photo_description",
        "photographer_username",
        "photographer_first_name",
        "photographer_last_name",
        "exif_camera_make",
        "exif_camera_model",
        "exif_aperture_value",
        "exif_focal_length",
        "exif_exposure_time",
        "photo_location_name",
        "photo_location_country",
        "photo_location_city",
        "ai_description",
        "ai_primary_landmark_name",
        "blur_hash",
        "text_for_cls",
        "rule_top1_label",
        "rule_scores_json",
        "category",
        "category_source",
    ]
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype("string")

    # photo_submitted_at 保持 datetime
    if "photo_submitted_at" in df.columns:
        df["photo_submitted_at"] = pd.to_datetime(df["photo_submitted_at"], errors="coerce")

    return df


def main() -> None:
    ensure_exists(PHOTOS_NO_ART_FILE, "请先运行 02_filter_art.py")

    # 如果之前跑炸过，建议手动先删掉旧文件
    if PRECLASSIFIED_FILE.exists():
        print(f"[WARN] 检测到旧文件，将覆盖写入：{PRECLASSIFIED_FILE}")
        PRECLASSIFIED_FILE.unlink()

    if NEED_LLM_FILE.exists():
        print(f"[WARN] 检测到旧文件，将覆盖写入：{NEED_LLM_FILE}")
        NEED_LLM_FILE.unlink()

    parquet_file = pq.ParquetFile(PHOTOS_NO_ART_FILE)
    writer_all = None
    llm_rows = []

    fixed_schema = None

    for batch in tqdm(parquet_file.iter_batches(batch_size=PARQUET_BATCH_SIZE), desc="rule_preclassify"):
        df = batch.to_pandas()

        extra = df.apply(score_one, axis=1, result_type="expand")
        out = pd.concat([df, extra], axis=1)

        # 统一 dtype，避免不同 batch schema 漂移
        out = normalize_batch_dtypes(out)

        table = pa.Table.from_pandas(out, preserve_index=False)

        if fixed_schema is None:
            fixed_schema = table.schema
            writer_all = pq.ParquetWriter(PRECLASSIFIED_FILE, fixed_schema, compression="zstd")
        else:
            # 强制 cast 到首个 batch 的 schema
            table = table.cast(fixed_schema)

        writer_all.write_table(table)

        pending = out[out["needs_llm"] == True]
        if len(pending) > 0:
            llm_rows.append(pending)

    if writer_all is not None:
        writer_all.close()

    if llm_rows:
        need_llm_df = pd.concat(llm_rows, ignore_index=True)
        need_llm_df = normalize_batch_dtypes(need_llm_df)
    else:
        need_llm_df = pd.DataFrame()

    need_llm_df.to_parquet(NEED_LLM_FILE, index=False)

    print(f"[OK] 预分类文件：{PRECLASSIFIED_FILE}")
    print(f"[OK] 待送 Ollama 文件：{NEED_LLM_FILE}")
    print(f"[INFO] 待送 Ollama 数量：{len(need_llm_df)}")


if __name__ == "__main__":
    main()