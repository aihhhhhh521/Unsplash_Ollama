from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from config import (
    LABELS,
    PARQUET_BATCH_SIZE,
    PHOTOS_NO_ART_FILE,
    PRECLASSIFIED_FILE,
    NEED_LLM_FILE,
    TEXT_MAX_CHARS,
    WORK_DIR,
)
from utils import ensure_exists, norm_text, safe_str, truncate_text, json_dumps

# -------------------------------
# 更严格的规则筛选配置
# -------------------------------
# 目标不是“尽量多分”，而是“先把明显不属于五类/证据不足的样本挡在外面”。
DIRECT_MIN_SCORE = 8
DIRECT_MIN_MARGIN = 4
SCREEN_MIN_SCORE = 3
SCREEN_MIN_STRONG_HITS = 1
SCREEN_MIN_TOTAL_HITS = 2
MAX_CANDIDATE_LABELS_FOR_LLM = 3

RULE_REJECT_FILE = WORK_DIR / "rule_rejected.parquet"

# 只保留更稳的词，去掉 cafe/restaurant/table/chair/car 等高混淆项
STRICT_VOCAB = {
    "人像": {
        "strong": {
            "portrait", "selfie", "headshot", "face", "close up face",
            "close-up face", "profile portrait", "bride", "groom",
        },
        "weak": {
            "person", "people", "human", "woman", "man", "girl", "boy",
            "child", "baby", "couple", "family", "male", "female",
        },
    },
    "室内": {
        "strong": {
            "interior", "indoor", "bedroom", "living room", "kitchen",
            "bathroom", "office", "library", "hotel room", "apartment",
            "hallway", "classroom", "indoor room",
        },
        "weak": {
            "room", "indoors",
        },
    },
    "城市、建筑": {
        "strong": {
            "architecture", "skyline", "skyscraper", "building exterior",
            "street scene", "city street", "urban landscape", "bridge",
            "tower", "cathedral", "temple", "facade",
        },
        "weak": {
            "urban", "building", "downtown", "road", "station", "landmark",
        },
    },
    "自然": {
        "strong": {
            "landscape", "mountain", "waterfall", "forest", "beach", "ocean",
            "sea", "river", "lake", "sunset", "sunrise", "wildlife",
        },
        "weak": {
            "nature", "tree", "flower", "plant", "animal", "bird", "dog",
            "cat", "horse", "snow", "desert", "valley", "grass",
        },
    },
    "静物": {
        "strong": {
            "still life", "product photo", "food photography", "tabletop",
            "product shot",
        },
        "weak": {
            "food", "drink", "coffee", "tea", "fruit", "camera", "phone",
            "laptop", "watch", "bottle", "cup", "plate", "product", "object",
            "book",
        },
    },
}

# 剩余的艺术类、图形类、文档类、抽象类，直接拦掉
HARD_REJECT_TERMS = {
    "art", "artwork", "painting", "illustration", "drawing", "sculpture",
    "mural", "graffiti", "sketch", "cartoon", "anime", "poster", "collage",
    "calligraphy", "watercolor", "oil painting", "installation", "graphic",
    "logo", "icon", "typography", "screenshot", "diagram", "map", "document",
    "flyer", "menu", "brochure", "handwriting", "pattern", "texture",
    "wallpaper", "abstract", "render", "rendering", "cgi", "3d render",
}

# 对“并非主体”的弱人像做更强抑制，避免街景/室内里只因出现 person 就被当成人像
PORTRAIT_ANCHORS = {"portrait", "selfie", "headshot", "face", "close up face", "close-up face"}

# 按字段打不同权重：location_city/country 不参与规则打分，只留给 LLM 看
SCORING_FIELDS = [
    ("photo_description", 1.0),
    ("ai_description", 1.0),
    ("location_name", 0.5),
    ("landmark", 0.7),
]


_word_pat_cache: dict[str, re.Pattern] = {}


def term_in_text(text: str, term: str) -> bool:
    term = term.strip().lower()
    if not term:
        return False
    if " " in term or "-" in term:
        return term in text
    pat = _word_pat_cache.get(term)
    if pat is None:
        pat = re.compile(rf"(?<!\w){re.escape(term)}(?!\w)")
        _word_pat_cache[term] = pat
    return pat.search(text) is not None


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
    return truncate_text("\n".join(parts), TEXT_MAX_CHARS)


def get_field_texts(row: pd.Series) -> dict[str, str]:
    out = {
        "photo_description": norm_text(safe_str(row.get("photo_description"))),
        "ai_description": norm_text(safe_str(row.get("ai_description"))),
        "location_name": norm_text(safe_str(row.get("photo_location_name"))),
        "location_city": norm_text(safe_str(row.get("photo_location_city"))),
        "location_country": norm_text(safe_str(row.get("photo_location_country"))),
        "landmark": norm_text(safe_str(row.get("ai_primary_landmark_name"))),
    }
    return out


def detect_hard_reject(full_text: str) -> list[str]:
    hits = []
    for term in HARD_REJECT_TERMS:
        if term_in_text(full_text, term):
            hits.append(term)
    return sorted(set(hits))


def score_one(row: pd.Series) -> dict:
    field_texts = get_field_texts(row)
    full_text = norm_text(build_text_for_cls(row))

    scores = defaultdict(int)
    strong_hits = defaultdict(int)
    weak_hits = defaultdict(int)
    matched_terms = defaultdict(list)

    for label, groups in STRICT_VOCAB.items():
        for field_name, field_weight in SCORING_FIELDS:
            text = field_texts.get(field_name, "")
            if not text:
                continue

            for term in groups["strong"]:
                if term_in_text(text, term):
                    strong_hits[label] += 1
                    matched_terms[label].append(f"{field_name}:strong:{term}")
                    scores[label] += int(round(4 * field_weight))

            for term in groups["weak"]:
                if term_in_text(text, term):
                    weak_hits[label] += 1
                    matched_terms[label].append(f"{field_name}:weak:{term}")
                    scores[label] += int(round(1 * field_weight))

    # 更强的人像抑制：只有弱人像词、且存在其他场景/物体证据时，不让人像轻易胜出
    portrait_has_anchor = any(term_in_text(full_text, x) for x in PORTRAIT_ANCHORS)
    other_non_portrait_score = max(scores[l] for l in LABELS if l != "人像") if LABELS else 0
    if scores["人像"] > 0 and not portrait_has_anchor and strong_hits["人像"] == 0:
        if other_non_portrait_score >= scores["人像"]:
            scores["人像"] = max(0, scores["人像"] - 3)
        else:
            scores["人像"] = max(0, scores["人像"] - 2)

    # location_city / country 只用于 LLM 文本，不计入规则。若全文只剩这些字段，直接倾向拒绝。
    hard_reject_hits = detect_hard_reject(full_text)
    total_hits = sum(strong_hits.values()) + sum(weak_hits.values())

    ranked = sorted(((k, scores.get(k, 0)) for k in LABELS), key=lambda x: x[1], reverse=True)
    top_label, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0
    margin = top_score - second_score

    candidate_labels = [
        label for label, score in ranked
        if score >= max(SCREEN_MIN_SCORE, top_score - 2)
    ]

    gate_pass = True
    reject_reason = ""

    if hard_reject_hits and top_score < 8:
        gate_pass = False
        reject_reason = f"hard_reject_terms={','.join(hard_reject_hits[:6])}"
    elif top_score < SCREEN_MIN_SCORE:
        gate_pass = False
        reject_reason = "insufficient_top_score"
    elif sum(strong_hits.values()) < SCREEN_MIN_STRONG_HITS and total_hits < SCREEN_MIN_TOTAL_HITS:
        gate_pass = False
        reject_reason = "insufficient_evidence"
    elif len(candidate_labels) > MAX_CANDIDATE_LABELS_FOR_LLM:
        gate_pass = False
        reject_reason = "too_many_candidate_labels"
    elif top_label == "人像" and not portrait_has_anchor and strong_hits["人像"] == 0 and top_score < 5:
        gate_pass = False
        reject_reason = "weak_portrait_only"

    # 只有非常明确的样本才让 rule 直接定类
    direct = (
        gate_pass
        and top_score >= DIRECT_MIN_SCORE
        and margin >= DIRECT_MIN_MARGIN
        and (
            strong_hits[top_label] >= 1
            or (top_label != "人像" and weak_hits[top_label] >= 3)
        )
        and not (top_label == "人像" and not portrait_has_anchor and strong_hits["人像"] == 0)
    )

    confidence = None
    if direct:
        confidence = min(0.98, 0.58 + top_score * 0.035 + margin * 0.02 + strong_hits[top_label] * 0.03)

    return {
        "text_for_cls": build_text_for_cls(row),
        "rule_top1_label": top_label,
        "rule_top1_score": top_score,
        "rule_top2_score": second_score,
        "rule_margin": margin,
        "rule_scores_json": json_dumps({k: scores.get(k, 0) for k in LABELS}),
        "rule_candidate_labels_json": json_dumps(candidate_labels),
        "rule_matched_terms_json": json_dumps({k: matched_terms.get(k, []) for k in LABELS}),
        "rule_positive_strong_hits": int(sum(strong_hits.values())),
        "rule_positive_weak_hits": int(sum(weak_hits.values())),
        "rule_gate_pass": bool(gate_pass),
        "rule_reject_reason": reject_reason,
        "needs_llm": bool(gate_pass and not direct),
        "category": top_label if direct else None,
        "category_confidence": round(confidence, 3) if confidence is not None else None,
        "category_source": "rule" if direct else None,
    }


def normalize_batch_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    int_cols = [
        "photo_width",
        "photo_height",
        "stats_views",
        "stats_downloads",
        "rule_top1_score",
        "rule_top2_score",
        "rule_margin",
        "rule_positive_strong_hits",
        "rule_positive_weak_hits",
    ]
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

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

    bool_cols = [
        "photo_featured",
        "needs_llm",
        "rule_gate_pass",
    ]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype("boolean")

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
        "rule_candidate_labels_json",
        "rule_matched_terms_json",
        "rule_reject_reason",
        "category",
        "category_source",
    ]
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype("string")

    if "photo_submitted_at" in df.columns:
        df["photo_submitted_at"] = pd.to_datetime(df["photo_submitted_at"], errors="coerce")

    return df


def main() -> None:
    ensure_exists(PHOTOS_NO_ART_FILE, "请先运行 02_filter_art.py")

    for path in [PRECLASSIFIED_FILE, NEED_LLM_FILE, RULE_REJECT_FILE]:
        if path.exists():
            print(f"[WARN] 检测到旧文件，将覆盖写入：{path}")
            path.unlink()

    parquet_file = pq.ParquetFile(PHOTOS_NO_ART_FILE)
    writer_all = None
    fixed_schema = None
    llm_rows = []
    reject_rows = []

    for batch in tqdm(parquet_file.iter_batches(batch_size=PARQUET_BATCH_SIZE), desc="strict_rule_preclassify"):
        df = batch.to_pandas()

        extra = df.apply(score_one, axis=1, result_type="expand")
        out = pd.concat([df, extra], axis=1)
        out = normalize_batch_dtypes(out)

        table = pa.Table.from_pandas(out, preserve_index=False)
        if fixed_schema is None:
            fixed_schema = table.schema
            writer_all = pq.ParquetWriter(PRECLASSIFIED_FILE, fixed_schema, compression="zstd")
        else:
            table = table.cast(fixed_schema)
        writer_all.write_table(table)

        pending = out[out["needs_llm"] == True]
        if len(pending) > 0:
            llm_rows.append(pending)

        rejected = out[out["rule_gate_pass"] != True]
        if len(rejected) > 0:
            reject_rows.append(rejected)

    if writer_all is not None:
        writer_all.close()

    if llm_rows:
        need_llm_df = pd.concat(llm_rows, ignore_index=True)
        need_llm_df = normalize_batch_dtypes(need_llm_df)
    else:
        need_llm_df = pd.DataFrame()
    need_llm_df.to_parquet(NEED_LLM_FILE, index=False)

    if reject_rows:
        reject_df = pd.concat(reject_rows, ignore_index=True)
        reject_df = normalize_batch_dtypes(reject_df)
    else:
        reject_df = pd.DataFrame()
    reject_df.to_parquet(RULE_REJECT_FILE, index=False)

    print(f"[OK] 严格预分类文件：{PRECLASSIFIED_FILE}")
    print(f"[OK] 规则拒绝文件：{RULE_REJECT_FILE}")
    print(f"[OK] 待送 Ollama 文件：{NEED_LLM_FILE}")
    print(f"[INFO] 待送 Ollama 数量：{len(need_llm_df)}")
    print(f"[INFO] 规则直接通过数量：{int(((pd.read_parquet(PRECLASSIFIED_FILE, columns=['category'])['category'].notna()).sum()) if PRECLASSIFIED_FILE.exists() else 0)}")
    print(f"[INFO] 规则拒绝数量：{len(reject_df)}")


if __name__ == "__main__":
    main()
