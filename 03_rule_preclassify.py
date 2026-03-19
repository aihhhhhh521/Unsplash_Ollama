from collections import defaultdict
from pathlib import Path

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
from utils import ensure_exists, norm_text, pipe_keywords_to_list, safe_str, truncate_text, json_dumps

def build_text_for_cls(row: pd.Series) -> str:
    parts = []
    fields = [
        ("photo_description", row.get("photo_description")),
        ("ai_description", row.get("ai_description")),
        ("keywords", row.get("keywords_text")),
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
    keywords = pipe_keywords_to_list(row.get("keywords_text"))
    full_text = norm_text(build_text_for_cls(row))
    keyword_set = set(keywords)

    scores = defaultdict(int)

    for label, groups in CATEGORY_VOCAB.items():
        strong = groups["strong"]
        weak = groups["weak"]

        for term in strong:
            t = term.lower()
            if t in keyword_set:
                scores[label] += RULE_WEIGHTS["keyword_strong"]
            if t in full_text:
                scores[label] += RULE_WEIGHTS["text_strong"]

        for term in weak:
            t = term.lower()
            if t in keyword_set:
                scores[label] += RULE_WEIGHTS["keyword_weak"]
            if t in full_text:
                scores[label] += RULE_WEIGHTS["text_weak"]

    # 一个小修正：generic people 不要太容易压过 architecture/nature
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

def main() -> None:
    ensure_exists(PHOTOS_NO_ART_FILE, "请先运行 02_filter_art.py")

    parquet_file = pq.ParquetFile(PHOTOS_NO_ART_FILE)
    writer_all = None
    llm_rows = []

    for batch in tqdm(parquet_file.iter_batches(batch_size=PARQUET_BATCH_SIZE), desc="rule_preclassify"):
        df = batch.to_pandas()
        extra = df.apply(score_one, axis=1, result_type="expand")
        out = pd.concat([df, extra], axis=1)

        table = pa.Table.from_pandas(out, preserve_index=False)
        if writer_all is None:
            writer_all = pq.ParquetWriter(PRECLASSIFIED_FILE, table.schema, compression="zstd")
        writer_all.write_table(table)

        pending = out[out["needs_llm"] == True]
        if len(pending) > 0:
            llm_rows.append(pending)

    if writer_all is not None:
        writer_all.close()

    if llm_rows:
        need_llm_df = pd.concat(llm_rows, ignore_index=True)
    else:
        need_llm_df = pd.DataFrame(columns=[])

    need_llm_df.to_parquet(NEED_LLM_FILE, index=False)

    total = sum(1 for _ in pq.ParquetFile(PRECLASSIFIED_FILE).iter_batches(batch_size=100000))
    print(f"[OK] 预分类文件：{PRECLASSIFIED_FILE}")
    print(f"[OK] 待送 Ollama 文件：{NEED_LLM_FILE}")
    print(f"[INFO] 待送 Ollama 数量：{len(need_llm_df)}")

if __name__ == "__main__":
    main()
