from __future__ import annotations

import json
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from config import (
    LABELS,
    MAX_RETRIES,
    MAX_WORKERS,
    NEED_LLM_FILE,
    OLLAMA_BASE_URL,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_MAX_OUTPUT_TOKENS_PER_ITEM,
    OLLAMA_MODEL,
    OLLAMA_OPTIONS,
    OLLAMA_READ_BATCH_SIZE,
    OLLAMA_RESULTS_FILE,
    OLLAMA_RESULTS_JSONL,
    OLLAMA_RETURN_REASON,
    OLLAMA_TEXT_MAX_CHARS,
    OLLAMA_USE_GENERATE_API,
    REQUEST_TIMEOUT,
)
from utils import ensure_exists, json_dumps, load_done_ids_from_jsonl, safe_str, truncate_text

REJECT_LABEL = "拒绝"
ALL_OUTPUT_LABELS = LABELS + [REJECT_LABEL]

SCHEMA = {
    "type": "object",
    "properties": {
        "is_target": {"type": "boolean"},
        "label": {"type": "string", "enum": ALL_OUTPUT_LABELS},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["is_target", "label", "confidence", "reason"],
}

SYSTEM_PROMPT = (
    "你是一个非常保守的 Unsplash 图片元数据审核器。"
    "输入不是图片本身，而是图片的文本元数据与规则筛查提示。"
    "你的首要任务不是强行五分类，而是先判断它是否真的可以稳定归入目标五大类。"
    "只要元数据证据不足、主体不明确、多个大类同等合理、或明显不像真实摄影照片主体，就必须拒绝。"
    "你必须只输出符合 JSON Schema 的 JSON，不得输出任何额外文字。"
)


def short_text(s: str, max_chars: int = OLLAMA_TEXT_MAX_CHARS) -> str:
    return truncate_text(safe_str(s).strip(), max_chars)


# 对明显没法靠元数据稳定判定的样本，先本地拦掉，省一次 LLM 调用
LOCAL_REJECT_TERMS = {
    "illustration", "painting", "drawing", "poster", "logo", "graphic", "diagram",
    "map", "screenshot", "menu", "brochure", "abstract", "pattern", "texture",
    "render", "rendering", "cgi", "3d render", "wallpaper",
}


def has_local_reject_hint(text: str) -> list[str]:
    low = safe_str(text).lower()
    hits = [x for x in LOCAL_REJECT_TERMS if x in low]
    return sorted(set(hits))


CATEGORY_GUIDE = """
目标五类定义（按“主视觉主体”判定）：
1. 城市、建筑：城市室外、建筑外观、街景、桥梁、地标、校园外景、天际线；主体是 built environment，而不是室内空间、人物或单个物体。
2. 室内：室内空间本身是主体，如房间、客厅、卧室、厨房、走廊、办公室、图书馆、酒店房间；如果只是桌面小物或餐食，不算室内。
3. 自然：自然风景、山水、森林、海洋、河流、天空、沙漠、花草、动物；主体是自然环境或自然生物。
4. 静物：食物、饮品、产品、器具、桌面物品、书本、相机、手机、手表、车辆特写等 object-centric 画面；主体是物体，不是空间，不是人。
5. 人像：人或人群是主要主体，包含单人、多人、半身、全身、特写、街拍人像；只要“人明显是主体”，优先归到人像。

必须拒绝的情况：
- 元数据无法支持“主视觉主体”判断
- 同时像多个大类，且没有明显主类
- 明显像插画、海报、Logo、UI 截图、图表、文档、抽象纹理、渲染图
- 看起来可能是普通记录照，但主体并不稳定落在上述五类之一

严格要求：
- 不要因为出现城市名、国家名、地标名，就自动判“城市、建筑”
- 不要因为出现 person / people / woman / man，就自动判“人像”；只有“人是主体”才算
- 不要因为出现 room / hotel / indoor，就自动判“室内”；如果更像桌面物品、餐食、产品，则应判“静物”
- 不要因为出现 tree / flower / animal 单词，就自动判“自然”；必须看起来主体确实是自然环境或自然生物
- 证据不够就拒绝，不要硬判
""".strip()


def build_user_prompt(row: dict) -> str:
    text_for_cls = short_text(row.get("text_for_cls", ""))
    rule_top1 = safe_str(row.get("rule_top1_label"))
    rule_margin = safe_str(row.get("rule_margin"))
    rule_candidates = safe_str(row.get("rule_candidate_labels_json"))
    matched_terms = short_text(row.get("rule_matched_terms_json", ""), 1000)
    reject_reason = safe_str(row.get("rule_reject_reason"))

    reason_instruction = "reason 用一句极简中文说明主依据；若拒绝，说明拒绝原因。" if OLLAMA_RETURN_REASON else "reason 固定输出空字符串。"

    return f"""
请先判断：这条元数据是否足以支持“它属于目标五类之一”。
只有当你能相对稳定地判断主主体时，才允许输出五类标签；否则必须拒绝。

{CATEGORY_GUIDE}

输出 JSON 字段：
- is_target: true / false
- label: 若 is_target=true，则必须是五类之一；若 is_target=false，则输出“{REJECT_LABEL}”
- confidence: 0 到 1 之间，表示你对“属于目标五类且标签正确”的信心；保守打分
- reason: {reason_instruction}

规则筛查提示（仅作参考，不可盲从）：
- rule_top1_label: {rule_top1 or '无'}
- rule_margin: {rule_margin or '无'}
- rule_candidate_labels_json: {rule_candidates or '[]'}
- rule_matched_terms_json: {matched_terms or '{}'}
- rule_reject_reason: {reject_reason or '无'}

待判断元数据：
{text_for_cls}
""".strip()


EMPTY_RESULT_COLUMNS = [
    "photo_id",
    "ollama_label",
    "ollama_confidence",
    "ollama_reason",
    "ollama_raw_response",
    "ollama_ok",
    "ollama_error",
    "ollama_in_scope",
    "ollama_reject_reason",
    "ollama_called",
    "ollama_source_hint",
    "ollama_finished_at",
]


def parse_response_obj(obj: dict) -> dict:
    is_target = bool(obj.get("is_target", False))
    label = safe_str(obj.get("label", "")).strip()
    confidence = float(obj.get("confidence", 0.0))
    reason = safe_str(obj.get("reason", "")).strip()

    if label not in ALL_OUTPUT_LABELS:
        raise ValueError(f"非法 label: {label}")

    confidence = max(0.0, min(1.0, confidence))

    # 后处理：只要模型给了“拒绝”或 is_target=False，就统一视为非目标五类候选
    if (not is_target) or label == REJECT_LABEL:
        return {
            "ollama_label": None,
            "ollama_confidence": round(confidence, 3),
            "ollama_reason": reason if OLLAMA_RETURN_REASON else "",
            "ollama_in_scope": False,
            "ollama_reject_reason": reason or "llm_reject",
        }

    return {
        "ollama_label": label,
        "ollama_confidence": round(confidence, 3),
        "ollama_reason": reason if OLLAMA_RETURN_REASON else "",
        "ollama_in_scope": True,
        "ollama_reject_reason": "",
    }


def call_ollama_generate(session: requests.Session, prompt: str) -> dict:
    url = f"{OLLAMA_BASE_URL}/generate"
    options = {"temperature": 0, "num_predict": OLLAMA_MAX_OUTPUT_TOKENS_PER_ITEM}
    options.update(OLLAMA_OPTIONS or {})
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "system": SYSTEM_PROMPT,
        "stream": False,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "format": SCHEMA,
        "options": options,
    }
    resp = session.post(url, json=payload, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    content = data["response"]
    obj = json.loads(content)
    out = parse_response_obj(obj)
    out["ollama_raw_response"] = content
    return out


def call_ollama_chat(session: requests.Session, prompt: str) -> dict:
    url = f"{OLLAMA_BASE_URL}/chat"
    options = {"temperature": 0, "num_predict": OLLAMA_MAX_OUTPUT_TOKENS_PER_ITEM}
    options.update(OLLAMA_OPTIONS or {})
    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "format": SCHEMA,
        "options": options,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    }
    resp = session.post(url, json=payload, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    content = data["message"]["content"]
    obj = json.loads(content)
    out = parse_response_obj(obj)
    out["ollama_raw_response"] = content
    return out


def call_ollama_once(session: requests.Session, row: dict) -> dict:
    prompt = build_user_prompt(row)
    if OLLAMA_USE_GENERATE_API:
        return call_ollama_generate(session, prompt)
    return call_ollama_chat(session, prompt)


def local_fast_reject(row: dict) -> dict | None:
    photo_id = safe_str(row.get("photo_id"))
    text_for_cls = short_text(row.get("text_for_cls", ""))
    hits = has_local_reject_hint(text_for_cls)

    if not text_for_cls or len(text_for_cls) < 20:
        return {
            "photo_id": photo_id,
            "ollama_label": None,
            "ollama_confidence": 0.0,
            "ollama_reason": "",
            "ollama_raw_response": "",
            "ollama_ok": True,
            "ollama_error": "",
            "ollama_in_scope": False,
            "ollama_reject_reason": "metadata_too_short",
            "ollama_called": False,
            "ollama_source_hint": safe_str(row.get("rule_top1_label")),
            "ollama_finished_at": datetime.now().isoformat(timespec="seconds"),
        }

    if hits:
        return {
            "photo_id": photo_id,
            "ollama_label": None,
            "ollama_confidence": 0.0,
            "ollama_reason": "",
            "ollama_raw_response": "",
            "ollama_ok": True,
            "ollama_error": "",
            "ollama_in_scope": False,
            "ollama_reject_reason": f"local_reject_terms={','.join(hits[:6])}",
            "ollama_called": False,
            "ollama_source_hint": safe_str(row.get("rule_top1_label")),
            "ollama_finished_at": datetime.now().isoformat(timespec="seconds"),
        }

    return None


def classify_one(row: dict) -> dict:
    photo_id = safe_str(row.get("photo_id"))

    local_reject = local_fast_reject(row)
    if local_reject is not None:
        return local_reject

    session = requests.Session()
    last_err = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            out = call_ollama_once(session, row)
            return {
                "photo_id": photo_id,
                "ollama_label": out["ollama_label"],
                "ollama_confidence": out["ollama_confidence"],
                "ollama_reason": out["ollama_reason"],
                "ollama_raw_response": out["ollama_raw_response"],
                "ollama_ok": True,
                "ollama_error": "",
                "ollama_in_scope": out["ollama_in_scope"],
                "ollama_reject_reason": out["ollama_reject_reason"],
                "ollama_called": True,
                "ollama_source_hint": safe_str(row.get("rule_top1_label")),
                "ollama_finished_at": datetime.now().isoformat(timespec="seconds"),
            }
        except Exception as e:
            last_err = str(e)
            time.sleep(min(6, attempt * 1.5))

    return {
        "photo_id": photo_id,
        "ollama_label": None,
        "ollama_confidence": 0.0,
        "ollama_reason": "",
        "ollama_raw_response": "",
        "ollama_ok": False,
        "ollama_error": last_err or "unknown_error",
        "ollama_in_scope": False,
        "ollama_reject_reason": "ollama_call_failed",
        "ollama_called": True,
        "ollama_source_hint": safe_str(row.get("rule_top1_label")),
        "ollama_finished_at": datetime.now().isoformat(timespec="seconds"),
    }


def load_jsonl_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=EMPTY_RESULT_COLUMNS)

    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue

    if not records:
        return pd.DataFrame(columns=EMPTY_RESULT_COLUMNS)

    out_df = pd.DataFrame(records)
    if "photo_id" not in out_df.columns:
        return pd.DataFrame(columns=EMPTY_RESULT_COLUMNS)

    # 保留最后一次结果
    out_df = out_df.drop_duplicates(subset=["photo_id"], keep="last")
    return out_df


def main() -> None:
    ensure_exists(NEED_LLM_FILE, "请先运行 03_rule_preclassify.py")
    OLLAMA_RESULTS_JSONL.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(NEED_LLM_FILE)
    if len(df) == 0:
        print("[INFO] 没有需要送给 Ollama 的样本。")
        pd.DataFrame(columns=EMPTY_RESULT_COLUMNS).to_parquet(OLLAMA_RESULTS_FILE, index=False)
        return

    done_ids = load_done_ids_from_jsonl(OLLAMA_RESULTS_JSONL)
    todo_df = df[~df["photo_id"].astype(str).isin(done_ids)].copy()

    print(f"[INFO] 待审核总数：{len(df)}")
    print(f"[INFO] 已完成（断点续跑）：{len(done_ids)}")
    print(f"[INFO] 本轮待处理：{len(todo_df)}")
    print(f"[INFO] OLLAMA_USE_GENERATE_API={OLLAMA_USE_GENERATE_API}")

    if len(todo_df) > 0:
        keep_cols = [
            c for c in [
                "photo_id", "text_for_cls", "rule_top1_label", "rule_margin",
                "rule_candidate_labels_json", "rule_matched_terms_json", "rule_reject_reason",
            ]
            if c in todo_df.columns
        ]
        rows = []
        for chunk_start in range(0, len(todo_df), OLLAMA_READ_BATCH_SIZE):
            chunk = todo_df.iloc[chunk_start: chunk_start + OLLAMA_READ_BATCH_SIZE]
            rows.extend(chunk[keep_cols].to_dict(orient="records"))

        with OLLAMA_RESULTS_JSONL.open("a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = [ex.submit(classify_one, row) for row in rows]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="ollama_strict"):
                result = fut.result()
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()

    out_df = load_jsonl_results(OLLAMA_RESULTS_JSONL)
    if len(out_df) == 0:
        out_df = pd.DataFrame(columns=EMPTY_RESULT_COLUMNS)

    # 统一列顺序，兼容后续 merge
    for col in EMPTY_RESULT_COLUMNS:
        if col not in out_df.columns:
            out_df[col] = None
    out_df = out_df[EMPTY_RESULT_COLUMNS]
    out_df.to_parquet(OLLAMA_RESULTS_FILE, index=False)

    ok_n = int(out_df["ollama_ok"].fillna(False).sum()) if len(out_df) else 0
    in_scope_n = int(out_df["ollama_in_scope"].fillna(False).sum()) if len(out_df) else 0
    reject_n = len(out_df) - in_scope_n

    print(f"[OK] Ollama 结果文件：{OLLAMA_RESULTS_FILE}")
    print(f"[INFO] 调用成功条数：{ok_n} / {len(out_df)}")
    print(f"[INFO] 通过五类审核：{in_scope_n}")
    print(f"[INFO] 拒绝/排除：{reject_n}")


if __name__ == "__main__":
    main()
