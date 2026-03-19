import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    OLLAMA_MODEL,
    OLLAMA_RESULTS_FILE,
    OLLAMA_RESULTS_JSONL,
    REQUEST_TIMEOUT,
)
from utils import ensure_exists, load_done_ids_from_jsonl, safe_str

SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": LABELS},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["label", "confidence"],
}

SYSTEM_PROMPT = (
    "你是一个严格的图片元数据分类器。"
    "你的输入不是图片本身，而是图片的文本元数据。"
    "你必须只输出符合 JSON Schema 的 JSON，不要输出任何额外解释。"
)

def build_user_prompt(text_for_cls: str) -> str:
    return f"""请把下面这条 Unsplash 图片元数据归为且仅归为以下 5 类之一：
1. 城市、建筑
2. 室内
3. 自然
4. 静物
5. 人像

类别定义：
- 城市、建筑：城市室外、建筑外观、街道、桥梁、地标、天际线、校园外景
- 室内：室内空间是主体，如房间、咖啡馆、办公室、图书馆、酒店、走廊
- 自然：山水、森林、河流、海洋、天空、花草、动植物等自然主体
- 静物：食物、饮品、产品、器具、桌面物品、交通工具特写等物体主体
- 人像：人是主体，包括单人、多人、半身、全身、特写、抓拍

冲突判定优先级：
- 如果人物是主体，优先判为“人像”
- 否则如果室内空间是主体，判为“室内”
- 否则如果城市室外或建筑主体明显，判为“城市、建筑”
- 否则如果自然环境或动植物主体明显，判为“自然”
- 否则判为“静物”

请输出 JSON，字段：
- label: 五选一
- confidence: 0 到 1 之间的小数
- reason: 一句极简中文理由

元数据如下：
{text_for_cls}
""".strip()

def call_ollama_once(session: requests.Session, text_for_cls: str) -> dict:
    url = f"{OLLAMA_BASE_URL}/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "format": SCHEMA,
        "options": {"temperature": 0},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(text_for_cls)},
        ],
    }
    resp = session.post(url, json=payload, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    content = data["message"]["content"]
    obj = json.loads(content)

    label = obj.get("label")
    confidence = float(obj.get("confidence", 0.0))
    reason = obj.get("reason", "")

    if label not in LABELS:
        raise ValueError(f"非法 label: {label}")

    confidence = max(0.0, min(1.0, confidence))
    return {
        "label": label,
        "confidence": round(confidence, 3),
        "reason": safe_str(reason).strip(),
        "raw_response": content,
    }

def classify_one(row: dict) -> dict:
    session = requests.Session()
    photo_id = safe_str(row.get("photo_id"))
    text_for_cls = safe_str(row.get("text_for_cls"))

    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            out = call_ollama_once(session, text_for_cls)
            return {
                "photo_id": photo_id,
                "ollama_label": out["label"],
                "ollama_confidence": out["confidence"],
                "ollama_reason": out["reason"],
                "ollama_raw_response": out["raw_response"],
                "ollama_ok": True,
                "ollama_error": "",
            }
        except Exception as e:
            last_err = str(e)
            time.sleep(min(5, attempt * 1.5))

    return {
        "photo_id": photo_id,
        "ollama_label": None,
        "ollama_confidence": 0.0,
        "ollama_reason": "",
        "ollama_raw_response": "",
        "ollama_ok": False,
        "ollama_error": last_err or "unknown_error",
    }

def main() -> None:
    ensure_exists(NEED_LLM_FILE, "请先运行 03_rule_preclassify.py")

    df = pd.read_parquet(NEED_LLM_FILE)
    if len(df) == 0:
        print("[INFO] 没有需要送给 Ollama 的样本。")
        pd.DataFrame(columns=[
            "photo_id", "ollama_label", "ollama_confidence", "ollama_reason",
            "ollama_raw_response", "ollama_ok", "ollama_error"
        ]).to_parquet(OLLAMA_RESULTS_FILE, index=False)
        return

    done_ids = load_done_ids_from_jsonl(OLLAMA_RESULTS_JSONL)
    todo_df = df[~df["photo_id"].astype(str).isin(done_ids)].copy()

    print(f"[INFO] 待分类总数：{len(df)}")
    print(f"[INFO] 已完成（断点续跑）：{len(done_ids)}")
    print(f"[INFO] 本轮待调用 Ollama：{len(todo_df)}")

    if len(todo_df) > 0:
        rows = todo_df[["photo_id", "text_for_cls"]].to_dict(orient="records")
        with OLLAMA_RESULTS_JSONL.open("a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = [ex.submit(classify_one, row) for row in rows]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="ollama"):
                result = fut.result()
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()

    results = []
    with OLLAMA_RESULTS_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except Exception:
                continue

    out_df = pd.DataFrame(results).drop_duplicates(subset=["photo_id"], keep="last")
    out_df.to_parquet(OLLAMA_RESULTS_FILE, index=False)

    ok_n = int(out_df["ollama_ok"].fillna(False).sum()) if len(out_df) else 0
    print(f"[OK] Ollama 结果文件：{OLLAMA_RESULTS_FILE}")
    print(f"[INFO] 成功条数：{ok_n} / {len(out_df)}")

if __name__ == "__main__":
    main()
