
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from tqdm import tqdm

import config as cfg
from utils import ensure_exists, load_done_ids_from_jsonl, safe_str


LABELS = cfg.LABELS
NEED_LLM_FILE = cfg.NEED_LLM_FILE
OLLAMA_BASE_URL = cfg.OLLAMA_BASE_URL.rstrip("/")
OLLAMA_MODEL = cfg.OLLAMA_MODEL
OLLAMA_KEEP_ALIVE = getattr(cfg, "OLLAMA_KEEP_ALIVE", "-1")
REQUEST_TIMEOUT = getattr(cfg, "REQUEST_TIMEOUT", 180)
MAX_RETRIES = getattr(cfg, "MAX_RETRIES", 3)
MAX_WORKERS = getattr(cfg, "MAX_WORKERS", 4)
OLLAMA_RESULTS_FILE = cfg.OLLAMA_RESULTS_FILE
OLLAMA_RESULTS_JSONL = cfg.OLLAMA_RESULTS_JSONL

READ_BATCH_SIZE = getattr(cfg, "OLLAMA_READ_BATCH_SIZE", 10000)
RECORDS_PER_REQUEST = getattr(cfg, "OLLAMA_RECORDS_PER_REQUEST", 8)
MAX_IN_FLIGHT_BATCHES = getattr(cfg, "OLLAMA_MAX_IN_FLIGHT_BATCHES", max(8, MAX_WORKERS * 4))
WRITE_BUFFER_SIZE = getattr(cfg, "OLLAMA_WRITE_BUFFER_SIZE", 1000)
RETURN_REASON = getattr(cfg, "OLLAMA_RETURN_REASON", False)
USE_GENERATE_API = getattr(cfg, "OLLAMA_USE_GENERATE_API", True)

_thread_local = threading.local()

SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "photo_id": {"type": "string"},
                    "label": {"type": "string", "enum": LABELS},
                    "confidence": {"type": "number"},
                    **({"reason": {"type": "string"}} if RETURN_REASON else {}),
                },
                "required": ["photo_id", "label", "confidence"] + (["reason"] if RETURN_REASON else []),
            },
        }
    },
    "required": ["items"],
}

SYSTEM_PROMPT = (
    "你是一个严格的图片元数据分类器。"
    "输入不是图片本身，而是图片的文本元数据。"
    "你必须只返回符合 JSON Schema 的 JSON。"
    "不要输出 markdown，不要输出解释，不要输出多余文本。"
)


def get_session() -> requests.Session:
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=max(8, MAX_WORKERS * 2),
            pool_maxsize=max(8, MAX_WORKERS * 2),
            max_retries=0,
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        _thread_local.session = session
    return session


def preload_model() -> None:
    try:
        session = get_session()
        if USE_GENERATE_API:
            url = f"{OLLAMA_BASE_URL}/generate"
            payload = {
                "model": OLLAMA_MODEL,
                "prompt": "",
                "stream": False,
                "keep_alive": OLLAMA_KEEP_ALIVE,
            }
        else:
            url = f"{OLLAMA_BASE_URL}/chat"
            payload = {
                "model": OLLAMA_MODEL,
                "messages": [],
                "stream": False,
                "keep_alive": OLLAMA_KEEP_ALIVE,
            }
        resp = session.post(url, json=payload, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
    except Exception:
        pass


def build_batch_prompt(items: list[dict]) -> str:
    lines = []
    for i, item in enumerate(items, start=1):
        pid = safe_str(item["photo_id"])
        txt = safe_str(item["text_for_cls"]).strip()
        lines.append(f"[{i}] photo_id={pid}\n{txt}")

    samples = "\n\n".join(lines)

    return f"""请把下面每一条 Unsplash 图片元数据分别归为且仅归为以下 5 类之一：
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

请对每一条输入都返回一项结果。
必须原样回填对应的 photo_id。
只输出 JSON：
{{
  "items": [
    {{
      "photo_id": "...",
      "label": "自然",
      "confidence": 0.91{', "reason": "..."' if RETURN_REASON else ''}
    }}
  ]
}}

待分类样本如下：

{samples}
""".strip()


def call_ollama_batch_once(items: list[dict]) -> list[dict]:
    session = get_session()
    prompt = build_batch_prompt(items)

    if USE_GENERATE_API:
        url = f"{OLLAMA_BASE_URL}/generate"
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "system": SYSTEM_PROMPT,
            "format": SCHEMA,
            "stream": False,
            "keep_alive": OLLAMA_KEEP_ALIVE,
            "options": {"temperature": 0},
        }
    else:
        url = f"{OLLAMA_BASE_URL}/chat"
        payload = {
            "model": OLLAMA_MODEL,
            "stream": False,
            "keep_alive": OLLAMA_KEEP_ALIVE,
            "format": SCHEMA,
            "options": {"temperature": 0},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        }

    resp = session.post(url, json=payload, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()

    content = data["response"] if USE_GENERATE_API else data["message"]["content"]
    obj = json.loads(content)
    raw_items = obj.get("items", [])
    if not isinstance(raw_items, list):
        raise ValueError("返回 JSON 缺少 items 数组")

    expected_ids = [safe_str(x["photo_id"]) for x in items]
    got = {}
    for r in raw_items:
        pid = safe_str(r.get("photo_id"))
        label = r.get("label")
        conf = float(r.get("confidence", 0.0))
        reason = safe_str(r.get("reason", "")).strip() if RETURN_REASON else ""

        if pid not in expected_ids:
            continue
        if label not in LABELS:
            continue

        got[pid] = {
            "photo_id": pid,
            "ollama_label": label,
            "ollama_confidence": round(max(0.0, min(1.0, conf)), 3),
            "ollama_reason": reason,
            "ollama_ok": True,
            "ollama_error": "",
        }

    if len(got) != len(expected_ids):
        missing = [pid for pid in expected_ids if pid not in got]
        raise ValueError(f"batch 返回不完整，缺少 {len(missing)} 条: {missing[:5]}")

    return [got[pid] for pid in expected_ids]


def fail_records(items: list[dict], err: str) -> list[dict]:
    out = []
    for item in items:
        out.append(
            {
                "photo_id": safe_str(item["photo_id"]),
                "ollama_label": None,
                "ollama_confidence": 0.0,
                "ollama_reason": "",
                "ollama_ok": False,
                "ollama_error": safe_str(err)[:1000],
            }
        )
    return out


def classify_batch(items: list[dict]) -> list[dict]:
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return call_ollama_batch_once(items)
        except Exception as e:
            last_err = str(e)
            time.sleep(min(5.0, attempt * 1.2))

    if len(items) == 1:
        return fail_records(items, last_err or "unknown_error")

    mid = len(items) // 2
    left = classify_batch(items[:mid])
    right = classify_batch(items[mid:])
    return left + right


def iter_todo_items(done_ids: set[str]):
    parquet_file = pq.ParquetFile(NEED_LLM_FILE)
    for batch in parquet_file.iter_batches(batch_size=READ_BATCH_SIZE, columns=["photo_id", "text_for_cls"]):
        df = batch.to_pandas()
        if done_ids:
            df = df[~df["photo_id"].astype(str).isin(done_ids)]
        if len(df) == 0:
            continue

        rows = df[["photo_id", "text_for_cls"]].to_dict(orient="records")
        for row in rows:
            yield row


def flush_jsonl(f, buffer: list[dict]) -> None:
    if not buffer:
        return
    f.write("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in buffer))
    f.flush()
    buffer.clear()


def jsonl_to_parquet_chunked(jsonl_path, parquet_path) -> None:
    writer = None
    try:
        reader = pd.read_json(jsonl_path, lines=True, chunksize=100_000)
        for chunk in reader:
            if len(chunk) == 0:
                continue

            for col in ["photo_id", "ollama_label", "ollama_reason", "ollama_error"]:
                if col in chunk.columns:
                    chunk[col] = chunk[col].astype("string")
            if "ollama_confidence" in chunk.columns:
                chunk["ollama_confidence"] = pd.to_numeric(chunk["ollama_confidence"], errors="coerce").astype("float64")
            if "ollama_ok" in chunk.columns:
                chunk["ollama_ok"] = chunk["ollama_ok"].astype("boolean")
            if "seq" in chunk.columns:
                chunk["seq"] = pd.to_numeric(chunk["seq"], errors="coerce").astype("Int64")

            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(parquet_path, table.schema, compression="zstd")
            else:
                table = table.cast(writer.schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()


def main() -> None:
    ensure_exists(NEED_LLM_FILE, "请先运行 03_rule_preclassify.py")

    pf = pq.ParquetFile(NEED_LLM_FILE)
    total_rows = pf.metadata.num_rows

    done_ids = load_done_ids_from_jsonl(OLLAMA_RESULTS_JSONL)
    remain_est = max(0, total_rows - len(done_ids))

    print(f"[INFO] 待分类总数（parquet 行数）：{total_rows}")
    print(f"[INFO] 已完成（断点续跑）：{len(done_ids)}")
    print(f"[INFO] 估计剩余：{remain_est}")
    print(f"[INFO] RECORDS_PER_REQUEST = {RECORDS_PER_REQUEST}")
    print(f"[INFO] MAX_IN_FLIGHT_BATCHES = {MAX_IN_FLIGHT_BATCHES}")
    print(f"[INFO] MAX_WORKERS = {MAX_WORKERS}")
    print(f"[INFO] RETURN_REASON = {RETURN_REASON}")
    print(f"[INFO] USE_GENERATE_API = {USE_GENERATE_API}")

    if remain_est == 0:
        print("[INFO] 没有需要送给 Ollama 的样本。")
        if OLLAMA_RESULTS_JSONL.exists() and OLLAMA_RESULTS_JSONL.stat().st_size > 0:
            jsonl_to_parquet_chunked(OLLAMA_RESULTS_JSONL, OLLAMA_RESULTS_FILE)
        else:
            pd.DataFrame(columns=[
                "photo_id", "ollama_label", "ollama_confidence", "ollama_reason",
                "ollama_ok", "ollama_error", "seq"
            ]).to_parquet(OLLAMA_RESULTS_FILE, index=False)
        return

    preload_model()

    seq = 0
    inflight = {}
    write_buffer = []
    batch_buffer = []

    with OLLAMA_RESULTS_JSONL.open("a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        pbar = tqdm(total=remain_est, desc="ollama_rows", unit="row")

        def submit_one_batch(items: list[dict]):
            fut = ex.submit(classify_batch, items)
            inflight[fut] = len(items)

        def drain_one():
            nonlocal seq
            fut = next(as_completed(list(inflight.keys())))
            inflight.pop(fut)
            results = fut.result()
            for r in results:
                seq += 1
                r["seq"] = seq
                write_buffer.append(r)
            if len(write_buffer) >= WRITE_BUFFER_SIZE:
                flush_jsonl(f, write_buffer)
            pbar.update(len(results))

        for item in iter_todo_items(done_ids):
            batch_buffer.append(item)
            if len(batch_buffer) >= RECORDS_PER_REQUEST:
                submit_one_batch(batch_buffer)
                batch_buffer = []

                while len(inflight) >= MAX_IN_FLIGHT_BATCHES:
                    drain_one()

        if batch_buffer:
            submit_one_batch(batch_buffer)

        while inflight:
            drain_one()

        flush_jsonl(f, write_buffer)
        pbar.close()

    print("[INFO] JSONL 已写完，开始转 parquet（分块，不整表进内存）...")
    jsonl_to_parquet_chunked(OLLAMA_RESULTS_JSONL, OLLAMA_RESULTS_FILE)
    print(f"[OK] Ollama 结果文件：{OLLAMA_RESULTS_FILE}")


if __name__ == "__main__":
    main()
