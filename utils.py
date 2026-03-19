import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd

def ensure_exists(path: Path, message: str | None = None) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(message or f"File not found: {path}")

def safe_str(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x)

def norm_text(s: Any) -> str:
    s = safe_str(s).lower()
    s = s.replace("|", " ")
    s = re.sub(r"[_/\\]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def truncate_text(s: str, max_chars: int) -> str:
    if len(s) <= max_chars:
        return s
    return s[:max_chars]

def json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)

def load_done_ids_from_jsonl(path: Path) -> set[str]:
    done = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                pid = str(obj.get("photo_id", "")).strip()
                if pid:
                    done.add(pid)
            except Exception:
                continue
    return done

def pipe_keywords_to_list(s: Any) -> list[str]:
    raw = safe_str(s)
    if not raw:
        return []
    return [x.strip().lower() for x in raw.split("|") if x.strip()]
