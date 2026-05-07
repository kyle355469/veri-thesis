from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def dumps_json(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(value, default=json_default, ensure_ascii=False, indent=indent)


def append_jsonl(path: str | Path, record: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(dumps_json(record) + "\n")


def preview_text(text: str, limit: int = 700) -> str:
    text = str(text).strip()
    return text if len(text) <= limit else text[:limit] + "...[truncated]"
