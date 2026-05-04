from __future__ import annotations

import json
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


class Monitor:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, payload: Dict[str, Any]) -> None:
        record = {
            "time": time.time(),
            "event": event,
            **payload,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=_json_default, ensure_ascii=False) + "\n")
