from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Dict

from .json_utils import dumps_json


class Monitor:
    _global_lock = threading.Lock()

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, payload: Dict[str, Any]) -> None:
        record = {
            "time": time.time(),
            "event": event,
            **payload,
        }
        with self._global_lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(dumps_json(record) + "\n")
