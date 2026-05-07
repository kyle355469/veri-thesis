from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


PathLike = str | Path


@dataclass(frozen=True)
class CacheConfig:
    path: PathLike = "data/history_cache.json"
    threshold: Optional[float] = None
    reuse_threshold: float = 0.95
    evidence_threshold: float = 0.88
    mode: str = "keywords"


@dataclass(frozen=True)
class RuntimeConfig:
    monitor_path: PathLike = "runs/monitor.jsonl"
    failed_log_path: PathLike = "runs/failed_attempts.jsonl"
    verbose_generation: bool = False


@dataclass(frozen=True)
class ToolCallingConfig:
    enabled: bool = False
    choice: Any = "auto"
    max_rounds: int = 4


@dataclass(frozen=True)
class FixedPipeConfig:
    yosys_bin: str = "yosys"
    yosys_timeout_s: int = 30
    second_edition_repair_attempts: int = 1
