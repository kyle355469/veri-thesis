from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class AgenticIpReuseConfig:
    target_hdl: str = "verilog"
    retrieve_k: int = 8
    context_k: int = 4
    max_repair_attempts: int = 2
    enable_functional_repair: bool = False
    max_functional_repair_attempts: int = 2
    temperature: float = 0.2
    max_tokens: int = 32768
    large_spec_threshold_chars: int = 40000
    large_spec_chunk_chars: int = 30000
    decomposition_mode: str = "original"
    recursive_decomposition: bool = True
    recursive_max_depth: int = 4
    recursive_max_nodes: int = 64
    testbench_dir: Optional[Path] = None
    max_generation_retries: int = 2

    def __post_init__(self) -> None:
        if self.decomposition_mode not in {"original", "chunking"}:
            raise ValueError("decomposition_mode must be 'original' or 'chunking'")
