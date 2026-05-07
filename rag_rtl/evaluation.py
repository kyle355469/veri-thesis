from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .config import CacheConfig, RuntimeConfig
from .embeddings import Embedder
from .json_utils import json_default
from .pipeline import RagRtlPipeline
from .types import RtlTask
from .vector_store import VectorStore


def iter_tasks(path: str | Path) -> Iterable[RtlTask]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            yield RtlTask(
                prompt=payload["prompt"],
                target_hdl=payload.get("target_hdl", "verilog"),
                module_signature=payload.get("module_signature"),
                constraints=payload.get("constraints", []),
                max_repair_attempts=int(payload.get("max_repair_attempts", 1)),
            )


def run_evaluation(
    tasks_path: str | Path,
    store: VectorStore,
    embedder: Embedder,
    mode: str,
    output_path: str | Path,
    llm_client: Any = None,
    verifier: Any = None,
    cache_mode: str = "keywords",
    cache_reuse_threshold: float = 0.95,
    cache_evidence_threshold: float = 0.88,
) -> Dict[str, Any]:
    if mode not in {"llm_only", "rag", "rag_cache_verify"}:
        raise ValueError("mode must be one of: llm_only, rag, rag_cache_verify")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if mode == "llm_only":
        store = VectorStore([], store.vectors[:0])

    with tempfile.TemporaryDirectory(prefix="rag_rtl_eval_") as tempdir:
        cache_path = Path(tempdir) / "cache.json" if mode != "rag_cache_verify" else "data/history_cache.json"
        reuse_threshold = 2.0 if mode in {"llm_only", "rag"} else cache_reuse_threshold
        evidence_threshold = 2.0 if mode in {"llm_only", "rag"} else cache_evidence_threshold
        pipeline = RagRtlPipeline(
            store=store,
            embedder=embedder,
            llm_client=llm_client,
            verifier=verifier,
            cache_config=CacheConfig(
                path=cache_path,
                mode=cache_mode,
                reuse_threshold=reuse_threshold,
                evidence_threshold=evidence_threshold,
            ),
            runtime_config=RuntimeConfig(
                monitor_path=Path(tempdir) / "monitor.jsonl",
                failed_log_path=Path(tempdir) / "failed_attempts.jsonl",
            ),
        )

        records: List[Dict[str, Any]] = []
        start = time.perf_counter()
        for task in iter_tasks(tasks_path):
            response = pipeline.run(task, context_k=0 if mode == "llm_only" else 4)
            records.append(
                {
                    "prompt": task.prompt,
                    "syntax_passed": response.verification.syntax_passed,
                    "lint_passed": response.verification.lint_passed,
                    "passed": response.verification.passed,
                    "repair_attempts": response.repair_attempts,
                    "cache_source": response.cache_source,
                    "retrieved_doc_ids": response.retrieved_doc_ids,
                    "cache_decision": response.metadata.get("cache_decision"),
                    "best_history_match": response.metadata.get("best_history_match"),
                    "timings": response.timings,
                }
            )

    count = max(len(records), 1)
    summary = {
        "mode": mode,
        "num_tasks": len(records),
        "syntax_pass_rate": sum(item["syntax_passed"] for item in records) / count,
        "lint_pass_rate": sum(item["lint_passed"] for item in records) / count,
        "pass_rate": sum(item["passed"] for item in records) / count,
        "avg_repair_attempts": sum(item["repair_attempts"] for item in records) / count,
        "total_s": time.perf_counter() - start,
        "records": records,
    }
    output_path.write_text(json.dumps(summary, default=json_default, indent=2), encoding="utf-8")
    return summary
