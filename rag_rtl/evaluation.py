from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from .config import CacheConfig, RuntimeConfig
from .embeddings import Embedder
from .json_utils import json_default
from .pipeline import RagRtlPipeline
from .types import RtlTask
from .vector_store import VectorStore

TASK_PROMPT_FIELDS = ("prompt", "spec", "problem", "instruction", "description")


def iter_tasks(path: str | Path) -> Iterable[RtlTask]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()

    if stripped.startswith("["):
        payload = json.loads(text)
        for index, record in enumerate(_records_from_json_payload(payload, path)):
            yield _task_from_record(record, path, index)
        return

    if stripped.startswith("{"):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            pass
        else:
            for index, record in enumerate(_records_from_json_payload(payload, path)):
                yield _task_from_record(record, path, index)
            return

    for index, record in enumerate(_records_from_jsonl(text, path)):
        yield _task_from_record(record, path, index)


def _records_from_json_payload(payload: Any, path: Path) -> Iterable[Dict[str, Any]]:
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        if _pick_text(payload, TASK_PROMPT_FIELDS):
            records = [payload]
        elif isinstance(payload.get("tasks"), list):
            records = payload["tasks"]
        elif isinstance(payload.get("records"), list):
            records = payload["records"]
        else:
            raise ValueError(
                f"{path}: JSON object must be a task with one of {TASK_PROMPT_FIELDS} "
                "or contain a 'tasks'/'records' list"
            )
    else:
        raise ValueError(f"{path}: expected a JSON array, JSON object, or JSONL task records")

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"{path}: task record {index} must be a JSON object")
        yield record


def _records_from_jsonl(text: str, path: Path) -> Iterable[Dict[str, Any]]:
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: expected one JSON task object per line") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number}: task record must be a JSON object")
        yield payload


def _task_from_record(payload: Dict[str, Any], path: Path, index: int) -> RtlTask:
    prompt = _pick_text(payload, TASK_PROMPT_FIELDS)
    if not prompt:
        raise ValueError(f"{path}: task record {index} is missing one of {TASK_PROMPT_FIELDS}")

    constraints = payload.get("constraints", [])
    if isinstance(constraints, str):
        constraints = [constraints]
    elif not isinstance(constraints, list):
        constraints = []

    return RtlTask(
        prompt=prompt,
        target_hdl=str(payload.get("target_hdl", "verilog")),
        module_signature=_stringify_field(payload.get("module_signature")) or None,
        constraints=[item for item in constraints if isinstance(item, str)],
        max_repair_attempts=int(payload.get("max_repair_attempts", 1)),
        top_module=_stringify_field(payload.get("top_module")) or None,
    )


def _pick_text(record: Dict[str, Any], fields: Sequence[str]) -> str:
    for field in fields:
        text = _stringify_field(record.get(field))
        if text:
            return text
    return ""


def _stringify_field(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts: List[str] = []
        for item in value:
            if isinstance(item, dict):
                content = item.get("content", "")
                if isinstance(content, str):
                    parts.append(content)
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts).strip()
    return ""


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
