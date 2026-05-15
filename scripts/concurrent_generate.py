from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rag_rtl.config import CacheConfig, RuntimeConfig, ToolCallingConfig
from rag_rtl.json_utils import dumps_json
from rag_rtl.embeddings import make_embedder
from rag_rtl.pipeline import RagRtlPipeline
from rag_rtl.types import RtlTask
from rag_rtl.vector_store import VectorStore
from rag_rtl.verifier import RtlVerifier


def iter_tasks(path: str | Path, limit: int | None = None) -> Iterable[RtlTask]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                break
            if not line.strip():
                continue
            payload = json.loads(line)
            yield RtlTask(
                prompt=payload["prompt"],
                target_hdl=payload.get("target_hdl", "verilog"),
                module_signature=payload.get("module_signature"),
                constraints=payload.get("constraints", []),
                max_repair_attempts=int(payload.get("max_repair_attempts", 1)),
                top_module=payload.get("top_module"),
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Maintain concurrent RAG RTL generation requests")
    parser.add_argument("--tasks", required=True, help="JSONL file with at least a prompt field per row")
    parser.add_argument("--index", default="indexes/rtl_hash")
    parser.add_argument("--embedder", default="hash")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", default="runs/concurrent_generation.jsonl")
    parser.add_argument("--cache", default="data/history_cache.json")
    parser.add_argument("--monitor", default="runs/monitor.jsonl")
    parser.add_argument("--failed-log", default="runs/failed_attempts.jsonl")
    parser.add_argument("--cache-mode", choices=["keywords", "direct", "none"], default="keywords")
    parser.add_argument("--cache-reuse-threshold", type=float, default=0.95)
    parser.add_argument("--cache-evidence-threshold", type=float, default=0.88)
    parser.add_argument("--retrieve-k", type=int, default=8)
    parser.add_argument("--context-k", type=int, default=4)
    parser.add_argument("--verbose-generation", action="store_true")
    parser.add_argument("--generation-temperature", type=float, default=0.4)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--enable-tool-calling", action="store_true")
    parser.add_argument("--tool-choice", default="auto")
    parser.add_argument("--max-tool-rounds", type=int, default=4)
    parser.add_argument("--testbench")
    parser.add_argument("--top-module")
    parser.add_argument("--test-command")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    tasks = list(iter_tasks(args.tasks, limit=args.limit))
    if args.top_module:
        for task in tasks:
            task.top_module = task.top_module or args.top_module

    embedder = make_embedder(args.embedder)
    store = VectorStore.load(args.index)
    pipeline = RagRtlPipeline(
        store=store,
        embedder=embedder,
        verifier=RtlVerifier(testbench_path=args.testbench, test_command=args.test_command),
        cache_config=CacheConfig(
            path=args.cache,
            mode=args.cache_mode,
            reuse_threshold=args.cache_reuse_threshold,
            evidence_threshold=args.cache_evidence_threshold,
        ),
        runtime_config=RuntimeConfig(
            monitor_path=args.monitor,
            failed_log_path=args.failed_log,
            verbose_generation=args.verbose_generation,
            generation_temperature=args.generation_temperature,
            max_tokens=args.max_tokens,
        ),
        tool_config=ToolCallingConfig(
            enabled=args.enable_tool_calling,
            choice=args.tool_choice,
            max_rounds=args.max_tool_rounds,
        ),
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_lock = threading.Lock()

    def run_one(index: int, task: RtlTask) -> Dict[str, Any]:
        response = pipeline.run(task, retrieve_k=args.retrieve_k, context_k=args.context_k)
        return {"index": index, "response": response}

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(run_one, index, task) for index, task in enumerate(tasks)]
        for future in as_completed(futures):
            record = future.result()
            line = dumps_json(record)
            with output_lock:
                with output_path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
            print(f"completed request {record['index']}")


if __name__ == "__main__":
    main()
