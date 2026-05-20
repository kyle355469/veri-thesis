#!/usr/bin/env python3
# cd /home/kai/veri-thesis
# python3 -u scripts/run_benchmark_matrix.py \
#   --benchmark both \
#   --mode all \
#   --samples 5 \
#   --concurrency 256 \
#   --base-url http://localhost:18000/v1 \
#   --output-dir runs/benchmark_matrix_codeV > benchmark_matrix_codeV.log 
# For --mode tool, --mode full, or --mode all, start vLLM first with:
#   ENABLE_TOOL_CALLING=1 TOOL_CALL_PARSER=hermes bash vllm_deploy.sh
"""Run Verilog-Eval and RTLLM across model/RAG/tool/full pipeline modes."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rag_rtl.config import CacheConfig, FixedPipeConfig, RuntimeConfig, ToolCallingConfig
from rag_rtl.embeddings import make_embedder
from rag_rtl.json_utils import dumps_json
from rag_rtl.llm import VllmClient
from rag_rtl.pipeline import FixedPipeRtlPipeline, RagRtlPipeline
from rag_rtl.vector_store import VectorStore
from rag_rtl.verifier import RtlVerifier

PASS_AT_KS = (1, 3, 5)
BENCHMARKS = ("verilog-eval", "rtllm")
MODES = ("full", "model", "tool", "rag")
TOOL_PREFLIGHT_PROMPT = (
    "Tool calling preflight for the RTL benchmark runner. "
    "Reply with the single word OK, or call the noop tool if required."
)
TOOL_PREFLIGHT_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "noop_tool",
            "description": "No-op tool used only to verify that the vLLM server accepts tool schemas.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    }
]
TOOL_SERVER_RESTART_HINT = "ENABLE_TOOL_CALLING=1 TOOL_CALL_PARSER=hermes bash vllm_deploy.sh"
MODE_ALIASES = {
    "full-pipeline": "full",
    "only-model": "model",
    "only-tool": "tool",
    "only-rag": "rag",
}


class ToolPreflightError(RuntimeError):
    """Raised when a tool-enabled matrix mode cannot run against the current server."""


@dataclass(frozen=True)
class ModeConfig:
    pipeline: str
    tool_calling: bool
    retrieve_k: int
    context_k: int
    structure_retrieve_k: int
    structure_context_k: int
    description: str


MODE_CONFIGS: Dict[str, ModeConfig] = {
    "model": ModeConfig(
        pipeline="rag",
        tool_calling=False,
        retrieve_k=0,
        context_k=0,
        structure_retrieve_k=0,
        structure_context_k=0,
        description="direct model only: no initial RAG, no tool calls, no history cache",
    ),
    "rag": ModeConfig(
        pipeline="rag",
        tool_calling=False,
        retrieve_k=8,
        context_k=4,
        structure_retrieve_k=0,
        structure_context_k=0,
        description="model with initial spec RAG only",
    ),
    "tool": ModeConfig(
        pipeline="rag",
        tool_calling=True,
        retrieve_k=0,
        context_k=0,
        structure_retrieve_k=0,
        structure_context_k=0,
        description="model with tool calls only; tools may retrieve or verify when the model asks",
    ),
    "full": ModeConfig(
        pipeline="fixed-pipe",
        tool_calling=True,
        retrieve_k=8,
        context_k=4,
        structure_retrieve_k=8,
        structure_context_k=4,
        description="fixed-pipe: spec RAG, verification/repair, datapath RAG, second edition, tool calls",
    ),
}


class TrackingVllmClient(VllmClient):
    """OpenAI-compatible client that accumulates per-thread token usage."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "siliconmind-server",
        timeout_s: int = 1200,
        api_key: str = "EMPTY",
    ) -> None:
        super().__init__(base_url=base_url, model=model, timeout_s=timeout_s, api_key=api_key)
        self._local = threading.local()

    @classmethod
    def from_env(cls, base_url: Optional[str] = None) -> "TrackingVllmClient":
        client = cls.__new__(cls)
        base = VllmClient.from_env()
        VllmClient.__init__(
            client,
            base_url=base_url or base.base_url,
            model=base.model,
            timeout_s=base.timeout_s,
            api_key=base.api_key,
        )
        client._local = threading.local()
        return client

    def reset_usage(self) -> None:
        self._local.usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "llm_requests": 0,
            "api_usage_requests": 0,
            "estimated_usage_requests": 0,
        }

    def current_usage(self) -> Dict[str, int]:
        usage = getattr(self._local, "usage", None)
        if usage is None:
            self.reset_usage()
            usage = self._local.usage
        return dict(usage)

    def chat(
        self,
        messages: Sequence[Dict[str, Any]],
        temperature: float = 0.4,
        max_tokens: int = 32768,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        parallel_tool_calls: Optional[bool] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if parallel_tool_calls is not None:
            payload["parallel_tool_calls"] = parallel_tool_calls

        body = self._post_chat_completion(payload)
        message = body["choices"][0]["message"]
        self._add_usage(body.get("usage"), payload, message)
        return message

    def _add_usage(
        self,
        usage: Optional[Dict[str, Any]],
        payload: Dict[str, Any],
        message: Dict[str, Any],
    ) -> None:
        current = getattr(self._local, "usage", None)
        if current is None:
            self.reset_usage()
            current = self._local.usage
        current["llm_requests"] += 1

        if usage:
            prompt = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
            completion = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
            total = int(usage.get("total_tokens") or prompt + completion)
            current["api_usage_requests"] += 1
        else:
            prompt = estimate_tokens(payload)
            completion = estimate_tokens(message)
            total = prompt + completion
            current["estimated_usage_requests"] += 1

        current["prompt_tokens"] += prompt
        current["completion_tokens"] += completion
        current["total_tokens"] += total


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Verilog-Eval and RTLLM with full/model/tool/RAG modes, "
            "compute pass@1/3/5, and write per-question token/correctness tables."
        )
    )
    parser.add_argument("--benchmark", action="append", choices=[*BENCHMARKS, "both"], default=[])
    parser.add_argument("--mode", action="append", choices=[*MODES, *MODE_ALIASES.keys(), "all"], default=[])
    parser.add_argument("--output-dir", default="runs/benchmark_matrix")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--verilog-eval-root", default="/home/kai/eval_dt/VerilogEval-v2-NTU")
    parser.add_argument("--verilog-eval-v1-root", default="/home/kai/eval_dt/verilog-eval")
    parser.add_argument("--verilog-task", choices=["spec-to-rtl", "code-complete-iccad2023"], default="spec-to-rtl")
    parser.add_argument("--rtllm-root", default="/home/kai/eval_dt/RTLLM")
    parser.add_argument("--include", action="append", default=[], help="RTLLM include filter; repeatable.")

    parser.add_argument("--index", default="indexes/rtl_hash")
    parser.add_argument("--code-structure-index", default="indexes/rtl_datapath_hash")
    parser.add_argument("--embedder", default="hash")
    parser.add_argument("--cache-mode", choices=["keywords", "direct", "none"], default="none")
    parser.add_argument("--cache-reuse-threshold", type=float, default=0.95)
    parser.add_argument("--cache-evidence-threshold", type=float, default=0.88)
    parser.add_argument("--generation-temperature", type=float, default=0.4)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument(
        "--serving-url",
        "--base-url",
        dest="serving_url",
        help="OpenAI-compatible serving base URL. Overrides VLLM_BASE_URL.",
    )
    parser.add_argument("--max-repair-attempts", type=int, default=3)
    parser.add_argument("--second-edition-repair-attempts", type=int, default=2)
    parser.add_argument("--tool-choice", default="auto")
    parser.add_argument("--max-tool-rounds", type=int, default=4)
    parser.add_argument("--iverilog-bin", default="iverilog")
    parser.add_argument("--simulation-timeout-s", type=int, default=30)
    return parser


def load_script_module(name: str) -> Any:
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def normalize_benchmarks(values: Sequence[str]) -> List[str]:
    if not values or "both" in values:
        return list(BENCHMARKS)
    return list(dict.fromkeys(values))


def normalize_modes(values: Sequence[str]) -> List[str]:
    if not values or "all" in values:
        return list(MODES)
    return list(dict.fromkeys(MODE_ALIASES.get(value, value) for value in values))


def build_benchmark_args(module: Any, benchmark: str, mode: str, cli: argparse.Namespace, output_dir: Path) -> argparse.Namespace:
    args = module.build_parser().parse_args([])
    mode_config = MODE_CONFIGS[mode]
    args.output_dir = str(output_dir)
    args.pipeline = mode_config.pipeline
    args.prompt_profile = mode
    args.index = cli.index
    args.code_structure_index = cli.code_structure_index
    args.embedder = cli.embedder
    args.concurrency = cli.concurrency
    args.limit = cli.limit
    args.samples = cli.samples
    args.resume = cli.resume
    args.evaluate_only = cli.evaluate_only
    args.dry_run = cli.dry_run
    args.retrieve_k = mode_config.retrieve_k
    args.context_k = mode_config.context_k
    args.structure_retrieve_k = mode_config.structure_retrieve_k
    args.structure_context_k = mode_config.structure_context_k
    args.max_repair_attempts = cli.max_repair_attempts
    args.second_edition_repair_attempts = cli.second_edition_repair_attempts
    args.cache = str(output_dir / "history_cache.json")
    args.monitor = str(output_dir / "monitor.jsonl")
    args.failed_log = str(output_dir / "failed_attempts.jsonl")
    args.cache_mode = cli.cache_mode
    args.cache_reuse_threshold = cli.cache_reuse_threshold
    args.cache_evidence_threshold = cli.cache_evidence_threshold
    args.generation_temperature = cli.generation_temperature
    args.max_tokens = cli.max_tokens
    args.serving_url = cli.serving_url
    args.enable_tool_calling = mode_config.tool_calling
    args.tool_choice = cli.tool_choice
    args.max_tool_rounds = cli.max_tool_rounds
    args.iverilog_bin = cli.iverilog_bin
    args.simulation_timeout_s = cli.simulation_timeout_s

    if benchmark == "verilog-eval":
        args.verilog_eval_root = cli.verilog_eval_root
        args.verilog_eval_v1_root = cli.verilog_eval_v1_root
        args.task = cli.verilog_task
    else:
        args.rtllm_root = cli.rtllm_root
        args.include = cli.include
    return args


def build_pipeline(args: argparse.Namespace, client: TrackingVllmClient) -> Any:
    embedder = make_embedder(args.embedder)
    verifier = RtlVerifier()
    cache_config = CacheConfig(
        path=args.cache,
        mode=args.cache_mode,
        reuse_threshold=args.cache_reuse_threshold,
        evidence_threshold=args.cache_evidence_threshold,
    )
    runtime_config = RuntimeConfig(
        monitor_path=args.monitor,
        failed_log_path=args.failed_log,
        verbose_generation=getattr(args, "verbose_generation", False),
        generation_temperature=args.generation_temperature,
        max_tokens=args.max_tokens,
    )
    tool_config = ToolCallingConfig(
        enabled=args.enable_tool_calling,
        choice=args.tool_choice,
        max_rounds=args.max_tool_rounds,
    )
    spec_store = VectorStore.load(args.index)

    if args.pipeline == "fixed-pipe":
        structure_store = VectorStore.load(args.code_structure_index)
        return FixedPipeRtlPipeline(
            spec_store=spec_store,
            code_structure_store=structure_store,
            embedder=embedder,
            llm_client=client,
            verifier=verifier,
            cache_config=cache_config,
            runtime_config=runtime_config,
            tool_config=tool_config,
            fixed_pipe_config=FixedPipeConfig(second_edition_repair_attempts=args.second_edition_repair_attempts),
        )

    return RagRtlPipeline(
        store=spec_store,
        embedder=embedder,
        llm_client=client,
        verifier=verifier,
        cache_config=cache_config,
        runtime_config=runtime_config,
        tool_config=tool_config,
    )


def discover_work(module: Any, benchmark: str, args: argparse.Namespace, output_dir: Path) -> Tuple[List[Any], List[Any]]:
    if benchmark == "verilog-eval":
        root = module.prepare_verilog_eval_root(args, output_dir)
        problems = module.discover_problems(root, args.task, limit=args.limit)
    else:
        problems = module.discover_problems(args.rtllm_root, args.include, limit=args.limit)
    return problems, list(module.iter_work_items(problems, args.samples))


def preflight_tool_calling(client: TrackingVllmClient, args: argparse.Namespace, benchmark: str, mode: str) -> None:
    if not args.enable_tool_calling:
        return
    try:
        client.chat(
            [{"role": "user", "content": TOOL_PREFLIGHT_PROMPT}],
            temperature=0.0,
            max_tokens=8,
            tools=TOOL_PREFLIGHT_SCHEMA,
            tool_choice=args.tool_choice,
            parallel_tool_calls=False,
        )
    except RuntimeError as exc:
        raise ToolPreflightError(tool_preflight_error_message(str(exc), benchmark, mode)) from exc
    finally:
        client.reset_usage()


def tool_preflight_error_message(error: str, benchmark: str, mode: str) -> str:
    return (
        f"{benchmark}/{mode} requires a vLLM server started with tool calling enabled, "
        "but the preflight request failed.\n\n"
        f"Restart the server with:\n  {TOOL_SERVER_RESTART_HINT}\n\n"
        "This mode was skipped before submitting benchmark samples so the matrix does not "
        "report hundreds of generation failures with zero LLM requests.\n\n"
        f"Underlying error: {error}"
    )


def preflight_failure_summary(
    benchmark: str,
    mode: str,
    args: argparse.Namespace,
    output_dir: Path,
    problems: Sequence[Any],
    work_items: Sequence[Any],
    error: str,
    elapsed_s: float,
) -> Dict[str, Any]:
    skipped = len(work_items)
    return {
        "benchmark": benchmark,
        "mode": mode,
        "mode_description": MODE_CONFIGS[mode].description,
        "output_dir": str(output_dir),
        "pipeline": args.pipeline,
        "num_problems": len(problems),
        "num_records": skipped,
        "samples_per_problem": args.samples,
        "generated": 0,
        "rag_generation_passed": 0,
        "syntax_passed": 0,
        "lint_passed": 0,
        "iverilog_compiled": 0,
        "passed": 0,
        "accuracy": None,
        "pass@1": None,
        "pass@3": None,
        "pass@5": None,
        "pass_at_denominators": {str(k): 0 for k in PASS_AT_KS},
        "passfail_counts": {"preflight_error": skipped},
        "avg_prompt_tokens": 0.0,
        "avg_completion_tokens": 0.0,
        "avg_total_tokens": 0.0,
        "llm_requests": 0,
        "token_usage_sources": {"none": skipped} if skipped else {},
        "preflight_failed": True,
        "preflight_error": error,
        "total_s": elapsed_s,
        "records": [],
    }


def run_one_with_usage(
    module: Any,
    benchmark: str,
    mode: str,
    item: Any,
    pipeline: Any,
    args: argparse.Namespace,
    output_dir: Path,
    client: Optional[TrackingVllmClient],
) -> Dict[str, Any]:
    if client:
        client.reset_usage()
    t0 = time.perf_counter()
    record = module.run_one(item, pipeline, args, output_dir)
    wall_s = time.perf_counter() - t0
    usage = client.current_usage() if client else zero_usage()
    usage_source = usage_source_label(usage)
    record.update(
        {
            "benchmark": benchmark,
            "mode": mode,
            "wall_s": wall_s,
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "total_tokens": usage["total_tokens"],
            "llm_requests": usage["llm_requests"],
            "token_usage_source": usage_source,
        }
    )
    return record


def run_benchmark_mode(
    module: Any,
    benchmark: str,
    mode: str,
    cli: argparse.Namespace,
    root_output_dir: Path,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    output_dir = root_output_dir / benchmark / mode
    output_dir.mkdir(parents=True, exist_ok=True)
    args = build_benchmark_args(module, benchmark, mode, cli, output_dir)
    problems, work_items = discover_work(module, benchmark, args, output_dir)

    if args.dry_run:
        summary = {
            "benchmark": benchmark,
            "mode": mode,
            "mode_description": MODE_CONFIGS[mode].description,
            "output_dir": str(output_dir),
            "num_problems": len(problems),
            "num_records": len(work_items),
            "samples_per_problem": args.samples,
            "dry_run": True,
        }
        return summary, []

    records_path = output_dir / "records.jsonl"
    records_path.write_text("", encoding="utf-8")
    client = None if args.evaluate_only else TrackingVllmClient.from_env(base_url=cli.serving_url)
    start = time.perf_counter()

    if client is not None and args.enable_tool_calling:
        try:
            preflight_tool_calling(client, args, benchmark, mode)
        except ToolPreflightError as exc:
            elapsed_s = time.perf_counter() - start
            summary = preflight_failure_summary(
                benchmark=benchmark,
                mode=mode,
                args=args,
                output_dir=output_dir,
                problems=problems,
                work_items=work_items,
                error=str(exc),
                elapsed_s=elapsed_s,
            )
            print(f"[{benchmark}/{mode}] tool preflight failed; skipped {len(work_items)} samples.")
            print(str(exc))
            (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            (output_dir / "preflight_error.txt").write_text(str(exc) + "\n", encoding="utf-8")
            module.write_csv_summary(output_dir / "summary.csv", [])
            write_records_csv(output_dir / "records_with_tokens.csv", [])
            return summary, []

    pipeline = None if args.evaluate_only else build_pipeline(args, client)
    records: List[Dict[str, Any]] = []
    records_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=max(args.concurrency, 1)) as executor:
        futures = [
            executor.submit(run_one_with_usage, module, benchmark, mode, item, pipeline, args, output_dir, client)
            for item in work_items
        ]
        for future in as_completed(futures):
            record = future.result()
            with records_lock:
                records.append(record)
                with records_path.open("a", encoding="utf-8") as handle:
                    handle.write(dumps_json(record) + "\n")
            print(
                f"[{benchmark}/{mode}] completed {record['problem']} sample {int(record['sample']):02d}: "
                f"{record['passfail']} passed={record['passed']} tokens={record['total_tokens']}"
            )

    records.sort(key=record_sort_key)
    elapsed_s = time.perf_counter() - start
    summary = module.summarize(records, args, output_dir, elapsed_s)
    summary.update(
        {
            "benchmark": benchmark,
            "mode": mode,
            "mode_description": MODE_CONFIGS[mode].description,
            "avg_prompt_tokens": average(record.get("prompt_tokens") for record in records),
            "avg_completion_tokens": average(record.get("completion_tokens") for record in records),
            "avg_total_tokens": average(record.get("total_tokens") for record in records),
            "llm_requests": sum(int(record.get("llm_requests") or 0) for record in records),
            "token_usage_sources": count_values(record.get("token_usage_source") for record in records),
        }
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    module.write_csv_summary(output_dir / "summary.csv", records)
    write_records_csv(output_dir / "records_with_tokens.csv", records)
    return summary, records


def record_sort_key(record: Dict[str, Any]) -> Tuple[str, str, str, int]:
    return (
        str(record.get("benchmark") or ""),
        str(record.get("category") or ""),
        str(record.get("problem") or ""),
        int(record.get("sample") or 0),
    )


def zero_usage() -> Dict[str, int]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "llm_requests": 0,
        "api_usage_requests": 0,
        "estimated_usage_requests": 0,
    }


def usage_source_label(usage: Dict[str, int]) -> str:
    if usage.get("llm_requests", 0) == 0:
        return "none"
    if usage.get("estimated_usage_requests", 0) == 0:
        return "api"
    if usage.get("api_usage_requests", 0) == 0:
        return "estimated"
    return "mixed"


def estimate_tokens(value: Any) -> int:
    text = json.dumps(value, ensure_ascii=False, default=str)
    return max(1, math.ceil(len(text) / 4))


def average(values: Iterable[Any]) -> float:
    numeric = [float(value) for value in values if value is not None]
    return (sum(numeric) / len(numeric)) if numeric else 0.0


def count_values(values: Iterable[Any]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for value in values:
        key = str(value or "none")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def grouped_records(records: Sequence[Dict[str, Any]]) -> Dict[Tuple[str, str, str, str], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = {}
    for record in records:
        key = (
            str(record.get("benchmark") or ""),
            str(record.get("mode") or ""),
            str(record.get("category") or ""),
            str(record.get("problem") or ""),
        )
        grouped.setdefault(key, []).append(record)
    for items in grouped.values():
        items.sort(key=lambda item: int(item.get("sample") or 0))
    return grouped


def estimate_pass_at_k(num_samples: int, num_correct: int, k: int) -> Optional[float]:
    if num_samples < k:
        return None
    if num_correct == 0:
        return 0.0
    if num_samples - num_correct < k:
        return 1.0
    probability_all_wrong = 1.0
    for value in range(num_samples - num_correct + 1, num_samples + 1):
        probability_all_wrong *= 1.0 - (k / value)
    return 1.0 - probability_all_wrong


def build_question_rows(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for (benchmark, mode, category, problem), items in sorted(grouped_records(records).items()):
        correct = sum(1 for item in items if item.get("passed"))
        row = {
            "benchmark": benchmark,
            "mode": mode,
            "category": category,
            "problem": problem,
            "samples": len(items),
            "correct_count": correct,
            "pass@1": estimate_pass_at_k(len(items), correct, 1),
            "pass@3": estimate_pass_at_k(len(items), correct, 3),
            "pass@5": estimate_pass_at_k(len(items), correct, 5),
            "avg_prompt_tokens": average(item.get("prompt_tokens") for item in items),
            "avg_completion_tokens": average(item.get("completion_tokens") for item in items),
            "avg_total_tokens": average(item.get("total_tokens") for item in items),
            "avg_wall_s": average(item.get("wall_s") for item in items),
            "generated_count": sum(1 for item in items if item.get("generated")),
            "syntax_passed_count": sum(1 for item in items if item.get("syntax_passed")),
            "lint_passed_count": sum(1 for item in items if item.get("lint_passed")),
            "compiled_count": sum(1 for item in items if item.get("compile_returncode") == 0),
            "passfail_counts": compact_counts(count_values(item.get("passfail") for item in items)),
        }
        rows.append(row)
    return rows


def build_matrix_rows(summaries: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for summary in summaries:
        rows.append(
            {
                "benchmark": summary.get("benchmark"),
                "mode": summary.get("mode"),
                "problems": summary.get("num_problems"),
                "samples": summary.get("num_records"),
                "correct_count": summary.get("passed"),
                "accuracy": summary.get("accuracy"),
                "pass@1": summary.get("pass@1"),
                "pass@3": summary.get("pass@3"),
                "pass@5": summary.get("pass@5"),
                "avg_total_tokens": summary.get("avg_total_tokens", 0.0),
                "avg_prompt_tokens": summary.get("avg_prompt_tokens", 0.0),
                "avg_completion_tokens": summary.get("avg_completion_tokens", 0.0),
                "llm_requests": summary.get("llm_requests", 0),
                "generated": summary.get("generated"),
                "compiled": summary.get("iverilog_compiled"),
                "syntax_passed": summary.get("syntax_passed"),
                "lint_passed": summary.get("lint_passed"),
                "passfail_counts": compact_counts(summary.get("passfail_counts", {})),
            }
        )
    return rows


def compact_counts(counts: Any) -> str:
    if not isinstance(counts, dict):
        return str(counts or "")
    return " ".join(f"{key}:{value}" for key, value in sorted(counts.items()))


def write_records_csv(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    fieldnames = [
        "benchmark",
        "mode",
        "problem",
        "category",
        "sample",
        "passed",
        "passfail",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "llm_requests",
        "token_usage_source",
        "generated",
        "syntax_passed",
        "lint_passed",
        "compile_returncode",
        "simulation_returncode",
        "wall_s",
        "generated_code_path",
    ]
    write_csv(path, records, fieldnames)


def write_table_outputs(output_dir: Path, summaries: Sequence[Dict[str, Any]], records: Sequence[Dict[str, Any]]) -> None:
    matrix_rows = build_matrix_rows(summaries)
    question_rows = build_question_rows(records)
    matrix_fields = [
        "benchmark",
        "mode",
        "problems",
        "samples",
        "correct_count",
        "accuracy",
        "pass@1",
        "pass@3",
        "pass@5",
        "avg_total_tokens",
        "avg_prompt_tokens",
        "avg_completion_tokens",
        "llm_requests",
        "generated",
        "compiled",
        "syntax_passed",
        "lint_passed",
        "passfail_counts",
    ]
    question_fields = [
        "benchmark",
        "mode",
        "category",
        "problem",
        "samples",
        "correct_count",
        "pass@1",
        "pass@3",
        "pass@5",
        "avg_total_tokens",
        "avg_prompt_tokens",
        "avg_completion_tokens",
        "avg_wall_s",
        "generated_count",
        "compiled_count",
        "syntax_passed_count",
        "lint_passed_count",
        "passfail_counts",
    ]
    write_csv(output_dir / "matrix_summary.csv", matrix_rows, matrix_fields)
    write_csv(output_dir / "question_table.csv", question_rows, question_fields)
    (output_dir / "matrix_summary.md").write_text(markdown_table(matrix_fields, matrix_rows), encoding="utf-8")
    (output_dir / "question_table.md").write_text(markdown_table(question_fields, question_rows), encoding="utf-8")
    (output_dir / "summary.json").write_text(
        json.dumps({"runs": list(summaries), "matrix": matrix_rows, "questions": question_rows}, indent=2),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field)) for field in fieldnames})


def markdown_table(fieldnames: Sequence[str], rows: Sequence[Dict[str, Any]]) -> str:
    lines = [
        "| " + " | ".join(fieldnames) + " |",
        "| " + " | ".join("---" for _ in fieldnames) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(markdown_value(row.get(field)) for field in fieldnames) + " |")
    return "\n".join(lines) + "\n"


def csv_value(value: Any) -> Any:
    if isinstance(value, float):
        return f"{value:.6f}"
    return "" if value is None else value


def markdown_value(value: Any) -> str:
    if isinstance(value, float):
        text = f"{value:.4f}"
    elif value is None:
        text = "n/a"
    else:
        text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def print_matrix(rows: Sequence[Dict[str, Any]]) -> None:
    fields = ["benchmark", "mode", "problems", "samples", "correct_count", "accuracy", "pass@1", "pass@3", "pass@5", "avg_total_tokens"]
    print(markdown_table(fields, rows))


def main() -> None:
    cli = build_parser().parse_args()
    benchmarks = normalize_benchmarks(cli.benchmark)
    modes = normalize_modes(cli.mode)
    output_dir = Path(cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    modules = {
        "verilog-eval": load_script_module("run_verilog_eval"),
        "rtllm": load_script_module("run_rtllm_eval"),
    }
    summaries: List[Dict[str, Any]] = []
    all_records: List[Dict[str, Any]] = []

    for benchmark in benchmarks:
        for mode in modes:
            print(f"=== {benchmark} / {mode}: {MODE_CONFIGS[mode].description} ===")
            summary, records = run_benchmark_mode(modules[benchmark], benchmark, mode, cli, output_dir)
            summaries.append(summary)
            all_records.extend(records)

    write_table_outputs(output_dir, summaries, all_records)
    matrix_rows = build_matrix_rows(summaries)
    print_matrix(matrix_rows)
    print(f"wrote matrix and per-question tables under {output_dir}")


if __name__ == "__main__":
    main()
