#!/usr/bin/env python3
# cd /home/kai/veri-thesis
# python3 -u scripts/run_agentic_benchmark_matrix.py \
#   --benchmark both \
#   --samples 5 \
#   --concurrency 16 \
#   --base-url http://localhost:18000/v1 \
#   --output-dir runs/agentic_benchmark_matrix
# Start vLLM first with tool calling enabled, for example:
#   ENABLE_TOOL_CALLING=1 TOOL_CALL_PARSER=qwen3_xml bash vllm_deploy.sh
"""Run the full agentic RTL generator on VerilogEval-v2-NTU and RTLLM-v2."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rag_rtl.embeddings import make_embedder
from rag_rtl.json_utils import dumps_json, json_default
from rag_rtl.llm import VllmClient
from rag_rtl.retrieval_context import RetrievalContext
from rag_rtl.types import PipelineResponse, RtlTask
from rag_rtl.vector_store import VectorStore
from rag_rtl.verifier import RtlVerifier
from rtl_agent.agent import AgentConfig, AgentResult, AgenticRtlAgent
from rtl_agent.harness import (
    DEFAULT_ALLOWED_COMMANDS,
    WORKSPACE_TOOL_SCHEMAS,
    CompositeToolExecutor,
    WorkspaceToolExecutor,
)
from rag_rtl.tool_calling import RTL_TOOL_SCHEMAS

from scripts.run_benchmark_matrix import (
    ToolPreflightError,
    TrackingVllmClient,
    average,
    build_matrix_rows,
    count_values,
    preflight_tool_calling,
    print_matrix,
    usage_source_label,
    write_records_csv,
    write_table_outputs,
)


PASS_AT_KS = (1, 3, 5)
MODE = "agentic-full"
BENCHMARK_ALIASES = {
    "verilog-eval": "verilog-eval-v2-ntu",
    "verilog-eval-v2": "verilog-eval-v2-ntu",
    "verilog-eval-v2-ntu": "verilog-eval-v2-ntu",
    "rtllm": "rtllm-v2",
    "rtllm-v2": "rtllm-v2",
}
BENCHMARKS = ("verilog-eval-v2-ntu", "rtllm-v2")


class AgentFactory:
    """Build per-sample agents while sharing immutable retrieval state and the LLM client."""

    def __init__(self, cli: argparse.Namespace, client: TrackingVllmClient) -> None:
        self.cli = cli
        self.client = client
        self.embedder = make_embedder(cli.embedder)
        self.store = VectorStore.load(cli.index)
        self.retrieval_context = RetrievalContext.from_store(self.store, self.embedder)

    def build(self, *, workspace_root: Path, top_module: Optional[str]) -> AgenticRtlAgent:
        verifier = RtlVerifier(
            yosys_bin=self.cli.yosys_bin,
            verilator_bin=self.cli.verilator_bin,
            timeout_s=self.cli.timeout_s,
        )
        rtl_tools = self.retrieval_context.tool_executor(verifier=verifier, default_top_module=top_module)
        workspace_tools = WorkspaceToolExecutor(
            root=workspace_root,
            allowed_commands=sorted(DEFAULT_ALLOWED_COMMANDS | set(self.cli.allow_command or [])),
            timeout_s=self.cli.command_timeout_s,
            max_output_chars=self.cli.command_max_output_chars,
        )
        return AgenticRtlAgent(
            llm_client=self.client,
            tool_executor=CompositeToolExecutor(rtl_tools, workspace_tools),
            verifier=verifier,
            config=AgentConfig(
                temperature=self.cli.temperature,
                max_tokens=self.cli.max_tokens,
                tool_choice=self.cli.tool_choice,
                max_steps=self.cli.max_steps,
                target_hdl="verilog",
                final_verify=True,
            ),
            tool_schemas=[*RTL_TOOL_SCHEMAS, *WORKSPACE_TOOL_SCHEMAS],
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate only the full agentic RTL generator on VerilogEval-v2-NTU "
            "and RTLLM-v2, with pass@1/3/5 tables like run_benchmark_matrix.py."
        )
    )
    parser.add_argument("--benchmark", action="append", choices=[*BENCHMARK_ALIASES.keys(), "both"], default=[])
    parser.add_argument("--output-dir", default="runs/agentic_benchmark_matrix")
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
    parser.add_argument("--embedder", default="hash")
    parser.add_argument(
        "--serving-url",
        "--base-url",
        dest="serving_url",
        help="OpenAI-compatible serving base URL. Overrides VLLM_BASE_URL.",
    )
    parser.add_argument("--model", help="Served model name. Defaults to VLLM_MODEL.")
    parser.add_argument("--api-key", help="API key. Defaults to VLLM_API_KEY or EMPTY.")
    parser.add_argument("--llm-timeout-s", type=int, default=1200)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--tool-choice", default="auto")
    parser.add_argument("--max-steps", type=int, default=8)

    parser.add_argument("--yosys-bin", default="yosys")
    parser.add_argument("--verilator-bin", default="verilator")
    parser.add_argument("--timeout-s", type=int, default=30)
    parser.add_argument("--iverilog-bin", default="iverilog")
    parser.add_argument("--simulation-timeout-s", type=int, default=30)
    parser.add_argument("--keep-vcd", action="store_true", help="Keep VerilogEval VCD output.")
    parser.add_argument("--keep-waveforms", action="store_true", help="Keep RTLLM waveform output.")
    parser.add_argument("--allow-command", action="append", default=[])
    parser.add_argument("--command-timeout-s", type=int, default=20)
    parser.add_argument("--command-max-output-chars", type=int, default=6000)
    return parser


def normalize_benchmarks(values: Sequence[str]) -> List[str]:
    if not values or "both" in values:
        return list(BENCHMARKS)
    normalized = [BENCHMARK_ALIASES[value] for value in values]
    return list(dict.fromkeys(normalized))


def load_script_module(name: str) -> Any:
    import importlib.util

    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def build_tracking_client(cli: argparse.Namespace) -> TrackingVllmClient:
    client = TrackingVllmClient.from_env(base_url=cli.serving_url)
    if cli.model:
        client.model = cli.model
    if cli.api_key:
        client.api_key = cli.api_key
    client.timeout_s = cli.llm_timeout_s
    return client


def build_benchmark_args(module: Any, benchmark: str, cli: argparse.Namespace, output_dir: Path) -> argparse.Namespace:
    args = module.build_parser().parse_args([])
    args.output_dir = str(output_dir)
    args.pipeline = MODE
    args.prompt_profile = MODE
    args.index = cli.index
    args.embedder = cli.embedder
    args.concurrency = cli.concurrency
    args.limit = cli.limit
    args.samples = cli.samples
    args.resume = cli.resume
    args.evaluate_only = cli.evaluate_only
    args.dry_run = cli.dry_run
    args.iverilog_bin = cli.iverilog_bin
    args.simulation_timeout_s = cli.simulation_timeout_s
    args.keep_vcd = cli.keep_vcd
    args.keep_waveforms = cli.keep_waveforms
    args.max_repair_attempts = 0

    if benchmark == "verilog-eval-v2-ntu":
        args.verilog_eval_root = cli.verilog_eval_root
        args.verilog_eval_v1_root = cli.verilog_eval_v1_root
        args.task = cli.verilog_task
    else:
        args.rtllm_root = cli.rtllm_root
        args.include = cli.include
    return args


def discover_work(module: Any, benchmark: str, args: argparse.Namespace, output_dir: Path) -> Tuple[List[Any], List[Any]]:
    if benchmark == "verilog-eval-v2-ntu":
        root = module.prepare_verilog_eval_root(args, output_dir)
        problems = module.discover_problems(root, args.task, limit=args.limit)
    else:
        problems = module.discover_problems(args.rtllm_root, args.include, limit=args.limit)
    return problems, list(module.iter_work_items(problems, args.samples))


def run_benchmark(
    module: Any,
    benchmark: str,
    cli: argparse.Namespace,
    root_output_dir: Path,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    output_dir = root_output_dir / benchmark / MODE
    output_dir.mkdir(parents=True, exist_ok=True)
    args = build_benchmark_args(module, benchmark, cli, output_dir)
    problems, work_items = discover_work(module, benchmark, args, output_dir)

    if args.dry_run:
        return {
            "benchmark": benchmark,
            "mode": MODE,
            "mode_description": "full agentic generation with RTL and workspace tools",
            "output_dir": str(output_dir),
            "num_problems": len(problems),
            "num_records": len(work_items),
            "samples_per_problem": args.samples,
            "dry_run": True,
        }, []

    records_path = output_dir / "records.jsonl"
    records_path.write_text("", encoding="utf-8")
    client = None if args.evaluate_only else build_tracking_client(cli)
    start = time.perf_counter()

    if client is not None:
        try:
            preflight_tool_calling(
                client,
                argparse.Namespace(enable_tool_calling=True, tool_choice=cli.tool_choice),
                benchmark,
                MODE,
            )
        except ToolPreflightError as exc:
            elapsed_s = time.perf_counter() - start
            summary = preflight_failure_summary(benchmark, args, output_dir, problems, work_items, str(exc), elapsed_s)
            print(f"[{benchmark}/{MODE}] tool preflight failed; skipped {len(work_items)} samples.")
            print(str(exc))
            (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            (output_dir / "preflight_error.txt").write_text(str(exc) + "\n", encoding="utf-8")
            module.write_csv_summary(output_dir / "summary.csv", [])
            write_records_csv(output_dir / "records_with_tokens.csv", [])
            return summary, []

    factory = None if args.evaluate_only else AgentFactory(cli, client)
    records: List[Dict[str, Any]] = []
    records_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=max(args.concurrency, 1)) as executor:
        futures = [
            executor.submit(run_one_agentic, module, benchmark, item, args, output_dir, factory, client)
            for item in work_items
        ]
        for future in as_completed(futures):
            record = future.result()
            with records_lock:
                records.append(record)
                with records_path.open("a", encoding="utf-8") as handle:
                    handle.write(dumps_json(record) + "\n")
            print(
                f"[{benchmark}/{MODE}] completed {record['problem']} sample {int(record['sample']):02d}: "
                f"{record['passfail']} passed={record['passed']} "
                f"repairs={repair_attempts_label(record.get('repair_attempts'))} "
                f"tokens={record['total_tokens']}"
            )

    records.sort(key=record_sort_key)
    elapsed_s = time.perf_counter() - start
    summary = module.summarize(records, args, output_dir, elapsed_s)
    summary.update(
        {
            "benchmark": benchmark,
            "mode": MODE,
            "mode_description": "full agentic generation with RTL and workspace tools",
            "avg_prompt_tokens": average(record.get("prompt_tokens") for record in records),
            "avg_completion_tokens": average(record.get("completion_tokens") for record in records),
            "avg_total_tokens": average(record.get("total_tokens") for record in records),
            "llm_requests": sum(int(record.get("llm_requests") or 0) for record in records),
            "token_usage_sources": count_values(record.get("token_usage_source") for record in records),
            "avg_agent_steps": average(record.get("agent_steps") for record in records),
            "agent_used_tools": sum(1 for record in records if record.get("agent_used_tools")),
        }
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    module.write_csv_summary(output_dir / "summary.csv", records)
    write_records_csv(output_dir / "records_with_tokens.csv", records)
    write_agent_records_csv(output_dir / "agent_records.csv", records)
    return summary, records


def run_one_agentic(
    module: Any,
    benchmark: str,
    item: Any,
    args: argparse.Namespace,
    output_dir: Path,
    factory: Optional[AgentFactory],
    client: Optional[TrackingVllmClient],
) -> Dict[str, Any]:
    code_path = module.generated_code_path(output_dir, item)
    gen_log = generation_log_path(module, benchmark, output_dir, item)
    eval_log = evaluation_log_path(module, benchmark, output_dir, item)
    code_path.parent.mkdir(parents=True, exist_ok=True)

    response: Optional[PipelineResponse] = None
    agent_result: Optional[AgentResult] = None
    generation_error: Optional[str] = None
    reused_existing = False
    usage = zero_usage()
    usage_source = "none"
    wall_s = 0.0

    if (args.resume or args.evaluate_only) and code_path.exists():
        code = code_path.read_text(encoding="utf-8")
        reused_existing = True
    elif args.evaluate_only:
        code = ""
        generation_error = f"missing generated file: {code_path}"
    else:
        assert factory is not None
        assert client is not None
        client.reset_usage()
        t0 = time.perf_counter()
        code, response, agent_result, generation_error = run_agent_generation(
            module=module,
            benchmark=benchmark,
            item=item,
            args=args,
            output_dir=output_dir,
            factory=factory,
        )
        wall_s = time.perf_counter() - t0
        usage = client.current_usage()
        usage_source = usage_source_label(usage)
        if code:
            code_path.write_text(code, encoding="utf-8")

    module.write_generation_log(gen_log, item, response, generation_error, reused_existing)
    agent_report_path = write_agent_report(output_dir, benchmark, item, agent_result, generation_error)

    if not code:
        eval_result = empty_generation_result(module, benchmark, generation_error or "generation produced empty code")
        write_empty_evaluation_log(module, benchmark, eval_log, eval_result)
    else:
        eval_result = module.evaluate_with_iverilog(item, code_path, eval_log, args)

    record = build_record(
        module=module,
        benchmark=benchmark,
        item=item,
        code_path=code_path,
        gen_log=gen_log,
        eval_log=eval_log,
        response=response,
        agent_result=agent_result,
        agent_report_path=agent_report_path,
        generation_error=generation_error,
        generated=bool(code),
        reused_existing=reused_existing,
        eval_result=eval_result,
        usage=usage,
        usage_source=usage_source,
        wall_s=wall_s,
    )
    return record


def run_agent_generation(
    *,
    module: Any,
    benchmark: str,
    item: Any,
    args: argparse.Namespace,
    output_dir: Path,
    factory: AgentFactory,
) -> Tuple[str, Optional[PipelineResponse], Optional[AgentResult], Optional[str]]:
    task: RtlTask = module.build_task(item.problem, args)
    workspace_root = agent_workspace_root(output_dir, benchmark, item)
    workspace_root.mkdir(parents=True, exist_ok=True)
    try:
        agent = factory.build(workspace_root=workspace_root, top_module=task.top_module)
        result = agent.run(task)
    except Exception as exc:  # noqa: BLE001 - keep the benchmark moving.
        return "", None, None, f"{exc}\n{traceback.format_exc()[-4000:]}"

    code = normalize_agent_code(module, benchmark, result.rtl, item, args)
    response = PipelineResponse(
        rtl=code,
        verification=result.verification,
        retrieved_doc_ids=[],
        cache_source="agentic",
        repair_attempts=0,
        llm_actions=[event.to_dict() for event in result.events],
        prompt=task.prompt,
        timings={},
        metadata={
            "agent_steps": result.steps,
            "agent_used_tools": result.used_tools,
            "agent_stopped_reason": result.stopped_reason,
            "agent_workspace_root": str(workspace_root),
        },
    )
    return code, response, result, None


def normalize_agent_code(module: Any, benchmark: str, rtl: str, item: Any, args: argparse.Namespace) -> str:
    if not rtl:
        return ""
    if benchmark == "verilog-eval-v2-ntu":
        return module.normalize_generated_code(rtl, item.problem.module_signature, args.top_module)
    return module.normalize_generated_code(rtl, item.problem.top_module)


def build_record(
    *,
    module: Any,
    benchmark: str,
    item: Any,
    code_path: Path,
    gen_log: Path,
    eval_log: Path,
    response: Optional[PipelineResponse],
    agent_result: Optional[AgentResult],
    agent_report_path: Optional[Path],
    generation_error: Optional[str],
    generated: bool,
    reused_existing: bool,
    eval_result: Any,
    usage: Dict[str, int],
    usage_source: str,
    wall_s: float,
) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "benchmark": benchmark,
        "mode": MODE,
        "problem": item.problem.problem_id,
        "category": getattr(item.problem, "category", ""),
        "sample": item.sample,
        "generated_code_path": str(code_path),
        "generation_log_path": str(gen_log),
        "generated": generated,
        "generation_error": generation_error,
        "reused_existing": reused_existing,
        **module.response_metadata(response),
        "verification_diagnostics": module.verification_diagnostics(response.verification if response else None),
        "passed": eval_result.passed,
        "passfail": eval_result.passfail,
        "compile_returncode": eval_result.compile_returncode,
        "simulation_returncode": eval_result.simulation_returncode,
        "compile_s": eval_result.compile_s,
        "simulation_s": eval_result.simulation_s,
        "compile_command": eval_result.compile_command,
        "run_command": eval_result.run_command,
        "stdout_tail": eval_result.stdout[-4000:],
        "stderr_tail": eval_result.stderr[-4000:],
        "evaluation_error": eval_result.error,
        "agent_report_path": str(agent_report_path) if agent_report_path else None,
        "agent_steps": agent_result.steps if agent_result else None,
        "agent_used_tools": agent_result.used_tools if agent_result else False,
        "agent_stopped_reason": agent_result.stopped_reason if agent_result else None,
        "agent_event_counts": agent_event_counts(agent_result),
        "wall_s": wall_s,
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "total_tokens": usage["total_tokens"],
        "llm_requests": usage["llm_requests"],
        "token_usage_source": usage_source,
    }

    if benchmark == "verilog-eval-v2-ntu":
        base.update(
            {
                "prompt_path": str(item.problem.prompt_path),
                "testbench_path": str(item.problem.testbench_path),
                "reference_path": str(item.problem.reference_path),
                "compile_log_path": str(eval_log),
                "mismatches": eval_result.mismatches,
                "num_test_samples": eval_result.samples,
            }
        )
    else:
        base.update(
            {
                "top_module": item.problem.top_module,
                "description_path": str(item.problem.description_path),
                "testbench_path": str(item.problem.testbench_path),
                "reference_path": str(item.problem.reference_path),
                "simulation_log_path": str(eval_log),
                "failures": eval_result.failures,
            }
        )
    return base


def preflight_failure_summary(
    benchmark: str,
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
        "mode": MODE,
        "mode_description": "full agentic generation with RTL and workspace tools",
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


def generation_log_path(module: Any, benchmark: str, output_dir: Path, item: Any) -> Path:
    if benchmark == "verilog-eval-v2-ntu":
        return module.generate_log_path(output_dir, item)
    return module.generation_log_path(output_dir, item)


def evaluation_log_path(module: Any, benchmark: str, output_dir: Path, item: Any) -> Path:
    if benchmark == "verilog-eval-v2-ntu":
        return module.compile_log_path(output_dir, item)
    return module.simulation_log_path(output_dir, item)


def empty_generation_result(module: Any, benchmark: str, error: str) -> Any:
    if benchmark == "verilog-eval-v2-ntu":
        return module.IverilogResult(
            passed=False,
            passfail="G",
            compile_returncode=None,
            simulation_returncode=None,
            mismatches=None,
            samples=None,
            compile_command=[],
            run_command=[],
            error=error,
        )
    return module.SimulationResult(
        passed=False,
        passfail="G",
        compile_returncode=None,
        simulation_returncode=None,
        failures=None,
        compile_command=[],
        run_command=[],
        error=error,
    )


def write_empty_evaluation_log(module: Any, benchmark: str, path: Path, result: Any) -> None:
    if benchmark == "verilog-eval-v2-ntu":
        module.write_compile_log(path, result)
    else:
        module.write_simulation_log(path, result)


def write_agent_report(
    output_dir: Path,
    benchmark: str,
    item: Any,
    result: Optional[AgentResult],
    error: Optional[str],
) -> Optional[Path]:
    if result is None and not error:
        return None
    path = agent_report_path(output_dir, benchmark, item)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "benchmark": benchmark,
        "mode": MODE,
        "problem": item.problem.problem_id,
        "category": getattr(item.problem, "category", ""),
        "sample": item.sample,
        "error": error,
        "result": result.to_dict() if result else None,
    }
    path.write_text(json.dumps(payload, default=json_default, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def agent_report_path(output_dir: Path, benchmark: str, item: Any) -> Path:
    return agent_workspace_root(output_dir, benchmark, item).parent / f"{sample_stem(item)}-agent-report.json"


def agent_workspace_root(output_dir: Path, benchmark: str, item: Any) -> Path:
    category = getattr(item.problem, "category", "")
    parts = ["_agent_workspaces", benchmark]
    if category:
        parts.extend(Path(category).parts)
    parts.extend([item.problem.problem_id, f"sample{int(item.sample):02d}"])
    return output_dir.joinpath(*parts)


def sample_stem(item: Any) -> str:
    return f"{item.problem.problem_id}_sample{int(item.sample):02d}"


def agent_event_counts(result: Optional[AgentResult]) -> Dict[str, int]:
    if result is None:
        return {}
    return dict(Counter(event.event for event in result.events))


def zero_usage() -> Dict[str, int]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "llm_requests": 0,
        "api_usage_requests": 0,
        "estimated_usage_requests": 0,
    }


def repair_attempts_label(value: Any) -> str:
    return "n/a" if value is None else str(value)


def record_sort_key(record: Dict[str, Any]) -> Tuple[str, str, str, int]:
    return (
        str(record.get("benchmark") or ""),
        str(record.get("category") or ""),
        str(record.get("problem") or ""),
        int(record.get("sample") or 0),
    )


def write_agent_records_csv(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    fieldnames = [
        "benchmark",
        "mode",
        "category",
        "problem",
        "sample",
        "passed",
        "passfail",
        "agent_steps",
        "agent_used_tools",
        "agent_stopped_reason",
        "llm_requests",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "wall_s",
        "generated_code_path",
        "agent_report_path",
    ]
    rows = [{field: csv_value(record.get(field)) for field in fieldnames} for record in records]
    path.parent.mkdir(parents=True, exist_ok=True)
    import csv

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def csv_value(value: Any) -> Any:
    if isinstance(value, float):
        return f"{value:.6f}"
    return "" if value is None else value


def main() -> None:
    cli = build_parser().parse_args()
    benchmarks = normalize_benchmarks(cli.benchmark)
    output_dir = Path(cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    modules = {
        "verilog-eval-v2-ntu": load_script_module("run_verilog_eval"),
        "rtllm-v2": load_script_module("run_rtllm_eval"),
    }
    summaries: List[Dict[str, Any]] = []
    all_records: List[Dict[str, Any]] = []

    print(f"agentic model={cli.model or os.getenv('VLLM_MODEL', 'siliconmind-server')}")
    print(f"endpoint={cli.serving_url or os.getenv('VLLM_BASE_URL', VllmClient().base_url)}")
    for benchmark in benchmarks:
        print(f"=== {benchmark} / {MODE}: full agentic generation ===")
        summary, records = run_benchmark(modules[benchmark], benchmark, cli, output_dir)
        summaries.append(summary)
        all_records.extend(records)

    write_table_outputs(output_dir, summaries, all_records)
    matrix_rows = build_matrix_rows(summaries)
    print_matrix(matrix_rows)
    print(f"wrote agentic benchmark tables under {output_dir}")


if __name__ == "__main__":
    main()
