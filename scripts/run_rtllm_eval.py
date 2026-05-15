# runnable
# python3 scripts/run_rtllm_eval.py \
#   --pipeline fixed-pipe \
#   --index indexes/rtl_hash \
#   --code-structure-index indexes/rtl_datapath_hash \
#   --concurrency 4 \
#   --output-dir runs/rtllm_eval \

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
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
from rag_rtl.llm import extract_code
from rag_rtl.pipeline import FixedPipeRtlPipeline, RagRtlPipeline
from rag_rtl.types import PipelineResponse, RtlTask, VerificationReport
from rag_rtl.vector_store import VectorStore
from rag_rtl.verifier import RtlVerifier

MODULE_RE = re.compile(r"(?m)^\s*module\s+([A-Za-z_][A-Za-z0-9_$]*)\b")
TEST_DESIGN_RE = re.compile(r"(?m)^\s*TEST_DESIGN\s*=\s*([A-Za-z_][A-Za-z0-9_$]*)\s*$")
DESCRIPTION_MODULE_RE = re.compile(
    r"(?is)\bmodule\s+name\s*:\s*(?:\r?\n|\s)+([A-Za-z_][A-Za-z0-9_$]*)"
)
INSTANCE_RE = re.compile(
    r"(?ms)^\s*([A-Za-z_][A-Za-z0-9_$]*)\s*(?:#\s*\(.*?\)\s*)?([A-Za-z_][A-Za-z0-9_$]*)\s*\("
)
VERILOG_KEYWORDS = {
    "always",
    "assign",
    "begin",
    "case",
    "else",
    "end",
    "for",
    "forever",
    "function",
    "if",
    "initial",
    "module",
    "repeat",
    "task",
    "while",
}
PASS_RE = re.compile(r"Your\s+Design\s+Passed", re.IGNORECASE)
FAILURE_COUNT_RE = re.compile(r"Test\s+completed\s+with\s+(\d+)\s*(?:/|\s+)?", re.IGNORECASE)
PASS_AT_KS = (1, 3, 5)


@dataclass(frozen=True)
class RtlLmProblem:
    problem_id: str
    category: str
    root_dir: Path
    description_path: Path
    testbench_path: Path
    reference_path: Path
    prompt: str
    top_module: str


@dataclass(frozen=True)
class WorkItem:
    problem: RtlLmProblem
    sample: int


@dataclass
class SimulationResult:
    passed: bool
    passfail: str
    compile_returncode: Optional[int]
    simulation_returncode: Optional[int]
    failures: Optional[int]
    compile_command: List[str]
    run_command: List[str]
    stdout: str = ""
    stderr: str = ""
    compile_s: float = 0.0
    simulation_s: float = 0.0
    error: Optional[str] = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run RTLLM through the veri-thesis RTL generator in parallel, "
            "then evaluate each generated design with its RTLLM testbench."
        )
    )
    parser.add_argument("--rtllm-root", default="/home/kai/eval_dt/RTLLM")
    parser.add_argument("--output-dir", default="runs/rtllm_eval")
    parser.add_argument("--pipeline", choices=["rag", "fixed-pipe"], default="rag")
    parser.add_argument("--index", default="indexes/rtl_hash")
    parser.add_argument("--code-structure-index", default="indexes/rtl_datapath_hash")
    parser.add_argument("--embedder", default="hash")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Case-insensitive substring filter for problem id or category; repeatable.",
    )
    parser.add_argument("--resume", action="store_true", help="Reuse existing generated .v files in output-dir")
    parser.add_argument("--evaluate-only", action="store_true", help="Skip generation and evaluate existing .v files")
    parser.add_argument("--dry-run", action="store_true", help="Only discover dataset records and print the count")
    parser.add_argument("--retrieve-k", type=int, default=8)
    parser.add_argument("--context-k", type=int, default=4)
    parser.add_argument("--structure-retrieve-k", type=int, default=8)
    parser.add_argument("--structure-context-k", type=int, default=4)
    parser.add_argument("--max-repair-attempts", type=int, default=3)
    parser.add_argument("--second-edition-repair-attempts", type=int, default=1)
    parser.add_argument("--cache", default="data/history_cache.json")
    parser.add_argument("--monitor", default="runs/rtllm_eval_monitor.jsonl")
    parser.add_argument("--failed-log", default="runs/rtllm_eval_failed_attempts.jsonl")
    parser.add_argument("--cache-mode", choices=["keywords", "direct", "none"], default="keywords")
    parser.add_argument("--cache-reuse-threshold", type=float, default=0.95)
    parser.add_argument("--cache-evidence-threshold", type=float, default=0.88)
    parser.add_argument("--generation-temperature", type=float, default=0.4)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--verbose-generation", action="store_true")
    parser.add_argument("--enable-tool-calling", action="store_true")
    parser.add_argument("--tool-choice", default="auto")
    parser.add_argument("--max-tool-rounds", type=int, default=4)
    parser.add_argument("--iverilog-bin", default="iverilog")
    parser.add_argument("--simulation-timeout-s", type=int, default=30)
    parser.add_argument("--keep-waveforms", action="store_true")
    return parser


def discover_problems(root: str | Path, include: Sequence[str], limit: Optional[int] = None) -> List[RtlLmProblem]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"RTLLM root not found: {root}")

    filters = [item.lower() for item in include if item.strip()]
    problems: List[RtlLmProblem] = []
    for description_path in sorted(root.rglob("design_description.txt")):
        if any(part.startswith("_") or part == ".git" for part in description_path.relative_to(root).parts):
            continue
        problem_dir = description_path.parent
        testbench_path = problem_dir / "testbench.v"
        reference_candidates = sorted(problem_dir.glob("verified_*.v"))
        if not testbench_path.exists() or not reference_candidates:
            continue
        problem_id = problem_dir.name
        category = str(problem_dir.parent.relative_to(root))
        haystack = f"{problem_id} {category}".lower()
        if filters and not any(item in haystack for item in filters):
            continue
        prompt = description_path.read_text(encoding="utf-8").strip()
        top_module = infer_top_module(problem_dir, testbench_path, prompt, problem_id)
        problems.append(
            RtlLmProblem(
                problem_id=problem_id,
                category=category,
                root_dir=problem_dir,
                description_path=description_path,
                testbench_path=testbench_path,
                reference_path=reference_candidates[0],
                prompt=prompt,
                top_module=top_module,
            )
        )
        if limit is not None and len(problems) >= limit:
            break
    return problems


def infer_top_module(problem_dir: Path, testbench_path: Path, prompt: str, fallback: str) -> str:
    testbench_module = infer_instantiated_module(testbench_path)
    if testbench_module:
        return testbench_module
    match = DESCRIPTION_MODULE_RE.search(prompt)
    if match:
        return match.group(1)
    makefile = problem_dir / "makefile"
    if makefile.exists():
        match = TEST_DESIGN_RE.search(makefile.read_text(encoding="utf-8", errors="ignore"))
        if match:
            return match.group(1)
    return fallback


def infer_instantiated_module(testbench_path: Path) -> Optional[str]:
    text = testbench_path.read_text(encoding="utf-8", errors="ignore")
    for module_name, instance_name in INSTANCE_RE.findall(text):
        lower_module = module_name.lower()
        lower_instance = instance_name.lower()
        if lower_module in VERILOG_KEYWORDS:
            continue
        if lower_instance in {"uut", "dut", "u0"} or lower_instance.startswith("u_"):
            return module_name
    return None


def build_pipeline(args: argparse.Namespace) -> Any:
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
        verbose_generation=args.verbose_generation,
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
            verifier=verifier,
            cache_config=cache_config,
            runtime_config=runtime_config,
            tool_config=tool_config,
            fixed_pipe_config=FixedPipeConfig(second_edition_repair_attempts=args.second_edition_repair_attempts),
        )

    return RagRtlPipeline(
        store=spec_store,
        embedder=embedder,
        verifier=verifier,
        cache_config=cache_config,
        runtime_config=runtime_config,
        tool_config=tool_config,
    )


def build_task(problem: RtlLmProblem, args: argparse.Namespace) -> RtlTask:
    constraints = [
        f"Return a complete Verilog module named {problem.top_module}.",
        "Use exactly the port names and widths requested in the design description.",
        "Do not include a testbench, reference module, markdown fences, or explanatory text.",
    ]
    return RtlTask(
        prompt=problem.prompt,
        target_hdl="verilog",
        constraints=constraints,
        max_repair_attempts=args.max_repair_attempts,
        top_module=problem.top_module,
    )


def output_problem_dir(output_dir: Path, problem: RtlLmProblem) -> Path:
    return output_dir / problem.category / problem.problem_id


def sample_stem(problem: RtlLmProblem, sample: int) -> str:
    return f"{problem.problem_id}_sample{sample:02d}"


def generated_code_path(output_dir: Path, item: WorkItem) -> Path:
    return output_problem_dir(output_dir, item.problem) / f"{sample_stem(item.problem, item.sample)}.v"


def generation_log_path(output_dir: Path, item: WorkItem) -> Path:
    return output_problem_dir(output_dir, item.problem) / f"{sample_stem(item.problem, item.sample)}-generate.log"


def simulation_log_path(output_dir: Path, item: WorkItem) -> Path:
    return output_problem_dir(output_dir, item.problem) / f"{sample_stem(item.problem, item.sample)}-iverilog.log"


def run_generation(item: WorkItem, pipeline: Any, args: argparse.Namespace) -> Tuple[str, Optional[PipelineResponse], Optional[str]]:
    task = build_task(item.problem, args)
    try:
        if args.pipeline == "fixed-pipe":
            response = pipeline.run(
                task,
                retrieve_k=args.retrieve_k,
                context_k=args.context_k,
                structure_retrieve_k=args.structure_retrieve_k,
                structure_context_k=args.structure_context_k,
            )
        else:
            response = pipeline.run(task, retrieve_k=args.retrieve_k, context_k=args.context_k)
    except Exception as exc:  # noqa: BLE001 - keep the benchmark moving.
        return "", None, str(exc)

    code = normalize_generated_code(response.rtl, item.problem.top_module)
    return code, response, None


def normalize_generated_code(code: str, top_module: str) -> str:
    extracted = extract_code(code).strip()
    return ensure_top_module_name(extracted, top_module)


def ensure_top_module_name(code: str, top_module: str) -> str:
    module_names = MODULE_RE.findall(code)
    if not module_names or top_module in module_names:
        return code
    first_name = module_names[0]
    pattern = re.compile(rf"(?m)^(\s*module\s+){re.escape(first_name)}\b")
    return pattern.sub(lambda match: f"{match.group(1)}{top_module}", code, count=1)


def evaluate_with_iverilog(
    item: WorkItem,
    candidate_path: Path,
    log_path: Path,
    args: argparse.Namespace,
) -> SimulationResult:
    if shutil.which(args.iverilog_bin) is None and not Path(args.iverilog_bin).exists():
        result = SimulationResult(
            passed=False,
            passfail="I",
            compile_returncode=None,
            simulation_returncode=None,
            failures=None,
            compile_command=[args.iverilog_bin],
            run_command=[],
            error=f"iverilog binary not found: {args.iverilog_bin}",
        )
        write_simulation_log(log_path, result)
        return result

    work_dir = candidate_path.parent
    exe_path = (work_dir / sample_stem(item.problem, item.sample)).resolve()
    compile_command = [
        args.iverilog_bin,
        "-Wall",
        "-Winfloop",
        "-Wno-timescale",
        "-g2012",
        "-o",
        str(exe_path),
        str(candidate_path.resolve()),
        str(item.problem.testbench_path.resolve()),
    ]
    t0 = time.perf_counter()
    compile_completed = subprocess.run(
        compile_command,
        check=False,
        capture_output=True,
        text=True,
        cwd=work_dir,
    )
    compile_s = time.perf_counter() - t0
    stdout = compile_completed.stdout or ""
    stderr = compile_completed.stderr or ""

    run_command: List[str] = []
    simulation_returncode: Optional[int] = None
    simulation_s = 0.0
    timed_out = False
    if compile_completed.returncode == 0:
        run_command = [str(exe_path)]
        t0 = time.perf_counter()
        try:
            run_completed = subprocess.run(
                run_command,
                check=False,
                capture_output=True,
                text=True,
                timeout=args.simulation_timeout_s,
                cwd=work_dir,
            )
            simulation_returncode = run_completed.returncode
            stdout += run_completed.stdout or ""
            stderr += run_completed.stderr or ""
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout += exc.stdout or ""
            stderr += (exc.stderr or "") + f"\nTIMEOUT after {args.simulation_timeout_s}s"
        simulation_s = time.perf_counter() - t0

    failures = parse_failure_count(stdout + "\n" + stderr)
    passfail = classify_result(compile_completed.returncode, simulation_returncode, timed_out, stdout, stderr, failures)
    result = SimulationResult(
        passed=passfail == ".",
        passfail=passfail,
        compile_returncode=compile_completed.returncode,
        simulation_returncode=simulation_returncode,
        failures=failures,
        compile_command=compile_command,
        run_command=run_command,
        stdout=stdout,
        stderr=stderr,
        compile_s=compile_s,
        simulation_s=simulation_s,
    )
    write_simulation_log(log_path, result)
    cleanup_simulation_artifacts(work_dir, keep=args.keep_waveforms)
    return result


def parse_failure_count(text: str) -> Optional[int]:
    match = FAILURE_COUNT_RE.search(text)
    if not match:
        return None
    return int(match.group(1))


def classify_result(
    compile_returncode: int,
    simulation_returncode: Optional[int],
    timed_out: bool,
    stdout: str,
    stderr: str,
    failures: Optional[int],
) -> str:
    text = stdout + "\n" + stderr
    lowered = text.lower()
    if "syntax error" in lowered:
        return "S"
    if "unknown module type" in lowered or "unable to bind" in lowered:
        return "m"
    if compile_returncode != 0:
        return "C"
    if timed_out or "timeout" in lowered:
        return "T"
    if simulation_returncode not in {0, None}:
        return "X"
    if PASS_RE.search(text):
        return "."
    if failures == 0:
        return "."
    if failures is not None:
        return "R"
    if "error" in lowered or "failed" in lowered or "failure" in lowered:
        return "R"
    return "?"


def cleanup_simulation_artifacts(work_dir: Path, keep: bool) -> None:
    if keep:
        return
    for pattern in ("*.vcd", "*.vpd", "*.fst", "*.lxt", "*.lxt2"):
        for path in work_dir.glob(pattern):
            try:
                path.unlink()
            except OSError:
                pass


def write_generation_log(
    path: Path,
    item: WorkItem,
    response: Optional[PipelineResponse],
    error: Optional[str],
    reused: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"problem = {item.problem.problem_id}",
        f"category = {item.problem.category}",
        f"sample = {item.sample:02d}",
        f"top_module = {item.problem.top_module}",
        f"description = {item.problem.description_path}",
        "prompt_tokens = 0",
        "resp_tokens = 0",
        "cost = 0.0",
        f"reused_existing = {str(reused).lower()}",
    ]
    if error:
        lines.append(f"error = {error}")
    if response:
        lines.extend(
            [
                f"syntax_passed = {response.verification.syntax_passed}",
                f"lint_passed = {response.verification.lint_passed}",
                f"verification_passed = {response.verification.passed}",
                f"cache_source = {response.cache_source}",
                f"repair_attempts = {response.repair_attempts}",
                f"retrieved_doc_ids = {json.dumps(response.retrieved_doc_ids)}",
                f"timings = {dumps_json(response.timings)}",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_simulation_log(path: Path, result: SimulationResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    parts = [
        "$ " + " ".join(result.compile_command),
        result.stdout,
        result.stderr,
        f"compile_returncode = {result.compile_returncode}",
        f"simulation_returncode = {result.simulation_returncode}",
        f"passfail = {result.passfail}",
    ]
    if result.error:
        parts.append(f"error = {result.error}")
    path.write_text("\n".join(part for part in parts if part is not None), encoding="utf-8")


def response_metadata(response: Optional[PipelineResponse]) -> Dict[str, Any]:
    if not response:
        return {
            "rag_generation_passed": False,
            "syntax_passed": False,
            "lint_passed": False,
            "repair_attempts": None,
            "cache_source": None,
            "retrieved_doc_ids": [],
            "timings": {},
        }
    verification = response.verification
    return {
        "rag_generation_passed": verification.passed,
        "syntax_passed": verification.syntax_passed,
        "lint_passed": verification.lint_passed,
        "repair_attempts": response.repair_attempts,
        "cache_source": response.cache_source,
        "retrieved_doc_ids": response.retrieved_doc_ids,
        "timings": response.timings,
    }


def verification_diagnostics(report: Optional[VerificationReport]) -> List[Dict[str, Any]]:
    if not report:
        return []
    return [
        {
            "tool": diagnostic.tool,
            "passed": diagnostic.passed,
            "returncode": diagnostic.returncode,
            "missing": diagnostic.missing,
            "stdout_tail": diagnostic.stdout[-2000:],
            "stderr_tail": diagnostic.stderr[-2000:],
        }
        for diagnostic in report.diagnostics
    ]


def run_one(item: WorkItem, pipeline: Any, args: argparse.Namespace, output_dir: Path) -> Dict[str, Any]:
    out_dir = output_problem_dir(output_dir, item.problem)
    out_dir.mkdir(parents=True, exist_ok=True)
    code_path = generated_code_path(output_dir, item)
    gen_log = generation_log_path(output_dir, item)
    sim_log = simulation_log_path(output_dir, item)

    response: Optional[PipelineResponse] = None
    generation_error: Optional[str] = None
    reused_existing = False

    if (args.resume or args.evaluate_only) and code_path.exists():
        code = code_path.read_text(encoding="utf-8")
        reused_existing = True
    elif args.evaluate_only:
        code = ""
        generation_error = f"missing generated file: {code_path}"
    else:
        code, response, generation_error = run_generation(item, pipeline, args)
        if code:
            code_path.write_text(code, encoding="utf-8")

    write_generation_log(gen_log, item, response, generation_error, reused_existing)

    if not code:
        sim_result = SimulationResult(
            passed=False,
            passfail="G",
            compile_returncode=None,
            simulation_returncode=None,
            failures=None,
            compile_command=[],
            run_command=[],
            error=generation_error or "generation produced empty code",
        )
        write_simulation_log(sim_log, sim_result)
    else:
        sim_result = evaluate_with_iverilog(item, code_path, sim_log, args)

    return {
        "problem": item.problem.problem_id,
        "category": item.problem.category,
        "sample": item.sample,
        "top_module": item.problem.top_module,
        "description_path": str(item.problem.description_path),
        "testbench_path": str(item.problem.testbench_path),
        "reference_path": str(item.problem.reference_path),
        "generated_code_path": str(code_path),
        "generation_log_path": str(gen_log),
        "simulation_log_path": str(sim_log),
        "generated": bool(code),
        "generation_error": generation_error,
        "reused_existing": reused_existing,
        **response_metadata(response),
        "verification_diagnostics": verification_diagnostics(response.verification if response else None),
        "passed": sim_result.passed,
        "passfail": sim_result.passfail,
        "compile_returncode": sim_result.compile_returncode,
        "simulation_returncode": sim_result.simulation_returncode,
        "failures": sim_result.failures,
        "compile_s": sim_result.compile_s,
        "simulation_s": sim_result.simulation_s,
        "compile_command": sim_result.compile_command,
        "run_command": sim_result.run_command,
        "stdout_tail": sim_result.stdout[-4000:],
        "stderr_tail": sim_result.stderr[-4000:],
        "evaluation_error": sim_result.error,
    }


def summarize(records: Sequence[Dict[str, Any]], args: argparse.Namespace, output_dir: Path, elapsed_s: float) -> Dict[str, Any]:
    count = len(records)
    denom = max(count, 1)
    passfail_counts: Dict[str, int] = {}
    for record in records:
        key = str(record.get("passfail") or "?")
        passfail_counts[key] = passfail_counts.get(key, 0) + 1
    pass_at_rates, pass_at_denominators = compute_pass_at(records, PASS_AT_KS)

    syntax_success_by_problem = {
        record["problem"] for record in records if record.get("compile_returncode") == 0
    }
    func_success_by_problem = {
        record["problem"] for record in records if record.get("passed")
    }
    return {
        "rtllm_root": str(Path(args.rtllm_root)),
        "pipeline": args.pipeline,
        "output_dir": str(output_dir),
        "num_records": count,
        "num_problems": len({record["problem"] for record in records}),
        "samples_per_problem": args.samples,
        "generated": sum(1 for record in records if record.get("generated")),
        "rag_generation_passed": sum(1 for record in records if record.get("rag_generation_passed")),
        "syntax_passed": sum(1 for record in records if record.get("syntax_passed")),
        "lint_passed": sum(1 for record in records if record.get("lint_passed")),
        "iverilog_compiled": sum(1 for record in records if record.get("compile_returncode") == 0),
        "passed": sum(1 for record in records if record.get("passed")),
        "accuracy": sum(1 for record in records if record.get("passed")) / denom,
        "pass@1": pass_at_rates[1],
        "pass@3": pass_at_rates[3],
        "pass@5": pass_at_rates[5],
        "pass_at_denominators": {str(k): pass_at_denominators[k] for k in PASS_AT_KS},
        "syntax_success_problem_count": len(syntax_success_by_problem),
        "function_success_problem_count": len(func_success_by_problem),
        "passfail_counts": dict(sorted(passfail_counts.items())),
        "total_s": elapsed_s,
        "records": list(records),
    }


def group_records_by_problem(records: Sequence[Dict[str, Any]]) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for record in records:
        key = (str(record.get("category") or ""), str(record["problem"]))
        grouped.setdefault(key, []).append(record)
    for problem_records in grouped.values():
        problem_records.sort(key=lambda item: int(item.get("sample") or 0))
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


def compute_pass_at(records: Sequence[Dict[str, Any]], ks: Sequence[int]) -> Tuple[Dict[int, Optional[float]], Dict[int, int]]:
    grouped = group_records_by_problem(records)
    rates: Dict[int, Optional[float]] = {}
    denominators: Dict[int, int] = {}
    for k in ks:
        estimates = [
            estimate
            for problem_records in grouped.values()
            if (estimate := estimate_pass_at_k(
                len(problem_records),
                sum(1 for record in problem_records if record.get("passed")),
                k,
            ))
            is not None
        ]
        denominators[k] = len(estimates)
        rates[k] = (sum(estimates) / len(estimates)) if estimates else None
    return rates, denominators


def format_summary_metric(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def write_csv_summary(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    fieldnames = [
        "problem",
        "category",
        "sample",
        "top_module",
        "passed",
        "passfail",
        "failures",
        "generated",
        "rag_generation_passed",
        "syntax_passed",
        "lint_passed",
        "repair_attempts",
        "cache_source",
        "compile_returncode",
        "simulation_returncode",
        "generated_code_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field) for field in fieldnames})


def iter_work_items(problems: Sequence[RtlLmProblem], samples: int) -> Iterable[WorkItem]:
    for problem in problems:
        for sample in range(1, samples + 1):
            yield WorkItem(problem=problem, sample=sample)


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    problems = discover_problems(args.rtllm_root, args.include, limit=args.limit)
    work_items = list(iter_work_items(problems, args.samples))

    if args.dry_run:
        print(f"discovered {len(problems)} problems and {len(work_items)} work items in {output_dir}")
        for problem in problems[:10]:
            print(f"{problem.problem_id}: top={problem.top_module} category={problem.category}")
        return

    pipeline = None if args.evaluate_only else build_pipeline(args)
    records: List[Dict[str, Any]] = []
    records_path = output_dir / "records.jsonl"
    records_path.write_text("", encoding="utf-8")
    records_lock = threading.Lock()
    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=max(args.concurrency, 1)) as executor:
        futures = [executor.submit(run_one, item, pipeline, args, output_dir) for item in work_items]
        for future in as_completed(futures):
            record = future.result()
            with records_lock:
                records.append(record)
                with records_path.open("a", encoding="utf-8") as handle:
                    handle.write(dumps_json(record) + "\n")
            print(
                f"completed {record['problem']} sample {record['sample']:02d}: "
                f"{record['passfail']} passed={record['passed']}"
            )

    records.sort(key=lambda item: (item["category"], item["problem"], item["sample"]))
    summary = summarize(records, args, output_dir, time.perf_counter() - start)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv_summary(output_dir / "summary.csv", records)
    print(
        f"accuracy={summary['accuracy']:.4f} "
        f"pass@1={format_summary_metric(summary['pass@1'])} "
        f"pass@3={format_summary_metric(summary['pass@3'])} "
        f"pass@5={format_summary_metric(summary['pass@5'])} "
        f"passed={summary['passed']}/{summary['num_records']} "
        f"summary={output_dir / 'summary.json'}"
    )


if __name__ == "__main__":
    main()
