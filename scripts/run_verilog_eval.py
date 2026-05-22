# minimun runnable
# python3 scripts/run_verilog_eval.py \
#   --task spec-to-rtl \
#   --pipeline fixed-pipe \
#   --index indexes/rtl_hash \
#   --cache-mode none \
#   --code-structure-index indexes/rtl_datapath_hash \
#   --concurrency 16 \
#   --samples 5 \
#   --output-dir runs/verilog_eval_nchc/


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
from rag_rtl.llm import VllmClient, extract_code
from rag_rtl.pipeline import FixedPipeRtlPipeline, RagRtlPipeline
from rag_rtl.types import PipelineResponse, RtlTask, VerificationReport
from rag_rtl.vector_store import VectorStore
from rag_rtl.verifier import RtlVerifier

TOP_MODULE_RE = re.compile(r"(?ms)^\s*module\s+TopModule\s*\(.*?\);\s*$")
MISMATCH_RE = re.compile(r"^Mismatches:\s+(\d+)\s+in\s+(\d+)\s+samples", re.MULTILINE)
MODULE_RE = re.compile(r"(?m)^\s*module\s+([A-Za-z_][A-Za-z0-9_$]*)\b")
PASS_AT_KS = (1, 3, 5)


@dataclass(frozen=True)
class VerilogEvalProblem:
    problem_id: str
    prompt_path: Path
    testbench_path: Path
    reference_path: Path
    prompt: str
    module_signature: Optional[str]


@dataclass(frozen=True)
class WorkItem:
    problem: VerilogEvalProblem
    sample: int


@dataclass
class IverilogResult:
    passed: bool
    passfail: str
    compile_returncode: Optional[int]
    simulation_returncode: Optional[int]
    mismatches: Optional[int]
    samples: Optional[int]
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
            "Run Verilog-Eval through the veri-thesis RTL generator in parallel, "
            "then evaluate generated TopModule implementations with the dataset testbenches."
        )
    )
    parser.add_argument("--verilog-eval-root", default="/home/kai/eval_dt/VerilogEval-v2-NTU")
    parser.add_argument(
        "--verilog-eval-v1-root",
        default="/home/kai/eval_dt/verilog-eval",
        help="Original Verilog-Eval checkout used as the base when --verilog-eval-root points to VerilogEval-v2-NTU.",
    )
    parser.add_argument("--task", choices=["spec-to-rtl", "code-complete-iccad2023"], default="spec-to-rtl")
    parser.add_argument("--output-dir", default="runs/verilog_eval")
    parser.add_argument("--pipeline", choices=["rag", "fixed-pipe"], default="rag")
    parser.add_argument("--index", default="indexes/rtl_hash")
    parser.add_argument("--code-structure-index", default="indexes/rtl_datapath_hash")
    parser.add_argument("--embedder", default="hash")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--resume", action="store_true", help="Reuse existing generated .sv files in output-dir")
    parser.add_argument("--evaluate-only", action="store_true", help="Skip generation and evaluate existing .sv files")
    parser.add_argument("--dry-run", action="store_true", help="Only discover dataset records and print the count")
    parser.add_argument("--retrieve-k", type=int, default=4)
    parser.add_argument("--context-k", type=int, default=2)
    parser.add_argument("--structure-retrieve-k", type=int, default=4)
    parser.add_argument("--structure-context-k", type=int, default=2)
    parser.add_argument("--max-repair-attempts", type=int, default=3)
    parser.add_argument("--second-edition-repair-attempts", type=int, default=1)
    parser.add_argument("--cache", default="data/history_cache.json")
    parser.add_argument("--monitor", default="runs/verilog_eval_monitor.jsonl")
    parser.add_argument("--failed-log", default="runs/verilog_eval_failed_attempts.jsonl")
    parser.add_argument("--cache-mode", choices=["keywords", "direct", "none"], default="keywords")
    parser.add_argument("--cache-reuse-threshold", type=float, default=0.95)
    parser.add_argument("--cache-evidence-threshold", type=float, default=0.88)
    parser.add_argument("--generation-temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument(
        "--serving-url",
        "--base-url",
        dest="serving_url",
        help="OpenAI-compatible serving base URL. Overrides VLLM_BASE_URL.",
    )
    parser.add_argument("--verbose-generation", action="store_true")
    parser.add_argument("--enable-tool-calling", action="store_true")
    parser.add_argument("--tool-choice", default="auto")
    parser.add_argument("--max-tool-rounds", type=int, default=4)
    parser.add_argument("--iverilog-bin", default="iverilog")
    parser.add_argument("--simulation-timeout-s", type=int, default=30)
    parser.add_argument("--top-module", default="TopModule")
    parser.add_argument("--keep-vcd", action="store_true")
    return parser


def prepare_verilog_eval_root(args: argparse.Namespace, output_dir: Path) -> Path:
    root = Path(args.verilog_eval_root)
    if (root / f"dataset_{args.task}").exists():
        return root
    if is_verilog_eval_v2_ntu_root(root):
        if args.task != "spec-to-rtl":
            raise ValueError("VerilogEval-v2-NTU only provides the spec-to-rtl task")
        return materialize_verilog_eval_v2_ntu(root, Path(args.verilog_eval_v1_root), output_dir)
    return root


def is_verilog_eval_v2_ntu_root(root: Path) -> bool:
    return (root / "patches").is_dir() and (root / "data").is_dir()


def materialize_verilog_eval_v2_ntu(v2_root: Path, v1_root: Path, output_dir: Path) -> Path:
    source_dataset = v1_root / "dataset_spec-to-rtl"
    if not source_dataset.exists():
        raise FileNotFoundError(f"Original Verilog-Eval dataset directory not found: {source_dataset}")
    patch_dir = v2_root / "patches"
    if not patch_dir.exists():
        raise FileNotFoundError(f"VerilogEval-v2-NTU patch directory not found: {patch_dir}")

    materialized_root = output_dir / "_VerilogEval-v2-NTU"
    materialized_dataset = materialized_root / "dataset_spec-to-rtl"
    if materialized_dataset.exists():
        shutil.rmtree(materialized_dataset)
    materialized_dataset.mkdir(parents=True, exist_ok=True)

    for source_path in source_dataset.iterdir():
        if source_path.is_file():
            shutil.copy2(source_path, materialized_dataset / source_path.name)

    for patch_path in sorted(patch_dir.glob("*.patch")):
        completed = subprocess.run(
            ["patch", "--batch", "--forward", "--silent", "-d", str(materialized_dataset), "-i", str(patch_path)],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            message = (completed.stderr or completed.stdout or "").strip()
            raise RuntimeError(f"failed to apply VerilogEval-v2-NTU patch {patch_path.name}: {message}")

    return materialized_root


def discover_problems(root: str | Path, task: str, limit: Optional[int] = None) -> List[VerilogEvalProblem]:
    dataset_dir = Path(root) / f"dataset_{task}"
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Verilog-Eval dataset directory not found: {dataset_dir}")

    problems: List[VerilogEvalProblem] = []
    for prompt_path in sorted(dataset_dir.glob("*_prompt.txt")):
        problem_id = prompt_path.name.removesuffix("_prompt.txt")
        testbench_path = dataset_dir / f"{problem_id}_test.sv"
        reference_path = dataset_dir / f"{problem_id}_ref.sv"
        if not testbench_path.exists() or not reference_path.exists():
            continue
        prompt = prompt_path.read_text(encoding="utf-8").strip()
        problems.append(
            VerilogEvalProblem(
                problem_id=problem_id,
                prompt_path=prompt_path,
                testbench_path=testbench_path,
                reference_path=reference_path,
                prompt=prompt,
                module_signature=extract_topmodule_signature(prompt),
            )
        )
        if limit is not None and len(problems) >= limit:
            break
    return problems


def extract_topmodule_signature(prompt: str) -> Optional[str]:
    matches = TOP_MODULE_RE.findall(prompt)
    return matches[-1].strip() if matches else None


def build_pipeline(args: argparse.Namespace) -> Any:
    embedder = make_embedder(args.embedder)
    llm_client = build_llm_client(args)
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
            llm_client=llm_client,
            verifier=verifier,
            cache_config=cache_config,
            runtime_config=runtime_config,
            tool_config=tool_config,
            fixed_pipe_config=FixedPipeConfig(second_edition_repair_attempts=args.second_edition_repair_attempts),
        )

    return RagRtlPipeline(
        store=spec_store,
        embedder=embedder,
        llm_client=llm_client,
        verifier=verifier,
        cache_config=cache_config,
        runtime_config=runtime_config,
        tool_config=tool_config,
    )


def build_llm_client(args: argparse.Namespace) -> VllmClient:
    client = VllmClient.from_env()
    if args.serving_url:
        client.base_url = args.serving_url
    return client


def build_task(problem: VerilogEvalProblem, args: argparse.Namespace) -> RtlTask:
    constraints = [
        f"Return a complete Verilog module named {args.top_module}, including the module declaration and endmodule.",
        "Do not include the testbench, reference module, markdown fences, or explanatory text.",
    ]
    if problem.module_signature:
        constraints.append("Preserve the exact TopModule port list from the prompt.")
    return RtlTask(
        prompt=problem.prompt,
        target_hdl="verilog",
        module_signature=problem.module_signature,
        constraints=constraints,
        max_repair_attempts=args.max_repair_attempts,
        top_module=args.top_module,
        prompt_profile=getattr(args, "prompt_profile", "rag"),
    )


def problem_dir(output_dir: Path, problem_id: str) -> Path:
    return output_dir / problem_id


def sample_stem(problem_id: str, sample: int) -> str:
    return f"{problem_id}_sample{sample:02d}"


def generated_code_path(output_dir: Path, item: WorkItem) -> Path:
    return problem_dir(output_dir, item.problem.problem_id) / f"{sample_stem(item.problem.problem_id, item.sample)}.sv"


def generate_log_path(output_dir: Path, item: WorkItem) -> Path:
    return problem_dir(output_dir, item.problem.problem_id) / f"{sample_stem(item.problem.problem_id, item.sample)}-sv-generate.log"


def compile_log_path(output_dir: Path, item: WorkItem) -> Path:
    return problem_dir(output_dir, item.problem.problem_id) / f"{sample_stem(item.problem.problem_id, item.sample)}-sv-iv-test.log"


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
    except Exception as exc:  # noqa: BLE001 - keep the benchmark running.
        return "", None, str(exc)

    code = normalize_generated_code(response.rtl, item.problem.module_signature, args.top_module)
    return code, response, None


def normalize_generated_code(code: str, module_signature: Optional[str], top_module: str) -> str:
    extracted = extract_code(code).strip()
    if MODULE_RE.search(extracted):
        return ensure_top_module_name(extracted, top_module)
    if module_signature:
        body = extracted
        if re.search(r"\bendmodule\b", body) is None:
            body = body.rstrip() + "\nendmodule"
        return f"{module_signature}\n{body}".strip()
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
) -> IverilogResult:
    if shutil.which(args.iverilog_bin) is None and not Path(args.iverilog_bin).exists():
        result = IverilogResult(
            passed=False,
            passfail="I",
            compile_returncode=None,
            simulation_returncode=None,
            mismatches=None,
            samples=None,
            compile_command=[args.iverilog_bin],
            run_command=[],
            error=f"iverilog binary not found: {args.iverilog_bin}",
        )
        write_compile_log(log_path, result)
        return result

    work_dir = candidate_path.parent
    exe_path = (work_dir / sample_stem(item.problem.problem_id, item.sample)).resolve()
    compile_command = [
        args.iverilog_bin,
        "-Wall",
        "-Winfloop",
        "-Wno-timescale",
        "-g2012",
        "-s",
        "tb",
        "-o",
        str(exe_path),
        str(candidate_path.resolve()),
        str(item.problem.testbench_path.resolve()),
        str(item.problem.reference_path.resolve()),
    ]
    t0 = time.perf_counter()
    try:
        compile_completed = subprocess.run(
            compile_command,
            check=False,
            capture_output=True,
            text=True,
            cwd=work_dir,
        )
    except OSError as exc:
        compile_s = time.perf_counter() - t0
        result = IverilogResult(
            passed=False,
            passfail="I",
            compile_returncode=None,
            simulation_returncode=None,
            mismatches=None,
            samples=None,
            compile_command=compile_command,
            run_command=[],
            stderr=f"error: failed to start iverilog: {exc}",
            compile_s=compile_s,
            error=str(exc),
        )
        write_compile_log(log_path, result)
        return result
    compile_s = time.perf_counter() - t0
    stdout = compile_completed.stdout or ""
    stderr = compile_completed.stderr or ""

    run_command: List[str] = []
    simulation_returncode: Optional[int] = None
    simulation_s = 0.0
    timed_out = False
    run_error: Optional[str] = None
    if compile_completed.returncode == 0:
        run_command = [str(exe_path)]
        t0 = time.perf_counter()
        if not exe_path.exists():
            simulation_returncode = -1
            run_error = f"simulation executable missing after successful compile: {exe_path}"
            stderr += f"\nerror: {run_error}"
        else:
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
            except OSError as exc:
                simulation_returncode = -1
                run_error = f"failed to run simulation executable {exe_path}: {exc}"
                stderr += f"\nerror: {run_error}"
        simulation_s = time.perf_counter() - t0

    mismatches, nsamples = parse_mismatches(stdout + "\n" + stderr)
    passfail = classify_iverilog_result(
        candidate_path,
        compile_completed.returncode,
        timed_out,
        stdout,
        stderr,
        mismatches,
    )
    passed = passfail == "."
    result = IverilogResult(
        passed=passed,
        passfail=passfail,
        compile_returncode=compile_completed.returncode,
        simulation_returncode=simulation_returncode,
        mismatches=mismatches,
        samples=nsamples,
        compile_command=compile_command,
        run_command=run_command,
        stdout=stdout,
        stderr=stderr,
        compile_s=compile_s,
        simulation_s=simulation_s,
        error=run_error,
    )
    write_compile_log(log_path, result)
    if not args.keep_vcd:
        vcd_path = work_dir / "wave.vcd"
        try:
            vcd_path.unlink()
        except OSError:
            pass
    return result


def parse_mismatches(text: str) -> Tuple[Optional[int], Optional[int]]:
    match = MISMATCH_RE.search(text)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def classify_iverilog_result(
    candidate_path: Path,
    compile_returncode: int,
    timed_out: bool,
    stdout: str,
    stderr: str,
    mismatches: Optional[int],
) -> str:
    text = stdout + "\n" + stderr
    if "syntax error" in text:
        return "S"
    if "error: This assignment requires an explicit cast" in text:
        return "e"
    if "error: Sized numeric constant must have a size greater than zero" in text:
        return "0"
    if "warning: always_comb process has no sensitivities" in text or "found no sensitivities" in text:
        return "n"
    if "is declared here as wire" in text:
        return "w"
    if "Unknown module type" in text:
        return "m"
    if "Unable to bind wire/reg/memory `clk'" in text:
        return "c"
    if timed_out or "TIMEOUT" in text:
        return "T"
    if compile_returncode != 0 or "error" in text:
        return "C"
    if mismatches == 0:
        return "."
    if mismatches is not None:
        return "R"

    try:
        candidate_text = candidate_path.read_text(encoding="utf-8")
    except OSError:
        return "?"
    if "posedge reset" in candidate_text or "negedge reset" in candidate_text or "posedge r)" in candidate_text:
        return "r"
    return "R"


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
        f"sample = {item.sample:02d}",
        f"prompt = {item.problem.prompt_path}",
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


def write_compile_log(path: Path, result: IverilogResult) -> None:
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
    out_dir = problem_dir(output_dir, item.problem.problem_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    code_path = generated_code_path(output_dir, item)
    gen_log = generate_log_path(output_dir, item)
    iv_log = compile_log_path(output_dir, item)

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
        eval_result = IverilogResult(
            passed=False,
            passfail="G",
            compile_returncode=None,
            simulation_returncode=None,
            mismatches=None,
            samples=None,
            compile_command=[],
            run_command=[],
            error=generation_error or "generation produced empty code",
        )
        write_compile_log(iv_log, eval_result)
    else:
        eval_result = evaluate_with_iverilog(item, code_path, iv_log, args)

    record = {
        "problem": item.problem.problem_id,
        "sample": item.sample,
        "prompt_path": str(item.problem.prompt_path),
        "testbench_path": str(item.problem.testbench_path),
        "reference_path": str(item.problem.reference_path),
        "generated_code_path": str(code_path),
        "generation_log_path": str(gen_log),
        "compile_log_path": str(iv_log),
        "generated": bool(code),
        "generation_error": generation_error,
        "reused_existing": reused_existing,
        **response_metadata(response),
        "verification_diagnostics": verification_diagnostics(response.verification if response else None),
        "passed": eval_result.passed,
        "passfail": eval_result.passfail,
        "compile_returncode": eval_result.compile_returncode,
        "simulation_returncode": eval_result.simulation_returncode,
        "mismatches": eval_result.mismatches,
        "num_test_samples": eval_result.samples,
        "compile_s": eval_result.compile_s,
        "simulation_s": eval_result.simulation_s,
        "compile_command": eval_result.compile_command,
        "run_command": eval_result.run_command,
        "stdout_tail": eval_result.stdout[-4000:],
        "stderr_tail": eval_result.stderr[-4000:],
        "evaluation_error": eval_result.error,
    }
    return record


def summarize(records: Sequence[Dict[str, Any]], args: argparse.Namespace, output_dir: Path, elapsed_s: float) -> Dict[str, Any]:
    count = len(records)
    denom = max(count, 1)
    passfail_counts: Dict[str, int] = {}
    for record in records:
        key = str(record.get("passfail") or "?")
        passfail_counts[key] = passfail_counts.get(key, 0) + 1
    pass_at_rates, pass_at_denominators = compute_pass_at(records, PASS_AT_KS)

    return {
        "verilog_eval_root": str(Path(args.verilog_eval_root)),
        "task": args.task,
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
        "passfail_counts": dict(sorted(passfail_counts.items())),
        "total_s": elapsed_s,
        "records": list(records),
    }


def group_records_by_problem(records: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record["problem"]), []).append(record)
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


def repair_attempts_label(value: Any) -> str:
    return "n/a" if value is None else str(value)


def write_csv_summary(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    fieldnames = [
        "problem",
        "sample",
        "passed",
        "passfail",
        "mismatches",
        "num_test_samples",
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


def iter_work_items(problems: Sequence[VerilogEvalProblem], samples: int) -> Iterable[WorkItem]:
    for problem in problems:
        for sample in range(1, samples + 1):
            yield WorkItem(problem=problem, sample=sample)


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir) / args.task
    output_dir.mkdir(parents=True, exist_ok=True)
    effective_root = prepare_verilog_eval_root(args, output_dir)
    problems = discover_problems(effective_root, args.task, limit=args.limit)
    work_items = list(iter_work_items(problems, args.samples))

    if args.dry_run:
        print(f"discovered {len(problems)} problems and {len(work_items)} work items in {output_dir}")
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
                f"{record['passfail']} passed={record['passed']} "
                f"repairs={repair_attempts_label(record.get('repair_attempts'))}"
            )

    records.sort(key=lambda item: (item["problem"], item["sample"]))
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
