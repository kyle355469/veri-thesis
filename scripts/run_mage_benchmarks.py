#!/usr/bin/env python3
"""Run the MAGE multi-agent RTL generator over Verilog-Eval, RTLLM, or RealBench
against a local vLLM (OpenAI-compatible) server.

Generation is MAGE's own TopAgent loop (tb generation -> rtl generation -> sim
judge/repair), which is benchmark-agnostic; the benchmark only supplies the spec,
an optional golden testbench seed, and the final scoring:

* ``--benchmark verilog-eval`` -- tasks + final scoring via MAGE's native helpers
  (sim_review_golden_benchmark; iverilog with the dataset tb + ref, the paper setup).
* ``--benchmark rtllm``        -- tasks from the RTLLM tree (design_description.txt);
  final scoring replicates scripts/run_rtllm_eval.py's iverilog compile+simulate
  ("Your Design Passed" / failure count) with the same flags.
* ``--benchmark realbench``    -- tasks from RealBench problems.jsonl; final scoring
  replicates the native verification-Makefile eval of
  scripts/run_agentic_plan_legacy_realbench.py (verilator, -Wno-TIMESCALEMOD appended,
  %Error/%Warning -> syntax, mismatch hints -> function).

The RTLLM/RealBench scorers are self-contained (stdlib only) because this script
runs inside the MAGE venv (~/venv-mage), which has llama-index but not the
veri-thesis dependencies. Scoring commands and pass criteria mirror the native
runners so numbers are comparable.

The LLM is built with llama-index's OpenAILike (any served model name works),
pointed at ``--base-url``/$VLLM_BASE_URL. Requires iverilog v12 on PATH (and
verilator + make for --benchmark realbench).

Usage::

    ~/venv-mage/bin/python scripts/run_mage_benchmarks.py --benchmark rtllm \
        --base-url http://127.0.0.1:8000/v1 --model gpt-oss-20b --output-dir runs/mage_rtllm
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

MODULE_DECL_RE = re.compile(r"(?m)^\s*module\s+([A-Za-z_][A-Za-z0-9_$]*)\b")
RTLLM_PASS_RE = re.compile(r"Your\s+Design\s+Passed", re.IGNORECASE)
RTLLM_FAILURE_COUNT_RE = re.compile(r"Test\s+completed\s+with\s+(\d+)\s*(?:/|\s+)?", re.IGNORECASE)
RTLLM_TEST_DESIGN_RE = re.compile(r"(?m)^\s*TEST_DESIGN\s*=\s*([A-Za-z_][A-Za-z0-9_$]*)\s*$")
RTLLM_DESCRIPTION_MODULE_RE = re.compile(r"(?is)\bmodule\s+name\s*:\s*(?:\r?\n|\s)+([A-Za-z_][A-Za-z0-9_$]*)")
RTLLM_INSTANCE_RE = re.compile(r"(?ms)^\s*([A-Za-z_][A-Za-z0-9_$]*)\s*(?:#\s*\(.*?\)\s*)?([A-Za-z_][A-Za-z0-9_$]*)\s*\(")
VERILOG_KEYWORDS = {
    "always", "assign", "begin", "case", "else", "end", "for", "forever",
    "function", "if", "initial", "module", "repeat", "task", "while",
}
REALBENCH_PASS_HINT_RE = re.compile(r"Hint:\s+Output.*no mismatches", re.IGNORECASE)
REALBENCH_MISMATCH_HINT_RE = re.compile(r"Hint:\s+Output.*mismatches", re.IGNORECASE)


@dataclass
class MageTask:
    """One benchmark task fed to MAGE plus what its final scoring needs."""

    task_id: str
    spec: str
    top_module: str
    golden_tb_path: Optional[str] = None
    golden_rtl_blackbox_path: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run MAGE over Verilog-Eval / RTLLM / RealBench against an OpenAI-compatible endpoint."
    )
    parser.add_argument("--benchmark", choices=["verilog-eval", "rtllm", "realbench"], default="verilog-eval")
    parser.add_argument("--mage-root", default="/home/kai/MAGE")
    parser.add_argument("--output-dir", default="runs/mage_benchmark")
    parser.add_argument("--run-identifier", default="mage")
    parser.add_argument("--n", type=int, default=1, help="Independent MAGE rounds over the task set.")
    parser.add_argument("--limit", type=int, help="Cap the number of tasks after filtering.")
    parser.add_argument("--resume", action="store_true", help="Skip tasks already present in a round's record.json.")
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Case-insensitive substring filter on task id (rtllm/realbench); repeatable.",
    )
    parser.add_argument("--filter-instance", default="^(.*)$", help="RegEx task filter (verilog-eval only).")

    # verilog-eval
    parser.add_argument(
        "--path-benchmark",
        default="/home/kai/verilog-eval",
        help="Verilog-Eval checkout containing dataset_spec-to-rtl (MAGE's submodule layout).",
    )
    parser.add_argument("--type-benchmark", choices=["verilog_eval_v1", "verilog_eval_v2"], default="verilog_eval_v2")
    # rtllm
    parser.add_argument("--rtllm-root", default="/home/kai/eval_dt/RTLLM")
    parser.add_argument("--iverilog-bin", default="iverilog")
    parser.add_argument("--simulation-timeout-s", type=int, default=30)
    # realbench
    parser.add_argument("--realbench-root", default="/home/kai/eval_dt/real_bench")
    parser.add_argument("--task-level", choices=["module", "system", "both"], default="both")
    parser.add_argument("--make-bin", default="make")
    parser.add_argument("--verification-timeout-s", type=int, default=300)
    parser.add_argument("--realbench-spec-max-chars", type=int, default=120000,
                        help="Clip oversized RealBench specs before handing them to MAGE (no condensation here).")

    parser.add_argument("--base-url", help="OpenAI-compatible base URL. Defaults to VLLM_BASE_URL.")
    parser.add_argument("--model", help="Served model name. Defaults to VLLM_MODEL.")
    parser.add_argument("--api-key", help="API key. Defaults to VLLM_API_KEY or EMPTY.")
    parser.add_argument("--context-window", type=int, default=32768)
    parser.add_argument("--max-token", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--llm-timeout-s", type=int, default=2400)
    parser.add_argument(
        "--use-golden-tb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Seed MAGE's testbench generator with the benchmark's golden testbench "
        "(and, on verilog-eval, the blackbox golden RTL) -- the MAGE paper's setup.",
    )
    return parser


def make_llm(args: argparse.Namespace) -> Any:
    base_url = args.base_url or os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    model = args.model or os.getenv("VLLM_MODEL", "siliconmind-server")
    api_key = args.api_key or os.getenv("VLLM_API_KEY", "EMPTY")
    from llama_index.llms.openai_like import OpenAILike

    return OpenAILike(
        model=model,
        api_base=base_url,
        api_key=api_key,
        is_chat_model=True,
        is_function_calling_model=False,
        context_window=args.context_window,
        max_tokens=args.max_token,
        timeout=args.llm_timeout_s,
    )


def ensure_top_module_name(code: str, top_module: str) -> str:
    names = MODULE_DECL_RE.findall(code)
    if not names or top_module in names:
        return code
    pattern = re.compile(rf"(?m)^(\s*module\s+){re.escape(names[0])}\b")
    return pattern.sub(lambda match: f"{match.group(1)}{top_module}", code, count=1)


def strip_module_redeclarations(code: str, module_names: List[str]) -> str:
    for name in module_names:
        pattern = re.compile(rf"(?ms)^[ \t]*module\s+{re.escape(name)}\b.*?^[ \t]*endmodule[^\n]*\n?")
        code = pattern.sub("", code)
    stripped = code.strip()
    return stripped + "\n" if stripped else ""


def run_group_capture(command: List[str], cwd: Path, timeout_s: float) -> subprocess.CompletedProcess:
    """subprocess.run with the child in its own process group; on timeout the whole
    group is SIGKILLed so runaway simulations cannot wedge the run."""
    with subprocess.Popen(
        command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, start_new_session=True,
    ) as proc:
        try:
            stdout, stderr = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
            stdout, stderr = proc.communicate()
            raise subprocess.TimeoutExpired(command, timeout_s, output=stdout, stderr=stderr)
    return subprocess.CompletedProcess(command, proc.returncode, stdout, stderr)


# --------------------------------------------------------------------------- #
# Task loading
# --------------------------------------------------------------------------- #


def include_matches(task_id: str, include: List[str]) -> bool:
    filters = [item.lower() for item in include if item.strip()]
    return not filters or any(item in task_id.lower() for item in filters)


def load_verilog_eval_tasks(args: argparse.Namespace) -> Tuple[List[MageTask], str]:
    from mage.benchmark_read_helper import TypeBenchmark, TypeBenchmarkFile, get_benchmark_contents

    type_benchmark = TypeBenchmark[args.type_benchmark.upper()]
    specs = get_benchmark_contents(type_benchmark, TypeBenchmarkFile.SPEC, args.path_benchmark, args.filter_instance)
    tbs = get_benchmark_contents(type_benchmark, TypeBenchmarkFile.TEST_PATH, args.path_benchmark, args.filter_instance)
    goldens = get_benchmark_contents(type_benchmark, TypeBenchmarkFile.GOLDEN_PATH, args.path_benchmark, args.filter_instance)
    tasks = [
        MageTask(
            task_id=task_id,
            spec=specs[task_id],
            top_module="TopModule",
            golden_tb_path=tbs.get(task_id) if args.use_golden_tb else None,
            golden_rtl_blackbox_path=goldens.get(task_id) if args.use_golden_tb else None,
        )
        for task_id in sorted(specs)
    ]
    return tasks, type_benchmark.name


def rtllm_infer_top_module(problem_dir: Path, testbench_path: Path, prompt: str, fallback: str) -> str:
    text = testbench_path.read_text(encoding="utf-8", errors="ignore")
    for module_name, instance_name in RTLLM_INSTANCE_RE.findall(text):
        if module_name.lower() in VERILOG_KEYWORDS:
            continue
        lower_instance = instance_name.lower()
        if lower_instance in {"uut", "dut", "u0"} or lower_instance.startswith("u_"):
            return module_name
    match = RTLLM_DESCRIPTION_MODULE_RE.search(prompt)
    if match:
        return match.group(1)
    makefile = problem_dir / "makefile"
    if makefile.exists():
        match = RTLLM_TEST_DESIGN_RE.search(makefile.read_text(encoding="utf-8", errors="ignore"))
        if match:
            return match.group(1)
    return fallback


def load_rtllm_tasks(args: argparse.Namespace) -> Tuple[List[MageTask], str]:
    root = Path(args.rtllm_root)
    if not root.exists():
        raise FileNotFoundError(f"RTLLM root not found: {root}")
    tasks: List[MageTask] = []
    for description_path in sorted(root.rglob("design_description.txt")):
        if any(part.startswith("_") or part == ".git" for part in description_path.relative_to(root).parts):
            continue
        problem_dir = description_path.parent
        testbench_path = problem_dir / "testbench.v"
        golden_candidates = sorted(problem_dir.glob("verified_*.v"))
        if not testbench_path.exists() or not golden_candidates:
            continue
        task_id = problem_dir.name
        if not include_matches(task_id, args.include):
            continue
        prompt = description_path.read_text(encoding="utf-8").strip()
        tasks.append(
            MageTask(
                task_id=task_id,
                spec=prompt,
                top_module=rtllm_infer_top_module(problem_dir, testbench_path, prompt, task_id),
                # Golden RTL declares the SAME module name as the candidate, so it can
                # never be compiled alongside as a blackbox; only the tb seeds MAGE.
                golden_tb_path=str(testbench_path) if args.use_golden_tb else None,
                golden_rtl_blackbox_path=None,
                meta={"testbench_path": str(testbench_path)},
            )
        )
    return tasks, "RTLLM"


def load_python_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def read_problem_jsonl(path: Path) -> Dict[str, str]:
    records: Dict[str, str] = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            task = str(payload.get("task") or "").strip()
            problem = str(payload.get("problem") or payload.get("text") or "").strip()
            if task and problem:
                records[task] = problem
    return records


def realbench_template_dir(root: Path, level: str, system: str, task: str) -> Path:
    if level == "system":
        return root / "system" / task
    return root / system / task / "verification"


def realbench_golden_tb(template_dir: Path) -> Optional[str]:
    for path in sorted([*template_dir.glob("*testbench*.sv"), *template_dir.glob("*testbench*.v")]):
        return str(path)
    return None


def load_realbench_tasks(args: argparse.Namespace) -> Tuple[List[MageTask], str]:
    root = Path(args.realbench_root)
    info = load_python_module(root / "benchmark_info.py", "mage_realbench_info")
    levels = ["module", "system"] if args.task_level == "both" else [args.task_level]
    tasks: List[MageTask] = []

    def add(level: str, system: str, task_name: str, prompt: str) -> None:
        task_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{level}__{system}__{task_name}").strip("_")
        if not include_matches(f"{task_id} {task_name}", args.include):
            return
        template = realbench_template_dir(root, level, system, task_name)
        if not template.exists():
            return
        spec = prompt
        if args.realbench_spec_max_chars and len(spec) > args.realbench_spec_max_chars:
            spec = spec[: args.realbench_spec_max_chars] + "\n... [truncated] ..."
        tasks.append(
            MageTask(
                task_id=task_id,
                spec=spec,
                top_module=task_name,
                golden_tb_path=realbench_golden_tb(template) if args.use_golden_tb else None,
                golden_rtl_blackbox_path=None,
                meta={"level": level, "system": system, "task": task_name, "template_dir": str(template)},
            )
        )

    if "module" in levels:
        for system, modules in info.benchmark_info.items():
            prompts = read_problem_jsonl(root / "problems" / system / "problems.jsonl")
            for module_name in modules:
                if module_name in prompts:
                    add("module", system, module_name, prompts[module_name])
    if "system" in levels:
        prompts = read_problem_jsonl(root / "problems" / "system" / "problems.jsonl")
        for system_name in info.system_info:
            if system_name in prompts:
                system = ("sdc" if system_name.startswith("sd") or system_name == "sdc_controller"
                          else "aes" if system_name.startswith("aes") else "e203_hbirdv2")
                add("system", system, system_name, prompts[system_name])
    if not tasks and not args.include:
        raise FileNotFoundError(
            f"no RealBench tasks found -- are problems.jsonl files prepared under {root / 'problems'}?"
        )
    return tasks, "REALBENCH"


# --------------------------------------------------------------------------- #
# Final scoring
# --------------------------------------------------------------------------- #


def read_generated_rtl(output_path: Path, benchmark_type_name: str, task: MageTask) -> str:
    rtl_path = output_path / f"{benchmark_type_name}_{task.task_id}" / "rtl.sv"
    if not rtl_path.exists():
        return ""
    return rtl_path.read_text(encoding="utf-8", errors="ignore")


def score_verilog_eval(task: MageTask, agent_output_path: Path, args: argparse.Namespace) -> Tuple[bool, str]:
    from mage.benchmark_read_helper import TypeBenchmark
    from mage.sim_reviewer import sim_review_golden_benchmark

    return sim_review_golden_benchmark(
        task_id=task.task_id,
        output_path=str(agent_output_path),
        benchmark_type=TypeBenchmark[args.type_benchmark.upper()],
        benchmark_path=args.path_benchmark,
    )


def score_rtllm(task: MageTask, code: str, args: argparse.Namespace) -> Tuple[bool, str]:
    """Replicates run_rtllm_eval.evaluate_with_iverilog: same compile flags, pass on
    'Your Design Passed' or a parsed failure count of 0."""
    code = ensure_top_module_name(code, task.top_module)
    if not code.strip():
        return False, "empty candidate RTL"
    with tempfile.TemporaryDirectory(prefix="mage_rtllm_") as temp_name:
        temp_dir = Path(temp_name)
        candidate = temp_dir / f"{task.top_module}.v"
        candidate.write_text(code, encoding="utf-8")
        exe = temp_dir / "sim"
        compile_cmd = [
            args.iverilog_bin, "-Wall", "-Winfloop", "-Wno-timescale", "-g2012",
            "-o", str(exe), str(candidate), task.meta["testbench_path"],
        ]
        try:
            compiled = subprocess.run(compile_cmd, cwd=temp_dir, check=False, capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return False, f"iverilog compile failed to run: {exc}"
        if compiled.returncode != 0:
            return False, f"compile failed:\n{(compiled.stderr or compiled.stdout)[-2000:]}"
        try:
            ran = run_group_capture([str(exe)], temp_dir, args.simulation_timeout_s)
        except subprocess.TimeoutExpired:
            return False, f"simulation timeout after {args.simulation_timeout_s}s"
        text = (ran.stdout or "") + "\n" + (ran.stderr or "")
    if RTLLM_PASS_RE.search(text):
        return True, text[-2000:]
    match = RTLLM_FAILURE_COUNT_RE.search(text)
    if match and int(match.group(1)) == 0:
        return True, text[-2000:]
    return False, text[-2000:]


def realbench_provided_modules(template_dir: Path, task_name: str) -> List[str]:
    top_filename = f"{task_name}_top.sv"
    names: set = set()
    for path in sorted([*template_dir.glob("*.v"), *template_dir.glob("*.sv")]):
        if path.name == top_filename:
            continue
        names.update(MODULE_DECL_RE.findall(path.read_text(encoding="utf-8", errors="ignore")))
    names.discard(task_name)
    return sorted(names)


def score_realbench(task: MageTask, code: str, args: argparse.Namespace) -> Tuple[bool, str]:
    """Replicates the native RealBench eval of run_agentic_plan_legacy_realbench.py:
    copy the verification template, append -Wno-TIMESCALEMOD, write <task>_top.sv,
    make all; %Error/%Warning lines fail syntax, mismatch hints fail function."""
    template_dir = Path(task.meta["template_dir"])
    task_name = task.meta["task"]
    code = ensure_top_module_name(code, task.top_module)
    code = strip_module_redeclarations(code, realbench_provided_modules(template_dir, task_name))
    if not code.strip():
        return False, "empty candidate RTL"
    try:
        with tempfile.TemporaryDirectory(prefix=f"mage_realbench_{task_name}_") as temp_name:
            temp_dir = Path(temp_name)
            for child in template_dir.iterdir():
                target = temp_dir / child.name
                if child.is_dir():
                    shutil.copytree(child, target)
                else:
                    shutil.copy2(child, target)
            makefile = temp_dir / "Makefile"
            if makefile.exists():
                with makefile.open("a", encoding="utf-8") as handle:
                    handle.write("\nVERILATOR_FLAGS += -Wno-TIMESCALEMOD\n")
            top_path = temp_dir / f"{task_name}_top.sv"
            top_path.unlink(missing_ok=True)
            top_path.write_text(code, encoding="utf-8")
            completed = run_group_capture([args.make_bin, "all"], temp_dir, args.verification_timeout_s)
    except subprocess.TimeoutExpired:
        return False, f"verification timeout after {args.verification_timeout_s}s"
    except Exception as exc:  # noqa: BLE001
        return False, f"verification failed: {exc}"
    stderr = completed.stderr or ""
    stdout = completed.stdout or ""
    syntax_info = "\n".join(
        line for line in stderr.splitlines() if line.startswith("%Error") or line.startswith("%Warning")
    )
    if completed.returncode != 0 or syntax_info:
        return False, f"syntax failure:\n{syntax_info[-2000:] or stderr[-2000:]}"
    mismatches = [
        line.removeprefix("Hint: ").strip()
        for line in stdout.splitlines()
        if REALBENCH_MISMATCH_HINT_RE.search(line) and not REALBENCH_PASS_HINT_RE.search(line)
    ]
    if mismatches:
        return False, "functional mismatch:\n" + "\n".join(mismatches)[-2000:]
    return True, stdout[-2000:]


def score_task(
    benchmark: str,
    task: MageTask,
    agent_output_path: Path,
    benchmark_type_name: str,
    args: argparse.Namespace,
) -> Tuple[bool, str]:
    if benchmark == "verilog-eval":
        return score_verilog_eval(task, agent_output_path, args)
    code = read_generated_rtl(agent_output_path, benchmark_type_name, task)
    if benchmark == "rtllm":
        return score_rtllm(task, code, args)
    return score_realbench(task, code, args)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def main(argv: Optional[list] = None) -> None:
    args = build_parser().parse_args(argv)
    mage_src = Path(args.mage_root) / "src"
    if not (mage_src / "mage").is_dir():
        raise FileNotFoundError(f"MAGE package not found under {mage_src}")
    sys.path.insert(0, str(mage_src))

    from mage.agent import TopAgent
    from mage.gen_config import set_exp_setting

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.benchmark == "verilog-eval":
        tasks, benchmark_type_name = load_verilog_eval_tasks(args)
    elif args.benchmark == "rtllm":
        tasks, benchmark_type_name = load_rtllm_tasks(args)
    else:
        tasks, benchmark_type_name = load_realbench_tasks(args)
    if args.limit is not None:
        tasks = tasks[: args.limit]
    print(f"[mage] benchmark={args.benchmark}: {len(tasks)} task(s)")

    llm = make_llm(args)
    set_exp_setting(temperature=args.temperature, top_p=args.top_p)

    rounds: list = []
    for round_index in range(args.n):
        run_id = f"{args.run_identifier}_{round_index}"
        round_dir = output_dir / run_id
        agent_output_path = round_dir / "output"
        agent_log_path = round_dir / "log"
        agent_output_path.mkdir(parents=True, exist_ok=True)
        agent_log_path.mkdir(parents=True, exist_ok=True)
        record_path = round_dir / "record.json"
        record: Dict[str, Any] = {"record_per_run": {}, "total_record": {}}
        if args.resume and record_path.exists():
            try:
                record = json.loads(record_path.read_text(encoding="utf-8"))
                record.setdefault("record_per_run", {})
                record.setdefault("total_record", {})
            except json.JSONDecodeError:
                pass

        agent = TopAgent(llm)
        agent.set_output_path(str(agent_output_path))
        agent.set_log_path(str(agent_log_path))
        agent.set_redirect_log(True)

        round_start = time.monotonic()
        for index, task in enumerate(tasks, start=1):
            if args.resume and task.task_id in record["record_per_run"]:
                print(f"[mage] ({index:03d}/{len(tasks):03d}) {task.task_id}: resumed, skipping")
                continue
            print(f"[mage] ({index:03d}/{len(tasks):03d}) round {round_index} task {task.task_id}")
            t0 = time.monotonic()
            error: Optional[str] = None
            try:
                agent.run(
                    benchmark_type_name=benchmark_type_name,
                    task_id=task.task_id,
                    spec=task.spec,
                    golden_tb_path=task.golden_tb_path,
                    golden_rtl_blackbox_path=task.golden_rtl_blackbox_path,
                )
            except Exception as exc:  # noqa: BLE001 - keep the benchmark moving.
                error = f"{exc}\n{traceback.format_exc()[-2000:]}"
                print(f"[mage] {task.task_id} agent error: {exc}", file=sys.stderr)
            wall_s = time.monotonic() - t0
            try:
                is_pass, sim_log = score_task(args.benchmark, task, agent_output_path, benchmark_type_name, args)
            except Exception as exc:  # noqa: BLE001
                is_pass, sim_log = False, f"final scoring failed: {exc}"
            token_count = agent.token_counter.get_sum_count()
            record["record_per_run"][task.task_id] = {
                "is_pass": bool(is_pass),
                "top_module": task.top_module,
                "wall_s": round(wall_s, 2),
                "in_tokens": token_count.in_token_cnt,
                "out_tokens": token_count.out_token_cnt,
                "agent_error": error,
                "sim_log_tail": str(sim_log)[-2000:],
            }
            record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
            print(f"[mage] {task.task_id}: is_pass={is_pass} wall={wall_s:.1f}s")

        per_run = record["record_per_run"]
        pass_count = sum(1 for entry in per_run.values() if entry.get("is_pass"))
        record["total_record"] = {
            "pass_cnt": pass_count,
            "total_cnt": len(per_run),
            "pass_rate": (pass_count / len(per_run)) if per_run else None,
            "total_run_time_s": round(time.monotonic() - round_start, 2),
        }
        record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        rounds.append(record["total_record"])
        print(f"[mage] round {round_index}: pass {pass_count}/{len(per_run)}")

    summary = {
        "benchmark": args.benchmark,
        "method": "mage",
        "model": args.model or os.getenv("VLLM_MODEL"),
        "num_tasks": len(tasks),
        "rounds": rounds,
        "pass_rate_mean": (
            sum(r["pass_rate"] for r in rounds if r["pass_rate"] is not None)
            / max(1, sum(1 for r in rounds if r["pass_rate"] is not None))
            if rounds else None
        ),
        "use_golden_tb": args.use_golden_tb,
        "temperature": args.temperature,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[mage] wrote {output_dir / 'summary.json'}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
