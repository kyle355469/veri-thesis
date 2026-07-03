#!/usr/bin/env python3
"""Model x benchmark evaluation matrix with automatic vLLM deploy/terminate cycles.

For each model in the matrix: deploy a vLLM OpenAI-compatible server (via
scripts/auto_vllm_deploy.sh), wait until it accepts requests, run every evaluation
configured for that model against it, then terminate the server (and wait for the
GPU to drain) before deploying the next model. One GPU, one model at a time.

Default matrix:

* ``codev-r1``       (zhuyaoyu/CodeV-R1-RL-Qwen-7B)              -- RealBench + RTLLM +
* ``siliconmind-v1`` (AS-SiliconMind/SiliconMind-V1-Qwen3-4B-T-2507) verilog-eval, each as
                      "pure" (direct single-shot, zero repair) AND "router" (cascade
                      router + agentic-plan/legacy-RTL pipeline).
* ``oss20b``         (openai/gpt-oss-20b) -- the MAGE multi-agent method on RealBench,
                      RTLLM, and verilog-eval (scripts/run_mage_benchmarks.py, needs
                      ~/venv-mage), plus our router pipeline on RTLLM and verilog-eval.

Eval arms map to existing runners:

* realbench-pure    -> run_realbench_direct_model.py       (zero-repair baseline)
* realbench-router  -> run_realbench_routed.py --router cascade
* rtllm-pure        -> run_agentic_plan_legacy_rtllm.py --router all_direct, 0 repairs
* rtllm-router      -> run_agentic_plan_legacy_rtllm.py --router cascade
* verilog-eval-pure / verilog-eval-router -> run_agentic_plan_legacy_verilog_eval.py
* mage-realbench / mage-rtllm / mage-verilog-eval -> run_mage_benchmarks.py
                      (run with --mage-python)

Layout: <output-dir>/<model>/<eval>/... plus <output-dir>/<model>/logs/<eval>.log and
<output-dir>/<model>/vllm/ (server logs). A progressive matrix_summary.json records
every deployment and eval outcome; --resume skips evals whose summary.json exists
(and forwards --resume to the underlying runners).

Usage::

    python scripts/run_model_eval_matrix.py --samples 5 --concurrency 8
    python scripts/run_model_eval_matrix.py --models codev-r1 --evals rtllm-router --dry-run
    python scripts/run_model_eval_matrix.py --config my_matrix.json --resume
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"

STANDARD_EVALS = [
    "realbench-pure",
    "realbench-router",
    "rtllm-pure",
    "rtllm-router",
    "verilog-eval-pure",
    "verilog-eval-router",
]

DEFAULT_MATRIX: List[Dict[str, Any]] = [
    {
        "name": "codev-r1",
        "model": "zhuyaoyu/CodeV-R1-RL-Qwen-7B",
        "served_name": "codev-r1",
        "max_model_len": 32768,
        "tool_calling": False,
        "evals": list(STANDARD_EVALS),
    },
    {
        "name": "siliconmind-v1",
        "model": "AS-SiliconMind/SiliconMind-V1-Qwen3-4B-T-2507",
        "served_name": "siliconmind-v1",
        "max_model_len": 32768,
        "tool_calling": False,
        "evals": list(STANDARD_EVALS),
    },
    {
        "name": "oss20b",
        "model": "openai/gpt-oss-20b",
        "served_name": "gpt-oss-20b",
        "max_model_len": 32768,
        "tool_calling": False,
        "evals": [
            "mage-realbench",
            "mage-rtllm",
            "mage-verilog-eval",
            "rtllm-router",
            "verilog-eval-router",
        ],
    },
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deploy each model on vLLM in turn and run its benchmark evaluations."
    )
    parser.add_argument("--output-dir", default="runs/model_eval_matrix")
    parser.add_argument("--config", help="JSON file with a custom matrix (list of model entries).")
    parser.add_argument("--models", action="append", default=[], help="Only run these model names (repeatable).")
    parser.add_argument("--evals", action="append", default=[], help="Only run these eval names (repeatable).")
    parser.add_argument("--samples", type=int, default=1, help="Samples per task for the benchmark runners.")
    parser.add_argument("--concurrency", type=int, default=4, help="Worker threads for the benchmark runners.")
    parser.add_argument("--decider", choices=["keyword", "llm"], default="keyword", help="Tier-0 decider for router arms.")
    parser.add_argument("--resume", action="store_true", help="Skip evals with an existing summary.json; forward --resume to runners.")
    parser.add_argument("--dry-run", action="store_true", help="Print deployments and commands without running anything.")
    parser.add_argument("--stop-on-error", action="store_true", help="Abort the matrix on the first failed eval (default: continue).")

    parser.add_argument("--python", default=str(Path.home() / "venv-verilog" / "bin" / "python"),
                        help="Python used for the veri-thesis benchmark runners.")
    parser.add_argument("--mage-python", default=str(Path.home() / "venv-mage" / "bin" / "python"),
                        help="Python with MAGE deps (llama-index etc.) for mage-verilog-eval.")
    parser.add_argument("--vllm-venv", default=str(Path.home() / "venv-verilog"),
                        help="Venv containing vllm, passed to auto_vllm_deploy.sh.")
    parser.add_argument("--port-start", type=int, default=8100)
    parser.add_argument("--deploy-timeout-s", type=int, default=1800)
    parser.add_argument("--gpu-idle-mb", type=int, default=3000,
                        help="GPU memory-used threshold treated as drained between deployments.")
    parser.add_argument("--gpu-drain-timeout-s", type=int, default=180)

    parser.add_argument("--realbench-root", default="/home/kai/eval_dt/real_bench")
    parser.add_argument("--rtllm-root", default="/home/kai/eval_dt/RTLLM")
    parser.add_argument("--verilog-eval-root", default="/home/kai/eval_dt/VerilogEval-v2-NTU")
    parser.add_argument("--mage-benchmark-path", default="/home/kai/verilog-eval",
                        help="Verilog-Eval checkout (dataset_spec-to-rtl layout) for MAGE.")
    parser.add_argument("--mage-rounds", type=int, default=1, help="Independent MAGE rounds (its --n).")
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Extra 'eval-name:flag' argument appended to that eval's command; repeatable, "
        "e.g. --extra-arg 'rtllm-router:--legacy-functional-repair'.",
    )
    return parser


# --------------------------------------------------------------------------- #
# Server lifecycle
# --------------------------------------------------------------------------- #


def find_free_port(start: int, count: int = 100) -> int:
    for port in range(start, start + count):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("", port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"no open port found in [{start}, {start + count})")


def gpu_memory_used_mb() -> Optional[int]:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            check=False, capture_output=True, text=True, timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    values = [int(line) for line in completed.stdout.split() if line.strip().isdigit()]
    return sum(values) if values else None


class ServerHandle:
    def __init__(self, entry: Dict[str, Any], port: int, log_dir: Path):
        self.entry = entry
        self.port = port
        self.base_url = f"http://127.0.0.1:{port}/v1"
        self.log_dir = log_dir
        self.pid_file = log_dir / f"vllm-{port}.pid"

    @property
    def pid(self) -> Optional[int]:
        try:
            return int(self.pid_file.read_text().strip())
        except (OSError, ValueError):
            return None


def deploy_model(entry: Dict[str, Any], args: argparse.Namespace, model_dir: Path) -> ServerHandle:
    """Start vLLM for this model via auto_vllm_deploy.sh (blocks until /v1/models responds)."""
    port = find_free_port(args.port_start)
    log_dir = model_dir / "vllm"
    log_dir.mkdir(parents=True, exist_ok=True)
    handle = ServerHandle(entry, port, log_dir)
    command = [
        "bash", str(SCRIPTS / "auto_vllm_deploy.sh"),
        "--model", entry["model"],
        "--served-name", entry["served_name"],
        "--port", str(port),
        "--log-dir", str(log_dir),
        "--wait-timeout-s", str(args.deploy_timeout_s),
    ]
    if args.vllm_venv and Path(args.vllm_venv, "bin", "activate").exists():
        command += ["--venv", args.vllm_venv]
    if not entry.get("tool_calling", False):
        command += ["--no-tool-calling"]
    for extra in entry.get("vllm_extra_args", []):
        command += ["--extra-arg", extra]
    env = dict(os.environ)
    env["MAX_MODEL_LEN"] = str(entry.get("max_model_len", 32768))
    if entry.get("gpu_memory_utilization"):
        env["GPU_MEMORY_UTILIZATION"] = str(entry["gpu_memory_utilization"])
    deploy_log = model_dir / "logs" / "deploy.log"
    deploy_log.parent.mkdir(parents=True, exist_ok=True)
    print(f"[matrix] deploying {entry['name']} ({entry['model']}) on port {port} ...")
    with deploy_log.open("a", encoding="utf-8") as log_handle:
        completed = subprocess.run(
            command, cwd=REPO_ROOT, env=env, stdout=log_handle, stderr=subprocess.STDOUT,
            timeout=args.deploy_timeout_s + 300, check=False,
        )
    if completed.returncode != 0:
        tail = deploy_log.read_text(encoding="utf-8", errors="ignore")[-3000:]
        raise RuntimeError(f"vLLM deploy failed for {entry['name']} (see {deploy_log}):\n{tail}")
    print(f"[matrix] {entry['name']} ready at {handle.base_url}")
    return handle


def stop_server(handle: ServerHandle, args: argparse.Namespace) -> None:
    """Terminate the vLLM server and wait until the GPU has drained."""
    pid = handle.pid
    if pid is not None:
        print(f"[matrix] stopping vLLM pid {pid} (port {handle.port})")
        for sig, grace_s in ((signal.SIGTERM, 60), (signal.SIGKILL, 30)):
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                break
            deadline = time.time() + grace_s
            while time.time() < deadline:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(2)
            else:
                continue
            break
    handle.pid_file.unlink(missing_ok=True)
    deadline = time.time() + args.gpu_drain_timeout_s
    while time.time() < deadline:
        used = gpu_memory_used_mb()
        if used is None or used <= args.gpu_idle_mb:
            print(f"[matrix] GPU drained (used={used} MiB)")
            return
        time.sleep(5)
    print(f"[matrix] WARNING: GPU still holds {gpu_memory_used_mb()} MiB after "
          f"{args.gpu_drain_timeout_s}s; continuing anyway", file=sys.stderr)


# --------------------------------------------------------------------------- #
# Eval commands
# --------------------------------------------------------------------------- #


def eval_run_dir(model_dir: Path, eval_name: str) -> Path:
    return model_dir / eval_name


def eval_summary_path(eval_name: str, run_dir: Path) -> Path:
    if eval_name.startswith("verilog-eval-"):
        return run_dir / "spec-to-rtl" / "summary.json"
    return run_dir / "summary.json"


def build_eval_command(
    eval_name: str,
    entry: Dict[str, Any],
    args: argparse.Namespace,
    run_dir: Path,
    base_url: str,
) -> List[str]:
    served = entry["served_name"]
    py = args.python
    common_llm = ["--base-url", base_url, "--model", served]
    resume = ["--resume"] if args.resume else []

    if eval_name == "realbench-pure":
        command = [
            py, str(SCRIPTS / "run_realbench_direct_model.py"),
            "--realbench-root", args.realbench_root,
            "--output-dir", str(run_dir),
            "--solution-name", f"{entry['name']}_direct",
            "--samples", str(args.samples),
            "--concurrency", str(args.concurrency),
            *common_llm, *resume,
        ]
    elif eval_name == "realbench-router":
        command = [
            py, str(SCRIPTS / "run_realbench_routed.py"),
            "--router", "cascade",
            "--decider", args.decider,
            "--realbench-root", args.realbench_root,
            "--output-dir", str(run_dir),
            "--solution-name", f"{entry['name']}_routed",
            "--samples", str(args.samples),
            # forwarded to both underlying runners:
            "--concurrency", str(args.concurrency),
            *common_llm, *resume,
        ]
    elif eval_name in ("rtllm-pure", "rtllm-router"):
        command = [
            py, str(SCRIPTS / "run_agentic_plan_legacy_rtllm.py"),
            "--rtllm-root", args.rtllm_root,
            "--output-dir", str(run_dir),
            "--samples", str(args.samples),
            "--concurrency", str(args.concurrency),
            *common_llm, *resume,
        ]
        if eval_name == "rtllm-pure":
            command += ["--router", "all_direct", "--legacy-max-repair-attempts", "0"]
        else:
            command += ["--router", "cascade", "--decider", args.decider]
    elif eval_name in ("verilog-eval-pure", "verilog-eval-router"):
        command = [
            py, str(SCRIPTS / "run_agentic_plan_legacy_verilog_eval.py"),
            "--verilog-eval-root", args.verilog_eval_root,
            "--output-dir", str(run_dir),
            "--samples", str(args.samples),
            "--concurrency", str(args.concurrency),
            *common_llm, *resume,
        ]
        if eval_name == "verilog-eval-pure":
            command += ["--router", "all_direct", "--legacy-max-repair-attempts", "0"]
        else:
            command += ["--router", "cascade", "--decider", args.decider]
    elif eval_name in ("mage-verilog-eval", "mage-rtllm", "mage-realbench"):
        benchmark = eval_name.removeprefix("mage-")
        command = [
            args.mage_python, str(SCRIPTS / "run_mage_benchmarks.py"),
            "--benchmark", benchmark,
            "--output-dir", str(run_dir),
            "--run-identifier", entry["name"],
            "--n", str(args.mage_rounds),
            *common_llm, *resume,
        ]
        if benchmark == "verilog-eval":
            command += ["--path-benchmark", args.mage_benchmark_path]
        elif benchmark == "rtllm":
            command += ["--rtllm-root", args.rtllm_root]
        else:
            command += ["--realbench-root", args.realbench_root]
    else:
        raise ValueError(f"unknown eval {eval_name!r}")

    for extra in args.extra_arg:
        prefix, _, flag = extra.partition(":")
        if prefix == eval_name and flag:
            command.append(flag)
    command += entry.get("eval_extra_args", {}).get(eval_name, [])
    return command


def run_eval(
    eval_name: str,
    entry: Dict[str, Any],
    args: argparse.Namespace,
    model_dir: Path,
    handle: ServerHandle,
) -> Dict[str, Any]:
    run_dir = eval_run_dir(model_dir, eval_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = eval_summary_path(eval_name, run_dir)
    if args.resume and summary_path.exists():
        print(f"[matrix] {entry['name']}/{eval_name}: summary exists, skipping (resume)")
        return {"eval": eval_name, "status": "skipped", "summary_path": str(summary_path),
                "metrics": extract_metrics(eval_name, summary_path)}

    command = build_eval_command(eval_name, entry, args, run_dir, handle.base_url)
    log_path = model_dir / "logs" / f"{eval_name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["VLLM_BASE_URL"] = handle.base_url
    env["VLLM_MODEL"] = entry["served_name"]
    env.setdefault("VLLM_API_KEY", "EMPTY")
    print(f"[matrix] {entry['name']}/{eval_name}: {' '.join(command)}")
    t0 = time.time()
    with log_path.open("a", encoding="utf-8") as log_handle:
        completed = subprocess.run(
            command, cwd=REPO_ROOT, env=env, stdout=log_handle, stderr=subprocess.STDOUT, check=False,
        )
    wall_s = round(time.time() - t0, 1)
    status = "ok" if completed.returncode == 0 else f"failed(rc={completed.returncode})"
    print(f"[matrix] {entry['name']}/{eval_name}: {status} in {wall_s}s (log: {log_path})")
    return {
        "eval": eval_name,
        "status": status,
        "returncode": completed.returncode,
        "wall_s": wall_s,
        "log_path": str(log_path),
        "summary_path": str(summary_path) if summary_path.exists() else None,
        "metrics": extract_metrics(eval_name, summary_path) if summary_path.exists() else None,
    }


def extract_metrics(eval_name: str, summary_path: Path) -> Optional[Dict[str, Any]]:
    """Pull the headline numbers out of each runner's summary.json."""
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if eval_name == "realbench-pure":
        return {k: summary.get(k) for k in ("num_records", "syntax_rate", "function_rate", "pass_rate")}
    if eval_name == "realbench-router":
        combined = summary.get("combined") or {}
        return {
            "num_tasks": summary.get("num_tasks"),
            "pass_rate": combined.get("pass_rate"),
            "pass_at_k": combined.get("pass_at_k"),
            "misroutes_total": summary.get("misroutes_total"),
        }
    if eval_name.startswith("mage-"):
        return {k: summary.get(k) for k in ("num_tasks", "pass_rate_mean", "rounds")}
    if eval_name.startswith(("rtllm-", "verilog-eval-")):
        return {k: summary.get(k) for k in ("num_records", "accuracy", "pass@1", "pass@5", "flows", "wasted_plans")}
    return None


# --------------------------------------------------------------------------- #
# Matrix driver
# --------------------------------------------------------------------------- #


def load_matrix(args: argparse.Namespace) -> List[Dict[str, Any]]:
    matrix = copy.deepcopy(DEFAULT_MATRIX)
    if args.config:
        matrix = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.models:
        wanted = set(args.models)
        matrix = [entry for entry in matrix if entry["name"] in wanted]
        missing = wanted - {entry["name"] for entry in matrix}
        if missing:
            raise ValueError(f"unknown model name(s): {sorted(missing)}")
    if args.evals:
        wanted_evals = set(args.evals)
        for entry in matrix:
            entry["evals"] = [name for name in entry["evals"] if name in wanted_evals]
        matrix = [entry for entry in matrix if entry["evals"]]
    return matrix


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix = load_matrix(args)
    matrix_summary_path = output_dir / "matrix_summary.json"
    matrix_summary: Dict[str, Any] = {"models": {}}
    if args.resume and matrix_summary_path.exists():
        try:
            matrix_summary = json.loads(matrix_summary_path.read_text(encoding="utf-8"))
            matrix_summary.setdefault("models", {})
        except json.JSONDecodeError:
            pass

    print(f"[matrix] {len(matrix)} model(s): " + ", ".join(f"{m['name']}({len(m['evals'])} evals)" for m in matrix))
    if args.dry_run:
        for entry in matrix:
            fake = ServerHandle(entry, args.port_start, output_dir / entry["name"] / "vllm")
            print(f"\n[dry-run] deploy: {entry['model']} served as {entry['served_name']}")
            for eval_name in entry["evals"]:
                run_dir = eval_run_dir(output_dir / entry["name"], eval_name)
                print("  " + " ".join(build_eval_command(eval_name, entry, args, run_dir, fake.base_url)))
            print(f"[dry-run] terminate server for {entry['name']}")
        return

    for entry in matrix:
        model_dir = output_dir / entry["name"]
        model_dir.mkdir(parents=True, exist_ok=True)
        model_record = matrix_summary["models"].setdefault(
            entry["name"], {"model": entry["model"], "evals": {}}
        )
        pending = [
            name for name in entry["evals"]
            if not (args.resume and eval_summary_path(name, eval_run_dir(model_dir, name)).exists()
                    and model_record["evals"].get(name, {}).get("status") in ("ok", "skipped"))
        ]
        if not pending:
            print(f"[matrix] {entry['name']}: all evals already complete, skipping deployment")
            continue

        try:
            handle = deploy_model(entry, args, model_dir)
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            print(f"[matrix] ERROR deploying {entry['name']}: {exc}", file=sys.stderr)
            model_record["deploy_error"] = str(exc)[-2000:]
            matrix_summary_path.write_text(json.dumps(matrix_summary, indent=2), encoding="utf-8")
            if args.stop_on_error:
                raise
            continue

        model_record["base_url"] = handle.base_url
        try:
            for eval_name in entry["evals"]:
                record = run_eval(eval_name, entry, args, model_dir, handle)
                model_record["evals"][eval_name] = record
                matrix_summary_path.write_text(json.dumps(matrix_summary, indent=2), encoding="utf-8")
                if args.stop_on_error and record.get("returncode") not in (None, 0):
                    raise RuntimeError(f"eval {eval_name} failed for {entry['name']}")
        finally:
            stop_server(handle, args)

    matrix_summary_path.write_text(json.dumps(matrix_summary, indent=2), encoding="utf-8")
    print(f"\n[matrix] done. Summary: {matrix_summary_path}")
    for model_name, model_record in matrix_summary["models"].items():
        print(f"  {model_name}:")
        for eval_name, record in model_record.get("evals", {}).items():
            metrics = record.get("metrics") or {}
            headline = ", ".join(f"{k}={v}" for k, v in metrics.items() if not isinstance(v, (dict, list)))
            print(f"    {eval_name:22s} {record.get('status', '?'):14s} {headline}")


if __name__ == "__main__":
    main()
