#!/usr/bin/env python3
"""Local web UI for the agentic IP reuse RealBench demo."""

from __future__ import annotations

import argparse
import importlib.util
import json
import mimetypes
import socket
import subprocess
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_agentic_ip_reuse_realbench as realbench  # noqa: E402
from rag_rtl.json_utils import dumps_json, json_default  # noqa: E402


@dataclass(frozen=True)
class WebSettings:
    host: str
    port: int
    realbench_root: Path
    rtl_mosaic_root: Path
    chipbench_root: Path
    output_dir: Path
    prepare_problems: bool
    embedder: str
    retrieve_k: int
    context_k: int
    max_repair_attempts: int
    base_url: Optional[str]
    model: Optional[str]
    api_key: Optional[str]
    llm_timeout_s: int
    temperature: float
    max_tokens: int
    agent_timeout_s: int
    verification_timeout_s: int


TASK_CACHE_LOCK = threading.Lock()
TASK_CACHE: Dict[str, Any] = {"key": None, "tasks": [], "error": None}
JOB_LOCK = threading.Lock()
JOBS: Dict[str, Dict[str, Any]] = {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local Agentic IP Reuse RealBench web UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8780)
    parser.add_argument("--realbench-root", default="/home/kai/eval_dt/real_bench")
    parser.add_argument("--rtl-mosaic-root", default="/home/kai/eval_dt/rtl-mosaic")
    parser.add_argument("--chipbench-root", default="/home/kai/eval_dt/ChipBench/Verilog Gen")
    parser.add_argument("--output-dir", default="runs/agentic_ip_reuse_web")
    parser.add_argument(
        "--prepare-problems",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Decrypt/generate RealBench problem files when missing.",
    )
    parser.add_argument("--embedder", default="hash")
    parser.add_argument("--retrieve-k", type=int, default=8)
    parser.add_argument("--context-k", type=int, default=4)
    parser.add_argument("--max-repair-attempts", type=int, default=2)
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--api-key")
    parser.add_argument("--llm-timeout-s", type=int, default=1200)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=30000)
    parser.add_argument("--agent-timeout-s", type=int, default=30)
    parser.add_argument("--verification-timeout-s", type=int, default=300)
    args = parser.parse_args()

    settings = WebSettings(
        host=args.host,
        port=_first_open_port(args.host, args.port),
        realbench_root=Path(args.realbench_root),
        rtl_mosaic_root=Path(args.rtl_mosaic_root),
        chipbench_root=Path(args.chipbench_root),
        output_dir=Path(args.output_dir),
        prepare_problems=args.prepare_problems,
        embedder=args.embedder,
        retrieve_k=args.retrieve_k,
        context_k=args.context_k,
        max_repair_attempts=args.max_repair_attempts,
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        llm_timeout_s=args.llm_timeout_s,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        agent_timeout_s=args.agent_timeout_s,
        verification_timeout_s=args.verification_timeout_s,
    )
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((settings.host, settings.port), _make_handler(settings))
    print(f"Agentic IP reuse UI running at http://{settings.host}:{settings.port}")
    print(f"Demo links are available from http://{settings.host}:{settings.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Agentic IP reuse UI.")


def _make_handler(settings: WebSettings):
    class AgenticIpReuseWebHandler(BaseHTTPRequestHandler):
        server_version = "AgenticIpReuseWeb/1.0"

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path in {"/", "/demo"}:
                self._send_text(INDEX_HTML, "text/html; charset=utf-8")
                return
            if parsed.path == "/style.css":
                self._send_text(STYLE_CSS, "text/css; charset=utf-8")
                return
            if parsed.path == "/app.js":
                self._send_text(APP_JS, "text/javascript; charset=utf-8")
                return
            if parsed.path == "/api/config":
                self._send_json(_config_payload(settings))
                return
            if parsed.path == "/api/tasks":
                query = parse_qs(parsed.query)
                refresh = _bool_query(query, "refresh")
                self._send_json(_tasks_payload(settings, refresh=refresh))
                return
            if parsed.path == "/api/task":
                query = parse_qs(parsed.query)
                task_id = query.get("task_id", [""])[0]
                task = _find_task(settings, task_id)
                if task is None:
                    self._send_error(HTTPStatus.NOT_FOUND, f"Unknown task_id: {task_id}")
                    return
                self._send_json({"ok": True, "task": _task_payload(task, settings)})
                return
            if parsed.path == "/api/job":
                query = parse_qs(parsed.query)
                job_id = query.get("job_id", [""])[0]
                payload = _job_payload(job_id)
                if payload is None:
                    self._send_error(HTTPStatus.NOT_FOUND, f"Unknown job_id: {job_id}")
                    return
                self._send_json(payload)
                return
            if parsed.path == "/api/rtl-mosaic":
                self._send_json(_rtl_mosaic_payload(settings))
                return
            self._send_error(HTTPStatus.NOT_FOUND, "Not found")

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path not in {"/api/index", "/api/run", "/api/rtl-mosaic/run"}:
                self._send_error(HTTPStatus.NOT_FOUND, "Not found")
                return
            try:
                payload = self._read_json()
                if parsed.path == "/api/index":
                    result = _build_index_payload(settings, str(payload.get("task_id") or ""))
                elif parsed.path == "/api/rtl-mosaic/run":
                    result = _start_rtl_mosaic_payload(settings, payload)
                else:
                    result = _start_run_payload(settings, payload)
            except Exception as exc:  # noqa: BLE001
                result = {
                    "ok": False,
                    "error": str(exc),
                    "traceback": traceback.format_exc()[-4000:],
                }
            self._send_json(result, status=HTTPStatus.OK if result.get("ok") else HTTPStatus.BAD_REQUEST)

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"[agentic-ip-web] {self.address_string()} - {fmt % args}")

        def _read_json(self) -> Dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length).decode("utf-8")
            return json.loads(body or "{}")

        def _send_json(self, payload: Dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(payload, default=json_default, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_text(self, text: str, content_type: str) -> None:
            body = text.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type or mimetypes.types_map.get(".txt", "text/plain"))
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_error(self, status: HTTPStatus, message: str) -> None:
            self._send_json({"ok": False, "error": message}, status=status)

    return AgenticIpReuseWebHandler


def _config_payload(settings: WebSettings) -> Dict[str, Any]:
    return {
        "ok": True,
        "defaults": {
            "realbench_root": str(settings.realbench_root),
            "rtl_mosaic_root": str(settings.rtl_mosaic_root),
            "chipbench_root": str(settings.chipbench_root),
            "output_dir": str(settings.output_dir),
            "embedder": settings.embedder,
            "retrieve_k": settings.retrieve_k,
            "context_k": settings.context_k,
            "max_repair_attempts": settings.max_repair_attempts,
            "base_url": settings.base_url or "http://localhost:8000/v1",
            "model": settings.model or "siliconmind-server",
        },
    }


def _rtl_mosaic_payload(settings: WebSettings) -> Dict[str, Any]:
    datasets = _chipbench_counts(settings.chipbench_root)
    problems = _chipbench_problem_summaries(settings)
    output_dir = (settings.output_dir / "rtl_mosaic_demo").resolve()
    command = [
        sys.executable,
        str(settings.rtl_mosaic_root / "eval" / "run_chipbench_eval.py"),
        "--engine",
        "agentic",
        "--veri-thesis-root",
        str(ROOT),
        "--datasets",
        "cpu_ip",
        "--limit",
        "1",
        "--output-dir",
        str(output_dir),
        "--dry-run",
    ]
    _append_agentic_args(command, settings)
    _append_llm_args(command, settings)
    return {
        "ok": True,
        "rtl_mosaic_root": str(settings.rtl_mosaic_root),
        "chipbench_root": str(settings.chipbench_root),
        "datasets": datasets,
        "problems": problems,
        "problem_count": sum(1 for problem in problems if problem.get("is_gold")),
        "downloaded_problem_count": len(problems),
        "default_command": command,
        "output_dir": str(output_dir),
        "evaluator_exists": (settings.rtl_mosaic_root / "eval" / "run_chipbench_eval.py").exists(),
    }


def _chipbench_counts(chipbench_root: Path) -> Dict[str, int]:
    mapping = {
        "cpu_ip": "dataset_cpu_ip",
        "self_contain": "dataset_self_contain",
        "not_self_contain": "dataset_not_self_contain",
    }
    counts: Dict[str, int] = {}
    for dataset, dirname in mapping.items():
        dataset_dir = chipbench_root / dirname
        counts[dataset] = len(list(dataset_dir.glob("*_prompt.txt"))) if dataset_dir.exists() else 0
    return counts


def _chipbench_problem_summaries(settings: WebSettings) -> List[Dict[str, Any]]:
    gold_ids = set(_rtl_mosaic_gold_problem_ids(settings.rtl_mosaic_root))
    dataset_dirs = {
        "cpu_ip": "dataset_cpu_ip",
        "self_contain": "dataset_self_contain",
        "not_self_contain": "dataset_not_self_contain",
    }
    labels = {
        "cpu_ip": "CPU IP",
        "self_contain": "Self-contained",
        "not_self_contain": "Hierarchical",
    }
    problems: List[Dict[str, Any]] = []
    for dataset, dirname in dataset_dirs.items():
        dataset_dir = settings.chipbench_root / dirname
        if not dataset_dir.exists():
            continue
        for prompt_path in sorted(dataset_dir.glob("*_prompt.txt")):
            problem = prompt_path.name[: -len("_prompt.txt")]
            prompt = _read_text_file(prompt_path) or ""
            problems.append(
                {
                    "dataset": dataset,
                    "dataset_label": labels[dataset],
                    "problem": problem,
                    "title": _problem_title(problem),
                    "prompt": prompt,
                    "prompt_preview": _preview(prompt, 180),
                    "tag": f"{dataset}__{problem}",
                    "is_gold": problem in gold_ids,
                }
            )
    return problems


def _rtl_mosaic_gold_problem_ids(rtl_mosaic_root: Path) -> List[str]:
    path = rtl_mosaic_root / "eval" / "gold_labels.py"
    if not path.exists():
        return []
    spec = importlib.util.spec_from_file_location("rtl_mosaic_gold_labels_web", path)
    if spec is None or spec.loader is None:
        return []
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, "all_problem_ids"):
        return list(module.all_problem_ids())
    return list(getattr(module, "GOLD", {}).keys())


def _problem_title(problem: str) -> str:
    text = problem
    if "_" in text:
        parts = text.split("_", 1)
        if len(parts) == 2:
            text = parts[1]
    return text.replace("_", " ").replace("-", " ")


def _tasks_payload(settings: WebSettings, *, refresh: bool = False) -> Dict[str, Any]:
    key = (str(settings.realbench_root), settings.prepare_problems)
    with TASK_CACHE_LOCK:
        if not refresh and TASK_CACHE["key"] == key:
            tasks = TASK_CACHE["tasks"]
            error = TASK_CACHE["error"]
        else:
            try:
                tasks = _discover_all_tasks(settings)
                error = None
            except Exception as exc:  # noqa: BLE001
                tasks = []
                error = str(exc)
            TASK_CACHE.update({"key": key, "tasks": tasks, "error": error})

    return {
        "ok": error is None,
        "error": error,
        "tasks": [_task_summary(task) for task in tasks],
        "counts": _task_counts(tasks),
    }


def _discover_all_tasks(settings: WebSettings) -> List[Any]:
    args = _realbench_args(settings, task_level="both")
    return realbench.discover_tasks(args)


def _find_task(settings: WebSettings, task_id: str) -> Optional[Any]:
    if not task_id:
        return None
    _tasks_payload(settings)
    with TASK_CACHE_LOCK:
        for task in TASK_CACHE["tasks"]:
            if task.task_id == task_id:
                return task
    return None


def _task_summary(task: Any) -> Dict[str, Any]:
    return {
        "task_id": task.task_id,
        "level": task.level,
        "system": task.system,
        "task": task.task,
        "dependencies": task.dependencies,
        "dependency_count": len(task.dependencies),
        "prompt_preview": _preview(task.prompt, 260),
        "demo_url": f"/demo?task={task.task_id}",
    }


def _task_payload(task: Any, settings: WebSettings) -> Dict[str, Any]:
    payload = _task_summary(task)
    payload.update(
        {
            "prompt": task.prompt,
            "top_module": task.top_module,
            "realbench_root": str(settings.realbench_root),
            "output_dir": str(settings.output_dir),
        }
    )
    return payload


def _task_counts(tasks: List[Any]) -> Dict[str, Any]:
    by_level: Dict[str, int] = {}
    by_system: Dict[str, int] = {}
    for task in tasks:
        by_level[task.level] = by_level.get(task.level, 0) + 1
        by_system[task.system] = by_system.get(task.system, 0) + 1
    return {"total": len(tasks), "by_level": by_level, "by_system": by_system}


def _build_index_payload(settings: WebSettings, task_id: str) -> Dict[str, Any]:
    task = _find_task(settings, task_id)
    if task is None:
        return {"ok": False, "error": f"Unknown task_id: {task_id}"}
    args = _realbench_args(settings, task_level=task.level)
    bundle = realbench.build_task_index(task, args, settings.output_dir)
    return {
        "ok": True,
        "index": _bundle_payload(bundle),
    }


def _start_run_payload(settings: WebSettings, payload: Dict[str, Any]) -> Dict[str, Any]:
    task_id = str(payload.get("task_id") or "")
    task = _find_task(settings, task_id)
    if task is None:
        return {"ok": False, "error": f"Unknown task_id: {task_id}"}
    sample = _int_value(payload.get("sample"), 1)
    max_repair_attempts = max(0, _int_value(payload.get("max_repair_attempts"), settings.max_repair_attempts))
    job_id = uuid.uuid4().hex[:12]
    job = {
        "ok": True,
        "job_id": job_id,
        "state": "queued",
        "task_id": task_id,
        "task": task.task,
        "sample": sample,
        "max_repair_attempts": max_repair_attempts,
        "started_at": time.time(),
        "updated_at": time.time(),
        "stages": [],
        "result": None,
        "error": None,
    }
    with JOB_LOCK:
        JOBS[job_id] = job
    thread = threading.Thread(
        target=_run_job_thread,
        args=(settings, task, sample, bool(payload.get("resume")), max_repair_attempts, job_id),
        daemon=True,
    )
    thread.start()
    return {"ok": True, "job_id": job_id}


def _start_rtl_mosaic_payload(settings: WebSettings, payload: Dict[str, Any]) -> Dict[str, Any]:
    job_id = uuid.uuid4().hex[:12]
    dry_run = bool(payload.get("dry_run", True))
    limit = _int_value(payload.get("limit"), 1)
    dataset = str(payload.get("dataset") or "cpu_ip")
    problems = [str(item) for item in payload.get("problems") or [] if str(item).strip()]
    engine = str(payload.get("engine") or "agentic")
    if engine not in {"agentic", "harness"}:
        engine = "agentic"
    job = {
        "ok": True,
        "job_id": job_id,
        "state": "queued",
        "task_id": "rtl-mosaic-chipbench",
        "task": "RTL-Mosaic ChipBench eval",
        "sample": 1,
        "started_at": time.time(),
        "updated_at": time.time(),
        "stages": [],
        "result": None,
        "error": None,
    }
    with JOB_LOCK:
        JOBS[job_id] = job
    thread = threading.Thread(
        target=_run_rtl_mosaic_job_thread,
        args=(settings, dataset, problems, limit, dry_run, engine, job_id),
        daemon=True,
    )
    thread.start()
    return {"ok": True, "job_id": job_id}


def _run_rtl_mosaic_job_thread(
    settings: WebSettings,
    dataset: str,
    problems: List[str],
    limit: int,
    dry_run: bool,
    engine: str,
    job_id: str,
) -> None:
    _update_job(job_id, state="running")
    output_dir = (settings.output_dir / "rtl_mosaic_demo").resolve()
    script = settings.rtl_mosaic_root / "eval" / "run_chipbench_eval.py"
    datasets = ["cpu_ip", "self_contain"] if dataset == "gold" else [dataset]
    command = [
        sys.executable,
        str(script),
        "--chipbench-root",
        str(settings.chipbench_root),
        "--engine",
        engine,
        "--veri-thesis-root",
        str(ROOT),
        "--datasets",
        *datasets,
        "--output-dir",
        str(output_dir),
        "--workers",
        "1",
    ]
    if problems:
        command.extend(["--problems", *problems])
    else:
        command.extend(["--limit", str(max(limit, 1))])
    if dry_run:
        command.append("--dry-run")
    _append_agentic_args(command, settings)
    _append_llm_args(command, settings)
    _append_stage(
        job_id,
        {
            "stage": "rtl_mosaic_eval",
            "status": "running",
            "dataset": dataset,
            "problems": problems,
            "limit": limit,
            "dry_run": dry_run,
            "engine": engine,
        },
    )
    try:
        completed = subprocess.run(
            command,
            cwd=settings.rtl_mosaic_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        summary = _read_json_file(output_dir / "summary.json")
        records = _read_jsonl_file(output_dir / "records.jsonl", limit=20)
        agent_steps = _rtl_mosaic_agent_steps(records, summary)
        generated_code = _first_generated_code(records)
        result = {
            "ok": completed.returncode == 0,
            "command": command,
            "returncode": completed.returncode,
            "output_dir": str(output_dir),
            "summary": summary,
            "records": records,
            "agent_steps": agent_steps,
            "generated_code": generated_code,
            "stdout_tail": completed.stdout[-4000:],
            "stderr_tail": completed.stderr[-4000:],
        }
        if completed.returncode == 0:
            _append_stage(job_id, {"stage": "rtl_mosaic_eval", "status": "complete", "summary": summary})
            _update_job(job_id, state="complete", result=result)
        else:
            _append_stage(
                job_id,
                {"stage": "rtl_mosaic_eval", "status": "error", "error": completed.stderr[-1000:]},
            )
            _update_job(job_id, state="error", error=completed.stderr[-1000:] or "rtl-mosaic eval failed", result=result)
    except Exception as exc:  # noqa: BLE001
        _append_stage(job_id, {"stage": "rtl_mosaic_eval", "status": "error", "error": str(exc)})
        _update_job(
            job_id,
            state="error",
            error=str(exc),
            result={"ok": False, "error": str(exc), "traceback": traceback.format_exc()[-4000:]},
        )


def _append_llm_args(command: List[str], settings: WebSettings) -> None:
    if settings.base_url:
        command.extend(["--base-url", settings.base_url])
    if settings.model:
        command.extend(["--model", settings.model])
    if settings.api_key:
        command.extend(["--api-key", settings.api_key])
    command.extend(["--llm-timeout-s", str(settings.llm_timeout_s)])
    command.extend(["--temperature", str(settings.temperature)])
    command.extend(["--max-tokens", str(settings.max_tokens)])


def _append_agentic_args(command: List[str], settings: WebSettings) -> None:
    command.extend(["--embedder", settings.embedder])
    command.extend(["--retrieve-k", str(settings.retrieve_k)])
    command.extend(["--context-k", str(settings.context_k)])
    command.extend(["--max-repair-attempts", str(settings.max_repair_attempts)])
    command.extend(["--index-jobs", "1"])


def _run_job_thread(
    settings: WebSettings,
    task: Any,
    sample: int,
    resume: bool,
    max_repair_attempts: int,
    job_id: str,
) -> None:
    _update_job(job_id, state="running")
    started = time.perf_counter()

    def stage_callback(event: Dict[str, Any]) -> None:
        _append_stage(job_id, event)

    try:
        args = _realbench_args(settings, task_level=task.level)
        args.samples = 1
        args.resume = resume
        args.evaluate_only = False
        args.max_repair_attempts = max_repair_attempts
        _append_stage(job_id, {"stage": "ip_database", "status": "running", "task": task.task})
        bundle = realbench.build_task_index(task, args, settings.output_dir)
        _append_stage(
            job_id,
            {
                "stage": "ip_database",
                "status": "complete",
                "doc_count": len(bundle.documents),
                "missing_dependencies": bundle.missing_dependencies,
                "dependency_access": bundle.access,
            },
        )
        record = realbench.run_one(
            realbench.WorkItem(task=task, sample=sample),
            args,
            settings.output_dir,
            {task.task_id: bundle},
            stage_callback=stage_callback,
        )
        result = {
            "ok": True,
            "elapsed_s": time.perf_counter() - started,
            "index": _bundle_payload(bundle),
            "record": record,
            "report": _read_json_file(record.get("agent_report_path")),
            "generated_code": _read_text_file(record.get("generated_code_path")),
        }
        _append_stage(job_id, {"stage": "web_job", "status": "complete", "passed": record.get("passed")})
        _update_job(job_id, state="complete", result=result)
    except Exception as exc:  # noqa: BLE001
        _append_stage(job_id, {"stage": "web_job", "status": "error", "error": str(exc)})
        _update_job(
            job_id,
            state="error",
            error=str(exc),
            result={"ok": False, "error": str(exc), "traceback": traceback.format_exc()[-4000:]},
        )


def _job_payload(job_id: str) -> Optional[Dict[str, Any]]:
    with JOB_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return None
        return json.loads(dumps_json(job))


def _append_stage(job_id: str, event: Dict[str, Any]) -> None:
    payload = {
        "time_s": time.time(),
        "stage": event.get("stage", "unknown"),
        "status": event.get("status", "running"),
        "detail": {key: value for key, value in event.items() if key not in {"stage", "status"}},
    }
    with JOB_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return
        job["stages"].append(payload)
        job["updated_at"] = time.time()


def _update_job(job_id: str, **updates: Any) -> None:
    with JOB_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return
        job.update(updates)
        job["updated_at"] = time.time()


def _run_task_payload(settings: WebSettings, payload: Dict[str, Any]) -> Dict[str, Any]:
    task_id = str(payload.get("task_id") or "")
    task = _find_task(settings, task_id)
    if task is None:
        return {"ok": False, "error": f"Unknown task_id: {task_id}"}
    args = _realbench_args(settings, task_level=task.level)
    args.samples = 1
    args.resume = bool(payload.get("resume"))
    args.evaluate_only = False
    args.max_repair_attempts = max(0, _int_value(payload.get("max_repair_attempts"), settings.max_repair_attempts))
    sample = _int_value(payload.get("sample"), 1)
    started = time.perf_counter()
    bundle = realbench.build_task_index(task, args, settings.output_dir)
    record = realbench.run_one(realbench.WorkItem(task=task, sample=sample), args, settings.output_dir, {task.task_id: bundle})
    return {
        "ok": True,
        "elapsed_s": time.perf_counter() - started,
        "index": _bundle_payload(bundle),
        "record": record,
        "report": _read_json_file(record.get("agent_report_path")),
        "generated_code": _read_text_file(record.get("generated_code_path")),
    }


def _bundle_payload(bundle: Any) -> Dict[str, Any]:
    return {
        "index_dir": str(bundle.index_dir),
        "doc_count": len(bundle.documents),
        "dependency_paths": bundle.dependency_paths,
        "support_paths": bundle.support_paths,
        "missing_dependencies": bundle.missing_dependencies,
        "dependency_access": bundle.access,
        "all_dependencies_retrievable": all(bundle.access.values()) if bundle.access else True,
        "documents": [
            {
                "doc_id": document.doc_id,
                "tags": document.tags,
                "source_path": document.metadata.get("source_path"),
                "chars": len(document.solution),
            }
            for document in bundle.documents
        ],
    }


def _realbench_args(settings: WebSettings, *, task_level: str) -> argparse.Namespace:
    return argparse.Namespace(
        benchmark="realbench",
        realbench_root=str(settings.realbench_root),
        rtl_mosaic_root="/home/kai/eval_dt/rtl-mosaic",
        output_dir=str(settings.output_dir),
        solution_name="agentic_ip_reuse_web",
        task_level=task_level,
        include=[],
        limit=None,
        samples=1,
        concurrency=1,
        resume=False,
        evaluate_only=False,
        dry_run=False,
        prepare_problems=settings.prepare_problems,
        embedder=settings.embedder,
        retrieve_k=settings.retrieve_k,
        context_k=settings.context_k,
        index_jobs=1,
        base_url=settings.base_url,
        model=settings.model,
        api_key=settings.api_key,
        llm_timeout_s=settings.llm_timeout_s,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        max_repair_attempts=settings.max_repair_attempts,
        yosys_bin="yosys",
        verilator_bin="verilator",
        agent_timeout_s=settings.agent_timeout_s,
        verification_timeout_s=settings.verification_timeout_s,
        make_bin="make",
    )


def _read_text_file(path_value: Any) -> Optional[str]:
    if not path_value:
        return None
    path = Path(str(path_value))
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_json_file(path_value: Any) -> Optional[Dict[str, Any]]:
    text = _read_text_file(path_value)
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _read_jsonl_file(path_value: Any, *, limit: int) -> List[Dict[str, Any]]:
    text = _read_text_file(path_value)
    if not text:
        return []
    rows: List[Dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
        if len(rows) >= limit:
            break
    return rows


def _rtl_mosaic_agent_steps(records: List[Dict[str, Any]], summary: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not records:
        engine = (summary or {}).get("engine", "agentic")
        if (summary or {}).get("dry_run"):
            return [
                _step("Discover ChipBench", "complete", {"engine": engine}),
                _step("Build IP Index", "complete", {"docs": (summary or {}).get("ip_db_doc_count")}),
                _step("Dry Run", "complete", {"llm_calls": 0}),
            ]
        return []

    steps: List[Dict[str, Any]] = []
    first = records[0]
    engine = first.get("engine") or (summary or {}).get("engine")
    if first.get("dry_run"):
        return [
            _step("Discover ChipBench", "complete", {"engine": engine, "dataset": first.get("dataset"), "problem": first.get("problem")}),
            _step("Build IP Index", "complete", {"docs": first.get("ip_db_doc_count"), "index": first.get("ip_db_index")}),
            _step("Dry Run", "complete", {"llm_calls": 0, "compiled": False, "simulated": False}),
        ]
    if engine == "agentic":
        steps.extend(_agentic_steps_from_record(first))
    else:
        steps.extend(_harness_steps_from_record(first))
    if len(records) > 1:
        steps.append(_step("Batch", "complete", {"records_shown": len(records), "total_records": (summary or {}).get("num_records")}))
    return steps


def _first_generated_code(records: List[Dict[str, Any]]) -> Optional[str]:
    for record in records:
        text = _read_text_file(record.get("generated_code_path"))
        if text:
            return text
    return None


def _agentic_steps_from_record(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    stage_path = record.get("agent_stage_path")
    events = _read_jsonl_file(stage_path, limit=200) if stage_path else []
    if not events:
        return [
            _step("Build IP Index", "complete", {"docs": record.get("ip_db_doc_count")}),
            _step("Agentic Generation", "error" if record.get("generation_error") else "complete", {
                "generated": record.get("generated"),
                "error": record.get("generation_error"),
            }),
            _step("ChipBench Simulation", "complete" if record.get("compiled") else "error", {
                "compiled": record.get("compiled"),
                "passed": record.get("passed"),
                "mismatch": record.get("mismatch_line"),
            }),
        ]

    steps = [_step("Build IP Index", "complete", {"docs": record.get("ip_db_doc_count"), "index": record.get("ip_db_index")})]
    for event in events:
        steps.append(
            _step(
                _stage_label(str(event.get("stage") or "unknown")),
                str(event.get("status") or "running"),
                {
                    key: value
                    for key, value in event.items()
                    if key not in {"stage", "status"}
                },
            )
        )
    steps.append(
        _step(
            "ChipBench Simulation",
            "complete" if record.get("compiled") else "error",
            {
                "compiled": record.get("compiled"),
                "passed": record.get("passed"),
                "mismatch": record.get("mismatch_line"),
                "selected_doc_ids": record.get("selected_doc_ids"),
                "retrieved_doc_ids": record.get("retrieved_doc_ids"),
                "generated_code_path": record.get("generated_code_path"),
                "agent_report_path": record.get("agent_report_path"),
            },
        )
    )
    return steps


def _harness_steps_from_record(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        _step("Planner", "complete", {"engine": "legacy harness"}),
        _step("IP Router", "complete", {"reused": record.get("n_reused"), "generated": record.get("n_generated")}),
        _step("Integrator", "complete", {"combined_path": record.get("combined_path")}),
        _step("ChipBench Simulation", "complete" if record.get("compiled") else "error", {
            "compiled": record.get("compiled"),
            "passed": record.get("passed"),
            "mismatch": record.get("mismatch_line"),
        }),
    ]


def _step(name: str, status: str, detail: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "detail": {key: value for key, value in detail.items() if value not in (None, "", [])},
    }


def _stage_label(stage: str) -> str:
    labels = {
        "agent_start": "Agent Start",
        "requirements": "Requirements",
        "decomposition": "Decomposition",
        "ip_search": "IP Search",
        "ip_evaluation": "IP Evaluation",
        "rtl_generation": "RTL Generation",
        "verification": "Syntax/Lint Verification",
        "repair": "Repair",
        "agent_complete": "Agent Complete",
    }
    if stage.startswith("llm:"):
        return "LLM " + stage[4:].replace("_", " ").title()
    return labels.get(stage, stage.replace("_", " ").title())


def _preview(text: str, limit: int) -> str:
    text = str(text).strip()
    return text if len(text) <= limit else text[:limit] + "...[truncated]"


def _int_value(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _bool_query(query: Dict[str, List[str]], key: str) -> bool:
    value = (query.get(key) or [""])[0].lower()
    return value in {"1", "true", "yes", "on"}


def _first_open_port(host: str, preferred: int) -> int:
    for port in range(preferred, preferred + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind((host, port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"No open port found from {preferred} to {preferred + 99}")


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Agentic IP Reuse</title>
  <link rel="stylesheet" href="/style.css">
</head>
<body>
  <aside class="sidebar">
    <div class="brand">
      <div class="brand-mark">IP</div>
      <div>
        <h1>Agentic IP Reuse</h1>
        <p id="serverLine">Loading server settings</p>
      </div>
    </div>
    <div class="mode-switch" role="tablist" aria-label="Benchmark category">
      <button id="realbenchModeBtn" type="button" class="active" role="tab">RealBench</button>
      <button id="mosaicModeBtn" type="button" role="tab">RTL-Mosaic</button>
    </div>
    <div id="realbenchBrowser" class="browser-pane">
      <div class="filters">
        <label>
          Search
          <input id="searchInput" type="search" placeholder="aes, e203, fifo">
        </label>
        <label>
          Level
          <select id="levelFilter">
            <option value="">All</option>
            <option value="module">Module</option>
            <option value="system">System</option>
          </select>
        </label>
        <label>
          System
          <select id="systemFilter">
            <option value="">All</option>
          </select>
        </label>
      </div>
      <div class="task-list" id="taskList"></div>
    </div>
    <div id="mosaicBrowser" class="browser-pane" hidden>
      <div class="filters">
        <label>
          Dataset
          <select id="mosaicDataset">
            <option value="gold">Gold eval set</option>
            <option value="cpu_ip">CPU IP</option>
            <option value="self_contain">Self-contained</option>
            <option value="not_self_contain">Hierarchical</option>
          </select>
        </label>
        <label>
          Search
          <input id="mosaicSearch" type="search" placeholder="alu, counter, fifo">
        </label>
        <label>
          Engine
          <select id="mosaicEngine">
            <option value="agentic">Agentic IP Reuse</option>
            <option value="harness">Legacy Harness</option>
          </select>
        </label>
        <label>
          Limit
          <input id="mosaicLimit" type="number" min="1" max="45" value="1">
        </label>
        <label class="checkbox-label">
          <input id="mosaicDryRun" type="checkbox" checked>
          Dry run
        </label>
      </div>
      <div class="task-list" id="mosaicQuestionList"></div>
    </div>
  </aside>

  <main id="realbenchView" class="main">
    <header class="toolbar">
      <div>
        <div class="eyebrow" id="selectedMeta">RealBench</div>
        <h2 id="selectedTitle">Select a question</h2>
      </div>
      <div class="toolbar-actions">
        <label class="inline-number">
          Retries
          <input id="maxRepairAttempts" type="number" min="0" max="20" value="2">
        </label>
        <button id="copyDemoBtn" title="Copy demo link" disabled>Copy Link</button>
        <button id="indexBtn" title="Build dependency-only IP database" disabled>Build IP DB</button>
        <button id="runBtn" title="Run agentic IP reuse" disabled>Run</button>
      </div>
    </header>

    <section class="summary-grid">
      <div class="metric">
        <span>Tasks</span>
        <strong id="taskCount">0</strong>
      </div>
      <div class="metric">
        <span>Dependencies</span>
        <strong id="depCount">0</strong>
      </div>
      <div class="metric">
        <span>IP DB Docs</span>
        <strong id="docCount">-</strong>
      </div>
      <div class="metric">
        <span>Result</span>
        <strong id="resultMetric">-</strong>
      </div>
    </section>

    <section class="workspace">
      <div class="panel prompt-panel">
        <div class="panel-head">
          <h3>Question</h3>
          <span id="demoUrl"></span>
        </div>
        <pre id="promptText">Choose a RealBench question from the left.</pre>
      </div>
      <div class="panel">
        <div class="panel-head">
          <h3>Supplied IP</h3>
        </div>
        <div id="deps"></div>
      </div>
      <div class="panel stages-panel">
        <div class="panel-head">
          <h3>Agent Stages</h3>
        </div>
        <div id="stageList" class="stage-list"></div>
      </div>
    </section>

    <section class="workspace output-row">
      <div class="panel">
        <div class="panel-head">
          <h3>Run Status</h3>
        </div>
        <pre id="statusText">Idle.</pre>
      </div>
      <div class="panel">
        <div class="panel-head">
          <h3>Generated RTL</h3>
        </div>
        <pre id="codeText"></pre>
      </div>
    </section>
  </main>

  <main id="mosaicView" class="main" hidden>
    <header class="toolbar">
      <div>
        <div class="eyebrow">RTL-Mosaic</div>
        <h2>ChipBench Evaluation</h2>
      </div>
      <div class="toolbar-actions">
        <button id="mosaicRunBtn" type="button">Run Check</button>
      </div>
    </header>

    <section class="summary-grid">
      <div class="metric">
        <span>CPU IP</span>
        <strong id="mosaicCpuCount">0</strong>
      </div>
      <div class="metric">
        <span>Self-contained</span>
        <strong id="mosaicSelfCount">0</strong>
      </div>
      <div class="metric">
        <span>Hierarchical</span>
        <strong id="mosaicHierCount">0</strong>
      </div>
      <div class="metric">
        <span>Result</span>
        <strong id="mosaicResult">-</strong>
      </div>
    </section>

    <section class="workspace mosaic-workspace">
      <div class="panel">
        <div class="panel-head">
          <h3>Question Prompt</h3>
          <span id="mosaicSelectedMeta">No question selected</span>
        </div>
        <pre id="mosaicPromptText">Select an RTL-Mosaic question from the left.</pre>
      </div>
      <div class="panel">
        <div class="panel-head">
          <h3>ChipBench Status</h3>
          <span id="mosaicInfo">Loading</span>
        </div>
        <pre id="mosaicStatus"></pre>
      </div>
    </section>

    <section class="workspace output-row">
      <div class="panel">
        <div class="panel-head">
          <h3>Run Status</h3>
        </div>
        <pre id="mosaicRunStatus">Idle.</pre>
      </div>
      <div class="panel">
        <div class="panel-head">
          <h3>Selected Question</h3>
        </div>
        <pre id="mosaicQuestionSummary">No question selected.</pre>
      </div>
    </section>

    <section class="panel mosaic-agent-panel">
      <div class="panel-head">
        <h3>Agent Steps</h3>
        <span id="mosaicStepSummary">No run yet</span>
      </div>
      <div id="mosaicStepList" class="mosaic-step-list"></div>
    </section>

    <section class="panel">
      <div class="panel-head">
        <h3>Result Code</h3>
        <span id="mosaicCodeMeta">No generated RTL</span>
      </div>
      <pre id="mosaicCodeText"></pre>
    </section>
  </main>

  <script src="/app.js"></script>
</body>
</html>
"""


STYLE_CSS = r""":root {
  color-scheme: light;
  --bg: #f7f8fb;
  --panel: #ffffff;
  --ink: #17202a;
  --muted: #657080;
  --line: #d9dee8;
  --accent: #0f766e;
  --accent-2: #b42318;
  --soft: #e7f4f2;
  --code: #111827;
}

* { box-sizing: border-box; }

body {
  margin: 0;
  min-height: 100vh;
  display: grid;
  grid-template-columns: 360px 1fr;
  background: var(--bg);
  color: var(--ink);
  font: 14px/1.45 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.sidebar {
  border-right: 1px solid var(--line);
  background: #fbfcfe;
  min-height: 100vh;
  display: grid;
  grid-template-rows: auto auto 1fr;
}

.brand {
  display: grid;
  grid-template-columns: 44px 1fr;
  gap: 12px;
  align-items: center;
  padding: 20px;
  border-bottom: 1px solid var(--line);
}

.brand-mark {
  width: 44px;
  height: 44px;
  display: grid;
  place-items: center;
  border: 1px solid #93c5bd;
  background: var(--soft);
  color: #115e59;
  font-weight: 800;
  border-radius: 8px;
}

h1, h2, h3, p { margin: 0; }
h1 { font-size: 18px; }
h2 { font-size: 24px; }
h3 { font-size: 15px; }
p, .eyebrow, label, .task-meta, .metric span, #demoUrl { color: var(--muted); }

.mode-switch {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 6px;
  padding: 14px 20px;
  border-bottom: 1px solid var(--line);
}

.mode-switch button {
  min-width: 0;
}

.mode-switch button.active {
  background: var(--accent);
  border-color: var(--accent);
  color: #fff;
}

.browser-pane {
  min-height: 0;
  display: grid;
  grid-template-rows: auto 1fr;
}

.filters {
  display: grid;
  gap: 12px;
  padding: 16px 20px;
  border-bottom: 1px solid var(--line);
}

label {
  display: grid;
  gap: 6px;
  font-size: 12px;
  font-weight: 650;
}

.checkbox-label {
  grid-template-columns: auto 1fr;
  align-items: center;
  color: var(--ink);
}

.checkbox-label input {
  width: auto;
  min-height: auto;
}

input, select, button {
  min-height: 36px;
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 0 10px;
  background: #fff;
  color: var(--ink);
  font: inherit;
}

button {
  cursor: pointer;
  font-weight: 700;
}

button:not(:disabled):hover { border-color: var(--accent); color: var(--accent); }
button:disabled { opacity: .45; cursor: not-allowed; }

.task-list {
  overflow: auto;
  padding: 10px;
}

.task-item {
  width: 100%;
  text-align: left;
  display: grid;
  gap: 5px;
  min-height: 84px;
  padding: 10px;
  border-radius: 7px;
  border: 1px solid transparent;
  background: transparent;
}

.task-item.active {
  border-color: #99d2cb;
  background: var(--soft);
}

.task-name {
  font-weight: 750;
  overflow-wrap: anywhere;
}

.task-meta {
  font-size: 12px;
}

.main {
  min-width: 0;
  padding: 22px;
  display: grid;
  gap: 16px;
  align-content: start;
}

.toolbar {
  display: flex;
  justify-content: space-between;
  gap: 18px;
  align-items: start;
}

.toolbar-actions {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  align-items: end;
}

.toolbar-actions button:last-child {
  background: var(--accent);
  color: white;
  border-color: var(--accent);
}

.inline-number {
  min-width: 84px;
  gap: 4px;
}

.inline-number input {
  width: 84px;
}

.summary-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(130px, 1fr));
  gap: 12px;
}

.metric, .panel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
}

.metric {
  min-height: 72px;
  display: grid;
  align-content: center;
  gap: 4px;
  padding: 12px 14px;
}

.metric strong {
  font-size: 22px;
}

.workspace {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(280px, .62fr) minmax(260px, .58fr);
  gap: 14px;
}

.output-row {
  grid-template-columns: minmax(280px, .8fr) minmax(0, 1.2fr);
}

.mosaic-workspace {
  grid-template-columns: minmax(0, 1fr) minmax(320px, .8fr);
}

.mosaic-agent-panel {
  min-height: 180px;
}

.panel {
  min-height: 220px;
  overflow: hidden;
}

.panel-head {
  min-height: 44px;
  padding: 12px 14px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  border-bottom: 1px solid var(--line);
}

pre {
  margin: 0;
  padding: 14px;
  white-space: pre-wrap;
  overflow: auto;
  max-height: 520px;
  color: var(--code);
  font: 12px/1.5 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}

#deps {
  padding: 12px;
  display: grid;
  gap: 8px;
}

.stage-list {
  padding: 12px;
  display: grid;
  gap: 8px;
  max-height: 520px;
  overflow: auto;
}

.mosaic-step-list {
  padding: 12px;
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
  max-height: 540px;
  overflow: auto;
}

.dep, .stage, .agent-step {
  border: 1px solid var(--line);
  border-radius: 7px;
  padding: 9px 10px;
  display: grid;
  gap: 4px;
}

.agent-step.complete { border-color: #99d2cb; background: #f1fbf9; }
.agent-step.running { border-color: #9ab7ff; background: #f2f6ff; }
.agent-step.error { border-color: #f5b5ac; background: #fff5f3; }
.agent-step strong { overflow-wrap: anywhere; }
.agent-step small { color: var(--muted); overflow-wrap: anywhere; }
.agent-step pre {
  padding: 8px;
  max-height: 120px;
  background: #fbfcfe;
  border: 1px solid var(--line);
  border-radius: 6px;
}

.dep strong { overflow-wrap: anywhere; }
.dep small, .stage small { color: var(--muted); overflow-wrap: anywhere; }
.stage.running { border-color: #9ab7ff; background: #f2f6ff; }
.stage.complete { border-color: #99d2cb; background: #f1fbf9; }
.stage.error { border-color: #f5b5ac; background: #fff5f3; }
.stage strong { text-transform: capitalize; }
.ok { color: var(--accent); }
.bad { color: var(--accent-2); }

[hidden] { display: none !important; }

@media (max-width: 980px) {
  body { grid-template-columns: 1fr; }
  .sidebar { min-height: auto; max-height: 55vh; }
  .workspace, .output-row, .summary-grid, .mosaic-workspace, .mosaic-step-list { grid-template-columns: 1fr; }
  .toolbar { display: grid; }
}
"""


APP_JS = r"""const state = {
  tasks: [],
  selected: null,
  selectedId: new URLSearchParams(location.search).get("task") || "",
  mode: new URLSearchParams(location.search).get("bench") === "rtl-mosaic" ? "mosaic" : "realbench",
  mosaic: null,
  mosaicSelectedProblem: "",
};

const els = {
  serverLine: document.getElementById("serverLine"),
  realbenchModeBtn: document.getElementById("realbenchModeBtn"),
  mosaicModeBtn: document.getElementById("mosaicModeBtn"),
  realbenchBrowser: document.getElementById("realbenchBrowser"),
  mosaicBrowser: document.getElementById("mosaicBrowser"),
  realbenchView: document.getElementById("realbenchView"),
  mosaicView: document.getElementById("mosaicView"),
  taskList: document.getElementById("taskList"),
  searchInput: document.getElementById("searchInput"),
  levelFilter: document.getElementById("levelFilter"),
  systemFilter: document.getElementById("systemFilter"),
  mosaicDataset: document.getElementById("mosaicDataset"),
  mosaicSearch: document.getElementById("mosaicSearch"),
  mosaicEngine: document.getElementById("mosaicEngine"),
  mosaicLimit: document.getElementById("mosaicLimit"),
  mosaicDryRun: document.getElementById("mosaicDryRun"),
  mosaicQuestionList: document.getElementById("mosaicQuestionList"),
  selectedMeta: document.getElementById("selectedMeta"),
  selectedTitle: document.getElementById("selectedTitle"),
  taskCount: document.getElementById("taskCount"),
  depCount: document.getElementById("depCount"),
  docCount: document.getElementById("docCount"),
  resultMetric: document.getElementById("resultMetric"),
  demoUrl: document.getElementById("demoUrl"),
  promptText: document.getElementById("promptText"),
  deps: document.getElementById("deps"),
  stageList: document.getElementById("stageList"),
  statusText: document.getElementById("statusText"),
  codeText: document.getElementById("codeText"),
  maxRepairAttempts: document.getElementById("maxRepairAttempts"),
  indexBtn: document.getElementById("indexBtn"),
  runBtn: document.getElementById("runBtn"),
  copyDemoBtn: document.getElementById("copyDemoBtn"),
  mosaicInfo: document.getElementById("mosaicInfo"),
  mosaicSelectedMeta: document.getElementById("mosaicSelectedMeta"),
  mosaicPromptText: document.getElementById("mosaicPromptText"),
  mosaicQuestionSummary: document.getElementById("mosaicQuestionSummary"),
  mosaicCpuCount: document.getElementById("mosaicCpuCount"),
  mosaicSelfCount: document.getElementById("mosaicSelfCount"),
  mosaicHierCount: document.getElementById("mosaicHierCount"),
  mosaicResult: document.getElementById("mosaicResult"),
  mosaicStatus: document.getElementById("mosaicStatus"),
  mosaicRunStatus: document.getElementById("mosaicRunStatus"),
  mosaicRunBtn: document.getElementById("mosaicRunBtn"),
  mosaicStepSummary: document.getElementById("mosaicStepSummary"),
  mosaicStepList: document.getElementById("mosaicStepList"),
  mosaicCodeMeta: document.getElementById("mosaicCodeMeta"),
  mosaicCodeText: document.getElementById("mosaicCodeText"),
};

async function init() {
  const config = await getJSON("/api/config");
  els.serverLine.textContent = `${config.defaults.model} at ${config.defaults.base_url}`;
  els.maxRepairAttempts.value = String(config.defaults.max_repair_attempts ?? 2);
  await loadMosaicStatus();
  await loadTasks();
  bindEvents();
  switchMode(state.mode, false);
}

function bindEvents() {
  els.realbenchModeBtn.addEventListener("click", () => switchMode("realbench", true));
  els.mosaicModeBtn.addEventListener("click", () => switchMode("mosaic", true));
  els.searchInput.addEventListener("input", renderTasks);
  els.levelFilter.addEventListener("change", renderTasks);
  els.systemFilter.addEventListener("change", renderTasks);
  els.mosaicDataset.addEventListener("change", () => {
    state.mosaicSelectedProblem = "";
    renderMosaicQuestions();
  });
  els.mosaicSearch.addEventListener("input", renderMosaicQuestions);
  els.indexBtn.addEventListener("click", buildIndex);
  els.runBtn.addEventListener("click", runSelected);
  els.copyDemoBtn.addEventListener("click", copyDemoLink);
  els.mosaicRunBtn.addEventListener("click", runMosaicSelected);
}

function switchMode(mode, updateUrl) {
  state.mode = mode === "mosaic" ? "mosaic" : "realbench";
  const mosaic = state.mode === "mosaic";
  els.realbenchModeBtn.classList.toggle("active", !mosaic);
  els.mosaicModeBtn.classList.toggle("active", mosaic);
  els.realbenchModeBtn.setAttribute("aria-selected", String(!mosaic));
  els.mosaicModeBtn.setAttribute("aria-selected", String(mosaic));
  els.realbenchBrowser.hidden = mosaic;
  els.mosaicBrowser.hidden = !mosaic;
  els.realbenchView.hidden = mosaic;
  els.mosaicView.hidden = !mosaic;
  if (updateUrl) {
    const params = new URLSearchParams(location.search);
    params.set("bench", mosaic ? "rtl-mosaic" : "realbench");
    if (!mosaic && state.selectedId) params.set("task", state.selectedId);
    if (mosaic) params.delete("task");
    history.pushState({}, "", `/demo?${params.toString()}`);
  }
}

async function loadMosaicStatus() {
  const payload = await getJSON("/api/rtl-mosaic");
  state.mosaic = payload;
  const counts = payload.datasets || {};
  const total = payload.problem_count || 0;
  els.mosaicInfo.textContent = `${total} gold questions · ${payload.downloaded_problem_count || 0} downloaded prompts`;
  els.mosaicCpuCount.textContent = String(counts.cpu_ip || 0);
  els.mosaicSelfCount.textContent = String(counts.self_contain || 0);
  els.mosaicHierCount.textContent = String(counts.not_self_contain || 0);
  els.mosaicRunBtn.disabled = !payload.evaluator_exists;
  els.mosaicStatus.textContent = JSON.stringify({
    evaluator_exists: payload.evaluator_exists,
    rtl_mosaic_root: payload.rtl_mosaic_root,
    chipbench_root: payload.chipbench_root,
    problem_count: payload.problem_count,
    default_command: payload.default_command,
  }, null, 2);
  renderMosaicQuestions();
}

function filteredMosaicProblems() {
  const selectedDataset = els.mosaicDataset.value;
  const query = els.mosaicSearch.value.trim().toLowerCase();
  return (state.mosaic?.problems || []).filter(problem => {
    const datasetOk = selectedDataset === "gold" ? problem.is_gold : problem.dataset === selectedDataset;
    const hay = `${problem.problem} ${problem.title} ${problem.dataset_label} ${problem.prompt_preview}`.toLowerCase();
    return datasetOk && (!query || hay.includes(query));
  });
}

function renderMosaicQuestions() {
  const problems = filteredMosaicProblems();
  if (!state.mosaicSelectedProblem || !problems.some(problem => problem.problem === state.mosaicSelectedProblem)) {
    state.mosaicSelectedProblem = problems[0]?.problem || "";
  }
  updateMosaicPrompt();
  els.mosaicQuestionList.replaceChildren(...problems.map(problem => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `task-item${state.mosaicSelectedProblem === problem.problem ? " active" : ""}`;
    button.innerHTML = `
      <span class="task-name">${escapeHTML(problem.problem)}</span>
      <span class="task-meta">${escapeHTML(problem.dataset_label)} · ${problem.is_gold ? "gold eval" : "extra"} · ${escapeHTML(problem.title)}</span>
      <span class="task-meta">${escapeHTML(problem.prompt_preview)}</span>
    `;
    button.addEventListener("click", () => {
      state.mosaicSelectedProblem = problem.problem;
      updateMosaicPrompt();
      renderMosaicQuestions();
    });
    return button;
  }));
  if (!problems.length) {
    els.mosaicQuestionList.innerHTML = `<div class="task-item"><span class="task-name">No questions matched</span><span class="task-meta">Adjust the dataset or search filter.</span></div>`;
  }
}

function selectedMosaicProblem() {
  return (state.mosaic?.problems || []).find(problem => problem.problem === state.mosaicSelectedProblem) || null;
}

function updateMosaicPrompt() {
  const problem = selectedMosaicProblem();
  if (!problem) {
    els.mosaicSelectedMeta.textContent = "No question selected";
    els.mosaicPromptText.textContent = "Select an RTL-Mosaic question from the left.";
    els.mosaicQuestionSummary.textContent = "No question selected.";
    return;
  }
  els.mosaicSelectedMeta.textContent = `${problem.dataset_label} · ${problem.is_gold ? "gold eval" : "extra"}`;
  els.mosaicPromptText.textContent = problem.prompt || problem.prompt_preview || "";
  els.mosaicQuestionSummary.textContent = JSON.stringify({
    problem: problem.problem,
    dataset: problem.dataset,
    title: problem.title,
    is_gold: problem.is_gold,
    tag: problem.tag,
  }, null, 2);
}

async function runMosaicSelected() {
  els.mosaicRunBtn.disabled = true;
  els.mosaicResult.textContent = "running";
  els.mosaicRunStatus.textContent = "Starting RTL-Mosaic ChipBench run.";
  els.mosaicCodeMeta.textContent = "Waiting for generated RTL";
  els.mosaicCodeText.textContent = "";
  renderMosaicSteps([{name: "Launch Evaluation", status: "running", detail: {
    engine: els.mosaicEngine.value,
    dataset: els.mosaicDataset.value,
    problem: state.mosaicSelectedProblem,
    dry_run: els.mosaicDryRun.checked,
  }}]);
  try {
    const started = await postJSON("/api/rtl-mosaic/run", {
      dataset: els.mosaicDataset.value,
      engine: els.mosaicEngine.value,
      problems: state.mosaicSelectedProblem ? [state.mosaicSelectedProblem] : [],
      limit: Number(els.mosaicLimit.value || 1),
      dry_run: els.mosaicDryRun.checked,
    });
    await pollMosaicJob(started.job_id);
  } catch (err) {
    els.mosaicResult.textContent = "error";
    els.mosaicRunStatus.textContent = err.message;
  } finally {
    els.mosaicRunBtn.disabled = false;
  }
}

async function pollMosaicJob(jobId) {
  for (;;) {
    const payload = await getJSON(`/api/job?job_id=${encodeURIComponent(jobId)}`);
    if (payload.state === "complete" || payload.state === "error") {
      const result = payload.result || {};
      const summary = result.summary || {};
      els.mosaicResult.textContent = payload.state === "error" ? "error" : (summary.dry_run ? "checked" : `${summary.passed || 0}/${summary.num_records || 0}`);
      els.mosaicRunStatus.textContent = JSON.stringify({
        state: payload.state,
        output_dir: result.output_dir,
        returncode: result.returncode,
        summary: result.summary,
        records: result.records,
        agent_steps: result.agent_steps,
        generated_code_path: result.records?.[0]?.generated_code_path,
        stdout_tail: result.stdout_tail,
        stderr_tail: result.stderr_tail,
      }, null, 2);
      renderMosaicSteps(result.agent_steps || []);
      renderMosaicCode(result);
      if (payload.state === "error") throw new Error(payload.error || "RTL-Mosaic eval failed");
      return;
    }
    const last = (payload.stages || [])[payload.stages.length - 1];
    els.mosaicRunStatus.textContent = JSON.stringify({
      state: payload.state,
      current_stage: last ? stageName(last.stage) : "queued",
      detail: last?.detail || {},
    }, null, 2);
    renderMosaicSteps([{name: last ? stageName(last.stage) : "Queued", status: last?.status || "running", detail: last?.detail || {}}]);
    await sleep(900);
  }
}

function renderMosaicCode(result) {
  const code = result.generated_code || "";
  const path = result.records?.find(record => record.generated_code_path)?.generated_code_path || "";
  if (!code) {
    els.mosaicCodeMeta.textContent = result.summary?.dry_run ? "Dry run produced no code" : "No generated RTL";
    els.mosaicCodeText.textContent = "";
    return;
  }
  els.mosaicCodeMeta.textContent = path || `${code.length} chars`;
  els.mosaicCodeText.textContent = code;
}

function renderMosaicSteps(steps) {
  if (!steps.length) {
    els.mosaicStepSummary.textContent = "No agent trace";
    els.mosaicStepList.innerHTML = `<div class="agent-step"><strong>No steps available</strong><small>Run an RTL-Mosaic agentic evaluation to populate this trace.</small></div>`;
    return;
  }
  const counts = steps.reduce((acc, step) => {
    const key = step.status === "error" ? "error" : step.status === "complete" ? "complete" : "running";
    acc[key] = (acc[key] || 0) + 1;
    return acc;
  }, {});
  els.mosaicStepSummary.textContent = `${steps.length} steps · ${counts.complete || 0} complete · ${counts.error || 0} errors`;
  els.mosaicStepList.replaceChildren(...steps.map(step => {
    const div = document.createElement("div");
    const cls = step.status === "error" ? "error" : step.status === "complete" ? "complete" : "running";
    div.className = `agent-step ${cls}`;
    div.innerHTML = `
      <strong>${escapeHTML(step.name || "Step")}</strong>
      <small>${escapeHTML(step.status || "running")}</small>
      <pre>${escapeHTML(formatStepDetail(step.detail || {}))}</pre>
    `;
    return div;
  }));
}

function formatStepDetail(detail) {
  const compact = {};
  for (const [key, value] of Object.entries(detail || {})) {
    if (value === null || value === undefined || value === "" || (Array.isArray(value) && !value.length)) continue;
    compact[key] = value;
  }
  return JSON.stringify(compact, null, 2);
}

async function loadTasks() {
  const payload = await getJSON("/api/tasks");
  if (!payload.ok) {
    els.statusText.textContent = payload.error || "Unable to load RealBench tasks.";
    return;
  }
  state.tasks = payload.tasks;
  els.taskCount.textContent = String(payload.counts.total);
  fillSystems(payload.counts.by_system || {});
  renderTasks();
  const initial = state.tasks.find(t => t.task_id === state.selectedId) || state.tasks[0];
  if (initial) selectTask(initial.task_id, false);
}

function fillSystems(bySystem) {
  for (const system of Object.keys(bySystem).sort()) {
    const option = document.createElement("option");
    option.value = system;
    option.textContent = `${system} (${bySystem[system]})`;
    els.systemFilter.appendChild(option);
  }
}

function filteredTasks() {
  const q = els.searchInput.value.trim().toLowerCase();
  const level = els.levelFilter.value;
  const system = els.systemFilter.value;
  return state.tasks.filter(task => {
    const hay = `${task.task} ${task.system} ${task.level} ${task.dependencies.join(" ")}`.toLowerCase();
    return (!q || hay.includes(q)) && (!level || task.level === level) && (!system || task.system === system);
  });
}

function renderTasks() {
  const tasks = filteredTasks();
  els.taskList.replaceChildren(...tasks.map(taskButton));
}

function taskButton(task) {
  const button = document.createElement("button");
  button.className = `task-item${state.selectedId === task.task_id ? " active" : ""}`;
  button.type = "button";
  button.innerHTML = `
    <span class="task-name">${escapeHTML(task.task)}</span>
    <span class="task-meta">${escapeHTML(task.level)} · ${escapeHTML(task.system)} · ${task.dependency_count} IP</span>
    <span class="task-meta">${escapeHTML(task.prompt_preview)}</span>
  `;
  button.addEventListener("click", () => selectTask(task.task_id, true));
  return button;
}

async function selectTask(taskId, updateUrl) {
  state.selectedId = taskId;
  renderTasks();
  const payload = await getJSON(`/api/task?task_id=${encodeURIComponent(taskId)}`);
  if (!payload.ok) {
    els.statusText.textContent = payload.error || "Unable to load task.";
    return;
  }
  state.selected = payload.task;
  if (updateUrl) history.pushState({}, "", `/demo?task=${encodeURIComponent(taskId)}`);
  els.selectedMeta.textContent = `${payload.task.level} · ${payload.task.system}`;
  els.selectedTitle.textContent = payload.task.task;
  els.depCount.textContent = String(payload.task.dependencies.length);
  els.docCount.textContent = "-";
  els.resultMetric.textContent = "-";
  els.demoUrl.textContent = payload.task.demo_url;
  els.promptText.textContent = payload.task.prompt;
  els.codeText.textContent = "";
  els.statusText.textContent = "Ready.";
  renderStages([]);
  els.indexBtn.disabled = false;
  els.runBtn.disabled = false;
  els.copyDemoBtn.disabled = false;
  renderDeps(payload.task.dependencies.map(name => ({name})));
}

function renderDeps(deps, index = null) {
  if (!deps.length) {
    els.deps.innerHTML = `<div class="dep"><strong>No declared dependencies</strong><small>The task will use new RTL unless retrieval finds support docs.</small></div>`;
    return;
  }
  els.deps.replaceChildren(...deps.map(dep => {
    const div = document.createElement("div");
    const access = index?.dependency_access?.[dep.name];
    const path = index?.dependency_paths?.[dep.name] || "";
    const cls = access === false ? "bad" : "ok";
    const label = access === undefined ? "not checked" : access ? "retrievable" : "not retrieved";
    div.className = "dep";
    div.innerHTML = `<strong>${escapeHTML(dep.name)}</strong><small class="${cls}">${label}</small><small>${escapeHTML(path)}</small>`;
    return div;
  }));
}

async function buildIndex() {
  if (!state.selected) return;
  els.indexBtn.disabled = true;
  els.statusText.textContent = "Building dependency-only IP database...";
  renderStages([{stage: "ip_database", status: "running", detail: {task: state.selected.task}}]);
  try {
    const payload = await postJSON("/api/index", {task_id: state.selected.task_id});
    els.docCount.textContent = String(payload.index.doc_count);
    renderDeps(state.selected.dependencies.map(name => ({name})), payload.index);
    renderStages([
      {stage: "ip_database", status: "complete", detail: {
        doc_count: payload.index.doc_count,
        dependency_access: payload.index.dependency_access,
      }},
    ]);
    els.statusText.textContent = formatIndex(payload.index);
  } catch (err) {
    els.statusText.textContent = err.message;
  } finally {
    els.indexBtn.disabled = false;
  }
}

async function runSelected() {
  if (!state.selected) return;
  const maxRepairAttempts = Math.max(0, Number.parseInt(els.maxRepairAttempts.value || "0", 10) || 0);
  els.runBtn.disabled = true;
  els.indexBtn.disabled = true;
  els.maxRepairAttempts.disabled = true;
  els.resultMetric.textContent = "running";
  els.statusText.textContent = "Starting background run.";
  renderStages([{stage: "web_job", status: "queued", detail: {
    task: state.selected.task,
    max_repair_attempts: maxRepairAttempts,
  }}]);
  try {
    const started = await postJSON("/api/run", {
      task_id: state.selected.task_id,
      sample: 1,
      max_repair_attempts: maxRepairAttempts,
    });
    els.statusText.textContent = `Running job ${started.job_id}.`;
    await pollJob(started.job_id);
  } catch (err) {
    els.resultMetric.textContent = "error";
    els.statusText.textContent = err.message;
  } finally {
    els.runBtn.disabled = false;
    els.indexBtn.disabled = false;
    els.maxRepairAttempts.disabled = false;
  }
}

async function pollJob(jobId) {
  for (;;) {
    const payload = await getJSON(`/api/job?job_id=${encodeURIComponent(jobId)}`);
    renderStages(payload.stages || []);
    if (payload.state === "complete" || payload.state === "error") {
      if (payload.state === "error") throw new Error(payload.error || "Run failed");
      const result = payload.result;
      els.docCount.textContent = String(result.index.doc_count);
      renderDeps(state.selected.dependencies.map(name => ({name})), result.index);
      els.resultMetric.textContent = result.record.passed ? "pass" : "fail";
      els.codeText.textContent = result.generated_code || "";
      els.statusText.textContent = formatRun(result);
      return;
    }
    els.statusText.textContent = formatLiveJob(payload);
    await sleep(900);
  }
}

function renderStages(stages) {
  if (!stages.length) {
    els.stageList.innerHTML = `<div class="stage"><strong>Idle</strong><small>No run is active.</small></div>`;
    return;
  }
  els.stageList.replaceChildren(...stages.map(stage => {
    const div = document.createElement("div");
    const cls = stage.status === "error" ? "error" : stage.status === "complete" ? "complete" : "running";
    div.className = `stage ${cls}`;
    div.innerHTML = `
      <strong>${escapeHTML(stageName(stage.stage))}</strong>
      <small>${escapeHTML(stage.status || "running")}</small>
      <small>${escapeHTML(stageDetail(stage.detail || {}))}</small>
    `;
    return div;
  }));
  els.stageList.scrollTop = els.stageList.scrollHeight;
}

function stageName(name) {
  return String(name || "unknown").replace(/^llm:/, "LLM ").replace(/_/g, " ");
}

function stageDetail(detail) {
  const parts = [];
  if (detail.module) parts.push(`module=${detail.module}`);
  if (detail.task) parts.push(`task=${detail.task}`);
  if (detail.query) parts.push(`query=${detail.query}`);
  if (detail.candidate_count !== undefined) parts.push(`candidates=${detail.candidate_count}`);
  if (detail.doc_count !== undefined) parts.push(`docs=${detail.doc_count}`);
  if (detail.selected_doc_id) parts.push(`selected=${detail.selected_doc_id}`);
  if (detail.action) parts.push(`action=${detail.action}`);
  if (detail.rtl_chars !== undefined) parts.push(`rtl_chars=${detail.rtl_chars}`);
  if (detail.max_repair_attempts !== undefined) parts.push(`retries=${detail.max_repair_attempts}`);
  if (detail.passed !== undefined) parts.push(`passed=${detail.passed}`);
  if (detail.syntax !== undefined) parts.push(`syntax=${detail.syntax}`);
  if (detail.function !== undefined) parts.push(`function=${detail.function}`);
  if (detail.error) parts.push(detail.error);
  return parts.join(" · ");
}

function formatLiveJob(payload) {
  const last = (payload.stages || [])[payload.stages.length - 1];
  return JSON.stringify({
    job_id: payload.job_id,
    state: payload.state,
    max_repair_attempts: payload.max_repair_attempts,
    current_stage: last ? stageName(last.stage) : "queued",
    status: last?.status || "queued",
    detail: last?.detail || {},
  }, null, 2);
}

function formatIndex(index) {
  return JSON.stringify({
    index_dir: index.index_dir,
    doc_count: index.doc_count,
    missing_dependencies: index.missing_dependencies,
    dependency_access: index.dependency_access,
    support_paths: index.support_paths,
  }, null, 2);
}

function formatRun(payload) {
  return JSON.stringify({
    elapsed_s: Number(payload.elapsed_s || 0).toFixed(2),
    generated: payload.record.generated,
    passed: payload.record.passed,
    syntax: payload.record.syntax,
    function: payload.record.function,
    selected_doc_ids: payload.record.selected_doc_ids,
    retrieved_doc_ids: payload.record.retrieved_doc_ids,
    repair_attempts: payload.record.repair_attempts,
    generated_code_path: payload.record.generated_code_path,
    agent_report_path: payload.record.agent_report_path,
    generation_error: payload.record.generation_error,
    evaluation_error: payload.record.evaluation_error,
  }, null, 2);
}

async function copyDemoLink() {
  if (!state.selected) return;
  const url = `${location.origin}/demo?task=${encodeURIComponent(state.selected.task_id)}`;
  await navigator.clipboard.writeText(url);
  els.statusText.textContent = `Copied ${url}`;
}

async function getJSON(url) {
  const res = await fetch(url);
  return await res.json();
}

async function postJSON(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body),
  });
  const payload = await res.json();
  if (!res.ok || !payload.ok) throw new Error(payload.error || "Request failed");
  return payload;
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function escapeHTML(value) {
  return String(value ?? "").replace(/[&<>"']/g, ch => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  }[ch]));
}

init().catch(err => {
  els.statusText.textContent = err.stack || err.message;
});
"""


if __name__ == "__main__":
    main()
