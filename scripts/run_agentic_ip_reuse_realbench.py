#!/usr/bin/env python3
"""Evaluate agentic_ip_reuse on RealBench with dependency-only IP indexes."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ip_reuse_legacy.agent import AgenticIpReuseAgent, AgenticIpReuseConfig, dumps_result
from rag_rtl.embeddings import encode_texts, make_embedder
from rag_rtl.json_utils import dumps_json
from rag_rtl.llm import VllmClient, extract_code
from rag_rtl.retrieval_context import RetrievalContext
from rag_rtl.types import RtlDocument
from rag_rtl.vector_store import VectorStore, build_vector_store
from rag_rtl.verifier import RtlVerifier

MODULE_DECL_RE = re.compile(r"(?m)^\s*module\s+([A-Za-z_][A-Za-z0-9_$]*)\b")
PASS_HINT_RE = re.compile(r"Hint:\s+Output.*no mismatches", re.IGNORECASE)
MISMATCH_HINT_RE = re.compile(r"Hint:\s+Output.*mismatches", re.IGNORECASE)


@dataclass(frozen=True)
class RealBenchTask:
    level: str
    system: str
    task: str
    prompt: str
    dependencies: List[str]
    root_dir: Path

    @property
    def task_id(self) -> str:
        return safe_name(f"{self.level}__{self.system}__{self.task}")

    @property
    def top_module(self) -> str:
        return self.task


@dataclass(frozen=True)
class WorkItem:
    task: RealBenchTask
    sample: int


@dataclass
class SourceDoc:
    doc_id: str
    name: str
    path: Path
    kind: str


@dataclass
class IndexBundle:
    index_dir: Path
    documents: List[RtlDocument]
    declared_doc_ids: Dict[str, str]
    dependency_paths: Dict[str, str]
    support_paths: Dict[str, str]
    missing_dependencies: List[str] = field(default_factory=list)
    access: Dict[str, bool] = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        return not self.missing_dependencies


@dataclass
class RealBenchEvalResult:
    syntax: int
    function: int
    syntax_info: str = ""
    function_info: str = ""
    compile_returncode: Optional[int] = None
    run_returncode: Optional[int] = None
    stdout_tail: str = ""
    stderr_tail: str = ""
    error: Optional[str] = None

    @property
    def passed(self) -> bool:
        return self.syntax == 1 and self.function == 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate agentic_ip_reuse on RealBench with per-task dependency-only IP indexes."
    )
    parser.add_argument("--benchmark", choices=["realbench", "rtl-mosaic"], default="realbench")
    parser.add_argument("--realbench-root", default="/home/kai/eval_dt/real_bench")
    parser.add_argument("--rtl-mosaic-root", default="/home/kai/eval_dt/rtl-mosaic")
    parser.add_argument("--chipbench-root", default="/home/kai/eval_dt/ChipBench/Verilog Gen")
    parser.add_argument("--rtl-mosaic-engine", choices=["agentic", "harness"], default="agentic")
    parser.add_argument(
        "--realbench-verifier",
        choices=["native", "harness"],
        default="native",
        help=(
            "RealBench testbench evaluator. native copies the verification directory directly; "
            "harness delegates to RealBench run_verify.py."
        ),
    )
    parser.add_argument("--output-dir", default="runs/agentic_ip_reuse_realbench")
    parser.add_argument("--solution-name", default="agentic_ip_reuse")
    parser.add_argument("--task-level", choices=["module", "system", "both"], default="both")
    parser.add_argument("--include", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--prepare-problems",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Decrypt RealBench markdown and generate problems.jsonl when needed.",
    )

    parser.add_argument("--embedder", default="hash")
    parser.add_argument("--retrieve-k", type=int, default=8)
    parser.add_argument("--context-k", type=int, default=4)
    parser.add_argument("--index-jobs", type=int, default=1)

    parser.add_argument("--base-url", help="OpenAI-compatible serving base URL. Overrides VLLM_BASE_URL.")
    parser.add_argument("--model", help="Served model name. Defaults to VLLM_MODEL.")
    parser.add_argument("--api-key", help="API key. Defaults to VLLM_API_KEY or EMPTY.")
    parser.add_argument("--llm-timeout-s", type=int, default=1200)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=30000)

    parser.add_argument("--max-repair-attempts", type=int, default=2)
    parser.add_argument("--max-generation-retries", type=int, default=2)
    parser.add_argument("--large-spec-threshold-chars", type=int, default=40000)
    parser.add_argument("--large-spec-chunk-chars", type=int, default=30000)
    parser.add_argument("--decomposition-mode", choices=["original", "chunking"], default="original")
    parser.add_argument("--recursive-decomposition", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--recursive-max-depth", type=int, default=4)
    parser.add_argument("--recursive-max-nodes", type=int, default=64)
    parser.add_argument("--yosys-bin", default="yosys")
    parser.add_argument("--verilator-bin", default="verilator")
    parser.add_argument("--agent-timeout-s", type=int, default=30)
    parser.add_argument("--verification-timeout-s", type=int, default=300)
    parser.add_argument("--make-bin", default="make")
    return parser


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def load_python_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_benchmark_info(realbench_root: str | Path) -> Tuple[Dict[str, Dict[str, List[str]]], Dict[str, List[str]]]:
    root = Path(realbench_root)
    module = load_python_module(root / "benchmark_info.py", f"realbench_info_{id(root)}")
    return module.benchmark_info, module.system_info


def selected_levels(task_level: str) -> List[str]:
    return ["module", "system"] if task_level == "both" else [task_level]


def ensure_problem_files(realbench_root: str | Path, levels: Sequence[str], prepare: bool) -> None:
    root = Path(realbench_root)
    missing = []
    if "module" in levels:
        benchmark_info, _ = load_benchmark_info(root)
        for system in benchmark_info:
            path = root / "problems" / system / "problems.jsonl"
            if not path.exists():
                missing.append(path)
    if "system" in levels and not (root / "problems" / "system" / "problems.jsonl").exists():
        missing.append(root / "problems" / "system" / "problems.jsonl")

    if not missing:
        return
    if not prepare:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"RealBench problem files are missing:\n{missing_text}")

    run_checked([shutil.which("make") or "make", "decrypt"], cwd=root)
    for level in levels:
        run_checked([sys.executable, "generate_problem.py", "--task_level", level], cwd=root)


def run_checked(command: List[str], cwd: Path) -> None:
    completed = subprocess.run(command, cwd=cwd, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed in {cwd}: {' '.join(command)}\n"
            f"stdout:\n{completed.stdout[-2000:]}\n"
            f"stderr:\n{completed.stderr[-2000:]}"
        )


def read_problem_jsonl(path: Path) -> Dict[str, str]:
    records: Dict[str, str] = {}
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


def discover_tasks(args: argparse.Namespace) -> List[RealBenchTask]:
    root = Path(args.realbench_root)
    levels = selected_levels(args.task_level)
    ensure_problem_files(root, levels, args.prepare_problems)
    benchmark_info, system_info = load_benchmark_info(root)
    filters = [item.lower() for item in args.include if item.strip()]
    tasks: List[RealBenchTask] = []

    if "module" in levels:
        for system, modules in benchmark_info.items():
            prompts = read_problem_jsonl(root / "problems" / system / "problems.jsonl")
            for module_name, dependencies in modules.items():
                prompt = prompts.get(module_name)
                if prompt is None:
                    continue
                task = RealBenchTask(
                    level="module",
                    system=system,
                    task=module_name,
                    prompt=prompt,
                    dependencies=[item for item in dependencies if item != module_name],
                    root_dir=root,
                )
                if include_task(task, filters):
                    tasks.append(task)

    if "system" in levels:
        prompts = read_problem_jsonl(root / "problems" / "system" / "problems.jsonl")
        for system_name, dependencies in system_info.items():
            prompt = prompts.get(system_name)
            if prompt is None:
                continue
            task = RealBenchTask(
                level="system",
                system=system_family(system_name),
                task=system_name,
                prompt=prompt,
                dependencies=[item for item in dependencies if item != system_name],
                root_dir=root,
            )
            if include_task(task, filters):
                tasks.append(task)

    if args.limit is not None:
        tasks = tasks[: args.limit]
    return tasks


def include_task(task: RealBenchTask, filters: Sequence[str]) -> bool:
    if not filters:
        return True
    haystack = f"{task.level} {task.system} {task.task} {task.task_id}".lower()
    return any(item in haystack for item in filters)


def system_family(system_or_task: str) -> str:
    if system_or_task.startswith("sd") or system_or_task == "sdc_controller":
        return "sdc"
    if system_or_task.startswith("aes"):
        return "aes"
    if system_or_task.startswith("e203"):
        return "e203_hbirdv2"
    raise ValueError(f"cannot infer RealBench system family for {system_or_task!r}")


def build_task_index(task: RealBenchTask, args: argparse.Namespace, output_dir: Path) -> IndexBundle:
    index_dir = output_dir / "indexes" / task.task_id
    declared_docs: List[SourceDoc] = []
    missing: List[str] = []
    seen_paths: set[Path] = set()

    for dependency in task.dependencies:
        source = resolve_dependency_source(task, dependency)
        if source is None:
            missing.append(dependency)
            continue
        if source.path not in seen_paths:
            declared_docs.append(source)
            seen_paths.add(source.path)

    support_docs = [
        source for source in resolve_support_sources(task) if source.path not in seen_paths
    ]
    documents = [source_doc_to_rtl(task, source) for source in [*declared_docs, *support_docs]]
    embedder = make_embedder(args.embedder)
    vectors = encode_texts(embedder, [document.retrieval_text for document in documents], jobs=args.index_jobs)
    store = build_vector_store(documents, vectors)
    store.save(index_dir)

    bundle = IndexBundle(
        index_dir=index_dir,
        documents=documents,
        declared_doc_ids={source.name: source.doc_id for source in declared_docs},
        dependency_paths={source.name: str(source.path) for source in declared_docs},
        support_paths={source.name: str(source.path) for source in support_docs},
        missing_dependencies=missing,
    )
    bundle.access = preflight_dependency_access(bundle, args)
    return bundle


def source_doc_to_rtl(task: RealBenchTask, source: SourceDoc) -> RtlDocument:
    code = source.path.read_text(encoding="utf-8", errors="ignore")
    return RtlDocument(
        doc_id=source.doc_id,
        problem=(
            f"RealBench reusable {source.kind} for task {task.task}. "
            f"Module or file name: {source.name}. Source path: {source.path}"
        ),
        solution=code,
        tags=["realbench", task.level, task.system, source.kind, source.name],
        metadata={
            "source": "realbench",
            "task_level": task.level,
            "task": task.task,
            "system": task.system,
            "dependency": source.name,
            "source_path": str(source.path),
            "license": "benchmark supplied",
            "verification_status": "supplied with RealBench verification collateral",
            "synthesis_support": "unknown",
            "documentation_quality": "unknown",
        },
    )


def resolve_dependency_source(task: RealBenchTask, dependency: str) -> Optional[SourceDoc]:
    for path in dependency_source_candidates(task, dependency):
        if path.exists() and path.is_file():
            return SourceDoc(
                doc_id=f"dep:{dependency}",
                name=dependency,
                path=path,
                kind="dependency",
            )
    scanned = scan_for_module(task_verification_dir(task), dependency)
    if scanned is not None:
        return SourceDoc(f"dep:{dependency}", dependency, scanned, "dependency")
    return None


def dependency_source_candidates(task: RealBenchTask, dependency: str) -> Iterable[Path]:
    verification_dir = task_verification_dir(task)
    for suffix in (".v", ".sv"):
        yield verification_dir / f"{dependency}{suffix}"

    family = task.system
    for suffix in (".v", ".sv"):
        yield task.root_dir / family / dependency / f"{dependency}{suffix}"

    if task.level == "system":
        system_dir = task.root_dir / "system" / task.task
        for suffix in (".v", ".sv"):
            yield system_dir / f"{dependency}{suffix}"

    for suffix in (".v", ".sv"):
        yield task.root_dir / "sdc" / dependency / f"{dependency}{suffix}"
        yield task.root_dir / "aes" / dependency / f"{dependency}{suffix}"
        yield task.root_dir / "e203_hbirdv2" / dependency / f"{dependency}{suffix}"


def task_verification_dir(task: RealBenchTask) -> Path:
    if task.level == "system":
        return task.root_dir / "system" / task.task
    return task.root_dir / task.system / task.task / "verification"


def scan_for_module(directory: Path, module_name: str) -> Optional[Path]:
    if not directory.exists():
        return None
    for path in sorted([*directory.glob("*.v"), *directory.glob("*.sv")]):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if module_name in MODULE_DECL_RE.findall(text):
            return path
    return None


def resolve_support_sources(task: RealBenchTask) -> List[SourceDoc]:
    sources: List[SourceDoc] = []
    if task.system == "sdc":
        path = task.root_dir / "sdc" / "sd_defines" / "sd_defines.v"
        if path.exists():
            sources.append(SourceDoc("support:sd_defines", "sd_defines", path, "support"))
    if task.system == "e203_hbirdv2":
        for name, path in {
            "e203_defines": task.root_dir / "e203_hbirdv2" / "e203_defines" / "e203_defines.v",
            "config": task.root_dir / "e203_hbirdv2" / "config" / "config.v",
        }.items():
            if path.exists():
                sources.append(SourceDoc(f"support:{name}", name, path, "support"))
    return sources


def preflight_dependency_access(bundle: IndexBundle, args: argparse.Namespace) -> Dict[str, bool]:
    if not bundle.declared_doc_ids:
        return {}
    store = VectorStore.load(bundle.index_dir)
    context = RetrievalContext.from_store(store, make_embedder(args.embedder))
    top_k = max(args.retrieve_k, len(bundle.declared_doc_ids), 1)
    access: Dict[str, bool] = {}
    for dependency, doc_id in bundle.declared_doc_ids.items():
        hits = context.prepare(query=dependency, retrieve_k=top_k, context_k=top_k)
        access[dependency] = doc_id in {hit.document.doc_id for hit in hits}
    return access


def build_agent(
    args: argparse.Namespace,
    index_dir: Path,
    stage_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    task: Optional[Any] = None,
) -> AgenticIpReuseAgent:
    embedder = make_embedder(args.embedder)
    store = VectorStore.load(index_dir)
    retrieval_context = RetrievalContext.from_store(store, embedder)
    verifier = RtlVerifier(
        yosys_bin=args.yosys_bin,
        verilator_bin=args.verilator_bin,
        timeout_s=args.agent_timeout_s,
    )
    llm = VllmClient(
        base_url=args.base_url or os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
        model=args.model or os.getenv("VLLM_MODEL", "siliconmind-server"),
        api_key=args.api_key or os.getenv("VLLM_API_KEY", "EMPTY"),
        timeout_s=args.llm_timeout_s,
    )
    testbench_dir = task_verification_dir(task) if task is not None else None
    return AgenticIpReuseAgent(
        llm,
        retrieval_context,
        verifier,
        AgenticIpReuseConfig(
            target_hdl="verilog",
            retrieve_k=args.retrieve_k,
            context_k=args.context_k,
            max_repair_attempts=args.max_repair_attempts,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            large_spec_threshold_chars=args.large_spec_threshold_chars,
            large_spec_chunk_chars=args.large_spec_chunk_chars,
            decomposition_mode=args.decomposition_mode,
            recursive_decomposition=args.recursive_decomposition,
            recursive_max_depth=args.recursive_max_depth,
            recursive_max_nodes=args.recursive_max_nodes,
            testbench_dir=testbench_dir,
            max_generation_retries=getattr(args, "max_generation_retries", 2),
        ),
        stage_callback=stage_callback,
    )


def work_items(tasks: Sequence[RealBenchTask], samples: int) -> List[WorkItem]:
    return [WorkItem(task=task, sample=sample) for task in tasks for sample in range(1, samples + 1)]


def generated_code_path(output_dir: Path, item: WorkItem) -> Path:
    return output_dir / "generated" / item.task.level / item.task.system / item.task.task / f"sample{item.sample:02d}.sv"


def report_path(output_dir: Path, item: WorkItem) -> Path:
    return output_dir / "reports" / item.task.level / item.task.system / f"{item.task.task}_sample{item.sample:02d}.json"


def agent_workspace_path(output_dir: Path, item: WorkItem) -> Path:
    return (
        output_dir
        / "workspaces"
        / (safe_name(item.task.level) or "unknown_level")
        / (safe_name(item.task.system) or "unknown_system")
        / (safe_name(item.task.task) or "unknown_task")
        / f"sample{item.sample:02d}"
    )


def collect_workspace_artifacts(workspace: Path) -> Dict[str, str]:
    if not workspace.exists():
        return {}
    return {
        str(path.relative_to(workspace)): str(path)
        for path in sorted(workspace.rglob("*"))
        if path.is_file()
    }


def run_one(
    item: WorkItem,
    args: argparse.Namespace,
    output_dir: Path,
    indexes: Dict[str, IndexBundle],
    stage_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    task = item.task
    bundle = indexes[task.task_id]
    code_path = generated_code_path(output_dir, item)
    code_path.parent.mkdir(parents=True, exist_ok=True)
    report = report_path(output_dir, item)
    report.parent.mkdir(parents=True, exist_ok=True)

    code = ""
    generation_error: Optional[str] = None
    agent_report_written = False
    reused_existing = False
    wall_s = 0.0
    selected_doc_ids: List[str] = []
    retrieved_doc_ids: List[str] = []
    repair_attempts: Optional[int] = None
    workspace = agent_workspace_path(output_dir, item)
    artifacts: Dict[str, str] = {}
    module_generation: List[Dict[str, Any]] = []
    large_spec_manifest: Optional[Dict[str, Any]] = None
    decomposition_tree: Optional[Dict[str, Any]] = None

    if not bundle.usable:
        generation_error = f"missing dependency source(s): {', '.join(bundle.missing_dependencies)}"
    elif (args.resume or args.evaluate_only) and code_path.exists():
        code = code_path.read_text(encoding="utf-8")
        reused_existing = True
    elif args.evaluate_only:
        generation_error = f"missing generated file: {code_path}"
    else:
        t0 = time.perf_counter()
        try:
            agent = build_agent(args, bundle.index_dir, stage_callback=stage_callback, task=task)
            result = agent.run(
                task.prompt,
                target_hdl="verilog",
                top_module=task.top_module,
                constraints=task_constraints(task),
                workspace_dir=workspace,
            )
            code = normalize_code(result.rtl, task.top_module)
            selected_doc_ids = [
                decision.selected_doc_id
                for decision in result.plan.decisions
                if decision.selected_doc_id
            ]
            retrieved_doc_ids = sorted(
                {
                    doc_id
                    for trace in result.retrieval_traces
                    for doc_id in trace.get("doc_ids", [])
                }
            )
            repair_attempts = result.repair_attempts
            artifacts = dict(result.artifacts)
            module_generation = list(result.module_generation)
            large_spec_manifest = result.large_spec_manifest
            decomposition_tree = result.decomposition_tree
            report.write_text(dumps_result(result), encoding="utf-8")
            agent_report_written = True
        except Exception as exc:  # noqa: BLE001 - keep the benchmark moving.
            generation_error = f"{exc}\n{traceback.format_exc()[-4000:]}"
        wall_s = time.perf_counter() - t0
        if code:
            code_path.write_text(code, encoding="utf-8")

    if code:
        if stage_callback:
            stage_callback({"stage": "realbench_verification", "status": "running", "task": task.task})
        eval_result = evaluate_realbench_code(task, code, args)
        if stage_callback:
            stage_callback(
                {
                    "stage": "realbench_verification",
                    "status": "complete",
                    "task": task.task,
                    "syntax": eval_result.syntax,
                    "function": eval_result.function,
                    "passed": eval_result.passed,
                }
            )
    else:
        eval_result = RealBenchEvalResult(
            syntax=0,
            function=0,
            error=generation_error or "generation produced empty code",
        )

    artifacts = {**collect_workspace_artifacts(workspace), **artifacts}
    return {
        "benchmark": "realbench",
        "task_level": task.level,
        "system": task.system,
        "task": task.task,
        "realbench_verifier": getattr(args, "realbench_verifier", "native"),
        "decomposition_mode": getattr(args, "decomposition_mode", "original"),
        "sample": item.sample,
        "top_module": task.top_module,
        "dependencies": task.dependencies,
        "dependency_paths": bundle.dependency_paths,
        "support_paths": bundle.support_paths,
        "missing_dependencies": bundle.missing_dependencies,
        "ip_db_index": str(bundle.index_dir),
        "ip_db_doc_count": len(bundle.documents),
        "dependency_access": bundle.access,
        "all_dependencies_retrievable": all(bundle.access.values()) if bundle.access else True,
        "selected_doc_ids": selected_doc_ids,
        "retrieved_doc_ids": retrieved_doc_ids,
        "generated": bool(code),
        "reused_existing": reused_existing,
        "generation_error": generation_error,
        "generated_code_path": str(code_path) if code else None,
        "agent_report_path": str(report) if agent_report_written else None,
        "agent_workspace_path": str(workspace),
        "agent_artifacts": artifacts,
        "module_generation": module_generation,
        "large_spec_manifest": large_spec_manifest,
        "decomposition_tree": decomposition_tree,
        "repair_attempts": repair_attempts,
        "syntax": eval_result.syntax,
        "function": eval_result.function,
        "passed": eval_result.passed,
        "syntax_info": eval_result.syntax_info,
        "function_info": eval_result.function_info,
        "compile_returncode": eval_result.compile_returncode,
        "run_returncode": eval_result.run_returncode,
        "stdout_tail": eval_result.stdout_tail,
        "stderr_tail": eval_result.stderr_tail,
        "evaluation_error": eval_result.error,
        "wall_s": wall_s,
    }


def task_constraints(task: RealBenchTask) -> List[str]:
    return [
        f"Return a complete Verilog implementation for module {task.top_module}.",
        "Use the exact public module name and port/interface described by the RealBench problem.",
        "Do not include a testbench, reference module, markdown fences, or explanatory text.",
        "Reuse the indexed dependency IP where it matches the requested behavior; create glue logic as needed.",
    ]


def normalize_code(code: str, top_module: str) -> str:
    extracted = extract_code(code).strip()
    if not extracted:
        return ""
    names = MODULE_DECL_RE.findall(extracted)
    if names and top_module not in names:
        pattern = re.compile(rf"(?m)^(\s*module\s+){re.escape(names[0])}\b")
        extracted = pattern.sub(lambda match: f"{match.group(1)}{top_module}", extracted, count=1)
    return extracted


def evaluate_realbench_code(task: RealBenchTask, code: str, args: argparse.Namespace) -> RealBenchEvalResult:
    if getattr(args, "realbench_verifier", "native") == "harness":
        return evaluate_realbench_code_with_harness(task, code, args)
    return evaluate_realbench_code_native(task, code, args)


def evaluate_realbench_code_native(task: RealBenchTask, code: str, args: argparse.Namespace) -> RealBenchEvalResult:
    template_dir = task_verification_dir(task)
    if not template_dir.exists():
        return RealBenchEvalResult(0, 0, error=f"verification directory not found: {template_dir}")
    top_filename = f"{task.task}_top.sv"
    try:
        with tempfile.TemporaryDirectory(prefix=f"realbench_{task.task}_") as temp_name:
            temp_dir = Path(temp_name)
            copy_verification_template(template_dir, temp_dir)
            top_path = temp_dir / top_filename
            if top_path.exists():
                top_path.unlink()
            top_path.write_text(code, encoding="utf-8")

            t0 = time.perf_counter()
            completed = subprocess.run(
                [args.make_bin, "all"],
                cwd=temp_dir,
                check=False,
                capture_output=True,
                text=True,
                timeout=args.verification_timeout_s,
            )
            elapsed = time.perf_counter() - t0
    except subprocess.TimeoutExpired as exc:
        return RealBenchEvalResult(0, 0, error=f"verification timeout: {exc}")
    except Exception as exc:  # noqa: BLE001
        return RealBenchEvalResult(0, 0, error=f"verification failed: {exc}")

    stderr = completed.stderr or ""
    stdout = completed.stdout or ""
    syntax_info = realbench_syntax_errors(stderr)
    syntax = 0 if completed.returncode != 0 or syntax_info else 1
    function_info = realbench_function_errors(stdout)
    function = 1 if syntax == 1 and not function_info else 0
    error = None if syntax == 1 else f"make all failed in {elapsed:.2f}s"
    return RealBenchEvalResult(
        syntax=syntax,
        function=function,
        syntax_info=syntax_info,
        function_info=function_info,
        compile_returncode=completed.returncode,
        run_returncode=completed.returncode,
        stdout_tail=stdout[-4000:],
        stderr_tail=stderr[-4000:],
        error=error,
    )


def evaluate_realbench_code_with_harness(
    task: RealBenchTask,
    code: str,
    args: argparse.Namespace,
) -> RealBenchEvalResult:
    root = Path(args.realbench_root)
    harness = root / "run_verify.py"
    if not harness.exists():
        return RealBenchEvalResult(0, 0, error=f"RealBench harness not found: {harness}")

    runner = (
        "import json, sys\n"
        "from pathlib import Path\n"
        "import run_verify\n"
        "level, system_name, task_name, code_path = sys.argv[1:5]\n"
        "code = Path(code_path).read_text(encoding='utf-8')\n"
        "if level == 'system':\n"
        "    result = run_verify.testbench_verification_system(code, task_name)\n"
        "else:\n"
        "    result = run_verify.testbench_verification(code, system_name, task_name)\n"
        "print(json.dumps({\n"
        "    'syntax': result[0],\n"
        "    'function': result[1],\n"
        "    'syntax_info': result[2],\n"
        "    'function_info': result[3],\n"
        "}))\n"
    )

    try:
        with tempfile.TemporaryDirectory(prefix=f"realbench_harness_{task.task}_") as temp_name:
            code_path = Path(temp_name) / f"{task.task}_top.sv"
            code_path.write_text(code, encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    runner,
                    task.level,
                    task.system,
                    task.task,
                    str(code_path),
                ],
                cwd=root,
                check=False,
                capture_output=True,
                text=True,
                timeout=args.verification_timeout_s + 30,
            )
    except subprocess.TimeoutExpired as exc:
        return RealBenchEvalResult(0, 0, error=f"RealBench harness timeout: {exc}")
    except Exception as exc:  # noqa: BLE001
        return RealBenchEvalResult(0, 0, error=f"RealBench harness failed: {exc}")

    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    payload = parse_harness_result(stdout)
    if completed.returncode != 0 or payload is None:
        return RealBenchEvalResult(
            0,
            0,
            compile_returncode=completed.returncode,
            run_returncode=completed.returncode,
            stdout_tail=stdout[-4000:],
            stderr_tail=stderr[-4000:],
            error=f"RealBench harness failed with return code {completed.returncode}",
        )

    syntax = int(payload.get("syntax") == 1)
    function = int(syntax == 1 and payload.get("function") == 1)
    return RealBenchEvalResult(
        syntax=syntax,
        function=function,
        syntax_info=str(payload.get("syntax_info") or ""),
        function_info=str(payload.get("function_info") or ""),
        compile_returncode=completed.returncode,
        run_returncode=completed.returncode,
        stdout_tail=stdout[-4000:],
        stderr_tail=stderr[-4000:],
        error=None if syntax == 1 else "RealBench harness reported syntax failure",
    )


def parse_harness_result(stdout: str) -> Optional[Dict[str, Any]]:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if {"syntax", "function", "syntax_info", "function_info"} <= set(payload):
            return payload
    return None


def copy_verification_template(source: Path, destination: Path) -> None:
    for child in source.iterdir():
        target = destination / child.name
        if child.is_dir():
            shutil.copytree(child, target)
        else:
            shutil.copy2(child, target)


def realbench_syntax_errors(stderr: str) -> str:
    lines = []
    for line in stderr.splitlines():
        if line.startswith("%Error") or line.startswith("%Warning"):
            lines.append(line)
    return "\n".join(lines)


def realbench_function_errors(stdout: str) -> str:
    lines = []
    for line in stdout.splitlines():
        if PASS_HINT_RE.search(line):
            continue
        if MISMATCH_HINT_RE.search(line):
            lines.append(line.removeprefix("Hint: ").strip())
    return "\n".join(lines)


def write_records(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in sorted(records, key=record_sort_key):
            handle.write(dumps_json(record) + "\n")


def write_solution_jsonl(output_dir: Path, solution_name: str, records: Sequence[Dict[str, Any]]) -> None:
    rows: Dict[str, List[Dict[str, Any]]] = {"aes": [], "e203_hbirdv2": [], "sdc": [], "system": []}
    for record in sorted(records, key=record_sort_key):
        code_path = record.get("generated_code_path")
        if not code_path or not Path(code_path).exists():
            continue
        key = "system" if record["task_level"] == "system" else record["system"]
        rows.setdefault(key, []).append(
            {
                "task": record["task"],
                "codeid": int(record["sample"]),
                "code": Path(code_path).read_text(encoding="utf-8"),
            }
        )

    sample_dir = output_dir / "samples" / solution_name
    sample_dir.mkdir(parents=True, exist_ok=True)
    for key, key_rows in rows.items():
        with (sample_dir / f"{key}.jsonl").open("w", encoding="utf-8") as handle:
            for row in key_rows:
                handle.write(dumps_json(row) + "\n")


def summarize(records: Sequence[Dict[str, Any]], tasks: Sequence[RealBenchTask], elapsed_s: float) -> Dict[str, Any]:
    groups: Dict[str, Dict[str, Any]] = {}
    for record in records:
        for key in (
            f"level:{record['task_level']}",
            f"system:{record['system']}",
            f"{record['task_level']}:{record['system']}",
        ):
            item = groups.setdefault(key, {"total": 0, "generated": 0, "syntax": 0, "function": 0, "passed": 0})
            item["total"] += 1
            item["generated"] += int(bool(record.get("generated")))
            item["syntax"] += int(record.get("syntax") == 1)
            item["function"] += int(record.get("function") == 1)
            item["passed"] += int(bool(record.get("passed")))
    for item in groups.values():
        total = item["total"] or 1
        item["syntax_rate"] = item["syntax"] / total
        item["function_rate"] = item["function"] / total
        item["pass_rate"] = item["passed"] / total

    total_records = len(records)
    return {
        "benchmark": "realbench",
        "num_tasks": len(tasks),
        "num_records": total_records,
        "samples_per_task": max((int(record["sample"]) for record in records), default=0),
        "generated": sum(1 for record in records if record.get("generated")),
        "syntax": sum(1 for record in records if record.get("syntax") == 1),
        "function": sum(1 for record in records if record.get("function") == 1),
        "passed": sum(1 for record in records if record.get("passed")),
        "syntax_rate": safe_rate(sum(1 for record in records if record.get("syntax") == 1), total_records),
        "function_rate": safe_rate(sum(1 for record in records if record.get("function") == 1), total_records),
        "pass_rate": safe_rate(sum(1 for record in records if record.get("passed")), total_records),
        "groups": groups,
        "total_s": elapsed_s,
    }


def safe_rate(numerator: int, denominator: int) -> Optional[float]:
    return None if denominator == 0 else numerator / denominator


def record_sort_key(record: Dict[str, Any]) -> Tuple[str, str, str, int]:
    return (
        str(record.get("task_level") or ""),
        str(record.get("system") or ""),
        str(record.get("task") or ""),
        int(record.get("sample") or 0),
    )


def run_realbench(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    tasks = discover_tasks(args)
    print(f"[realbench] discovered {len(tasks)} task(s)")

    indexes: Dict[str, IndexBundle] = {}
    for task in tasks:
        bundle = build_task_index(task, args, output_dir)
        indexes[task.task_id] = bundle
        print(
            f"[realbench] index {task.task_id}: docs={len(bundle.documents)} "
            f"deps={len(bundle.declared_doc_ids)} missing={bundle.missing_dependencies}"
        )

    items = work_items(tasks, args.samples)
    if args.dry_run:
        records = [dry_run_record(item, indexes[item.task.task_id], output_dir, args) for item in items]
    else:
        records = []
        records_lock = threading.Lock()
        with ThreadPoolExecutor(max_workers=max(args.concurrency, 1)) as executor:
            futures = [executor.submit(run_one, item, args, output_dir, indexes) for item in items]
            for future in as_completed(futures):
                record = future.result()
                with records_lock:
                    records.append(record)
                status = "PASS" if record["passed"] else "FAIL"
                print(
                    f"[realbench] {status} {record['task_level']}/{record['system']}/{record['task']} "
                    f"sample {int(record['sample']):02d} syntax={record['syntax']} function={record['function']}"
                )

    elapsed_s = time.perf_counter() - start
    write_records(output_dir / "records.jsonl", records)
    write_solution_jsonl(output_dir, args.solution_name, records)
    summary = summarize(records, tasks, elapsed_s)
    if args.dry_run:
        summary["dry_run"] = True
    (output_dir / "summary.json").write_text(dumps_json(summary, indent=2), encoding="utf-8")
    print(f"[realbench] wrote results under {output_dir}")
    return summary


def dry_run_record(
    item: WorkItem,
    bundle: IndexBundle,
    output_dir: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    task = item.task
    return {
        "benchmark": "realbench",
        "task_level": task.level,
        "system": task.system,
        "task": task.task,
        "realbench_verifier": getattr(args, "realbench_verifier", "native"),
        "sample": item.sample,
        "top_module": task.top_module,
        "dependencies": task.dependencies,
        "dependency_paths": bundle.dependency_paths,
        "support_paths": bundle.support_paths,
        "missing_dependencies": bundle.missing_dependencies,
        "ip_db_index": str(bundle.index_dir),
        "ip_db_doc_count": len(bundle.documents),
        "dependency_access": bundle.access,
        "all_dependencies_retrievable": all(bundle.access.values()) if bundle.access else True,
        "generated": False,
        "reused_existing": False,
        "generation_error": "dry run",
        "generated_code_path": None,
        "agent_report_path": None,
        "agent_workspace_path": str(agent_workspace_path(output_dir, item)),
        "agent_artifacts": {},
        "module_generation": [],
        "large_spec_manifest": None,
        "decomposition_tree": None,
        "repair_attempts": None,
        "syntax": 0,
        "function": 0,
        "passed": False,
        "evaluation_error": "dry run",
        "wall_s": 0.0,
    }


def run_rtl_mosaic(args: argparse.Namespace) -> Dict[str, Any]:
    root = Path(args.rtl_mosaic_root)
    script = root / "eval" / "run_chipbench_eval.py"
    if not script.exists():
        raise FileNotFoundError(f"rtl-mosaic ChipBench evaluator not found: {script}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rtl_output_dir = output_dir / "rtl_mosaic"
    command = [
        sys.executable,
        str(script),
        "--chipbench-root",
        str(Path(args.chipbench_root)),
        "--engine",
        args.rtl_mosaic_engine,
        "--output-dir",
        str(rtl_output_dir),
        "--workers",
        str(max(args.concurrency, 1)),
    ]
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    for item in args.include:
        command.extend(["--include", item])
    if args.resume:
        command.append("--resume")
    if args.dry_run:
        command.append("--dry-run")
    command.extend(["--veri-thesis-root", str(REPO_ROOT)])
    command.extend(["--embedder", args.embedder])
    command.extend(["--retrieve-k", str(args.retrieve_k)])
    command.extend(["--context-k", str(args.context_k)])
    command.extend(["--max-repair-attempts", str(args.max_repair_attempts)])
    command.extend(["--index-jobs", str(args.index_jobs)])
    if args.base_url:
        command.extend(["--base-url", args.base_url])
    if args.model:
        command.extend(["--model", args.model])
    if args.api_key:
        command.extend(["--api-key", args.api_key])
    if args.llm_timeout_s:
        command.extend(["--llm-timeout-s", str(args.llm_timeout_s)])
    command.extend(["--temperature", str(args.temperature)])
    command.extend(["--max-tokens", str(args.max_tokens)])

    print(f"[rtl-mosaic] running: {' '.join(command)}")
    completed = subprocess.run(command, cwd=root, check=False, capture_output=True, text=True)
    payload = {
        "benchmark": "rtl-mosaic",
        "root": str(root),
        "output_dir": str(rtl_output_dir),
        "command": command,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-8000:],
        "stderr_tail": completed.stderr[-8000:],
    }
    summary_path = rtl_output_dir / "summary.json"
    if summary_path.exists():
        payload["summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
    if completed.returncode != 0:
        (output_dir / "rtl_mosaic_error.json").write_text(dumps_json(payload, indent=2), encoding="utf-8")
        raise RuntimeError(
            f"rtl-mosaic evaluation failed with return code {completed.returncode}\n"
            f"stdout:\n{completed.stdout[-2000:]}\n"
            f"stderr:\n{completed.stderr[-2000:]}"
        )
    (output_dir / "rtl_mosaic_summary.json").write_text(dumps_json(payload, indent=2), encoding="utf-8")
    print(f"[rtl-mosaic] wrote results under {rtl_output_dir}")
    return payload


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.benchmark == "rtl-mosaic":
        payload = run_rtl_mosaic(args)
        print(dumps_json(payload.get("summary", payload), indent=2))
        return
    summary = run_realbench(args)
    print(dumps_json({key: value for key, value in summary.items() if key != "groups"}, indent=2))


if __name__ == "__main__":
    main()
