#!/usr/bin/env python3
"""Two-stage RealBench runner: agentic_ip_reuse plan -> ip_reuse_legacy RTL."""

from __future__ import annotations

import argparse
import hashlib
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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
PLANNER_ROOT = REPO_ROOT / "agentic_ip_reuse"
for path in (str(PLANNER_ROOT), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from agentic_ip_reuse.agent import AgentConfig, AgenticIpReuseAgent as PlanningAgent, dumps_result as dumps_plan_result
from agentic_ip_reuse.llm import VllmClient as PlanningVllmClient
from agentic_ip_reuse.repository import JsonIpRepository
from agentic_ip_reuse.semantic_repository import SemanticIpRepository
from agentic_ip_reuse.tools import AgentToolExecutor
from agentic_ip_reuse.types import AgentResult as PlanningResult, DesignTask
from ip_reuse_legacy.agent import AgenticIpReuseAgent as LegacyAgent
from ip_reuse_legacy.config import AgenticIpReuseConfig as LegacyConfig
from ip_reuse_legacy.plan_adapter import agentic_plan_from_payload
from ip_reuse_legacy.types import IpReusePlan
from rag_rtl.embeddings import HashingEmbedder, make_embedder_with_fallback
from rag_rtl.json_utils import dumps_json
from rag_rtl.llm import VllmClient as LegacyVllmClient, extract_code
from rag_rtl.repair_cache import RepairFixCache
from rag_rtl.retrieval import LexicalReranker, Retriever
from rag_rtl.retrieval_context import RetrievalContext
from rag_rtl.types import Diagnostic, RtlDocument, VerificationReport
from rag_rtl.vector_store import VectorStore

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
    ip_id: str
    name: str
    path: Path
    kind: str


@dataclass
class CatalogBundle:
    catalog_path: Path
    sources: List[SourceDoc]
    source_by_key: Dict[str, SourceDoc]
    dependency_paths: Dict[str, str]
    support_paths: Dict[str, str]
    missing_dependencies: List[str] = field(default_factory=list)

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


class NoopVerifier:
    def verify(self, rtl: str, top_module: str | None = None) -> VerificationReport:
        return VerificationReport(
            syntax_passed=True,
            lint_passed=True,
            diagnostics=[Diagnostic(tool="skipped", passed=True, stderr="legacy internal verification skipped")],
        )


class RealBenchWorkspaceVerifier:
    """Lint candidate RTL together with the dependency/defines files that the
    RealBench Makefile compiles, using the Makefile's warning suppressions, so the
    legacy repair loop sees the same MODDUP/port errors the final evaluation would."""

    WNO_FLAGS = [
        "-Wno-CASEOVERLAP",
        "-Wno-LATCH",
        "-Wno-UNOPTFLAT",
        "-Wno-MULTIDRIVEN",
        "-Wno-ASCRANGE",
        "-Wno-IMPLICIT",
        "-Wno-CASEINCOMPLETE",
        "-Wno-PINMISSING",
        "-Wno-WIDTHTRUNC",
        "-Wno-EOFNEWLINE",
        "-Wno-DECLFILENAME",
        "-Wno-WIDTHEXPAND",
    ]

    def __init__(
        self,
        verilator_bin: str,
        timeout_s: int,
        extra_sources: Sequence[Path],
        wno_flags: Optional[Sequence[str]] = None,
        lint_top: Optional[str] = None,
    ):
        self.verilator_bin = verilator_bin
        self.timeout_s = timeout_s
        self.extra_sources = list(extra_sources)
        self.wno_flags = list(wno_flags) if wno_flags else list(self.WNO_FLAGS)
        # When the testbench is part of the lint file set, the elaboration top is
        # the testbench module (as in the eval Makefile), not the candidate.
        self.lint_top = lint_top

    def verify(self, rtl: str, top_module: str | None = None) -> VerificationReport:
        if shutil.which(self.verilator_bin) is None:
            diagnostic = Diagnostic(
                tool="verilator", passed=False, missing=True, stderr="verilator not found on PATH"
            )
            return VerificationReport(syntax_passed=False, lint_passed=False, diagnostics=[diagnostic])
        try:
            with tempfile.TemporaryDirectory(prefix="realbench_lint_") as temp_name:
                temp_dir = Path(temp_name)
                file_names: List[str] = []
                for source in self.extra_sources:
                    target = temp_dir / source.name
                    if not target.exists():
                        shutil.copy2(source, target)
                        file_names.append(source.name)
                candidate_name = f"{top_module or 'candidate'}.gen.sv"
                (temp_dir / candidate_name).write_text(rtl, encoding="utf-8")
                file_names.append(candidate_name)
                command = [self.verilator_bin, "--lint-only", "--timing", *self.wno_flags]
                effective_top = self.lint_top or top_module
                if effective_top:
                    command += ["--top-module", effective_top]
                command += file_names
                completed = subprocess.run(
                    command,
                    cwd=temp_dir,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_s,
                )
        except subprocess.TimeoutExpired as exc:
            diagnostic = Diagnostic(tool="verilator", passed=False, stderr=str(exc), returncode=None)
            return VerificationReport(syntax_passed=False, lint_passed=False, diagnostics=[diagnostic])
        passed = completed.returncode == 0
        diagnostic = Diagnostic(
            tool="verilator",
            passed=passed,
            stdout=(completed.stdout or "")[-8000:],
            stderr=(completed.stderr or "")[-8000:],
            returncode=completed.returncode,
        )
        return VerificationReport(syntax_passed=passed, lint_passed=passed, diagnostics=[diagnostic])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run RealBench with a two-stage agentic planner plus legacy RTL generator pipeline."
    )
    parser.add_argument("--realbench-root", default="/home/kai/eval_dt/real_bench")
    parser.add_argument("--output-dir", default="runs/agentic_plan_legacy_realbench")
    parser.add_argument("--solution-name", default="agentic_plan_legacy")
    parser.add_argument("--task-level", choices=["module", "system", "both"], default="both")
    parser.add_argument("--include", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prepare-problems", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--realbench-verifier",
        choices=["native", "harness"],
        default="native",
        help="native runs RealBench verification Makefiles directly; harness delegates to run_verify.py.",
    )

    parser.add_argument("--base-url", help="OpenAI-compatible vLLM base URL. Defaults to VLLM_BASE_URL.")
    parser.add_argument("--model", help="Served model name. Defaults to VLLM_MODEL.")
    parser.add_argument("--api-key", help="API key. Defaults to VLLM_API_KEY or EMPTY.")
    parser.add_argument("--llm-timeout-s", type=int, default=2400)
    parser.add_argument("--temperature", type=float, default=0.2)

    parser.add_argument("--planner-max-tokens", type=int, default=100000)
    parser.add_argument("--planner-max-steps", type=int, default=16)
    parser.add_argument(
        "--planner-enable-tools",
        dest="planner_enable_tools",
        action="store_true",
        help="Enable agentic tool calling in the planner (attach search/inspect tool schemas). "
        "Off by default: the deployed reasoning model calls 0 tools and attaching them triggers "
        "its empty-content bug; catalog injection + grounding cover reuse selection instead.",
    )
    parser.set_defaults(planner_enable_tools=False)
    parser.add_argument("--planner-tool-choice", default="auto")
    parser.add_argument("--target-hdl", default="verilog")
    parser.add_argument(
        "--no-planner-inject-catalog",
        dest="planner_inject_catalog",
        action="store_false",
        help="Disable injecting the task catalog (real ip_id vocabulary) into the planner prompt.",
    )
    parser.add_argument(
        "--no-planner-ground-reuse",
        dest="planner_ground_reuse",
        action="store_false",
        help="Disable closed-vocabulary grounding of plan reuse decisions against the catalog.",
    )
    parser.add_argument(
        "--no-planner-completeness-gate",
        dest="planner_completeness_gate",
        action="store_false",
        help="Disable the one-shot re-prompt when a plan is missing reuse/integration sections.",
    )
    parser.add_argument("--planner-max-catalog-entries", type=int, default=60)
    parser.set_defaults(planner_inject_catalog=True, planner_ground_reuse=True, planner_completeness_gate=True)
    parser.add_argument(
        "--planner-search-mode",
        choices=["token", "semantic", "hybrid"],
        default="token",
        help="Backend for the planner's search_reuse_ip tool: token overlap (legacy), embedding "
        "retrieval over the task catalog, or semantic results topped up with token matches.",
    )
    parser.add_argument(
        "--embedder",
        default="auto",
        help="Embedder for retrieval and the repair cache: 'auto' (sentence-transformers MiniLM, "
        "falling back to the hashing embedder when unavailable), 'hash', or a sentence-transformers model name.",
    )
    parser.add_argument(
        "--planner-retrieval-min-score",
        type=float,
        default=0.70,
        help="Rerank-score floor for semantic IP search; results below it are flagged or dropped "
        "so weak retrievals never read as confident matches.",
    )
    parser.add_argument(
        "--planner-retrieval-below-threshold",
        choices=["flag", "drop"],
        default="flag",
        help="What to do with below-threshold candidates: tag them retrieval_confidence=low or remove them.",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Force re-embedding of the per-task catalog indexes even when cached ones match.",
    )
    parser.add_argument(
        "--repair-cache",
        choices=["off", "task", "run"],
        default="off",
        help="Semantic cache of verified Verilator-diagnostic->fix pairs injected as repair-prompt "
        "guidance: off, one cache per task, or one cache shared across the run.",
    )
    parser.add_argument(
        "--repair-cache-path",
        help="Persistence path for the run-scope repair cache (default: <output-dir>/repair_cache.json).",
    )
    parser.add_argument(
        "--repair-cache-evidence-threshold",
        type=float,
        default=0.85,
        help="Minimum similarity for a cached fix to be injected as guidance.",
    )
    parser.add_argument(
        "--repair-cache-reuse-threshold",
        type=float,
        default=0.95,
        help="Similarity labeled as near-exact in cache events; guidance-only either way, never auto-applied.",
    )
    parser.add_argument(
        "--repair-cache-max-hint-chars",
        type=int,
        default=1800,
        help="Cap on each cached fix excerpt injected into repair prompts.",
    )
    parser.add_argument(
        "--spec-condense-threshold-chars",
        type=int,
        default=200000,
        help="Specs longer than this are condensed with the ip_reuse_legacy chunk&merge pipeline before planning.",
    )
    parser.add_argument(
        "--spec-condense-threshold-tokens",
        type=int,
        default=45000,
        help="Specs whose estimated token count exceeds this are condensed even if under the char threshold; "
        "condensed views are also clipped to this token budget.",
    )
    parser.add_argument(
        "--condense-chunk-chars",
        type=int,
        default=30000,
        help="Chunk size used when condensing oversized specs.",
    )
    parser.add_argument(
        "--verbatim-excerpt-max-chars",
        type=int,
        default=24000,
        help="Budget for verbatim port tables / module headers carried unmodified into condensed specs.",
    )

    parser.add_argument("--legacy-max-tokens", type=int, default=80000)
    parser.add_argument("--legacy-max-repair-attempts", type=int, default=2)
    parser.add_argument(
        "--legacy-spec-max-chars",
        type=int,
        default=120000,
        help="Maximum characters of the original spec embedded in legacy RTL generation prompts.",
    )
    parser.add_argument(
        "--candidate-solution-max-chars",
        type=int,
        default=8000,
        help="Per-candidate cap on reused IP source embedded in the legacy plan prompt.",
    )
    parser.add_argument(
        "--candidate-total-budget-chars",
        type=int,
        default=80000,
        help="Total budget across all reused IP sources embedded in the legacy plan prompt.",
    )
    parser.add_argument(
        "--reuse-signature-max-chars",
        type=int,
        default=3000,
        help="Per-module cap on the port signature embedded for each provided reusable module.",
    )
    parser.add_argument(
        "--legacy-prompt-budget-chars",
        type=int,
        default=200000,
        help="Approximate total character budget for legacy RTL prompts; the spec excerpt shrinks to fit.",
    )
    parser.add_argument(
        "--legacy-skip-internal-verify",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip the internal Verilator lint (against dependency/defines files) that drives the repair loop.",
    )
    parser.add_argument(
        "--legacy-lint-with-testbench",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the testbench/reference files in the internal lint so it mirrors the eval compile "
        "exactly (catches PINNOTFOUND/MODDUP); only Verilator diagnostics reach the repair prompt, "
        "never the reference source.",
    )
    parser.add_argument(
        "--testbench-contract-max-chars",
        type=int,
        default=8000,
        help="Cap on the testbench DUT-instantiation snippet embedded in RTL prompts as the port contract.",
    )
    parser.add_argument("--yosys-bin", default="yosys")
    parser.add_argument("--verilator-bin", default="verilator")
    parser.add_argument("--agent-timeout-s", type=int, default=120)
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
        raise FileNotFoundError("RealBench problem files are missing:\n" + "\n".join(str(path) for path in missing))
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


def task_verification_dir(task: RealBenchTask) -> Path:
    if task.level == "system":
        return task.root_dir / "system" / task.task
    return task.root_dir / task.system / task.task / "verification"


def build_task_catalog(task: RealBenchTask, output_dir: Path) -> CatalogBundle:
    catalog_path = output_dir / "catalogs" / f"{task.task_id}.json"
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    declared_sources: List[SourceDoc] = []
    missing: List[str] = []
    seen_paths: set[Path] = set()
    for dependency in task.dependencies:
        source = resolve_dependency_source(task, dependency)
        if source is None:
            missing.append(dependency)
            continue
        if source.path not in seen_paths:
            declared_sources.append(source)
            seen_paths.add(source.path)
    support_sources = [source for source in resolve_support_sources(task) if source.path not in seen_paths]
    sources = [*declared_sources, *support_sources]
    payload = {"ips": [source_to_catalog_ip(task, source) for source in sources]}
    catalog_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    by_key: Dict[str, SourceDoc] = {}
    for source in sources:
        by_key[source.ip_id] = source
        by_key[source.name] = source
    return CatalogBundle(
        catalog_path=catalog_path,
        sources=sources,
        source_by_key=by_key,
        dependency_paths={source.name: str(source.path) for source in declared_sources},
        support_paths={source.name: str(source.path) for source in support_sources},
        missing_dependencies=missing,
    )


def source_to_catalog_ip(task: RealBenchTask, source: SourceDoc) -> Dict[str, Any]:
    code = source.path.read_text(encoding="utf-8", errors="ignore")
    signature = module_signature(code) or source.name
    return {
        "ip_id": source.ip_id,
        "name": source.name,
        "summary": f"RealBench {source.kind} RTL for {source.name}. Signature: {signature}",
        "category": source.kind,
        "interfaces": [signature],
        "parameters": {},
        "license": "benchmark supplied",
        "verification": ["supplied with RealBench verification collateral"],
        "synthesis": "Verilog/SystemVerilog source supplied by RealBench",
        "documentation": f"Source path: {source.path}",
        "tags": ["realbench", task.level, task.system, source.kind, source.name],
        "behavior": code,
        "integration_notes": [
            f"Use or instantiate this source when implementing {task.task}.",
            f"Original file: {source.path}",
        ],
        "known_limits": [],
    }


def resolve_dependency_source(task: RealBenchTask, dependency: str) -> Optional[SourceDoc]:
    for path in dependency_source_candidates(task, dependency):
        if path.exists() and path.is_file():
            return SourceDoc(ip_id=dependency, name=dependency, path=path, kind="dependency")
    scanned = scan_for_module(task_verification_dir(task), dependency)
    if scanned is not None:
        return SourceDoc(ip_id=dependency, name=dependency, path=scanned, kind="dependency")
    return None


def dependency_source_candidates(task: RealBenchTask, dependency: str) -> Iterable[Path]:
    verification_dir = task_verification_dir(task)
    for suffix in (".v", ".sv"):
        yield verification_dir / f"{dependency}{suffix}"
    for suffix in (".v", ".sv"):
        yield task.root_dir / task.system / dependency / f"{dependency}{suffix}"
    if task.level == "system":
        system_dir = task.root_dir / "system" / task.task
        for suffix in (".v", ".sv"):
            yield system_dir / f"{dependency}{suffix}"
    for suffix in (".v", ".sv"):
        yield task.root_dir / "sdc" / dependency / f"{dependency}{suffix}"
        yield task.root_dir / "aes" / dependency / f"{dependency}{suffix}"
        yield task.root_dir / "e203_hbirdv2" / dependency / f"{dependency}{suffix}"


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


def work_items(tasks: Sequence[RealBenchTask], samples: int) -> List[WorkItem]:
    return [WorkItem(task=task, sample=sample) for task in tasks for sample in range(1, samples + 1)]


def plan_output_dir(output_dir: Path, item: WorkItem) -> Path:
    return output_dir / "plans" / item.task.level / item.task.system / item.task.task / f"sample{item.sample:02d}"


def legacy_workspace_path(output_dir: Path, item: WorkItem) -> Path:
    return output_dir / "legacy_workspaces" / item.task.level / item.task.system / item.task.task / f"sample{item.sample:02d}"


def generated_code_path(output_dir: Path, item: WorkItem) -> Path:
    return output_dir / "generated" / item.task.level / item.task.system / item.task.task / f"sample{item.sample:02d}.sv"


def report_path(output_dir: Path, item: WorkItem) -> Path:
    return output_dir / "reports" / item.task.level / item.task.system / f"{item.task.task}_sample{item.sample:02d}.json"


def run_one(
    item: WorkItem,
    args: argparse.Namespace,
    output_dir: Path,
    catalogs: Dict[str, CatalogBundle],
    rag: RagRuntime,
) -> Dict[str, Any]:
    task = item.task
    catalog = catalogs[task.task_id]
    code_path = generated_code_path(output_dir, item)
    code_path.parent.mkdir(parents=True, exist_ok=True)
    report = report_path(output_dir, item)
    report.parent.mkdir(parents=True, exist_ok=True)
    plan_dir = plan_output_dir(output_dir, item)
    legacy_workspace = legacy_workspace_path(output_dir, item)
    plan_report_path = plan_dir / "agent_result.json"
    legacy_report_path = output_dir / "legacy_reports" / task.level / task.system / f"{task.task}_sample{item.sample:02d}.json"
    legacy_report_path.parent.mkdir(parents=True, exist_ok=True)

    code = ""
    generation_error: Optional[str] = None
    reused_existing = False
    wall_s = 0.0
    planner_result: Optional[PlanningResult] = None
    legacy_result: Any = None
    spec_condensed = False
    planner_repository: Any = None
    repair_cache = rag.repair_cache_for(task.task_id)

    if not catalog.usable:
        generation_error = f"missing dependency source(s): {', '.join(catalog.missing_dependencies)}"
    elif (args.resume or args.evaluate_only) and code_path.exists():
        code = code_path.read_text(encoding="utf-8")
        reused_existing = True
    elif args.evaluate_only:
        generation_error = f"missing generated file: {code_path}"
    elif args.dry_run:
        generation_error = "dry run"
    else:
        t0 = time.perf_counter()
        try:
            spec_bundle = effective_spec(task, catalog, args, output_dir)
            spec_condensed = spec_bundle.condensed
            planner_repository = make_planner_repository(task, catalog, args, rag)
            planner_result = run_planner(task, catalog, args, plan_dir, spec_bundle.planner, planner_repository)
            plan_report_path.write_text(dumps_plan_result(planner_result), encoding="utf-8")
            legacy_plan = agentic_plan_from_payload(planner_result.to_dict())
            enrich_legacy_plan_with_sources(legacy_plan, catalog, task, args)
            legacy_result = run_legacy_generator(
                task, legacy_plan, args, legacy_workspace, catalog, spec_bundle.generation, repair_cache
            )
            legacy_report_path.write_text(dumps_json(legacy_result.to_dict(), indent=2), encoding="utf-8")
            code = normalize_code(legacy_result.rtl, task.top_module)
            if code:
                code = strip_dependency_redeclarations(code, template_provided_module_names(task))
        except Exception as exc:  # noqa: BLE001 - keep benchmark moving.
            generation_error = f"{exc}\n{traceback.format_exc()[-4000:]}"
        wall_s = time.perf_counter() - t0
        if code:
            code_path.write_text(code, encoding="utf-8")

    eval_result = evaluate_realbench_code(task, code, args) if code else RealBenchEvalResult(
        syntax=0,
        function=0,
        error=generation_error or "generation produced empty code",
    )
    record = {
        "benchmark": "realbench",
        "pipeline": "agentic_ip_reuse_plan_to_ip_reuse_legacy_rtl",
        "task_level": task.level,
        "system": task.system,
        "task": task.task,
        "sample": item.sample,
        "top_module": task.top_module,
        "dependencies": task.dependencies,
        "dependency_paths": catalog.dependency_paths,
        "support_paths": catalog.support_paths,
        "missing_dependencies": catalog.missing_dependencies,
        "catalog_path": str(catalog.catalog_path),
        "catalog_doc_count": len(catalog.sources),
        "generated": bool(code),
        "reused_existing": reused_existing,
        "generation_error": generation_error,
        "generated_code_path": str(code_path) if code else None,
        "plan_dir": str(plan_dir),
        "plan_report_path": str(plan_report_path) if plan_report_path.exists() else None,
        "legacy_report_path": str(legacy_report_path) if legacy_report_path.exists() else None,
        "legacy_workspace_path": str(legacy_workspace),
        "planner_steps": planner_result.steps if planner_result else None,
        "planner_used_tools": planner_result.used_tools if planner_result else None,
        "legacy_repair_attempts": legacy_result.repair_attempts if legacy_result else None,
        "planner_search_mode": args.planner_search_mode,
        "embedder": rag.embedder_name,
        "retrieval_metrics": retrieval_metrics_from(planner_repository, rag.index_meta.get(task.task_id)),
        "repair_cache_metrics": repair_cache_metrics_from(args, legacy_result),
        "llm_token_estimate": llm_token_estimate_from(legacy_result),
        "spec_chars": len(task.prompt),
        "spec_condensed": spec_condensed,
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
    report.write_text(dumps_json(record, indent=2), encoding="utf-8")
    return record


def run_planner(
    task: RealBenchTask,
    catalog: CatalogBundle,
    args: argparse.Namespace,
    output_dir: Path,
    spec_text: str,
    repository: Any = None,
) -> PlanningResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    if repository is None:
        repository = JsonIpRepository(catalog.catalog_path)
    executor = AgentToolExecutor(repository, output_dir)
    llm = PlanningVllmClient(
        base_url=args.base_url or os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
        model=args.model or os.getenv("VLLM_MODEL", "siliconmind-server"),
        api_key=args.api_key or os.getenv("VLLM_API_KEY", "EMPTY"),
        timeout_s=args.llm_timeout_s,
    )
    agent = PlanningAgent(
        llm_client=llm,
        tool_executor=executor,
        config=AgentConfig(
            temperature=args.temperature,
            max_tokens=args.planner_max_tokens,
            use_tools=args.planner_enable_tools,
            tool_choice=args.planner_tool_choice,
            max_steps=args.planner_max_steps,
            inject_catalog=args.planner_inject_catalog,
            ground_reuse_decisions=args.planner_ground_reuse,
            completeness_gate=args.planner_completeness_gate,
            max_catalog_entries=args.planner_max_catalog_entries,
        ),
    )
    return agent.run(
        DesignTask(
            prompt=spec_text,
            target_hdl=args.target_hdl,
            constraints=task_constraints(task),
            known_interfaces=[],
            ppa_targets=[],
        )
    )


def make_legacy_llm(args: argparse.Namespace) -> LegacyVllmClient:
    return LegacyVllmClient(
        base_url=args.base_url or os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
        model=args.model or os.getenv("VLLM_MODEL", "siliconmind-server"),
        api_key=args.api_key or os.getenv("VLLM_API_KEY", "EMPTY"),
        timeout_s=args.llm_timeout_s,
    )


def make_legacy_agent(
    args: argparse.Namespace,
    verifier: Any,
    config: LegacyConfig,
    repair_cache: Any = None,
) -> LegacyAgent:
    embedder = HashingEmbedder(dim=128)
    retrieval_context = RetrievalContext.from_store(VectorStore([], embedder.encode([])), embedder)
    return LegacyAgent(make_legacy_llm(args), retrieval_context, verifier, config, repair_cache=repair_cache)


class LockedEmbedder:
    """Serializes encode() calls; sentence-transformers models are not guaranteed
    thread-safe and run_one executes inside a ThreadPoolExecutor."""

    def __init__(self, inner: Any):
        self.inner = inner
        self.dim = inner.dim
        self._lock = threading.Lock()

    def encode(self, texts):
        with self._lock:
            return self.inner.encode(texts)


@dataclass
class RagRuntime:
    """Run-wide retrieval/cache state shared across work items."""

    embedder: Any = None
    embedder_name: str = "none"
    stores: Dict[str, VectorStore] = field(default_factory=dict)
    index_meta: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    repair_caches: Dict[str, Any] = field(default_factory=dict)
    repair_cache_scope: str = "off"

    def repair_cache_for(self, task_id: str) -> Any:
        if self.repair_cache_scope == "run":
            return self.repair_caches.get("__run__")
        if self.repair_cache_scope == "task":
            return self.repair_caches.get(task_id)
        return None


def build_rag_runtime(
    args: argparse.Namespace,
    output_dir: Path,
    tasks: Sequence[RealBenchTask],
    catalogs: Dict[str, CatalogBundle],
) -> RagRuntime:
    rag = RagRuntime(repair_cache_scope=args.repair_cache)
    if args.planner_search_mode == "token" and args.repair_cache == "off":
        return rag
    embedder, embedder_name = make_embedder_with_fallback(args.embedder, warn=lambda message: print(f"[realbench] {message}"))
    rag.embedder = LockedEmbedder(embedder)
    rag.embedder_name = embedder_name
    if args.planner_search_mode != "token":
        for task in tasks:
            catalog = catalogs[task.task_id]
            store, meta = build_or_load_task_index(catalog.catalog_path, rag.embedder, embedder_name, args.reindex)
            rag.stores[task.task_id] = store
            rag.index_meta[task.task_id] = meta
            print(
                f"[realbench] index {task.task_id}: docs={meta['doc_count']} "
                f"embedder={embedder_name} rebuilt={meta.get('index_built', False)}"
            )
    if args.repair_cache == "run":
        cache_path = Path(args.repair_cache_path) if args.repair_cache_path else output_dir / "repair_cache.json"
        rag.repair_caches["__run__"] = make_repair_cache(args, rag.embedder, cache_path)
    elif args.repair_cache == "task":
        for task in tasks:
            cache_path = output_dir / "repair_cache" / f"{task.task_id}.json"
            rag.repair_caches[task.task_id] = make_repair_cache(args, rag.embedder, cache_path)
    return rag


def make_repair_cache(args: argparse.Namespace, embedder: Any, path: Path) -> RepairFixCache:
    return RepairFixCache(
        embedder=embedder,
        path=path,
        evidence_threshold=args.repair_cache_evidence_threshold,
        reuse_threshold=args.repair_cache_reuse_threshold,
        max_hint_chars=args.repair_cache_max_hint_chars,
    )


def catalog_index_documents(catalog_path: Path) -> List[RtlDocument]:
    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    documents: List[RtlDocument] = []
    for item in payload.get("ips", []):
        interfaces = " ".join(str(entry) for entry in item.get("interfaces", []))
        documents.append(
            RtlDocument(
                doc_id=str(item["ip_id"]),
                # Embed what the IP *is* (name, summary, interfaces) plus a slice of
                # its body; MiniLM truncates around 256 tokens so full RTL is wasted.
                problem=f"{item.get('name', '')}: {item.get('summary', '')}\n{interfaces}",
                solution=str(item.get("behavior", ""))[:1500],
                tags=[str(tag) for tag in item.get("tags", [])],
            )
        )
    return documents


def build_or_load_task_index(
    catalog_path: Path,
    embedder: Any,
    embedder_name: str,
    reindex: bool,
) -> Tuple[VectorStore, Dict[str, Any]]:
    index_dir = catalog_path.with_suffix(".index")
    meta_path = index_dir / "index_meta.json"
    catalog_sha = hashlib.sha256(catalog_path.read_bytes()).hexdigest()
    if not reindex and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            meta = {}
        if (
            meta.get("catalog_sha256") == catalog_sha
            and meta.get("embedder_name") == embedder_name
            and meta.get("dim") == embedder.dim
        ):
            return VectorStore.load(index_dir), {**meta, "index_built": False}
    documents = catalog_index_documents(catalog_path)
    t0 = time.perf_counter()
    vectors = embedder.encode([document.retrieval_text for document in documents])
    embed_s = round(time.perf_counter() - t0, 4)
    store = VectorStore(documents, vectors)
    store.save(index_dir)
    meta = {
        "embedder_name": embedder_name,
        "dim": embedder.dim,
        "catalog_sha256": catalog_sha,
        "doc_count": len(documents),
        "embed_s": embed_s,
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return store, {**meta, "index_built": True}


def make_planner_repository(task: RealBenchTask, catalog: CatalogBundle, args: argparse.Namespace, rag: RagRuntime) -> Any:
    repository = JsonIpRepository(catalog.catalog_path)
    if args.planner_search_mode == "token":
        return repository
    store = rag.stores.get(task.task_id)
    if store is None or rag.embedder is None:
        return repository
    return SemanticIpRepository(
        inner=repository,
        retriever=Retriever(store, rag.embedder),
        reranker=LexicalReranker(),
        mode=args.planner_search_mode,
        min_score=args.planner_retrieval_min_score,
        below_threshold=args.planner_retrieval_below_threshold,
    )


def retrieval_metrics_from(repository: Any, index_meta: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    traces = getattr(repository, "traces", None)
    if traces is None:
        return None
    top_scores = [trace.top_score for trace in traces if trace.top_score is not None]
    return {
        "searches": len(traces),
        "mean_top_score": round(sum(top_scores) / len(top_scores), 4) if top_scores else None,
        "low_confidence_searches": sum(1 for trace in traces if trace.low_confidence),
        "filtered_below_threshold": sum(trace.filtered_below_threshold for trace in traces),
        "retrieval_latency_s": round(sum(trace.latency_s for trace in traces), 4),
        "index_built": bool(index_meta and index_meta.get("index_built")),
        "index_embed_s": index_meta.get("embed_s") if index_meta else None,
    }


def repair_cache_metrics_from(args: argparse.Namespace, legacy_result: Any) -> Dict[str, Any]:
    if args.repair_cache == "off":
        return {"enabled": False}
    events = list(getattr(legacy_result, "repair_cache_events", []) or []) if legacy_result else []
    lookups = [event for event in events if event.get("event") == "lookup"]
    hits = [event for event in lookups if event.get("decision") in ("evidence", "reuse")]
    return {
        "enabled": True,
        "scope": args.repair_cache,
        "lookups": len(lookups),
        "hits": len(hits),
        "hit_scores": [round(float(event["score"]), 4) for event in hits if event.get("score") is not None],
        "hints_injected": sum(1 for event in lookups if event.get("injected")),
        "fixes_recorded": sum(1 for event in events if event.get("event") == "record"),
    }


# Calibrated against the one measured RealBench point (703,806 chars = 160,645 tokens).
EST_CHARS_PER_TOKEN = 4.38


def llm_token_estimate_from(legacy_result: Any) -> Optional[Dict[str, Any]]:
    traces = getattr(legacy_result, "llm_traces", None) if legacy_result else None
    if not traces:
        return None
    per_stage: Dict[str, Dict[str, int]] = {}
    total_prompt_chars = 0
    total_response_chars = 0
    for trace in traces:
        prompt_chars = int(getattr(trace, "prompt_chars", 0) or 0)
        response_chars = int(getattr(trace, "response_chars", 0) or 0)
        total_prompt_chars += prompt_chars
        total_response_chars += response_chars
        stage = per_stage.setdefault(trace.stage, {"prompt_chars": 0, "response_chars": 0})
        stage["prompt_chars"] += prompt_chars
        stage["response_chars"] += response_chars
    return {
        "prompt_tokens": int(total_prompt_chars / EST_CHARS_PER_TOKEN),
        "response_tokens": int(total_response_chars / EST_CHARS_PER_TOKEN),
        "prompt_chars": total_prompt_chars,
        "response_chars": total_response_chars,
        "llm_calls": len(traces),
        "per_stage": per_stage,
    }


_CONDENSE_LOCK = threading.Lock()

_TOKEN_PIECE_RE = re.compile(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]")
_TOKEN_WORD_RE = re.compile(r"[A-Za-z0-9_]+")


def estimate_tokens(text: str) -> int:
    """Tokenizer-free token estimate: one per word/punctuation piece plus a
    surcharge for long identifiers that BPE splits. Calibrated against the one
    measured RealBench point (e203_cpu_top: 703,806 chars = 160,645 tokens;
    estimate lands ~4% above, i.e. safely conservative)."""
    pieces = len(_TOKEN_PIECE_RE.findall(text))
    surcharge = sum((len(word) - 1) // 6 for word in _TOKEN_WORD_RE.findall(text))
    return pieces + surcharge


def clip_to_token_budget(text: str, max_tokens: int) -> str:
    estimated = estimate_tokens(text)
    if estimated <= max_tokens:
        return text
    chars_per_token = len(text) / max(estimated, 1)
    return clip_text(text, int(max_tokens * chars_per_token))


@dataclass(frozen=True)
class SpecBundle:
    planner: str
    generation: str
    condensed: bool


def effective_spec(
    task: RealBenchTask,
    catalog: CatalogBundle,
    args: argparse.Namespace,
    output_dir: Path,
) -> SpecBundle:
    """Return the task spec for each pipeline stage, condensed via the
    ip_reuse_legacy chunk&merge pipeline when it is too large to fit the model
    context (by chars or estimated tokens). Condensed views are cached per task."""
    if (
        len(task.prompt) <= args.spec_condense_threshold_chars
        and estimate_tokens(task.prompt) <= args.spec_condense_threshold_tokens
    ):
        return SpecBundle(planner=task.prompt, generation=task.prompt, condensed=False)
    cache_dir = output_dir / "condensed_specs"
    planner_path = cache_dir / f"{task.task_id}.planner.txt"
    generation_path = cache_dir / f"{task.task_id}.generation.txt"
    with _CONDENSE_LOCK:
        if planner_path.exists() and generation_path.exists():
            return SpecBundle(
                planner=planner_path.read_text(encoding="utf-8"),
                generation=generation_path.read_text(encoding="utf-8"),
                condensed=True,
            )
        agent = make_legacy_agent(
            args,
            verifier=NoopVerifier(),
            config=LegacyConfig(
                target_hdl="verilog",
                temperature=args.temperature,
                max_tokens=args.legacy_max_tokens,
                large_spec_chunk_chars=args.condense_chunk_chars,
                decomposition_mode="chunking",
            ),
        )
        views = agent.condense_spec_views(
            task.prompt,
            target_hdl=args.target_hdl,
            top_module=task.top_module,
            constraints=task_constraints(task),
            workspace_dir=output_dir / "condense_workspaces" / task.task_id,
            provided_modules=template_provided_module_names(task),
            excerpt_max_chars=args.verbatim_excerpt_max_chars,
        )
        # Sections are rendered most-important-first (top interface, verbatim
        # excerpts, module interfaces, then behavioral detail), so clipping to
        # the context budget drops the least useful tail.
        bundle = SpecBundle(
            planner=clip_to_token_budget(
                clip_text(views["planner"], args.spec_condense_threshold_chars),
                args.spec_condense_threshold_tokens,
            ),
            generation=clip_to_token_budget(
                clip_text(views["generation"], args.spec_condense_threshold_chars),
                args.spec_condense_threshold_tokens,
            ),
            condensed=True,
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        planner_path.write_text(bundle.planner, encoding="utf-8")
        generation_path.write_text(bundle.generation, encoding="utf-8")
        print(
            f"[realbench] condensed spec for {task.task_id}: {len(task.prompt)} chars -> "
            f"planner {len(bundle.planner)} / generation {len(bundle.generation)} chars "
            f"(~{estimate_tokens(bundle.generation)} tokens)"
        )
        return bundle


def run_legacy_generator(
    task: RealBenchTask,
    plan: IpReusePlan,
    args: argparse.Namespace,
    workspace_dir: Path,
    catalog: CatalogBundle,
    spec_text: str,
    repair_cache: Any = None,
) -> Any:
    if args.legacy_skip_internal_verify:
        verifier: Any = NoopVerifier()
    else:
        wno_flags, makefile_top = makefile_lint_settings(task)
        include_testbench = bool(args.legacy_lint_with_testbench and makefile_top)
        verifier = RealBenchWorkspaceVerifier(
            verilator_bin=args.verilator_bin,
            timeout_s=args.agent_timeout_s,
            extra_sources=internal_verify_sources(task, include_testbench=include_testbench),
            wno_flags=wno_flags or None,
            lint_top=makefile_top if include_testbench else None,
        )
    agent = make_legacy_agent(
        args,
        verifier,
        LegacyConfig(
            target_hdl="verilog",
            max_repair_attempts=args.legacy_max_repair_attempts,
            temperature=args.temperature,
            max_tokens=args.legacy_max_tokens,
        ),
        repair_cache=repair_cache,
    )
    provided_signatures, inline_signatures = reuse_module_signatures(task, catalog)
    provided_signatures = {
        name: clip_text(signature, args.reuse_signature_max_chars)
        for name, signature in provided_signatures.items()
    }
    signatures_chars = sum(len(name) + len(sig) + 8 for name, sig in provided_signatures.items())
    # Keep the whole prompt inside the model context: the spec excerpt yields
    # whatever the signatures and reused-source candidates do not consume.
    spec_cap = min(
        args.legacy_spec_max_chars,
        max(
            20000,
            args.legacy_prompt_budget_chars
            - signatures_chars
            - args.candidate_total_budget_chars
            - 30000,
        ),
    )
    return agent.run_from_plan(
        plan,
        target_hdl="verilog",
        top_module=task.top_module,
        workspace_dir=workspace_dir,
        original_spec=clip_text(spec_text, spec_cap),
        reuse_modules=provided_signatures,
        environment_notes=build_environment_notes(task, provided_signatures, inline_signatures, args),
    )


def template_resident(task: RealBenchTask, path: Path | str) -> bool:
    return Path(path).resolve().parent == task_verification_dir(task).resolve()


def reuse_module_signatures(task: RealBenchTask, catalog: CatalogBundle) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Split dependency modules into (provided-by-compile-environment, must-be-inlined)
    and return their module header signatures."""
    provided: Dict[str, str] = {}
    inline: Dict[str, str] = {}
    for source in catalog.sources:
        if source.kind != "dependency":
            continue
        code = source.path.read_text(encoding="utf-8", errors="ignore")
        signature = module_signature(code) or f"module {source.name}(/* see original source */);"
        if template_resident(task, source.path):
            provided[source.name] = signature
        else:
            inline[source.name] = signature
    return provided, inline


def template_defines_files(task: RealBenchTask) -> List[Path]:
    template_dir = task_verification_dir(task)
    if not template_dir.exists():
        return []
    return [
        path
        for path in sorted(template_dir.iterdir())
        if path.is_file()
        and not path.name.startswith("ref_")
        and (path.name == "config.v" or "defines" in path.name.lower())
    ]


def internal_verify_sources(task: RealBenchTask, include_testbench: bool = False) -> List[Path]:
    """Files compiled together with the candidate RTL during internal lint.

    Mirrors the eval Makefile (which compiles every *.v/*.sv in the template dir)
    minus the candidate slot. With include_testbench the file set matches the eval
    exactly (testbench/reference included), so the repair loop sees the same
    PINNOTFOUND/MODDUP errors the final evaluation would; otherwise the
    testbench/reference collateral is left out."""
    template_dir = task_verification_dir(task)
    if not template_dir.exists():
        return []
    excluded_suffixes = (
        "_ref.sv",
        "_ref.v",
        "_testbench.sv",
        "_testbench.v",
        "_stimulus_gen.sv",
        "_stimulus_gen.v",
    )
    top_filename = f"{task.task}_top.sv"
    defines: List[Path] = []
    others: List[Path] = []
    for path in sorted([*template_dir.glob("*.v"), *template_dir.glob("*.sv")]):
        name = path.name
        if name == top_filename:
            continue
        if not include_testbench and (name.startswith("ref_") or name.endswith(excluded_suffixes)):
            continue
        if path.name == "config.v" or "defines" in path.name.lower():
            defines.append(path)
        else:
            others.append(path)
    return [*defines, *others]


def makefile_lint_settings(task: RealBenchTask) -> Tuple[List[str], Optional[str]]:
    """Warning suppressions and elaboration top parsed from the task's eval Makefile."""
    makefile = task_verification_dir(task) / "Makefile"
    if not makefile.exists():
        return [], None
    text = makefile.read_text(encoding="utf-8", errors="ignore")
    wno_flags = list(dict.fromkeys(re.findall(r"-Wno-[A-Z0-9]+", text)))
    top_match = re.search(r"--top(?:-module)?\s+([A-Za-z_]\w*)", text)
    return wno_flags, top_match.group(1) if top_match else None


def testbench_dut_instantiation(task: RealBenchTask) -> str:
    """The testbench's instantiation of the candidate module, verbatim. This is
    the binding port contract: every connected pin must exist on the module."""
    template_dir = task_verification_dir(task)
    if not template_dir.exists():
        return ""
    for path in sorted([*template_dir.glob("*testbench*.sv"), *template_dir.glob("*testbench*.v")]):
        text = path.read_text(encoding="utf-8", errors="ignore")
        match = re.search(rf"(?m)^[ \t]*{re.escape(task.top_module)}\s+[A-Za-z_]\w*\s*\(", text)
        if not match:
            continue
        depth = 0
        for index in range(match.end() - 1, len(text)):
            char = text[index]
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    semicolon = text.find(";", index)
                    end = semicolon + 1 if semicolon != -1 else index + 1
                    return text[match.start() : end]
    return ""


def template_provided_module_names(task: RealBenchTask) -> List[str]:
    """All module names declared by template files other than the candidate slot.
    Generated code must never re-declare any of them (fatal MODDUP at eval)."""
    template_dir = task_verification_dir(task)
    if not template_dir.exists():
        return []
    top_filename = f"{task.task}_top.sv"
    names: set[str] = set()
    for path in sorted([*template_dir.glob("*.v"), *template_dir.glob("*.sv")]):
        if path.name == top_filename:
            continue
        names.update(MODULE_DECL_RE.findall(path.read_text(encoding="utf-8", errors="ignore")))
    names.discard(task.top_module)
    return sorted(names)


def build_environment_notes(
    task: RealBenchTask,
    provided_signatures: Dict[str, str],
    inline_signatures: Dict[str, str],
    args: argparse.Namespace,
) -> List[str]:
    notes: List[str] = []
    instantiation = testbench_dut_instantiation(task)
    if instantiation:
        notes.append(
            f"The verification testbench instantiates {task.top_module} exactly as shown below. Your module "
            "must declare every port connected here, with these exact names (a missing port is a fatal "
            "PINNOTFOUND elaboration error). Ports inside `ifdef blocks are required whenever the compile "
            "environment defines that macro:\n"
            + clip_text(instantiation, args.testbench_contract_max_chars)
        )
    defines = template_defines_files(task)
    if defines:
        names = ", ".join(path.name for path in defines)
        notes.append(
            f"The compile directory provides these defines/include files: {names}. "
            "Reference their macros with a backtick (e.g. `MACRO_NAME) and put the matching "
            "`include directive(s) at the top of your file if you use any of those macros."
        )
    else:
        notes.append(
            "No defines/include files are available in the compile directory. Do not use `include; "
            "if you need macros or constants, define them inline in your code."
        )
    if inline_signatures:
        names = ", ".join(sorted(inline_signatures))
        notes.append(
            f"These dependency modules are NOT provided by the compile environment and their full "
            f"implementation must be included in your output: {names}. Copy the original source from "
            "the reuse plan candidates verbatim when available."
        )
    if provided_signatures:
        notes.append(
            "All modules listed under provided reusable modules are compiled from their own source "
            "files; emitting a module with one of those names will cause a fatal duplicate-declaration error."
        )
    return notes


def clip_text(text: str, max_chars: int, marker: str = "\n... [truncated] ...") -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + marker


def strip_dependency_redeclarations(code: str, module_names: Iterable[str]) -> str:
    """Remove re-declarations of modules that the eval environment supplies as files."""
    for name in module_names:
        pattern = re.compile(rf"(?ms)^[ \t]*module\s+{re.escape(name)}\b.*?^[ \t]*endmodule[^\n]*\n?")
        code = pattern.sub("", code)
    stripped = code.strip()
    return stripped + "\n" if stripped else ""


def enrich_legacy_plan_with_sources(
    plan: IpReusePlan,
    catalog: CatalogBundle,
    task: RealBenchTask,
    args: argparse.Namespace,
) -> None:
    selected_count = sum(
        1
        for decision in plan.decisions
        if decision.selected_doc_id and catalog.source_by_key.get(decision.selected_doc_id)
    )
    per_candidate_cap = args.candidate_solution_max_chars
    if selected_count:
        per_candidate_cap = min(
            per_candidate_cap,
            max(1500, args.candidate_total_budget_chars // selected_count),
        )
    for decision in plan.decisions:
        if not decision.selected_doc_id:
            continue
        source = catalog.source_by_key.get(decision.selected_doc_id)
        if source is None:
            continue
        code = source.path.read_text(encoding="utf-8", errors="ignore")
        # Modules the compile environment does not supply must be reproduced verbatim,
        # so their source is never truncated.
        if template_resident(task, source.path):
            code = bounded_source_text(code, per_candidate_cap)
        for candidate in decision.candidates:
            if candidate.doc_id != decision.selected_doc_id:
                continue
            candidate.problem = (
                f"Reusable RealBench {source.kind} source {source.name} for module "
                f"{decision.module.name}. Original file: {source.path}"
            )
            candidate.solution = code
            candidate.tags = ["realbench", source.kind, source.name]
            candidate.metadata.update({"source_path": str(source.path), "source_kind": source.kind})
            break


def bounded_source_text(code: str, max_chars: int) -> str:
    if len(code) <= max_chars:
        return code
    return (
        code[:max_chars]
        + "\n// ... truncated; the complete file is supplied separately in the compile environment ..."
    )


def task_constraints(task: RealBenchTask) -> List[str]:
    return [
        f"Final RTL must implement exactly one public top module named {task.top_module}.",
        "Use the exact port names, directions, widths, parameters, reset polarity, and behavior from the RealBench problem.",
        "Do not include a testbench, markdown fences, or explanatory prose in generated RTL.",
        "Use catalog dependency IP when it matches required submodule behavior; generate glue or new RTL only where needed.",
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
    if args.realbench_verifier == "harness":
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
    return RealBenchEvalResult(
        syntax=syntax,
        function=function,
        syntax_info=syntax_info,
        function_info=function_info,
        compile_returncode=completed.returncode,
        run_returncode=completed.returncode,
        stdout_tail=stdout[-4000:],
        stderr_tail=stderr[-4000:],
        error=None if syntax == 1 else f"make all failed in {elapsed:.2f}s",
    )


def evaluate_realbench_code_with_harness(task: RealBenchTask, code: str, args: argparse.Namespace) -> RealBenchEvalResult:
    root = Path(args.realbench_root)
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
        "print(json.dumps({'syntax': result[0], 'function': result[1], 'syntax_info': result[2], 'function_info': result[3]}))\n"
    )
    try:
        with tempfile.TemporaryDirectory(prefix=f"realbench_harness_{task.task}_") as temp_name:
            code_path = Path(temp_name) / f"{task.task}_top.sv"
            code_path.write_text(code, encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, "-c", runner, task.level, task.system, task.task, str(code_path)],
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
    payload = parse_harness_result(completed.stdout or "")
    if completed.returncode != 0 or payload is None:
        return RealBenchEvalResult(
            0,
            0,
            compile_returncode=completed.returncode,
            run_returncode=completed.returncode,
            stdout_tail=(completed.stdout or "")[-4000:],
            stderr_tail=(completed.stderr or "")[-4000:],
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
        stdout_tail=(completed.stdout or "")[-4000:],
        stderr_tail=(completed.stderr or "")[-4000:],
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
    return "\n".join(
        line for line in stderr.splitlines() if line.startswith("%Error") or line.startswith("%Warning")
    )


def realbench_function_errors(stdout: str) -> str:
    lines = []
    for line in stdout.splitlines():
        if PASS_HINT_RE.search(line):
            continue
        if MISMATCH_HINT_RE.search(line):
            lines.append(line.removeprefix("Hint: ").strip())
    return "\n".join(lines)


def module_signature(rtl: str) -> str:
    match = re.search(r"(?ms)^\s*module\s+[A-Za-z_][A-Za-z0-9_$]*\b.*?;", rtl)
    return match.group(0).strip() if match else ""


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
            {"task": record["task"], "codeid": int(record["sample"]), "code": Path(code_path).read_text(encoding="utf-8")}
        )
    sample_dir = output_dir / "samples" / solution_name
    sample_dir.mkdir(parents=True, exist_ok=True)
    for key, key_rows in rows.items():
        with (sample_dir / f"{key}.jsonl").open("w", encoding="utf-8") as handle:
            for row in key_rows:
                handle.write(dumps_json(row) + "\n")


def summarize(records: Sequence[Dict[str, Any]], tasks: Sequence[RealBenchTask], elapsed_s: float) -> Dict[str, Any]:
    total = len(records)
    return {
        "benchmark": "realbench",
        "pipeline": "agentic_ip_reuse_plan_to_ip_reuse_legacy_rtl",
        "num_tasks": len(tasks),
        "num_records": total,
        "samples_per_task": max((int(record["sample"]) for record in records), default=0),
        "generated": sum(1 for record in records if record.get("generated")),
        "syntax": sum(1 for record in records if record.get("syntax") == 1),
        "function": sum(1 for record in records if record.get("function") == 1),
        "passed": sum(1 for record in records if record.get("passed")),
        "syntax_rate": safe_rate(sum(1 for record in records if record.get("syntax") == 1), total),
        "function_rate": safe_rate(sum(1 for record in records if record.get("function") == 1), total),
        "pass_rate": safe_rate(sum(1 for record in records if record.get("passed")), total),
        "total_s": elapsed_s,
        **rag_aggregates(records),
    }


def rag_aggregates(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Cross-record retrieval/cache aggregates for before-vs-after run comparison."""
    repair = [record.get("repair_cache_metrics") or {} for record in records]
    lookups = sum(metrics.get("lookups", 0) for metrics in repair if metrics.get("enabled"))
    hits = sum(metrics.get("hits", 0) for metrics in repair if metrics.get("enabled"))
    retrieval = [record.get("retrieval_metrics") or {} for record in records]
    searches = sum(metrics.get("searches", 0) for metrics in retrieval)
    low_confidence = sum(metrics.get("low_confidence_searches", 0) for metrics in retrieval)
    attempts = [
        int(record["legacy_repair_attempts"])
        for record in records
        if record.get("legacy_repair_attempts") is not None
    ]
    walls = [float(record.get("wall_s") or 0.0) for record in records]
    tokens = [record.get("llm_token_estimate") or {} for record in records]
    total_tokens = sum(item.get("prompt_tokens", 0) + item.get("response_tokens", 0) for item in tokens)
    return {
        "repair_cache_lookups": lookups,
        "repair_cache_hits": hits,
        "repair_cache_hit_rate": safe_rate(hits, lookups),
        "mean_repair_attempts": (sum(attempts) / len(attempts)) if attempts else None,
        "mean_wall_s": (sum(walls) / len(walls)) if walls else None,
        "total_estimated_tokens": total_tokens or None,
        "retrieval_searches": searches,
        "low_confidence_search_rate": safe_rate(low_confidence, searches),
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

    catalogs: Dict[str, CatalogBundle] = {}
    for task in tasks:
        bundle = build_task_catalog(task, output_dir)
        catalogs[task.task_id] = bundle
        print(
            f"[realbench] catalog {task.task_id}: docs={len(bundle.sources)} "
            f"deps={len(bundle.dependency_paths)} missing={bundle.missing_dependencies}"
        )

    items = work_items(tasks, args.samples)
    if args.dry_run:
        rag = RagRuntime(repair_cache_scope=args.repair_cache)
        records = [dry_run_record(item, catalogs[item.task.task_id], output_dir) for item in items]
    else:
        rag = build_rag_runtime(args, output_dir, tasks, catalogs)
        records = []
        with ThreadPoolExecutor(max_workers=max(args.concurrency, 1)) as executor:
            futures = [executor.submit(run_one, item, args, output_dir, catalogs, rag) for item in items]
            for future in as_completed(futures):
                record = future.result()
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
    rag_metrics = {
        "planner_search_mode": args.planner_search_mode,
        "embedder": rag.embedder_name,
        "repair_cache_scope": args.repair_cache,
        **rag_aggregates(records),
    }
    if rag.repair_caches:
        rag_metrics["repair_cache_stats"] = {key: cache.stats() for key, cache in rag.repair_caches.items()}
    (output_dir / "rag_metrics.json").write_text(dumps_json(rag_metrics, indent=2), encoding="utf-8")
    print(f"[realbench] wrote results under {output_dir}")
    return summary


def dry_run_record(item: WorkItem, bundle: CatalogBundle, output_dir: Path) -> Dict[str, Any]:
    task = item.task
    return {
        "benchmark": "realbench",
        "pipeline": "agentic_ip_reuse_plan_to_ip_reuse_legacy_rtl",
        "task_level": task.level,
        "system": task.system,
        "task": task.task,
        "sample": item.sample,
        "top_module": task.top_module,
        "dependencies": task.dependencies,
        "dependency_paths": bundle.dependency_paths,
        "support_paths": bundle.support_paths,
        "missing_dependencies": bundle.missing_dependencies,
        "catalog_path": str(bundle.catalog_path),
        "catalog_doc_count": len(bundle.sources),
        "generated": False,
        "reused_existing": False,
        "generation_error": "dry run",
        "generated_code_path": None,
        "plan_dir": str(plan_output_dir(output_dir, item)),
        "legacy_workspace_path": str(legacy_workspace_path(output_dir, item)),
        "syntax": 0,
        "function": 0,
        "passed": False,
        "evaluation_error": "dry run",
        "wall_s": 0.0,
    }


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    summary = run_realbench(args)
    print(dumps_json(summary, indent=2))


if __name__ == "__main__":
    main()
