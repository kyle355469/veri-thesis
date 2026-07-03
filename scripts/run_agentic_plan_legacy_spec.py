#!/usr/bin/env python3
"""Two-stage runner for a normal (free-form) spec file: router -> agentic plan -> legacy RTL.

Same router and pipeline as scripts/run_agentic_plan_legacy_realbench.py (and the
cascade router of scripts/run_realbench_routed.py), minus the RealBench benchmark
collateral:

* the spec comes from ``--spec <file>`` (or ``-`` for stdin) instead of problems.jsonl;
* reuse IP comes from an optional ``--catalog`` JSON (JsonIpRepository format) and/or
  ``--extra-source`` RTL files (auto-cataloged, linted alongside the candidate, and
  treated as compile-environment-provided modules exactly like RealBench dependencies);
* final scoring is a Verilator lint over candidate + extra sources (syntax), plus an
  optional user ``--test-command`` for a functional verdict.

Routing arms (``--router``), identical decision logic to the RealBench cascade router:

* ``all_pipeline`` / ``all_direct`` -- force one flow.
* ``pre``        -- Tier-0 spec features (``--decider {keyword,llm}``) -> decide_pre,
                    ``uncertain`` resolved by size fallback.
* ``plan_probe`` -- enter the pipeline; the Tier-1 plan-probe (rag_rtl.routing.decide_plan)
                    bails self-contained-algorithm samples to direct generation.
* ``cascade``    -- Tier-0 routes confident tasks; ``uncertain`` enters the pipeline with
                    the plan-probe enabled. (default)

The direct flow skips planning, not repair: directly generated RTL gets the same
syntax/lint (+ optional functional) repair budget via agent.repair_rtl(plan=None).

Usage::

    python scripts/run_agentic_plan_legacy_spec.py --spec my_block.md --top-module my_block
    python scripts/run_agentic_plan_legacy_spec.py --spec spec.md --top-module top \
        --catalog my_ips.json --extra-source deps/defines.v --extra-source deps/fifo.v \
        --router cascade --decider llm --samples 3
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
PLANNER_ROOT = REPO_ROOT / "agentic_ip_reuse"
for _path in (str(PLANNER_ROOT), str(REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from agentic_ip_reuse.agent import AgentConfig, AgenticIpReuseAgent as PlanningAgent, dumps_result as dumps_plan_result
from agentic_ip_reuse.hierarchical import HierarchicalAgent, HierarchicalConfig
from agentic_ip_reuse.llm import (
    VllmClient as PlanningVllmClient,
    get_request_log as get_planning_request_log,
    reset_request_log as reset_planning_request_log,
)
from agentic_ip_reuse.repository import JsonIpRepository
from agentic_ip_reuse.semantic_repository import SemanticIpRepository
from agentic_ip_reuse.tools import AgentToolExecutor
from agentic_ip_reuse.types import AgentResult as PlanningResult, DesignTask
from ip_reuse_legacy.config import AgenticIpReuseConfig as LegacyConfig
from ip_reuse_legacy.plan_adapter import agentic_plan_from_payload
from ip_reuse_legacy.types import IpReusePlan
from rag_rtl import routing
from rag_rtl.json_utils import dumps_json
from rag_rtl.llm import (
    get_request_log as get_legacy_request_log,
    reset_request_log as reset_legacy_request_log,
)
from rag_rtl.retrieval import LexicalReranker, Retriever
from rag_rtl.embeddings import make_embedder_with_fallback

# Benchmark-agnostic machinery shared verbatim with the RealBench runner, so the
# spec flow and the benchmark flow cannot drift apart.
from scripts.run_agentic_plan_legacy_realbench import (
    MODULE_DECL_RE,
    LockedEmbedder,
    NoopVerifier,
    RealBenchWorkspaceVerifier as WorkspaceLintVerifier,
    build_or_load_task_index,
    clip_text,
    clip_to_token_budget,
    estimate_tokens,
    make_legacy_agent,
    make_legacy_llm,
    make_repair_cache,
    module_signature,
    normalize_code,
    safe_name,
    safe_rate,
    strip_dependency_redeclarations,
)

ROUTERS = ["cascade", "pre", "plan_probe", "all_pipeline", "all_direct"]


@dataclass(frozen=True)
class SpecTask:
    """A free-form design task: one spec, one required top module."""

    name: str
    top_module: str
    prompt: str
    extra_sources: Tuple[Path, ...]

    @property
    def task_id(self) -> str:
        return safe_name(self.name)


@dataclass(frozen=True)
class SpecBundle:
    planner: str
    generation: str
    condensed: bool


@dataclass
class FunctionalReport:
    function_passed: bool
    function_info: str = ""
    syntax_ok: bool = True
    stdout_tail: str = ""
    error: Optional[str] = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Route a free-form spec to direct or the agentic-plan + legacy-RTL pipeline "
        "(same router and pipeline as run_agentic_plan_legacy_realbench.py)."
    )
    parser.add_argument("--spec", required=True, help="Path to the spec text/markdown file, or '-' for stdin.")
    parser.add_argument(
        "--top-module",
        help="Required public top module name. Defaults to the spec filename stem.",
    )
    parser.add_argument(
        "--catalog",
        help="Optional reuse-IP catalog JSON in JsonIpRepository format ({'ips': [...]}) fed to the planner.",
    )
    parser.add_argument(
        "--extra-source",
        action="append",
        default=[],
        help="RTL file compiled alongside the candidate (repeatable). Its modules are treated as "
        "compile-environment-provided (linted together, signatures injected, re-declarations "
        "stripped) and it is auto-added to the planner catalog as reusable IP.",
    )
    parser.add_argument("--output-dir", default="runs/agentic_plan_legacy_spec")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--resume", action="store_true")

    add_router_args(parser)
    add_shared_pipeline_args(parser)
    parser.add_argument(
        "--test-command",
        help="Optional functional check command with {rtl}, {sources}, {top} placeholders, e.g. "
        "\"vvp-run.sh {rtl} tb.sv\". Exit code 0 = functional pass; drives --legacy-functional-repair "
        "and the final functional verdict.",
    )
    return parser


def add_router_args(parser: argparse.ArgumentParser) -> None:
    """Router arms shared by the spec/RTLLM/verilog-eval runners."""
    parser.add_argument("--router", choices=ROUTERS, default="cascade")
    parser.add_argument("--decider", choices=["keyword", "llm"], default="keyword")
    parser.add_argument(
        "--confidence-tau",
        type=float,
        default=0.5,
        help="LLM-feature confidence below which Tier-0 says 'uncertain'.",
    )


def add_shared_pipeline_args(parser: argparse.ArgumentParser) -> None:
    """LLM/planner/legacy pipeline flags shared by the spec/RTLLM/verilog-eval runners
    (same names and defaults as run_agentic_plan_legacy_realbench.py)."""
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
        help="Enable agentic tool calling in the planner (see the RealBench runner for why this is off).",
    )
    parser.set_defaults(planner_enable_tools=False)
    parser.add_argument("--planner-tool-choice", default="auto")
    parser.add_argument("--target-hdl", default="verilog")
    parser.add_argument("--no-planner-inject-catalog", dest="planner_inject_catalog", action="store_false")
    parser.add_argument("--no-planner-ground-reuse", dest="planner_ground_reuse", action="store_false")
    parser.add_argument("--no-planner-completeness-gate", dest="planner_completeness_gate", action="store_false")
    parser.add_argument("--planner-max-catalog-entries", type=int, default=60)
    parser.set_defaults(planner_inject_catalog=True, planner_ground_reuse=True, planner_completeness_gate=True)
    parser.add_argument("--planner-hierarchical", dest="planner_hierarchical", action="store_true")
    parser.set_defaults(planner_hierarchical=False)
    parser.add_argument("--planner-max-depth", type=int, default=2)
    parser.add_argument(
        "--planner-search-mode",
        choices=["token", "semantic", "hybrid"],
        default="token",
        help="Backend for the planner's search_reuse_ip tool over the catalog.",
    )
    parser.add_argument("--embedder", default="auto")
    parser.add_argument("--planner-retrieval-min-score", type=float, default=0.70)
    parser.add_argument("--planner-retrieval-below-threshold", choices=["flag", "drop"], default="flag")
    parser.add_argument("--reindex", action="store_true")

    parser.add_argument("--repair-cache", choices=["off", "run"], default="off")
    parser.add_argument("--repair-cache-path")
    parser.add_argument("--repair-cache-evidence-threshold", type=float, default=0.85)
    parser.add_argument("--repair-cache-reuse-threshold", type=float, default=0.95)
    parser.add_argument("--repair-cache-max-hint-chars", type=int, default=1800)

    parser.add_argument("--spec-condense-threshold-chars", type=int, default=200000)
    parser.add_argument("--spec-condense-threshold-tokens", type=int, default=45000)
    parser.add_argument("--condense-chunk-chars", type=int, default=30000)
    parser.add_argument("--verbatim-excerpt-max-chars", type=int, default=24000)

    parser.add_argument("--legacy-max-tokens", type=int, default=80000)
    parser.add_argument("--legacy-max-repair-attempts", type=int, default=6)
    parser.add_argument(
        "--legacy-functional-repair",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run --test-command in-loop after syntax passes and add functional repair turns. "
        "Requires --test-command.",
    )
    parser.add_argument("--legacy-max-functional-repair-attempts", type=int, default=4)
    parser.add_argument("--legacy-repair-spec-slice", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--legacy-repair-spec-slice-max-chars", type=int, default=24000)
    parser.add_argument("--legacy-spec-max-chars", type=int, default=120000)
    parser.add_argument("--candidate-solution-max-chars", type=int, default=8000)
    parser.add_argument("--candidate-total-budget-chars", type=int, default=80000)
    parser.add_argument("--reuse-signature-max-chars", type=int, default=3000)
    parser.add_argument("--legacy-prompt-budget-chars", type=int, default=200000)
    parser.add_argument(
        "--legacy-skip-internal-verify",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip the internal Verilator lint that drives the syntax repair loop.",
    )

    parser.add_argument("--verilator-bin", default="verilator")
    parser.add_argument("--agent-timeout-s", type=int, default=120)
    parser.add_argument("--verification-timeout-s", type=int, default=300)


# --------------------------------------------------------------------------- #
# Task / catalog construction
# --------------------------------------------------------------------------- #


def load_spec_text(spec_arg: str) -> str:
    if spec_arg == "-":
        return sys.stdin.read()
    return Path(spec_arg).read_text(encoding="utf-8")


def make_task(args: argparse.Namespace) -> SpecTask:
    prompt = load_spec_text(args.spec).strip()
    if not prompt:
        raise ValueError(f"spec is empty: {args.spec}")
    stem = "spec" if args.spec == "-" else Path(args.spec).stem
    top_module = args.top_module or safe_name(stem)
    extra = tuple(Path(item).resolve() for item in args.extra_source)
    for path in extra:
        if not path.is_file():
            raise FileNotFoundError(f"--extra-source not found: {path}")
    return SpecTask(name=top_module, top_module=top_module, prompt=prompt, extra_sources=extra)


def extra_source_catalog_ip(task: SpecTask, path: Path) -> Dict[str, Any]:
    code = path.read_text(encoding="utf-8", errors="ignore")
    name = path.stem
    signature = module_signature(code) or name
    return {
        "ip_id": name,
        "name": name,
        "summary": f"User-supplied RTL source {name}. Signature: {signature}",
        "category": "dependency",
        "interfaces": [signature],
        "parameters": {},
        "license": "user supplied",
        "verification": ["user supplied"],
        "synthesis": "Verilog/SystemVerilog source supplied by the user",
        "documentation": f"Source path: {path}",
        "tags": ["spec", "dependency", name],
        "behavior": code,
        "integration_notes": [
            f"Use or instantiate this source when implementing {task.top_module}.",
            f"Original file: {path}",
        ],
        "known_limits": [],
    }


def build_catalog(task: SpecTask, args: argparse.Namespace, output_dir: Path) -> Tuple[Path, List[Dict[str, Any]]]:
    """Merge the user catalog (if any) with auto-cataloged --extra-source files and
    write the per-task catalog the planner searches, mirroring build_task_catalog."""
    ips: List[Dict[str, Any]] = []
    if args.catalog:
        payload = json.loads(Path(args.catalog).read_text(encoding="utf-8"))
        ips.extend(payload.get("ips", []))
    known_ids = {str(ip.get("ip_id")) for ip in ips}
    for path in task.extra_sources:
        if path.stem not in known_ids:
            ips.append(extra_source_catalog_ip(task, path))
    catalog_path = output_dir / "catalogs" / f"{task.task_id}.json"
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(json.dumps({"ips": ips}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return catalog_path, ips


def provided_module_names(task: SpecTask) -> List[str]:
    """Module names declared by --extra-source files; the candidate must never
    re-declare them (they are compiled alongside it)."""
    names: set[str] = set()
    for path in task.extra_sources:
        names.update(MODULE_DECL_RE.findall(path.read_text(encoding="utf-8", errors="ignore")))
    names.discard(task.top_module)
    return sorted(names)


def reuse_module_signatures(task: SpecTask, catalog_ips: Sequence[Dict[str, Any]]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """(provided-by-compile-environment, must-be-inlined) module signatures.

    Extra-source modules are provided; catalog-only IPs must be inlined into the output.
    """
    provided: Dict[str, str] = {}
    for path in task.extra_sources:
        code = path.read_text(encoding="utf-8", errors="ignore")
        provided[path.stem] = module_signature(code) or f"module {path.stem}(/* see original source */);"
    inline: Dict[str, str] = {}
    provided_stems = set(provided)
    for ip in catalog_ips:
        name = str(ip.get("name") or ip.get("ip_id") or "")
        if not name or name in provided_stems:
            continue
        behavior = str(ip.get("behavior") or "")
        interfaces = ip.get("interfaces") or []
        signature = module_signature(behavior) or (str(interfaces[0]) if interfaces else f"module {name}(...);")
        inline[name] = signature
    return provided, inline


def task_constraints(task: SpecTask) -> List[str]:
    return [
        f"Final RTL must implement exactly one public top module named {task.top_module}.",
        "Use the exact port names, directions, widths, parameters, reset polarity, and behavior from the specification.",
        "Do not include a testbench, markdown fences, or explanatory prose in generated RTL.",
        "Use catalog IP when it matches required submodule behavior; generate glue or new RTL only where needed.",
    ]


def build_environment_notes(
    task: SpecTask,
    provided_signatures: Dict[str, str],
    inline_signatures: Dict[str, str],
) -> List[str]:
    notes: List[str] = []
    if provided_signatures:
        names = ", ".join(sorted(provided_signatures))
        notes.append(
            f"These modules are compiled from their own source files alongside your code: {names}. "
            "Instantiate them as needed but never re-declare any of them (fatal duplicate-declaration error)."
        )
    else:
        notes.append(
            "No external source files are compiled alongside your code; every module you "
            "instantiate must be fully declared in your output."
        )
    if inline_signatures:
        names = ", ".join(sorted(inline_signatures))
        notes.append(
            f"These reuse-catalog modules are NOT provided by the compile environment; if you use "
            f"one, include its full implementation in your output: {names}. Copy the original "
            "source from the reuse plan candidates verbatim when available."
        )
    return notes


# --------------------------------------------------------------------------- #
# Spec condensation (same chunk&merge path as the RealBench runner)
# --------------------------------------------------------------------------- #


def effective_spec(task: SpecTask, args: argparse.Namespace, output_dir: Path) -> SpecBundle:
    if (
        len(task.prompt) <= args.spec_condense_threshold_chars
        and estimate_tokens(task.prompt) <= args.spec_condense_threshold_tokens
    ):
        return SpecBundle(planner=task.prompt, generation=task.prompt, condensed=False)
    cache_dir = output_dir / "condensed_specs"
    planner_path = cache_dir / f"{task.task_id}.planner.txt"
    generation_path = cache_dir / f"{task.task_id}.generation.txt"
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
        provided_modules=provided_module_names(task),
        excerpt_max_chars=args.verbatim_excerpt_max_chars,
    )
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
        f"[spec] condensed spec for {task.task_id}: {len(task.prompt)} chars -> "
        f"planner {len(bundle.planner)} / generation {len(bundle.generation)} chars"
    )
    return bundle


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #


def build_lint_verifier(task: SpecTask, args: argparse.Namespace) -> Any:
    if args.legacy_skip_internal_verify:
        return NoopVerifier()
    return WorkspaceLintVerifier(
        verilator_bin=args.verilator_bin,
        timeout_s=args.agent_timeout_s,
        extra_sources=list(task.extra_sources),
    )


class CommandFunctionalVerifier:
    """Run the user ``--test-command`` on candidate RTL and report a functional verdict.

    The command is formatted with ``{rtl}`` (path to the candidate file, written next to
    copies of the extra sources), ``{sources}`` (space-joined extra-source paths), and
    ``{top}``. Exit code 0 means the functional check passed. Duck-typed for the legacy
    agent's functional repair loop: ``verify_functional``.
    """

    def __init__(self, task: SpecTask, args: argparse.Namespace) -> None:
        self.task = task
        self.test_command = args.test_command
        self.timeout_s = args.verification_timeout_s

    def verify_functional(self, rtl: str, top_module: str | None = None) -> FunctionalReport:
        import shlex
        import shutil
        import subprocess
        import tempfile

        code = normalize_code(rtl, self.task.top_module)
        if not code:
            return FunctionalReport(
                function_passed=False,
                syntax_ok=False,
                error="functional verification skipped: empty candidate RTL",
            )
        code = strip_dependency_redeclarations(code, provided_module_names(self.task))
        try:
            with tempfile.TemporaryDirectory(prefix="spec_func_") as temp_name:
                temp_dir = Path(temp_name)
                for source in self.task.extra_sources:
                    shutil.copy2(source, temp_dir / source.name)
                rtl_path = temp_dir / f"{self.task.top_module}.gen.sv"
                rtl_path.write_text(code, encoding="utf-8")
                command_text = self.test_command.format(
                    rtl=str(rtl_path),
                    sources=" ".join(str(temp_dir / source.name) for source in self.task.extra_sources),
                    top=self.task.top_module,
                )
                completed = subprocess.run(
                    shlex.split(command_text),
                    cwd=temp_dir,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_s,
                )
        except subprocess.TimeoutExpired as exc:
            return FunctionalReport(function_passed=False, error=f"test command timeout: {exc}")
        except (KeyError, IndexError) as exc:
            return FunctionalReport(function_passed=False, error=f"bad --test-command placeholder: {exc}")
        except Exception as exc:  # noqa: BLE001
            return FunctionalReport(function_passed=False, error=f"test command failed to run: {exc}")
        passed = completed.returncode == 0
        tail = (completed.stdout or "")[-4000:]
        return FunctionalReport(
            function_passed=passed,
            function_info="" if passed else ((completed.stderr or "")[-2000:] or tail or "test command failed"),
            stdout_tail=tail,
        )


@dataclass
class EvalResult:
    syntax: int
    function: Optional[int]
    syntax_info: str = ""
    function_info: str = ""
    error: Optional[str] = None

    @property
    def passed(self) -> bool:
        return self.syntax == 1 and self.function != 0


def evaluate_code(task: SpecTask, code: str, args: argparse.Namespace) -> EvalResult:
    """Final scoring: Verilator lint (syntax), then the optional --test-command (function).
    ``function`` is None when no test command is configured (syntax-only run)."""
    lint = WorkspaceLintVerifier(
        verilator_bin=args.verilator_bin,
        timeout_s=args.verification_timeout_s,
        extra_sources=list(task.extra_sources),
    ).verify(code, top_module=task.top_module)
    syntax = 1 if lint.syntax_passed else 0
    syntax_info = "" if lint.syntax_passed else "\n".join(
        (diag.stderr or diag.stdout or "")[-4000:] for diag in lint.diagnostics
    ).strip()
    if syntax != 1:
        return EvalResult(syntax=0, function=0 if args.test_command else None, syntax_info=syntax_info)
    if not args.test_command:
        return EvalResult(syntax=1, function=None)
    report = CommandFunctionalVerifier(task, args).verify_functional(code, task.top_module)
    return EvalResult(
        syntax=1,
        function=1 if report.function_passed else 0,
        function_info=report.function_info,
        error=report.error,
    )


# --------------------------------------------------------------------------- #
# Flows
# --------------------------------------------------------------------------- #


def direct_prompt(task: SpecTask, spec_text: str, args: argparse.Namespace) -> str:
    constraints = "\n".join(f"- {item}" for item in task_constraints(task))
    provided = provided_module_names(task)
    provided_note = ""
    if provided:
        provided_note = (
            "\nThe compile environment already provides these modules from their own source files. "
            "Do not redeclare them in your output: " + ", ".join(provided) + "."
        )
    problem = clip_text(spec_text, args.legacy_spec_max_chars)
    return f"""Generate the requested RTL implementation directly from the specification.

Target HDL: {args.target_hdl}
Required public top module name: {task.top_module}

Constraints:
{constraints}
- Return only synthesizable RTL source code.
- Do not include a testbench, explanations, markdown fences, or analysis text.
{provided_note}

### Specification
{problem}
"""


def run_direct_flow(
    task: SpecTask,
    spec_bundle: SpecBundle,
    args: argparse.Namespace,
    workspace_dir: Path,
    catalog_ips: Sequence[Dict[str, Any]],
    llm_client: Any,
    repair_cache: Any = None,
    functional_verifier: Any = None,
) -> Tuple[str, Any]:
    """Single-shot generation, then the shared syntax/lint (+ optional functional)
    repair loops via agent.repair_rtl(plan=None) -- the direct flow skips planning,
    not repair. Returns (code, legacy_result_or_None). ``functional_verifier`` lets
    benchmark runners inject their testbench-backed verifier; without one, the
    user --test-command (if any) is used."""
    message = llm_client.chat(
        [{"role": "user", "content": direct_prompt(task, spec_bundle.generation, args)}],
        temperature=args.temperature,
        max_tokens=args.legacy_max_tokens,
    )
    code = normalize_code(str(message.get("content") or ""), task.top_module)
    legacy_result: Any = None
    functional_repair = bool(
        args.legacy_functional_repair
        and (functional_verifier is not None or getattr(args, "test_command", None))
    )
    if not functional_repair:
        functional_verifier = None
    if code and (args.legacy_max_repair_attempts > 0 or functional_repair):
        verifier = build_lint_verifier(task, args)
        if functional_repair and functional_verifier is None:
            functional_verifier = CommandFunctionalVerifier(task, args)
        agent = make_legacy_agent(
            args,
            verifier,
            LegacyConfig(
                target_hdl="verilog",
                max_repair_attempts=args.legacy_max_repair_attempts,
                temperature=args.temperature,
                max_tokens=args.legacy_max_tokens,
                enable_functional_repair=functional_repair,
                max_functional_repair_attempts=args.legacy_max_functional_repair_attempts,
                enable_repair_spec_slice=args.legacy_repair_spec_slice,
                repair_spec_slice_max_chars=args.legacy_repair_spec_slice_max_chars,
            ),
            repair_cache=repair_cache,
            functional_verifier=functional_verifier,
            llm_client=llm_client,
        )
        provided_signatures, inline_signatures = reuse_module_signatures(task, catalog_ips)
        provided_signatures = {
            name: clip_text(signature, args.reuse_signature_max_chars)
            for name, signature in provided_signatures.items()
        }
        legacy_result = agent.repair_rtl(
            code,
            plan=None,
            target_hdl="verilog",
            top_module=task.top_module,
            workspace_dir=workspace_dir,
            original_spec=clip_text(spec_bundle.generation, max(20000, args.legacy_repair_spec_slice_max_chars)),
            reuse_modules=provided_signatures,
            environment_notes=build_environment_notes(task, provided_signatures, inline_signatures),
        )
        code = normalize_code(legacy_result.rtl, task.top_module)
    return code, legacy_result


def make_planner_repository(catalog_path: Path, args: argparse.Namespace, rag: Dict[str, Any]) -> Any:
    repository = JsonIpRepository(catalog_path)
    if args.planner_search_mode == "token" or rag.get("store") is None or rag.get("embedder") is None:
        return repository
    return SemanticIpRepository(
        inner=repository,
        retriever=Retriever(rag["store"], rag["embedder"]),
        reranker=LexicalReranker(),
        mode=args.planner_search_mode,
        min_score=args.planner_retrieval_min_score,
        below_threshold=args.planner_retrieval_below_threshold,
    )


def run_planner(
    task: SpecTask,
    catalog_path: Path,
    args: argparse.Namespace,
    output_dir: Path,
    spec_text: str,
    repository: Any,
) -> PlanningResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    executor = AgentToolExecutor(repository, output_dir)
    llm = PlanningVllmClient(
        base_url=args.base_url or os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
        model=args.model or os.getenv("VLLM_MODEL", "siliconmind-server"),
        api_key=args.api_key or os.getenv("VLLM_API_KEY", "EMPTY"),
        timeout_s=args.llm_timeout_s,
    )
    agent_config = AgentConfig(
        temperature=args.temperature,
        max_tokens=args.planner_max_tokens,
        use_tools=args.planner_enable_tools,
        tool_choice=args.planner_tool_choice,
        max_steps=args.planner_max_steps,
        inject_catalog=args.planner_inject_catalog,
        ground_reuse_decisions=args.planner_ground_reuse,
        completeness_gate=args.planner_completeness_gate,
        max_catalog_entries=args.planner_max_catalog_entries,
    )
    design_task = DesignTask(
        prompt=spec_text,
        target_hdl=args.target_hdl,
        constraints=task_constraints(task),
        known_interfaces=[],
        ppa_targets=[],
    )
    if args.planner_hierarchical:
        h_agent = HierarchicalAgent(
            llm_client=llm,
            base_executor=executor,
            agent_config=agent_config,
            h_config=HierarchicalConfig(max_depth=args.planner_max_depth),
        )
        h_plan = h_agent.run(design_task)
        h_plan.write_hierarchical_summary(output_dir)
        return h_plan.result
    agent = PlanningAgent(llm_client=llm, tool_executor=executor, config=agent_config)
    return agent.run(design_task)


def enrich_legacy_plan_with_sources(
    plan: IpReusePlan,
    catalog_ips: Sequence[Dict[str, Any]],
    task: SpecTask,
    args: argparse.Namespace,
) -> None:
    """Attach the selected catalog IPs' source (``behavior``) to the plan candidates the
    legacy generator embeds, mirroring the RealBench enrichment. Extra-source-backed IPs
    (compile-environment-provided) are truncated to the per-candidate budget; catalog-only
    IPs must be reproduced verbatim, so they are never truncated."""
    by_key: Dict[str, Dict[str, Any]] = {}
    for ip in catalog_ips:
        for key in (str(ip.get("ip_id") or ""), str(ip.get("name") or "")):
            if key:
                by_key[key] = ip
    provided_stems = {path.stem for path in task.extra_sources}
    selected_count = sum(
        1 for decision in plan.decisions if decision.selected_doc_id and by_key.get(decision.selected_doc_id)
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
        ip = by_key.get(decision.selected_doc_id)
        if ip is None:
            continue
        name = str(ip.get("name") or ip.get("ip_id"))
        code = str(ip.get("behavior") or "")
        if name in provided_stems and len(code) > per_candidate_cap:
            code = (
                code[:per_candidate_cap]
                + "\n// ... truncated; the complete file is supplied separately in the compile environment ..."
            )
        for candidate in decision.candidates:
            if candidate.doc_id != decision.selected_doc_id:
                continue
            candidate.problem = (
                f"Reusable source {name} for module {decision.module.name}. "
                f"Documentation: {str(ip.get('documentation') or '')[:200]}"
            )
            candidate.solution = code
            candidate.tags = ["spec", "reuse", name]
            candidate.metadata.update({"source_kind": "dependency" if name in provided_stems else "catalog"})
            break


def run_pipeline_flow(
    task: SpecTask,
    plan_payload: Dict[str, Any],
    spec_bundle: SpecBundle,
    args: argparse.Namespace,
    workspace_dir: Path,
    catalog_ips: Sequence[Dict[str, Any]],
    repair_cache: Any = None,
    functional_verifier: Any = None,
) -> Tuple[str, Any]:
    """Legacy RTL generation + repair from the agentic plan (same construction as the
    RealBench runner's run_legacy_generator). Returns (code, legacy_result).
    ``functional_verifier`` lets benchmark runners inject their testbench-backed
    verifier; without one, the user --test-command (if any) is used."""
    legacy_plan = agentic_plan_from_payload(plan_payload)
    enrich_legacy_plan_with_sources(legacy_plan, catalog_ips, task, args)
    verifier = build_lint_verifier(task, args)
    functional_repair = bool(
        args.legacy_functional_repair
        and (functional_verifier is not None or getattr(args, "test_command", None))
    )
    if not functional_repair:
        functional_verifier = None
    elif functional_verifier is None:
        functional_verifier = CommandFunctionalVerifier(task, args)
    agent = make_legacy_agent(
        args,
        verifier,
        LegacyConfig(
            target_hdl="verilog",
            max_repair_attempts=args.legacy_max_repair_attempts,
            temperature=args.temperature,
            max_tokens=args.legacy_max_tokens,
            enable_functional_repair=functional_repair,
            max_functional_repair_attempts=args.legacy_max_functional_repair_attempts,
            enable_repair_spec_slice=args.legacy_repair_spec_slice,
            repair_spec_slice_max_chars=args.legacy_repair_spec_slice_max_chars,
        ),
        repair_cache=repair_cache,
        functional_verifier=functional_verifier,
    )
    provided_signatures, inline_signatures = reuse_module_signatures(task, catalog_ips)
    provided_signatures = {
        name: clip_text(signature, args.reuse_signature_max_chars)
        for name, signature in provided_signatures.items()
    }
    signatures_chars = sum(len(name) + len(sig) + 8 for name, sig in provided_signatures.items())
    spec_cap = min(
        args.legacy_spec_max_chars,
        max(
            20000,
            args.legacy_prompt_budget_chars - signatures_chars - args.candidate_total_budget_chars - 30000,
        ),
    )
    legacy_result = agent.run_from_plan(
        legacy_plan,
        target_hdl="verilog",
        top_module=task.top_module,
        workspace_dir=workspace_dir,
        original_spec=clip_text(spec_bundle.generation, spec_cap),
        reuse_modules=provided_signatures,
        environment_notes=build_environment_notes(task, provided_signatures, inline_signatures),
    )
    return normalize_code(legacy_result.rtl, task.top_module), legacy_result


# --------------------------------------------------------------------------- #
# Routing + per-sample driver
# --------------------------------------------------------------------------- #


def tier0_route(task: SpecTask, args: argparse.Namespace, output_dir: Path) -> Dict[str, Any]:
    """Tier-0 decision for the whole run (per-task, spec-only), same arms as
    run_realbench_routed.plan_routing. Returns flow/probe/meta."""
    router = args.router
    if router == "all_pipeline":
        return {"flow": "pipeline", "probe": False, "routed_by": "none", "route_decision": "pipeline", "route_features": None}
    if router == "all_direct":
        return {"flow": "direct", "probe": False, "routed_by": "none", "route_decision": "direct", "route_features": None}
    if router == "plan_probe":
        return {"flow": "pipeline", "probe": True, "routed_by": "plan_probe", "route_decision": None, "route_features": None}
    feature_client = None
    if args.decider == "llm":
        feature_client = PlanningVllmClient(
            base_url=args.base_url or os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
            model=args.model or os.getenv("VLLM_MODEL", "siliconmind-server"),
            api_key=args.api_key or os.getenv("VLLM_API_KEY", "EMPTY"),
            timeout_s=args.llm_timeout_s,
        )
    cache_dir = output_dir / "routing" / "cache"
    decision, feats = routing.route_pre(
        task.prompt, args.decider, feature_client, cache_dir, force=(router == "pre")
    )
    if router == "pre" or decision != "uncertain":
        return {
            "flow": decision,
            "probe": False,
            "routed_by": f"pre_{args.decider}",
            "route_decision": decision,
            "route_features": feats.to_dict(),
        }
    # cascade + uncertain -> pipeline with the Tier-1 plan-probe enabled.
    return {"flow": "pipeline", "probe": True, "routed_by": "plan_probe", "route_decision": None, "route_features": feats.to_dict()}


def generate_with_router(
    task: SpecTask,
    sample: int,
    args: argparse.Namespace,
    output_dir: Path,
    catalog_path: Path,
    catalog_ips: Sequence[Dict[str, Any]],
    route: Dict[str, Any],
    rag: Dict[str, Any],
    functional_verifier: Any = None,
) -> Dict[str, Any]:
    """Router-directed generation for one sample: direct flow, or planner ->
    (optional Tier-1 plan-probe) -> legacy pipeline / direct bail-out. Returns the
    code plus flow/plan/repair metadata. Shared by the spec, RTLLM, and verilog-eval
    runners; the caller injects the benchmark's functional verifier (if any)."""
    plan_dir = output_dir / "plans" / task.task_id / f"sample{sample:02d}"
    plan_report_path = plan_dir / "agent_result.json"
    legacy_workspace = output_dir / "legacy_workspaces" / task.task_id / f"sample{sample:02d}"
    legacy_report_path = output_dir / "legacy_reports" / f"{task.task_id}_sample{sample:02d}.json"
    legacy_report_path.parent.mkdir(parents=True, exist_ok=True)

    flow_taken = route["flow"]
    route_decision = route["route_decision"] or route["flow"]
    routed_by = route["routed_by"]
    planner_result: Optional[PlanningResult] = None
    legacy_result: Any = None

    spec_bundle = effective_spec(task, args, output_dir)
    if route["flow"] == "direct":
        code, legacy_result = run_direct_flow(
            task, spec_bundle, args, legacy_workspace, catalog_ips,
            llm_client=make_legacy_llm(args), repair_cache=rag.get("repair_cache"),
            functional_verifier=functional_verifier,
        )
    else:
        repository = make_planner_repository(catalog_path, args, rag)
        planner_result = run_planner(task, catalog_path, args, plan_dir, spec_bundle.planner, repository)
        plan_report_path.write_text(dumps_plan_result(planner_result), encoding="utf-8")
        if route["probe"]:
            routed_by = "plan_probe"
            route_decision = routing.decide_plan(planner_result.to_dict())
        if route_decision == "direct":
            flow_taken = "direct"
            code, legacy_result = run_direct_flow(
                task, spec_bundle, args, legacy_workspace, catalog_ips,
                llm_client=make_legacy_llm(args), repair_cache=rag.get("repair_cache"),
                functional_verifier=functional_verifier,
            )
        else:
            route_decision = "pipeline"
            code, legacy_result = run_pipeline_flow(
                task, planner_result.to_dict(), spec_bundle, args, legacy_workspace,
                catalog_ips, repair_cache=rag.get("repair_cache"),
                functional_verifier=functional_verifier,
            )
    if legacy_result is not None:
        legacy_report_path.write_text(dumps_json(legacy_result.to_dict(), indent=2), encoding="utf-8")
    if code:
        code = strip_dependency_redeclarations(code, provided_module_names(task))
    return {
        "code": code,
        "flow": flow_taken,
        "route_decision": route_decision,
        "routed_by": routed_by,
        "spec_condensed": spec_bundle.condensed,
        "planner_result": planner_result,
        "legacy_result": legacy_result,
        "plan_dir": plan_dir,
        "plan_report_path": plan_report_path,
        "legacy_report_path": legacy_report_path,
        "legacy_workspace": legacy_workspace,
        "wasted_plan": bool(routed_by == "plan_probe" and route_decision == "direct" and planner_result is not None),
    }


def run_one(
    task: SpecTask,
    sample: int,
    args: argparse.Namespace,
    output_dir: Path,
    catalog_path: Path,
    catalog_ips: Sequence[Dict[str, Any]],
    route: Dict[str, Any],
    rag: Dict[str, Any],
) -> Dict[str, Any]:
    code_path = output_dir / "generated" / task.task_id / f"sample{sample:02d}.sv"
    code_path.parent.mkdir(parents=True, exist_ok=True)
    plan_dir = output_dir / "plans" / task.task_id / f"sample{sample:02d}"
    plan_report_path = plan_dir / "agent_result.json"
    legacy_workspace = output_dir / "legacy_workspaces" / task.task_id / f"sample{sample:02d}"
    legacy_report_path = output_dir / "legacy_reports" / f"{task.task_id}_sample{sample:02d}.json"
    legacy_report_path.parent.mkdir(parents=True, exist_ok=True)
    report = output_dir / "reports" / f"{task.task_id}_sample{sample:02d}.json"
    report.parent.mkdir(parents=True, exist_ok=True)

    code = ""
    generation_error: Optional[str] = None
    reused_existing = False
    wall_s = 0.0
    planner_result: Optional[PlanningResult] = None
    legacy_result: Any = None
    flow_taken = route["flow"]
    route_decision = route["route_decision"] or route["flow"]
    routed_by = route["routed_by"]
    spec_condensed = False

    reset_planning_request_log()
    reset_legacy_request_log()

    if args.resume and code_path.exists():
        code = code_path.read_text(encoding="utf-8")
        reused_existing = True
    else:
        t0 = time.perf_counter()
        try:
            outcome = generate_with_router(task, sample, args, output_dir, catalog_path, catalog_ips, route, rag)
            code = outcome["code"]
            spec_condensed = outcome["spec_condensed"]
            planner_result = outcome["planner_result"]
            legacy_result = outcome["legacy_result"]
            flow_taken = outcome["flow"]
            route_decision = outcome["route_decision"]
            routed_by = outcome["routed_by"]
        except Exception as exc:  # noqa: BLE001 - keep remaining samples moving.
            generation_error = f"{exc}\n{traceback.format_exc()[-4000:]}"
        wall_s = time.perf_counter() - t0
        if code:
            code_path.write_text(code, encoding="utf-8")

    if code:
        eval_result = evaluate_code(task, code, args)
    else:
        eval_result = EvalResult(
            syntax=0,
            function=0 if args.test_command else None,
            error=generation_error or "generation produced empty code",
        )
    request_log = sorted(
        get_planning_request_log() + get_legacy_request_log(),
        key=lambda entry: entry.get("start_epoch") or 0.0,
    )
    record = {
        "benchmark": "spec",
        "pipeline": "agentic_ip_reuse_plan_to_ip_reuse_legacy_rtl",
        "task": task.name,
        "sample": sample,
        "top_module": task.top_module,
        "spec_path": args.spec,
        "catalog_path": str(catalog_path),
        "catalog_doc_count": len(catalog_ips),
        "extra_sources": [str(path) for path in task.extra_sources],
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
        "legacy_functional_repair": bool(args.legacy_functional_repair and args.test_command),
        "legacy_functional_repair_attempts": (
            getattr(legacy_result, "functional_repair_attempts", None) if legacy_result else None
        ),
        "planner_search_mode": args.planner_search_mode,
        "spec_chars": len(task.prompt),
        "spec_condensed": spec_condensed,
        "router": args.router,
        "decider": args.decider,
        "flow": flow_taken,
        "route_decision": route_decision,
        "routed_by": routed_by,
        "route_features": route["route_features"],
        "wasted_plan": bool(routed_by == "plan_probe" and route_decision == "direct" and planner_result is not None),
        "syntax": eval_result.syntax,
        "function": eval_result.function,
        "passed": eval_result.passed,
        "syntax_info": eval_result.syntax_info,
        "function_info": eval_result.function_info,
        "evaluation_error": eval_result.error,
        "wall_s": wall_s,
        "llm_request_log": request_log,
        "llm_latency_s": round(sum(float(r.get("latency_s") or 0) for r in request_log), 4),
    }
    report.write_text(dumps_json(record, indent=2), encoding="utf-8")
    return record


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #


def build_rag(args: argparse.Namespace, output_dir: Path, catalog_path: Path) -> Dict[str, Any]:
    """Optional semantic-search index over the catalog + optional run-scope repair cache."""
    rag: Dict[str, Any] = {"embedder": None, "embedder_name": "none", "store": None, "repair_cache": None}
    if args.planner_search_mode == "token" and args.repair_cache == "off":
        return rag
    embedder, embedder_name = make_embedder_with_fallback(args.embedder, warn=lambda message: print(f"[spec] {message}"))
    rag["embedder"] = LockedEmbedder(embedder)
    rag["embedder_name"] = embedder_name
    if args.planner_search_mode != "token":
        store, meta = build_or_load_task_index(catalog_path, rag["embedder"], embedder_name, args.reindex)
        rag["store"] = store
        print(f"[spec] index: docs={meta['doc_count']} embedder={embedder_name} rebuilt={meta.get('index_built', False)}")
    if args.repair_cache == "run":
        cache_path = Path(args.repair_cache_path) if args.repair_cache_path else output_dir / "repair_cache.json"
        rag["repair_cache"] = make_repair_cache(args, rag["embedder"], cache_path)
    return rag


def summarize(records: Sequence[Dict[str, Any]], args: argparse.Namespace, elapsed_s: float) -> Dict[str, Any]:
    total = len(records)
    functional_scored = [record for record in records if record.get("function") is not None]
    return {
        "benchmark": "spec",
        "pipeline": "agentic_ip_reuse_plan_to_ip_reuse_legacy_rtl",
        "router": args.router,
        "decider": args.decider,
        "num_records": total,
        "generated": sum(1 for record in records if record.get("generated")),
        "syntax": sum(1 for record in records if record.get("syntax") == 1),
        "function": sum(1 for record in functional_scored if record.get("function") == 1),
        "passed": sum(1 for record in records if record.get("passed")),
        "syntax_rate": safe_rate(sum(1 for record in records if record.get("syntax") == 1), total),
        "function_rate": safe_rate(
            sum(1 for record in functional_scored if record.get("function") == 1), len(functional_scored)
        ),
        "flows": {
            "direct": sum(1 for record in records if record.get("flow") == "direct"),
            "pipeline": sum(1 for record in records if record.get("flow") == "pipeline"),
        },
        "wasted_plans": sum(1 for record in records if record.get("wasted_plan")),
        "functional_check": bool(args.test_command),
        "total_s": elapsed_s,
        "total_llm_latency_s": round(sum(float(record.get("llm_latency_s") or 0.0) for record in records), 4),
    }


def run_spec(args: argparse.Namespace) -> Dict[str, Any]:
    if args.legacy_functional_repair and not args.test_command:
        print("[spec] --legacy-functional-repair has no effect without --test-command", file=sys.stderr)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    task = make_task(args)
    catalog_path, catalog_ips = build_catalog(task, args, output_dir)
    print(f"[spec] task {task.task_id}: spec {len(task.prompt)} chars, catalog docs={len(catalog_ips)}")

    route = tier0_route(task, args, output_dir)
    (output_dir / "routing").mkdir(parents=True, exist_ok=True)
    (output_dir / "routing" / "plan.json").write_text(
        dumps_json({"router": args.router, "decider": args.decider, "task": task.name, **route}, indent=2),
        encoding="utf-8",
    )
    probe_note = " (+plan-probe)" if route["probe"] else ""
    print(f"[spec] route: {route['flow']}{probe_note} via {route['routed_by']}")

    rag = build_rag(args, output_dir, catalog_path)

    records: List[Dict[str, Any]] = []
    samples = list(range(1, max(args.samples, 1) + 1))
    with ThreadPoolExecutor(max_workers=max(args.concurrency, 1)) as executor:
        futures = {
            executor.submit(run_one, task, sample, args, output_dir, catalog_path, catalog_ips, route, rag): sample
            for sample in samples
        }
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            status = "PASS" if record["passed"] else "FAIL"
            function_text = record["function"] if record["function"] is not None else "n/a"
            print(
                f"[spec] {status} {record['task']} sample {int(record['sample']):02d} "
                f"flow={record['flow']} syntax={record['syntax']} function={function_text}"
            )

    elapsed_s = time.perf_counter() - start
    records.sort(key=lambda record: int(record.get("sample") or 0))
    with (output_dir / "records.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(dumps_json(record) + "\n")
    summary = summarize(records, args, elapsed_s)
    (output_dir / "summary.json").write_text(dumps_json(summary, indent=2), encoding="utf-8")
    print(f"[spec] wrote results under {output_dir}")
    return summary


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    summary = run_spec(args)
    print(dumps_json(summary, indent=2))


if __name__ == "__main__":
    main()
