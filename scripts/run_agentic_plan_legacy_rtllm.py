#!/usr/bin/env python3
"""RTLLM benchmark through the router + two-stage agentic-plan/legacy-RTL pipeline.

Same router and pipeline as scripts/run_agentic_plan_legacy_realbench.py: Tier-0
spec pre-route (``--router cascade|pre|plan_probe|all_pipeline|all_direct``,
``--decider keyword|llm``, versioned decision rule via ``--route-rule
v1|v2-20b|v2-120b``, default ``v2-20b`` = the wrap-cleaned pipeline-default rule,
validated with ``--decider llm``), Tier-1 plan-probe on the generated plan, direct
flow with the shared repair loops (agent.repair_rtl(plan=None)), and the
plan-driven legacy generator for pipeline-routed tasks.

Benchmark collateral is reused verbatim from scripts/run_rtllm_eval.py: problem
discovery (design_description.txt + testbench.v + verified_*.v), top-module
inference, and the final iverilog compile+simulate scoring with its passfail
classification and pass@k. RTLLM problems have no reuse dependencies, so the
planner runs over an empty catalog. With ``--legacy-functional-repair`` the RTLLM
testbench itself drives in-loop functional repair (same testbench that scores the
benchmark -- report as a separate experimental arm, like the RealBench runner).

Usage::

    python scripts/run_agentic_plan_legacy_rtllm.py --samples 5 --concurrency 8
    python scripts/run_agentic_plan_legacy_rtllm.py --router cascade --decider llm \
        --legacy-functional-repair --include adder --samples 3
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
PLANNER_ROOT = REPO_ROOT / "agentic_ip_reuse"
for _path in (str(PLANNER_ROOT), str(REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from agentic_ip_reuse.llm import (
    get_request_log as get_planning_request_log,
    reset_request_log as reset_planning_request_log,
)
from rag_rtl.json_utils import dumps_json
from rag_rtl.llm import (
    get_request_log as get_legacy_request_log,
    reset_request_log as reset_legacy_request_log,
)

from scripts.run_agentic_plan_legacy_realbench import safe_rate
from scripts.run_agentic_plan_legacy_spec import (
    FunctionalReport,
    SpecTask,
    add_router_args,
    add_shared_pipeline_args,
    build_rag,
    generate_with_router,
    tier0_route,
)
from scripts.run_rtllm_eval import (
    PASS_AT_KS,
    RtlLmProblem,
    SimulationResult,
    WorkItem,
    compute_pass_at,
    discover_problems,
    evaluate_with_iverilog,
    format_summary_metric,
    generated_code_path,
    iter_work_items,
    normalize_generated_code,
    output_problem_dir,
    simulation_log_path,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run RTLLM through the cascade router + agentic-plan/legacy-RTL pipeline, "
        "then evaluate each generated design with its RTLLM testbench."
    )
    parser.add_argument("--rtllm-root", default="/home/kai/eval_dt/RTLLM")
    parser.add_argument("--output-dir", default="runs/agentic_plan_legacy_rtllm")
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Case-insensitive substring filter for problem id or category; repeatable.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--resume", action="store_true", help="Reuse existing generated .v files in output-dir")
    parser.add_argument("--evaluate-only", action="store_true", help="Skip generation and evaluate existing .v files")
    parser.add_argument("--dry-run", action="store_true", help="Only discover problems and print routing decisions")
    parser.add_argument("--iverilog-bin", default="iverilog")
    parser.add_argument("--simulation-timeout-s", type=int, default=30)
    parser.add_argument("--keep-waveforms", action="store_true")
    add_router_args(parser)
    add_shared_pipeline_args(parser)
    return parser


def to_spec_task(problem: RtlLmProblem) -> SpecTask:
    return SpecTask(
        name=f"{problem.category}__{problem.problem_id}",
        top_module=problem.top_module,
        prompt=problem.prompt,
        extra_sources=(),
    )


class RtllmFunctionalVerifier:
    """Run the problem's RTLLM testbench on candidate RTL inside the repair loop.

    Delegates to the same ``evaluate_with_iverilog`` path that scores the benchmark,
    so in-loop and final verdicts agree on identical code. Duck-typed for the legacy
    agent: ``verify_functional``."""

    def __init__(self, problem: RtlLmProblem, args: argparse.Namespace) -> None:
        self.problem = problem
        self.args = args

    def verify_functional(self, rtl: str, top_module: str | None = None) -> FunctionalReport:
        code = normalize_generated_code(rtl, self.problem.top_module)
        if not code:
            return FunctionalReport(
                function_passed=False,
                syntax_ok=False,
                error="functional verification skipped: empty candidate RTL",
            )
        with tempfile.TemporaryDirectory(prefix="rtllm_func_") as temp_name:
            temp_dir = Path(temp_name)
            candidate = temp_dir / f"{self.problem.top_module}.v"
            candidate.write_text(code, encoding="utf-8")
            item = WorkItem(problem=self.problem, sample=0)
            result = evaluate_with_iverilog(item, candidate, temp_dir / "sim.log", self.args)
        return FunctionalReport(
            function_passed=result.passed,
            function_info=(
                ""
                if result.passed
                else f"passfail={result.passfail} failures={result.failures}\n" + result.stdout[-2000:]
            ),
            syntax_ok=result.compile_returncode == 0,
            stdout_tail=result.stdout[-4000:],
            error=result.error,
        )


def run_one(
    item: WorkItem,
    args: argparse.Namespace,
    output_dir: Path,
    catalog_path: Path,
    route: Dict[str, Any],
    rag: Dict[str, Any],
) -> Dict[str, Any]:
    problem = item.problem
    task = to_spec_task(problem)
    output_problem_dir(output_dir, problem).mkdir(parents=True, exist_ok=True)
    code_path = generated_code_path(output_dir, item)
    sim_log = simulation_log_path(output_dir, item)

    code = ""
    generation_error: Optional[str] = None
    reused_existing = False
    wall_s = 0.0
    outcome: Dict[str, Any] = {}

    reset_planning_request_log()
    reset_legacy_request_log()

    if (args.resume or args.evaluate_only) and code_path.exists():
        code = code_path.read_text(encoding="utf-8")
        reused_existing = True
    elif args.evaluate_only:
        generation_error = f"missing generated file: {code_path}"
    else:
        t0 = time.perf_counter()
        try:
            functional_verifier = RtllmFunctionalVerifier(problem, args) if args.legacy_functional_repair else None
            outcome = generate_with_router(
                task, item.sample, args, output_dir, catalog_path, [], route, rag,
                functional_verifier=functional_verifier,
            )
            code = normalize_generated_code(outcome["code"], problem.top_module)
        except Exception as exc:  # noqa: BLE001 - keep the benchmark moving.
            generation_error = f"{exc}\n{traceback.format_exc()[-4000:]}"
        wall_s = time.perf_counter() - t0
        if code:
            code_path.write_text(code, encoding="utf-8")

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
    else:
        sim_result = evaluate_with_iverilog(item, code_path, sim_log, args)

    planner_result = outcome.get("planner_result")
    legacy_result = outcome.get("legacy_result")
    request_log = sorted(
        get_planning_request_log() + get_legacy_request_log(),
        key=lambda entry: entry.get("start_epoch") or 0.0,
    )
    return {
        "benchmark": "rtllm",
        "pipeline": "agentic_ip_reuse_plan_to_ip_reuse_legacy_rtl",
        "problem": problem.problem_id,
        "category": problem.category,
        "sample": item.sample,
        "top_module": problem.top_module,
        "description_path": str(problem.description_path),
        "testbench_path": str(problem.testbench_path),
        "generated_code_path": str(code_path),
        "simulation_log_path": str(sim_log),
        "generated": bool(code),
        "generation_error": generation_error,
        "reused_existing": reused_existing,
        "router": args.router,
        "decider": args.decider,
        "route_rule": route.get("route_rule"),
        "flow": outcome.get("flow", route["flow"]),
        "route_decision": outcome.get("route_decision", route["route_decision"] or route["flow"]),
        "routed_by": outcome.get("routed_by", route["routed_by"]),
        "route_features": route.get("route_features"),
        "wasted_plan": bool(outcome.get("wasted_plan")),
        "spec_condensed": bool(outcome.get("spec_condensed")),
        "plan_report_path": str(outcome["plan_report_path"]) if outcome.get("plan_report_path") and Path(outcome["plan_report_path"]).exists() else None,
        "legacy_report_path": str(outcome["legacy_report_path"]) if outcome.get("legacy_report_path") and Path(outcome["legacy_report_path"]).exists() else None,
        "planner_steps": planner_result.steps if planner_result else None,
        "legacy_repair_attempts": legacy_result.repair_attempts if legacy_result else None,
        "legacy_functional_repair": bool(args.legacy_functional_repair),
        "legacy_functional_repair_attempts": (
            getattr(legacy_result, "functional_repair_attempts", None) if legacy_result else None
        ),
        "passed": sim_result.passed,
        "passfail": sim_result.passfail,
        "compile_returncode": sim_result.compile_returncode,
        "simulation_returncode": sim_result.simulation_returncode,
        "failures": sim_result.failures,
        "stdout_tail": sim_result.stdout[-4000:],
        "stderr_tail": sim_result.stderr[-4000:],
        "evaluation_error": sim_result.error,
        "wall_s": wall_s,
        "llm_request_log": request_log,
        "llm_latency_s": round(sum(float(r.get("latency_s") or 0) for r in request_log), 4),
    }


def summarize(records: Sequence[Dict[str, Any]], args: argparse.Namespace, elapsed_s: float) -> Dict[str, Any]:
    count = len(records)
    denom = max(count, 1)
    passfail_counts: Dict[str, int] = {}
    for record in records:
        key = str(record.get("passfail") or "?")
        passfail_counts[key] = passfail_counts.get(key, 0) + 1
    pass_at_rates, pass_at_denominators = compute_pass_at(records, PASS_AT_KS)
    return {
        "benchmark": "rtllm",
        "pipeline": "agentic_ip_reuse_plan_to_ip_reuse_legacy_rtl",
        "router": args.router,
        "decider": args.decider,
        "route_rule": getattr(args, "route_rule", None),
        "num_records": count,
        "num_problems": len({record["problem"] for record in records}),
        "samples_per_problem": args.samples,
        "generated": sum(1 for record in records if record.get("generated")),
        "iverilog_compiled": sum(1 for record in records if record.get("compile_returncode") == 0),
        "passed": sum(1 for record in records if record.get("passed")),
        "accuracy": sum(1 for record in records if record.get("passed")) / denom,
        "pass@1": pass_at_rates[1],
        "pass@3": pass_at_rates[3],
        "pass@5": pass_at_rates[5],
        "pass_at_denominators": {str(k): pass_at_denominators[k] for k in PASS_AT_KS},
        "passfail_counts": dict(sorted(passfail_counts.items())),
        "flows": {
            "direct": sum(1 for record in records if record.get("flow") == "direct"),
            "pipeline": sum(1 for record in records if record.get("flow") == "pipeline"),
        },
        "per_flow_pass": {
            flow: safe_rate(
                sum(1 for record in records if record.get("flow") == flow and record.get("passed")),
                sum(1 for record in records if record.get("flow") == flow),
            )
            for flow in ("direct", "pipeline")
        },
        "wasted_plans": sum(1 for record in records if record.get("wasted_plan")),
        "total_s": elapsed_s,
        "total_llm_latency_s": round(sum(float(record.get("llm_latency_s") or 0.0) for record in records), 4),
    }


def plan_routes(
    problems: Sequence[RtlLmProblem], args: argparse.Namespace, output_dir: Path
) -> Dict[str, Dict[str, Any]]:
    """Tier-0 route per problem (samples of one problem share the decision, like the
    RealBench routed runner); evaluate-only skips routing entirely."""
    routes: Dict[str, Dict[str, Any]] = {}
    if args.evaluate_only:
        placeholder = {"flow": "pipeline", "probe": False, "routed_by": "none", "route_decision": None, "route_features": None}
        return {to_spec_task(problem).task_id: dict(placeholder) for problem in problems}
    for problem in problems:
        task = to_spec_task(problem)
        routes[task.task_id] = tier0_route(task, args, output_dir)
    return routes


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    problems = discover_problems(args.rtllm_root, args.include, limit=args.limit)
    work_items = list(iter_work_items(problems, args.samples))
    print(f"[rtllm] router={args.router} decider={args.decider} rule={args.route_rule} discovered {len(problems)} problem(s), {len(work_items)} work item(s)")

    routes = plan_routes(problems, args, output_dir)
    (output_dir / "routing").mkdir(parents=True, exist_ok=True)
    (output_dir / "routing" / "plan.json").write_text(
        dumps_json({"router": args.router, "decider": args.decider, "route_rule": args.route_rule, "routes": routes}, indent=2),
        encoding="utf-8",
    )
    flows = [route["flow"] for route in routes.values()]
    probed = sum(1 for route in routes.values() if route.get("probe"))
    print(f"[rtllm] routed: direct={flows.count('direct')} pipeline={flows.count('pipeline')} (probe on {probed})")

    if args.dry_run:
        for problem in problems[:20]:
            route = routes[to_spec_task(problem).task_id]
            print(f"  {problem.problem_id}: top={problem.top_module} -> {route['flow']}"
                  + (" (+probe)" if route.get("probe") else ""))
        return

    catalog_path = output_dir / "catalogs" / "empty.json"
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text('{"ips": []}\n', encoding="utf-8")
    rag = build_rag(args, output_dir, catalog_path)

    records: List[Dict[str, Any]] = []
    records_path = output_dir / "records.jsonl"
    records_path.write_text("", encoding="utf-8")
    records_lock = threading.Lock()
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(args.concurrency, 1)) as executor:
        futures = [
            executor.submit(run_one, item, args, output_dir, catalog_path, routes[to_spec_task(item.problem).task_id], rag)
            for item in work_items
        ]
        for future in as_completed(futures):
            record = future.result()
            with records_lock:
                records.append(record)
                with records_path.open("a", encoding="utf-8") as handle:
                    handle.write(dumps_json(record) + "\n")
            print(
                f"[rtllm] {record['passfail']} {record['problem']} sample {int(record['sample']):02d} "
                f"flow={record['flow']} passed={record['passed']}"
            )

    records.sort(key=lambda item: (item["category"], item["problem"], item["sample"]))
    with records_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(dumps_json(record) + "\n")
    summary = summarize(records, args, time.perf_counter() - start)
    (output_dir / "summary.json").write_text(dumps_json(summary, indent=2), encoding="utf-8")
    print(
        f"[rtllm] accuracy={summary['accuracy']:.4f} "
        f"pass@1={format_summary_metric(summary['pass@1'])} "
        f"pass@5={format_summary_metric(summary['pass@5'])} "
        f"passed={summary['passed']}/{summary['num_records']} "
        f"summary={output_dir / 'summary.json'}"
    )


if __name__ == "__main__":
    main()
