#!/usr/bin/env python3
"""Routed RealBench flow: decide per task whether to run the cheap ``direct`` model or
the full agentic ``pipeline``, instead of running both on every task.

Routing arms (``--router``):

* ``all_pipeline`` / ``all_direct`` -- baselines (every task to one flow).
* ``pre``        -- Tier-0 only: spec features (``--decider {keyword,llm}``) -> decide_pre,
                    ``uncertain`` resolved by size fallback. No plan generated for direct tasks.
* ``plan_probe`` -- every task enters the pipeline; the per-sample plan-probe (Tier-1) downgrades
                    wrapper/thin samples to direct (the discarded plan is the only waste).
* ``cascade``    -- Tier-0 sends confident A->direct and confident B->pipeline; ``uncertain``
                    tasks enter the pipeline with the plan-probe enabled. (main proposal)
* ``oracle``     -- route by golden ``own_cells`` labels (routing ceiling; eval only).

Forwarded flags (model, --concurrency, --legacy-functional-repair, ...) pass through to the
underlying runners, exactly like run_realbench_combined.py.

Usage::

    python scripts/run_realbench_routed.py --router cascade --decider llm --samples 10
    python scripts/run_realbench_routed.py --router oracle --route-labels routing/route_labels.json --samples 10
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
PLANNER_ROOT = REPO_ROOT / "agentic_ip_reuse"
for _path in (str(PLANNER_ROOT), str(REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from agentic_ip_reuse.llm import VllmClient as FeatureClient
from rag_rtl import routing
from rag_rtl.json_utils import dumps_json

from scripts.run_agentic_plan_legacy_realbench import (
    build_parser as build_pipeline_parser,
    discover_tasks,
    record_sort_key,
    run_realbench,
    safe_rate,
    write_records,
    write_solution_jsonl,
)
from scripts.run_realbench_combined import _rate_block, _sub_args, read_records, task_key
from scripts.run_realbench_direct_model import build_parser as build_direct_parser, run_realbench_direct

ROUTERS = ["all_pipeline", "all_direct", "pre", "plan_probe", "cascade", "oracle"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Route each RealBench task to direct or pipeline, then score the pooled samples.",
        epilog="Any flag not listed here is forwarded to both underlying runners (unknown flags ignored per-runner).",
    )
    parser.add_argument("--router", choices=ROUTERS, default="cascade")
    parser.add_argument("--decider", choices=["keyword", "llm"], default="keyword")
    parser.add_argument("--samples", type=int, default=10, help="Samples per task (each task runs in one flow).")
    parser.add_argument("--output-dir", default="runs/realbench_routed")
    parser.add_argument("--solution-name", default="routed")
    parser.add_argument("--realbench-root", default="/home/kai/eval_dt/real_bench")
    parser.add_argument("--route-labels", default=str(REPO_ROOT / "routing" / "route_labels.json"),
                        help="Golden A/B label table (compute_route_labels.py); drives the oracle arm and scoring.")
    parser.add_argument("--confidence-tau", type=float, default=0.5, help="LLM-feature confidence below which Tier-0 says 'uncertain'.")
    return parser


def load_labels(path: str | Path) -> Dict[str, str]:
    p = Path(path)
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    return {task: info.get("label", "B") for task, info in data.get("labels", {}).items()}


def make_feature_client(pipe_args: argparse.Namespace) -> FeatureClient:
    return FeatureClient(
        base_url=pipe_args.base_url or os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
        model=pipe_args.model or os.getenv("VLLM_MODEL", "siliconmind-server"),
        api_key=pipe_args.api_key or os.getenv("VLLM_API_KEY", "EMPTY"),
        timeout_s=pipe_args.llm_timeout_s,
    )


def plan_routing(
    args: argparse.Namespace,
    tasks: Sequence[Any],
    labels: Dict[str, str],
    feature_client: Optional[FeatureClient],
    cache_dir: Path,
) -> Tuple[List[str], List[str], List[str], Dict[str, Dict[str, Any]]]:
    """Return (direct_tasks, pipeline_tasks, probe_tasks, per-task routing metadata)."""
    direct_tasks: List[str] = []
    pipeline_tasks: List[str] = []
    probe_tasks: List[str] = []
    meta: Dict[str, Dict[str, Any]] = {}
    router = args.router

    for task in tasks:
        name = task.task
        if router == "all_pipeline":
            pipeline_tasks.append(name)
            meta[name] = {"routed_by": "none", "route_decision": "pipeline", "route_features": None}
        elif router == "all_direct":
            direct_tasks.append(name)
            meta[name] = {"routed_by": "none", "route_decision": "direct", "route_features": None}
        elif router == "plan_probe":
            probe_tasks.append(name)
            meta[name] = {"routed_by": "plan_probe", "route_decision": None, "route_features": None}
        elif router == "oracle":
            decision = "direct" if labels.get(name, "B") == "A" else "pipeline"
            (direct_tasks if decision == "direct" else pipeline_tasks).append(name)
            meta[name] = {"routed_by": "oracle", "route_decision": decision, "route_features": None}
        elif router == "pre":
            decision, feats = routing.route_pre(task.prompt, args.decider, feature_client, cache_dir, force=True)
            (direct_tasks if decision == "direct" else pipeline_tasks).append(name)
            meta[name] = {"routed_by": f"pre_{args.decider}", "route_decision": decision, "route_features": feats.to_dict()}
        elif router == "cascade":
            decision, feats = routing.route_pre(task.prompt, args.decider, feature_client, cache_dir, force=False)
            if decision == "direct":
                direct_tasks.append(name)
            elif decision == "pipeline":
                pipeline_tasks.append(name)
            else:  # uncertain -> pipeline with plan-probe
                probe_tasks.append(name)
            meta[name] = {"routed_by": f"pre_{args.decider}" if decision != "uncertain" else "plan_probe",
                          "route_decision": None if decision == "uncertain" else decision,
                          "route_features": feats.to_dict()}
        else:
            raise ValueError(f"unknown router {router!r}")
    return direct_tasks, pipeline_tasks, probe_tasks, meta


def run_routed(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    parser = build_parser()
    args, forwarded = parser.parse_known_args(argv)
    output_dir = Path(args.output_dir)
    pipeline_dir = output_dir / "pipeline"
    direct_dir = output_dir / "direct"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover the task set once, using a parsed pipeline-args view of the forwarded flags.
    discovery_args = _sub_args(build_pipeline_parser, forwarded, args.samples, pipeline_dir, args.solution_name)
    discovery_args.realbench_root = args.realbench_root
    tasks = discover_tasks(discovery_args)
    print(f"[routed] router={args.router} decider={args.decider} discovered {len(tasks)} task(s)")

    labels = load_labels(args.route_labels)
    cache_dir = output_dir / "routing" / "cache"
    feature_client = (
        make_feature_client(discovery_args)
        if args.router in ("pre", "cascade") and args.decider == "llm"
        else None
    )

    direct_tasks, pipeline_tasks, probe_tasks, meta = plan_routing(args, tasks, labels, feature_client, cache_dir)
    print(f"[routed] direct={len(direct_tasks)} pipeline={len(pipeline_tasks)} probe={len(probe_tasks)}")

    # Persist the routing plan for auditing.
    (output_dir / "routing").mkdir(parents=True, exist_ok=True)
    (output_dir / "routing" / "plan.json").write_text(
        dumps_json({"router": args.router, "decider": args.decider, "meta": meta}, indent=2), encoding="utf-8"
    )

    pipeline_records: List[Dict[str, Any]] = []
    direct_records: List[Dict[str, Any]] = []

    agentic_include = pipeline_tasks + probe_tasks
    if agentic_include:
        extra: List[str] = []
        for name in agentic_include:
            extra += ["--include-exact", name]
        if probe_tasks:
            extra += ["--plan-probe"]
            for name in probe_tasks:
                extra += ["--plan-probe-include", name]
        pipe_args = _sub_args(build_pipeline_parser, list(forwarded) + extra, args.samples, pipeline_dir, args.solution_name)
        pipe_args.realbench_root = args.realbench_root
        print(f"[routed] === agentic pipeline on {len(agentic_include)} task(s) ({len(probe_tasks)} probed) ===")
        run_realbench(pipe_args)
        pipeline_records = read_records(pipeline_dir / "records.jsonl")

    if direct_tasks:
        extra = []
        for name in direct_tasks:
            extra += ["--include-exact", name]
        direct_args = _sub_args(build_direct_parser, list(forwarded) + extra, args.samples, direct_dir, args.solution_name)
        direct_args.realbench_root = args.realbench_root
        print(f"[routed] === direct model on {len(direct_tasks)} task(s) ===")
        run_realbench_direct(direct_args)
        direct_records = read_records(direct_dir / "records.jsonl")

    merged = merge_routed(pipeline_records, direct_records, meta, labels)
    write_records(output_dir / "records.jsonl", merged)
    write_solution_jsonl(output_dir, args.solution_name, merged)

    summary = routed_summary(args, merged, labels)
    (output_dir / "summary.json").write_text(dumps_json(summary, indent=2), encoding="utf-8")
    print(f"[routed] wrote routed results under {output_dir}")
    return summary


def merge_routed(
    pipeline_records: Sequence[Dict[str, Any]],
    direct_records: Sequence[Dict[str, Any]],
    meta: Dict[str, Dict[str, Any]],
    labels: Dict[str, str],
) -> List[Dict[str, Any]]:
    """Pool both runners' records, honoring each record's own ``flow`` (set by plan-probe)
    and stamping the Tier-0/oracle routing metadata + the golden oracle label for scoring."""
    merged: List[Dict[str, Any]] = []

    for record in pipeline_records:
        rec = dict(record)
        info = meta.get(rec.get("task", ""), {})
        rec.setdefault("flow", "pipeline")
        # plan-probe records already carry routed_by/route_decision; only stamp Tier-0 meta
        # for records the runner did not route itself.
        if rec.get("routed_by", "none") == "none":
            rec["routed_by"] = info.get("routed_by", "none")
            rec["route_decision"] = info.get("route_decision") or rec.get("flow")
        rec["route_features"] = info.get("route_features")
        rec["oracle_label"] = labels.get(rec.get("task", ""))
        merged.append(rec)

    for record in direct_records:
        rec = dict(record)
        info = meta.get(rec.get("task", ""), {})
        rec["flow"] = "direct"
        rec["routed_by"] = info.get("routed_by", "none")
        rec["route_decision"] = "direct"
        rec["route_features"] = info.get("route_features")
        rec["oracle_label"] = labels.get(rec.get("task", ""))
        merged.append(rec)

    return sorted(merged, key=record_sort_key)


def _num(record: Dict[str, Any], key: str) -> float:
    value = record.get(key)
    return float(value) if isinstance(value, (int, float)) else 0.0


def routed_summary(args: argparse.Namespace, merged: Sequence[Dict[str, Any]], labels: Dict[str, str]) -> Dict[str, Any]:
    pipeline_recs = [r for r in merged if r.get("flow") == "pipeline"]
    direct_recs = [r for r in merged if r.get("flow") == "direct"]
    all_keys = {task_key(r) for r in merged}
    solved = {task_key(r) for r in merged if r.get("passed")}

    # cost
    plans_generated = sum(1 for r in merged if r.get("plan_report_path") or r.get("planner_steps") is not None or r.get("routed_by") == "plan_probe")
    wasted_plans = sum(1 for r in merged if r.get("wasted_plan"))

    # routing confusion vs oracle: rows=oracle label, cols=flow taken
    confusion = {"A->direct": 0, "A->pipeline": 0, "B->direct": 0, "B->pipeline": 0, "unlabeled": 0}
    for r in merged:
        lab = r.get("oracle_label")
        flow = r.get("flow")
        if lab not in ("A", "B"):
            confusion["unlabeled"] += 1
        else:
            confusion[f"{lab}->{flow}"] += 1

    return {
        "benchmark": "realbench",
        "flow": "routed",
        "router": args.router,
        "decider": args.decider,
        "samples": args.samples,
        "num_tasks": len(all_keys),
        "combined": {**_rate_block(merged), "solved_tasks": len(solved), "pass_at_k": safe_rate(len(solved), len(all_keys))},
        "per_flow": {
            "pipeline": {**_rate_block(pipeline_recs), "num_samples": len(pipeline_recs)},
            "direct": {**_rate_block(direct_recs), "num_samples": len(direct_recs)},
        },
        "cost": {
            "total_llm_tokens": sum(_num(r, "llm_token_estimate") for r in merged),
            "total_wall_s": sum(_num(r, "wall_s") for r in merged),
            "plans_generated": plans_generated,
            "wasted_plans": wasted_plans,
            "direct_samples": len(direct_recs),
            "pipeline_samples": len(pipeline_recs),
        },
        "routing_confusion": confusion,
        "dangerous_misroute_B_to_direct": confusion["B->direct"],
        "overspend_A_to_pipeline": confusion["A->pipeline"],
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    summary = run_routed(argv)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
