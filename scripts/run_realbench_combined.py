#!/usr/bin/env python3
"""Combined RealBench flow: split the sample budget across the two-stage agentic
pipeline and the direct-model baseline, then score the pooled samples together.

If you ask for ``--samples 20``, this runs the agentic pipeline for 10 samples and
the direct model for 10 samples (configurable split), writes each flow's artifacts
under its own subdirectory, and finally merges every generated sample into one pool
to report a single combined pass rate (and pass@k over the union of both flows).

Every flag that this script does not own is forwarded verbatim to BOTH underlying
runners, so pipeline-only knobs (``--legacy-functional-repair``, ``--repair-cache``,
``--planner-search-mode`` ...) keep working; flags a runner does not recognise are
ignored for that runner.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rag_rtl.json_utils import dumps_json

from scripts.run_agentic_plan_legacy_realbench import build_parser as build_pipeline_parser
from scripts.run_agentic_plan_legacy_realbench import (
    record_sort_key,
    run_realbench,
    safe_rate,
    write_records,
    write_solution_jsonl,
)
from scripts.run_realbench_direct_model import build_parser as build_direct_parser
from scripts.run_realbench_direct_model import run_realbench_direct


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the agentic pipeline and the direct-model baseline on a split sample "
            "budget, then compute the pass rate over the pooled samples."
        ),
        epilog=(
            "Any flag not listed here is passed through to both underlying runners "
            "(unknown flags are ignored per-runner)."
        ),
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=20,
        help="Total samples per task, split across the two flows (default: 20).",
    )
    parser.add_argument(
        "--pipeline-samples",
        type=int,
        help="Override the pipeline sample count (default: ceil(samples/2)).",
    )
    parser.add_argument(
        "--direct-samples",
        type=int,
        help="Override the direct-model sample count (default: floor(samples/2)).",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/realbench_combined",
        help="Parent directory; pipeline/ and direct/ subdirs hold each flow's run.",
    )
    parser.add_argument(
        "--solution-name",
        default="combined",
        help="Solution name for the merged samples/ jsonl set.",
    )
    parser.add_argument(
        "--skip-pipeline",
        action="store_true",
        help="Only run the direct model (still scores via the combined pool).",
    )
    parser.add_argument(
        "--skip-direct",
        action="store_true",
        help="Only run the agentic pipeline (still scores via the combined pool).",
    )
    parser.add_argument(
        "--parallel-flows",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run the pipeline and direct-model flows concurrently instead of one "
            "after the other. Both flows hit the served model at the same time, so "
            "their effective load is the sum of each flow's --concurrency."
        ),
    )
    return parser


def split_samples(args: argparse.Namespace) -> Tuple[int, int]:
    """Resolve how many samples each flow runs."""
    if args.pipeline_samples is not None or args.direct_samples is not None:
        pipe = args.pipeline_samples if args.pipeline_samples is not None else 0
        direct = args.direct_samples if args.direct_samples is not None else 0
    else:
        pipe = (args.samples + 1) // 2  # ceil -> pipeline gets the extra one
        direct = args.samples // 2
    if args.skip_pipeline:
        pipe = 0
    if args.skip_direct:
        direct = 0
    return pipe, direct


def _sub_args(
    build_sub_parser, forwarded: Sequence[str], samples: int, output_dir: Path, solution_name: str
) -> argparse.Namespace:
    """Build a fully-defaulted Namespace for one underlying runner.

    Forwarded flags are parsed leniently so that flags meant for the *other* runner
    are ignored rather than fatal.
    """
    overrides = [
        "--samples",
        str(samples),
        "--output-dir",
        str(output_dir),
        "--solution-name",
        solution_name,
    ]
    sub_args, _unknown = build_sub_parser().parse_known_args(list(forwarded) + overrides)
    return sub_args


def read_records(records_path: Path) -> List[Dict[str, Any]]:
    if not records_path.exists():
        return []
    records: List[Dict[str, Any]] = []
    with records_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _flow_pass_label(record: Dict[str, Any]) -> str:
    return str(record.get("pipeline") or "unknown")


def task_key(record: Dict[str, Any]) -> Tuple[str, str, str]:
    return (
        str(record.get("task_level") or ""),
        str(record.get("system") or ""),
        str(record.get("task") or ""),
    )


def merge_records(
    pipeline_records: Sequence[Dict[str, Any]],
    direct_records: Sequence[Dict[str, Any]],
    pipeline_samples: int,
) -> List[Dict[str, Any]]:
    """Pool both flows into a single sample space (direct sample ids are offset so
    every (task, sample) pair across the union is unique)."""
    merged: List[Dict[str, Any]] = []
    for record in pipeline_records:
        clone = dict(record)
        clone["flow"] = "pipeline"
        merged.append(clone)
    for record in direct_records:
        clone = dict(record)
        clone["flow"] = "direct"
        clone["sample"] = int(record.get("sample") or 0) + pipeline_samples
        merged.append(clone)
    return merged


def _rate_block(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(records)
    syntax = sum(1 for r in records if r.get("syntax") == 1)
    function = sum(1 for r in records if r.get("function") == 1)
    passed = sum(1 for r in records if r.get("passed"))
    return {
        "num_records": total,
        "syntax": syntax,
        "function": function,
        "passed": passed,
        "syntax_rate": safe_rate(syntax, total),
        "function_rate": safe_rate(function, total),
        "pass_rate": safe_rate(passed, total),
    }


def _solved_tasks(records: Sequence[Dict[str, Any]]) -> set:
    return {task_key(r) for r in records if r.get("passed")}


def combined_summary(
    merged: Sequence[Dict[str, Any]],
    pipeline_records: Sequence[Dict[str, Any]],
    direct_records: Sequence[Dict[str, Any]],
    pipeline_samples: int,
    direct_samples: int,
) -> Dict[str, Any]:
    all_keys = {task_key(r) for r in merged}
    num_tasks = len(all_keys)

    pipe_solved = _solved_tasks(pipeline_records)
    direct_solved = _solved_tasks(direct_records)
    combined_solved = pipe_solved | direct_solved

    per_task: List[Dict[str, Any]] = []
    for key in sorted(all_keys):
        level, system, task = key
        pipe_pass = sum(
            1 for r in pipeline_records if task_key(r) == key and r.get("passed")
        )
        direct_pass = sum(
            1 for r in direct_records if task_key(r) == key and r.get("passed")
        )
        per_task.append(
            {
                "task_level": level,
                "system": system,
                "task": task,
                "pipeline_pass": pipe_pass,
                "direct_pass": direct_pass,
                "combined_pass": pipe_pass + direct_pass,
                "solved": bool(pipe_pass or direct_pass),
            }
        )

    return {
        "benchmark": "realbench",
        "flow": "combined_pipeline_plus_direct",
        "samples_requested": pipeline_samples + direct_samples,
        "pipeline_samples": pipeline_samples,
        "direct_samples": direct_samples,
        "num_tasks": num_tasks,
        "combined": {
            **_rate_block(merged),
            "solved_tasks": len(combined_solved),
            "pass_at_k": safe_rate(len(combined_solved), num_tasks),
        },
        "per_flow": {
            "pipeline": {
                **_rate_block(pipeline_records),
                "solved_tasks": len(pipe_solved),
                "pass_at_k": safe_rate(len(pipe_solved), num_tasks),
            },
            "direct": {
                **_rate_block(direct_records),
                "solved_tasks": len(direct_solved),
                "pass_at_k": safe_rate(len(direct_solved), num_tasks),
            },
        },
        "per_task": per_task,
    }


def run_combined(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    parser = build_parser()
    args, forwarded = parser.parse_known_args(argv)
    pipeline_samples, direct_samples = split_samples(args)

    if pipeline_samples <= 0 and direct_samples <= 0:
        parser.error("nothing to run: both pipeline and direct sample counts are 0")

    output_dir = Path(args.output_dir)
    pipeline_dir = output_dir / "pipeline"
    direct_dir = output_dir / "direct"
    output_dir.mkdir(parents=True, exist_ok=True)

    mode = "concurrently" if args.parallel_flows else "sequentially"
    print(
        f"[combined] sample budget {pipeline_samples + direct_samples} ({mode}): "
        f"pipeline={pipeline_samples} direct={direct_samples}"
    )

    def run_pipeline_flow() -> List[Dict[str, Any]]:
        print(f"[combined] === running agentic pipeline ({pipeline_samples} samples/task) ===")
        pipe_args = _sub_args(
            build_pipeline_parser, forwarded, pipeline_samples, pipeline_dir, args.solution_name
        )
        run_realbench(pipe_args)
        return read_records(pipeline_dir / "records.jsonl")

    def run_direct_flow() -> List[Dict[str, Any]]:
        print(f"[combined] === running direct model ({direct_samples} samples/task) ===")
        direct_args = _sub_args(
            build_direct_parser, forwarded, direct_samples, direct_dir, args.solution_name
        )
        run_realbench_direct(direct_args)
        return read_records(direct_dir / "records.jsonl")

    flows = []
    if pipeline_samples > 0:
        flows.append(("pipeline", run_pipeline_flow))
    if direct_samples > 0:
        flows.append(("direct", run_direct_flow))

    results: Dict[str, List[Dict[str, Any]]] = {"pipeline": [], "direct": []}
    if args.parallel_flows and len(flows) > 1:
        with ThreadPoolExecutor(max_workers=len(flows)) as executor:
            futures = {label: executor.submit(fn) for label, fn in flows}
            for label, future in futures.items():
                results[label] = future.result()
    else:
        for label, fn in flows:
            results[label] = fn()

    pipeline_records = results["pipeline"]
    direct_records = results["direct"]

    merged = merge_records(pipeline_records, direct_records, pipeline_samples)
    write_records(output_dir / "records.jsonl", merged)
    write_solution_jsonl(output_dir, args.solution_name, merged)

    summary = combined_summary(
        sorted(merged, key=record_sort_key),
        pipeline_records,
        direct_records,
        pipeline_samples,
        direct_samples,
    )
    (output_dir / "summary.json").write_text(dumps_json(summary, indent=2), encoding="utf-8")
    print(f"[combined] wrote pooled results under {output_dir}")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    summary = run_combined(argv)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
