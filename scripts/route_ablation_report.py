#!/usr/bin/env python3
"""Cross-arm comparison table for the routing ablation.

Reads several arm run dirs (each a ``runs/<arm>/`` with ``records.jsonl``) produced by
run_realbench_routed.py / run_realbench_combined.py and emits one markdown table on the
**cost-at-fixed-quality** axis: combined pass@k (quality), planner+repair compute (cost),
and routing accuracy vs the golden oracle labels (A->direct and B->pipeline are the misroutes).

Usage::

    python scripts/route_ablation_report.py runs/arm_all_pipeline runs/arm_cascade runs/arm_oracle
    python scripts/route_ablation_report.py runs/arm_* --k 1 5 10 -o routing/ablation.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.module_pass_rate_report import pass_at_k  # HumanEval estimator


def load_records(run_dir: Path) -> List[Dict[str, Any]]:
    path = run_dir / "records.jsonl"
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def task_key(record: Dict[str, Any]):
    return (record.get("task_level", ""), record.get("system", ""), record.get("task", ""))


def combined_pass_at_k(records: Sequence[Dict[str, Any]], ks: Sequence[int]) -> Dict[int, float]:
    """Mean per-task pass@k over the pooled samples (a task's samples may span both flows)."""
    by_task: Dict[Any, List[Dict[str, Any]]] = {}
    for r in records:
        by_task.setdefault(task_key(r), []).append(r)
    out: Dict[int, float] = {}
    for k in ks:
        vals = []
        for samples in by_task.values():
            n = len(samples)
            c = sum(1 for s in samples if s.get("passed"))
            if n >= k:
                vals.append(pass_at_k(n, c, k))
        out[k] = sum(vals) / len(vals) if vals else 0.0
    return out


def _num(record: Dict[str, Any], key: str) -> float:
    value = record.get(key)
    return float(value) if isinstance(value, (int, float)) else 0.0


def arm_row(name: str, records: Sequence[Dict[str, Any]], ks: Sequence[int]) -> Dict[str, Any]:
    n = len(records)
    pak = combined_pass_at_k(records, ks)
    pipeline = sum(1 for r in records if r.get("flow") == "pipeline")
    direct = sum(1 for r in records if r.get("flow") == "direct")
    plans = sum(1 for r in records if r.get("plan_report_path") or r.get("planner_steps") is not None or r.get("routed_by") == "plan_probe")
    wasted = sum(1 for r in records if r.get("wasted_plan"))
    tokens = sum(_num(r, "llm_token_estimate") for r in records)
    wall = sum(_num(r, "wall_s") for r in records)
    # Correct routing: A(wrapper)->pipeline, B(self-contained)->direct.
    # Misroutes (each loses a would-be pass): A->direct and B->pipeline.
    mis_a = sum(1 for r in records if r.get("oracle_label") == "A" and r.get("flow") == "direct")
    mis_b = sum(1 for r in records if r.get("oracle_label") == "B" and r.get("flow") == "pipeline")
    return {
        "arm": name,
        "samples": n,
        "pass_at_k": pak,
        "pipeline": pipeline,
        "direct": direct,
        "plans": plans,
        "wasted_plans": wasted,
        "tokens": tokens,
        "wall_s": wall,
        "misroute_A_to_direct": mis_a,
        "misroute_B_to_pipeline": mis_b,
    }


def render(rows: List[Dict[str, Any]], ks: Sequence[int]) -> str:
    pak_cols = " | ".join(f"pass@{k}" for k in ks)
    lines = [
        "# Routing ablation",
        "",
        "The flows are complementary: wrapper/integration modules (A) pass under the pipeline,",
        "self-contained algorithmic modules (B) pass under direct. Correct routing = A→pipeline,",
        "B→direct; **both** misroutes (A→direct, B→pipeline) lose a would-be pass. Headline: reach",
        "the union pass@k (combined baseline) at the lowest compute while keeping misroutes near 0.",
        "",
        f"| arm | samples | pipe/direct | {pak_cols} | plans | wasted | tokens | wall_s | A→direct✗ | B→pipe✗ |",
        "|---|--:|--:|" + "--:|" * len(ks) + "--:|--:|--:|--:|--:|--:|",
    ]
    for r in rows:
        pak = " | ".join(f"{100 * r['pass_at_k'][k]:.1f}" for k in ks)
        lines.append(
            f"| {r['arm']} | {r['samples']} | {r['pipeline']}/{r['direct']} | {pak} | "
            f"{r['plans']} | {r['wasted_plans']} | {r['tokens']:.0f} | {r['wall_s']:.0f} | "
            f"{r['misroute_A_to_direct']} | {r['misroute_B_to_pipeline']} |"
        )
    lines += [
        "",
        "Reading: `pre_keyword` vs `pre_llm` = value of the LLM feature extractor; `pre_llm` vs",
        "`cascade` = value of the plan-probe; `cascade` vs `oracle` = remaining gap (magnitude residual).",
    ]
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare routing-ablation arms on cost vs quality.")
    parser.add_argument("run_dirs", nargs="+", help="Arm run directories (each with records.jsonl).")
    parser.add_argument("--k", nargs="+", type=int, default=[1, 5, 10], help="pass@k values.")
    parser.add_argument("-o", "--output", help="Write the markdown report here (default: stdout).")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    rows = []
    for run_dir in args.run_dirs:
        path = Path(run_dir)
        records = load_records(path)
        if not records:
            print(f"[ablation] warning: no records in {path}")
            continue
        rows.append(arm_row(path.name, records, args.k))
    report = render(rows, args.k)
    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
        print(f"[ablation] wrote {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
