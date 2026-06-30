#!/usr/bin/env python3
"""Precompute ground-truth A/B routing labels for RealBench tasks (offline, eval only).

For each module-level task it synthesizes the golden RTL (own local cell count) and
applies the structural guard in :func:`rag_rtl.routing.oracle_label`, writing a table
to ``routing/route_labels.json``. This table is the *ceiling* the cascade router is
scored against (and drives the ``oracle`` ablation arm). It reads the golden answer,
so it must never feed an inference-time routing decision.

Usage::

    python scripts/compute_route_labels.py
    python scripts/compute_route_labels.py --realbench-root /path/to/real_bench -o routing/route_labels.json
    python scripts/compute_route_labels.py --include sdc --include aes        # substring filter
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rag_rtl import routing
from rag_rtl.json_utils import dumps_json

DEFAULT_ROOT = "/home/kai/eval_dt/real_bench"


def load_benchmark_info(root: Path):
    """Load the ``benchmark_info``/``system_info`` dicts from the RealBench repo."""
    spec = importlib.util.spec_from_file_location("realbench_benchmark_info", root / "benchmark_info.py")
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module.benchmark_info, getattr(module, "system_info", {})


def family(task: str) -> str:
    if task.startswith("aes"):
        return "aes"
    if task.startswith("sd"):
        return "sdc"
    return "e203_hbirdv2"


def find_module_v(root: Path, module: str) -> Path | None:
    """Resolve a module name to its golden ``.v`` (generic sirv_* live bundled in
    e203_hbirdv2/general and are added wholesale by oracle_label, so they are skipped)."""
    for fam in ("e203_hbirdv2", "aes", "sdc"):
        cand = root / fam / module / f"{module}.v"
        if cand.exists():
            return cand
    return None


def dependency_files(root: Path, task: str, depmap: Dict[str, List[str]]) -> List[Path]:
    """Recursive dependency closure -> resolvable golden ``.v`` files (general/* excluded)."""
    seen: set[str] = set()
    files: List[Path] = []
    stack = list(depmap.get(task, []))
    while stack:
        dep = stack.pop()
        if dep in seen:
            continue
        seen.add(dep)
        path = find_module_v(root, dep)
        if path is not None:
            files.append(path)
        stack.extend(depmap.get(dep, []))
    return files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Precompute golden A/B routing labels for RealBench.")
    parser.add_argument("--realbench-root", default=DEFAULT_ROOT)
    parser.add_argument("-o", "--output", default=str(REPO_ROOT / "routing" / "route_labels.json"))
    parser.add_argument("--include", action="append", default=[], help="Only label tasks whose name contains a filter (repeatable).")
    parser.add_argument("--include-system", action="store_true", help="Also label system-level tasks (slow; synthesizes whole IP).")
    return parser


def main(argv: List[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    root = Path(args.realbench_root)
    benchmark_info, system_info = load_benchmark_info(root)

    # flatten the per-family dependency map (+ system map) into one task -> deps dict
    depmap: Dict[str, List[str]] = {}
    for _family, comps in benchmark_info.items():
        depmap.update(comps)

    tasks: List[str] = [module for comps in benchmark_info.values() for module in comps]
    if args.include_system:
        tasks += list(system_info.keys())
        depmap.update(system_info)
    if args.include:
        tasks = [t for t in tasks if any(f.lower() in t.lower() for f in args.include)]

    labels: Dict[str, Any] = {}
    counts = {"A": 0, "B": 0}
    for task in tasks:
        vpath = find_module_v(root, task)
        if vpath is None:
            print(f"[labels] skip {task}: no golden .v found")
            continue
        deps = dependency_files(root, task, depmap)
        result = routing.oracle_label(task, root, dep_files=deps)
        labels[task] = result
        counts[result["label"]] += 1
        print(f"[labels] {result['label']} {task:28s} own_cells={result['own_cells']} ({result['reason']})")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(dumps_json({"realbench_root": str(root), "labels": labels}, indent=2), encoding="utf-8")
    print(f"\n[labels] wrote {len(labels)} labels (A={counts['A']} B={counts['B']}) -> {out_path}")


if __name__ == "__main__":
    main()
