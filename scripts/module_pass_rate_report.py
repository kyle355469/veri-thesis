#!/usr/bin/env python3
"""Generate a module pass-rate analysis report (like module_pass_rate_analysis.md).

Reads a run's ``records.jsonl`` (the canonical per-sample results), restricts to
module-level tasks, and reports syntax/function pass rates plus pass@k broken
down by family and per module.

pass@k uses the standard order-independent (HumanEval) estimator over the
``n`` samples of each module::

    pass@k = 1 - C(n-c, k) / C(n, k)

where ``c`` is the number of correct samples. Family and overall pass@k values
are the mean of the per-module pass@k. The aggregate "Pass Rate" column is the
pooled count over all samples in the group.

Usage::

    python scripts/module_pass_rate_report.py runs/<run_dir>
    python scripts/module_pass_rate_report.py runs/<run_dir> -o report.md --k 1 5 10
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict

# Pretty display names for known RealBench systems; unknown ones are upper-cased.
FAMILY_DISPLAY = {
    "sdc": "SDC",
    "aes": "AES",
    "e203_hbirdv2": "E203",
}


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased probability that >=1 of k samples (drawn without replacement) is correct."""
    if c <= 0:
        return 0.0
    if k >= n:
        return 1.0 if c > 0 else 0.0
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def load_records(run_dir: str) -> list[dict]:
    path = os.path.join(run_dir, "records.jsonl")
    if not os.path.exists(path):
        sys.exit(f"error: no records.jsonl in {run_dir}")
    records = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def module_stats(records: list[dict]) -> dict[tuple[str, str], dict]:
    """Aggregate per (system, task): sample count + correct counts for syntax/function."""
    by_mod: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"n": 0, "syntax": 0, "function": 0}
    )
    for r in records:
        if r.get("task_level") != "module":
            continue
        key = (r.get("system", "?"), r.get("task", "?"))
        s = by_mod[key]
        s["n"] += 1
        s["syntax"] += int(r.get("syntax") or 0)
        # `function`==1 implies syntax==1 in RealBench; this is the end-to-end pass.
        s["function"] += int(r.get("function") or 0)
    return by_mod


def pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def rate_cell(c: int, n: int) -> str:
    return f"{c}/{n} ({pct(c / n if n else 0)})"


def row_metrics(c: int, n: int, ks: list[int]) -> list[str]:
    """Pass-rate cell + pass@k cells for one metric (syntax or function)."""
    cells = [rate_cell(c, n)]
    cells += [pct(pass_at_k(n, c, k)) for k in ks]
    return cells


def mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def family_passk(mods: list[dict], metric: str, ks: list[int]) -> list[float]:
    """Mean per-module pass@k across a family for a metric."""
    return [mean([pass_at_k(m["n"], m[metric], k) for m in mods]) for k in ks]


def build_report(run_dir: str, ks: list[int]) -> str:
    records = load_records(run_dir)
    by_mod = module_stats(records)
    if not by_mod:
        sys.exit("error: no module-level records found")

    # Group modules by family.
    fams: dict[str, list[dict]] = defaultdict(list)
    for (system, task), s in by_mod.items():
        fams[system].append({"task": task, **s})

    def fam_func_rate(system: str) -> float:
        mods = fams[system]
        tot = sum(m["n"] for m in mods)
        return sum(m["function"] for m in mods) / tot if tot else 0.0

    # Families ordered by function pass rate (desc), like the original report.
    fam_order = sorted(fams, key=lambda s: (-fam_func_rate(s), s))

    ks_label = lambda metric: " | ".join(f"{metric} pass@{k}" for k in ks)
    overall_hdr = (
        f"| Family | Modules | Samples | Syntax Pass Rate | {ks_label('Syntax')} "
        f"| Function Pass Rate | {ks_label('Function')} |"
    )
    sep = "|---|---:|" + "---:|" * (3 + 2 * len(ks))

    src = os.path.join(run_dir, "records.jsonl")
    out: list[str] = []
    out.append("# Module Pass Rate Analysis")
    out.append("")
    out.append(f"Source: `{os.path.abspath(src)}`")
    out.append("")
    out.append(
        "Scope: module tasks only; `system/...` tasks are ignored. pass@k uses the "
        "standard order-independent estimator over the samples of each module; "
        "family and overall pass@k values are averages across modules."
    )
    out.append("")

    # ----- Overall table -----
    out.append("## Overall")
    out.append("")
    out.append(overall_hdr)
    out.append(sep)

    all_mods = [m for mods in fams.values() for m in mods]
    for system in fam_order:
        mods = fams[system]
        n = sum(m["n"] for m in mods)
        c_syn = sum(m["syntax"] for m in mods)
        c_fn = sum(m["function"] for m in mods)
        syn_k = family_passk(mods, "syntax", ks)
        fn_k = family_passk(mods, "function", ks)
        cells = (
            [FAMILY_DISPLAY.get(system, system.upper()), str(len(mods)), str(n)]
            + [rate_cell(c_syn, n)] + [pct(v) for v in syn_k]
            + [rate_cell(c_fn, n)] + [pct(v) for v in fn_k]
        )
        out.append("| " + " | ".join(cells) + " |")

    # "All modules" row: pooled rates, mean per-module pass@k.
    n = sum(m["n"] for m in all_mods)
    c_syn = sum(m["syntax"] for m in all_mods)
    c_fn = sum(m["function"] for m in all_mods)
    syn_k = family_passk(all_mods, "syntax", ks)
    fn_k = family_passk(all_mods, "function", ks)
    cells = (
        ["**All modules**", str(len(all_mods)), str(n)]
        + [rate_cell(c_syn, n)] + [pct(v) for v in syn_k]
        + [rate_cell(c_fn, n)] + [pct(v) for v in fn_k]
    )
    out.append("| " + " | ".join(cells) + " |")
    out.append("")

    # ----- Per-family sections -----
    mod_hdr = (
        f"| Module | Samples | Syntax Pass Rate | {ks_label('Syntax')} "
        f"| Function Pass Rate | {ks_label('Function')} |"
    )
    mod_sep = "|---|---:|" + "---:|" * (2 + 2 * len(ks))
    for system in fam_order:
        out.append(f"## {FAMILY_DISPLAY.get(system, system.upper())} Modules")
        out.append("")
        out.append(mod_hdr)
        out.append(mod_sep)
        # Sort: function rate desc, then syntax rate desc, then name.
        mods = sorted(
            fams[system],
            key=lambda m: (-m["function"] / m["n"], -m["syntax"] / m["n"], m["task"]),
        )
        for m in mods:
            cells = (
                [f"`{m['task']}`", str(m["n"])]
                + row_metrics(m["syntax"], m["n"], ks)
                + row_metrics(m["function"], m["n"], ks)
            )
            out.append("| " + " | ".join(cells) + " |")
        out.append("")

    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", help="run directory containing records.jsonl")
    ap.add_argument(
        "-o", "--output",
        help="output path (default: <run_dir>/module_pass_rate_analysis.md; '-' for stdout)",
    )
    ap.add_argument(
        "--k", type=int, nargs="+", default=[1, 5, 10],
        help="pass@k values to report (default: 1 5 10)",
    )
    args = ap.parse_args()

    report = build_report(args.run_dir.rstrip("/"), sorted(set(args.k)))

    if args.output == "-":
        sys.stdout.write(report)
        return
    out_path = args.output or os.path.join(args.run_dir, "module_pass_rate_analysis.md")
    with open(out_path, "w") as fh:
        fh.write(report)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
