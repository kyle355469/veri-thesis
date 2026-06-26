#!/usr/bin/env python3
"""Generate a pass-rate report from a hybrid combined run's ``runtime.log``.

The combined RealBench flow (``run_realbench_combined.py``) runs every task
twice -- once through the agentic *pipeline* and once through the *direct*
model -- and pools the samples.  Its stdout/``runtime.log`` emits one line per
sample::

    [realbench]        PASS module/sdc/sd_crc_16   sample 06 syntax=1 function=1
    [realbench-direct] FAIL module/sdc/sd_tx_fifo  sample 10 syntax=0 function=0

``[realbench]`` lines belong to the pipeline flow, ``[realbench-direct]`` lines
to the direct flow.  This script parses those lines (no ``records.jsonl``
needed) and reports syntax / function pass rates plus pass@k for the pipeline
flow, the direct flow, and their pooled *combined* result -- the union a hybrid
run actually delivers.

pass@k uses the standard order-independent (HumanEval) estimator::

    pass@k = 1 - C(n-c, k) / C(n, k)

over the ``n`` samples of each task (``c`` correct).  Family / overall pass@k
are the mean of the per-task values.  The "Pass Rate" columns are pooled counts
over all samples in the group.

Usage::

    python scripts/hybrid_pass_rate_report.py runs/hybrid_oss20b/runtime.log
    python scripts/hybrid_pass_rate_report.py runs/hybrid_oss20b        # finds runtime.log
    python scripts/hybrid_pass_rate_report.py runs/hybrid_oss20b -o report.md --k 1 5 10
"""
from __future__ import annotations

import argparse
import math
import os
import re
import sys
from collections import defaultdict

# Pretty display names for known RealBench systems; unknown ones are upper-cased.
FAMILY_DISPLAY = {
    "sdc": "SDC",
    "aes": "AES",
    "e203_hbirdv2": "E203",
}

# [realbench] PASS module/sdc/sd_crc_16 sample 06 syntax=1 function=1
# [realbench-direct] FAIL module/sdc/sd_tx_fifo sample 10 syntax=0 function=0
RESULT_RE = re.compile(
    r"\[realbench(?P<direct>-direct)?\]\s+"
    r"(?P<verdict>PASS|FAIL)\s+"
    r"(?P<level>\w+)/(?P<system>[^/]+)/(?P<task>\S+)\s+"
    r"sample\s+(?P<sample>\d+)\s+"
    r"syntax=(?P<syntax>[01])\s+function=(?P<function>[01])"
)


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased probability that >=1 of k samples (drawn without replacement) is correct."""
    if c <= 0:
        return 0.0
    if k >= n:
        return 1.0 if c > 0 else 0.0
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def family_name(system: str) -> str:
    return FAMILY_DISPLAY.get(system, system.upper())


def resolve_log(path: str) -> str:
    """Accept either a runtime.log file or a run directory containing one."""
    if os.path.isdir(path):
        candidate = os.path.join(path, "runtime.log")
        if not os.path.exists(candidate):
            sys.exit(f"error: no runtime.log in {path}")
        return candidate
    if not os.path.exists(path):
        sys.exit(f"error: {path} not found")
    return path


def parse_log(log_path: str) -> list[dict]:
    """Return one record per parsed sample line.

    Each record: flow, system, task, sample, syntax (bool), function (bool).
    The last occurrence of a (flow, task, sample) wins so re-runs / repair
    passes that re-emit a line don't double count.
    """
    by_key: dict[tuple, dict] = {}
    with open(log_path) as fh:
        for line in fh:
            m = RESULT_RE.search(line)
            if not m:
                continue
            flow = "direct" if m.group("direct") else "pipeline"
            rec = {
                "flow": flow,
                "system": m.group("system"),
                "task": m.group("task"),
                "sample": int(m.group("sample")),
                "syntax": m.group("syntax") == "1",
                "function": m.group("function") == "1",
            }
            by_key[(flow, rec["system"], rec["task"], rec["sample"])] = rec
    return list(by_key.values())


def task_table(records: list[dict]) -> dict[tuple, dict]:
    """Aggregate records into per-(system, task) stats keyed by flow.

    Returns {(system, task): {flow: {n, syntax, correct}, ...}} where flow is
    one of 'pipeline', 'direct', 'combined' (combined = pooled samples).
    """
    tasks: dict[tuple, dict] = defaultdict(
        lambda: {
            "pipeline": {"n": 0, "syntax": 0, "correct": 0},
            "direct": {"n": 0, "syntax": 0, "correct": 0},
            "combined": {"n": 0, "syntax": 0, "correct": 0},
        }
    )
    for r in records:
        key = (r["system"], r["task"])
        for flow in (r["flow"], "combined"):
            agg = tasks[key][flow]
            agg["n"] += 1
            agg["syntax"] += int(r["syntax"])
            agg["correct"] += int(r["function"])
    return tasks


def group_summary(tasks: dict[tuple, dict], flow: str, ks: list[int]) -> dict:
    """Pooled syntax/function rates + mean per-task pass@k for one flow."""
    n = syntax = correct = 0
    solved = 0
    ntasks = 0
    patk = {k: 0.0 for k in ks}
    for stats in tasks.values():
        f = stats[flow]
        if f["n"] == 0:
            continue
        ntasks += 1
        n += f["n"]
        syntax += f["syntax"]
        correct += f["correct"]
        if f["correct"] > 0:
            solved += 1
        for k in ks:
            patk[k] += pass_at_k(f["n"], f["correct"], k)
    return {
        "tasks": ntasks,
        "n": n,
        "syntax": syntax,
        "correct": correct,
        "solved": solved,
        "syntax_rate": syntax / n if n else 0.0,
        "pass_rate": correct / n if n else 0.0,
        "pass_at_k": {k: (patk[k] / ntasks if ntasks else 0.0) for k in ks},
    }


def family_summaries(tasks: dict[tuple, dict], flow: str, ks: list[int]) -> dict[str, dict]:
    fams: dict[str, dict] = defaultdict(dict)
    for (system, task), stats in tasks.items():
        fams[system][(system, task)] = stats
    return {sys_: group_summary(sub, flow, ks) for sys_, sub in fams.items()}


def fmt_pct(x: float) -> str:
    return f"{100 * x:5.1f}%"


def render(tasks: dict[tuple, dict], log_path: str, ks: list[int]) -> str:
    flows = ["pipeline", "direct", "combined"]
    flow_label = {"pipeline": "Pipeline", "direct": "Direct", "combined": "Combined (hybrid)"}
    out: list[str] = []
    out.append(f"# Hybrid pass-rate report\n")
    out.append(f"Source: `{log_path}`\n")

    # --- Overall ------------------------------------------------------------
    summaries = {f: group_summary(tasks, f, ks) for f in flows}
    ntasks = summaries["combined"]["tasks"]
    out.append(f"**Tasks:** {ntasks}\n")
    out.append("## Overall\n")
    k_cols = "".join(f" pass@{k} |" for k in ks)
    out.append(f"| Flow | Samples | Syntax | Func pass | Solved |{k_cols}")
    sep = "|------|--------:|-------:|----------:|-------:|" + "".join("------:|" for _ in ks)
    out.append(sep)
    for f in flows:
        s = summaries[f]
        row = (
            f"| {flow_label[f]} | {s['n']} | {fmt_pct(s['syntax_rate'])} "
            f"| {fmt_pct(s['pass_rate'])} | {s['solved']}/{s['tasks']} |"
        )
        row += "".join(f" {fmt_pct(s['pass_at_k'][k])} |" for k in ks)
        out.append(row)
    out.append("")

    # --- Per family ---------------------------------------------------------
    out.append("## By family\n")
    fam_data = {f: family_summaries(tasks, f, ks) for f in flows}
    systems = sorted({s for (s, _t) in tasks.keys()}, key=family_name)
    for f in flows:
        out.append(f"### {flow_label[f]}\n")
        out.append(f"| Family | Samples | Syntax | Func pass | Solved | pass@1 | pass@{ks[-1]} |")
        out.append("|--------|--------:|-------:|----------:|-------:|-------:|-------:|")
        for sys_ in systems:
            s = fam_data[f][sys_]
            out.append(
                f"| {family_name(sys_)} | {s['n']} | {fmt_pct(s['syntax_rate'])} "
                f"| {fmt_pct(s['pass_rate'])} | {s['solved']}/{s['tasks']} "
                f"| {fmt_pct(s['pass_at_k'][1])} | {fmt_pct(s['pass_at_k'][ks[-1]])} |"
            )
        out.append("")

    # --- Per task -----------------------------------------------------------
    out.append("## Per task (correct samples)\n")
    out.append("| Family | Task | Pipeline | Direct | Combined | Solved |")
    out.append("|--------|------|---------:|-------:|---------:|:------:|")
    for (system, task) in sorted(tasks.keys(), key=lambda kt: (family_name(kt[0]), kt[1])):
        st = tasks[(system, task)]
        p, d, c = st["pipeline"], st["direct"], st["combined"]
        solved = "✅" if c["correct"] > 0 else "—"
        out.append(
            f"| {family_name(system)} | {task} "
            f"| {p['correct']}/{p['n']} | {d['correct']}/{d['n']} "
            f"| {c['correct']}/{c['n']} | {solved} |"
        )
    out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", help="runtime.log file or a run directory containing one")
    ap.add_argument("-o", "--output", help="write the markdown report here (default: stdout)")
    ap.add_argument(
        "--k", type=int, nargs="+", default=[1, 5, 10],
        help="pass@k values to report (default: 1 5 10)",
    )
    args = ap.parse_args(argv)

    log_path = resolve_log(args.log)
    records = parse_log(log_path)
    if not records:
        sys.exit(f"error: no '[realbench] PASS/FAIL ...' result lines found in {log_path}")

    tasks = task_table(records)
    ks = sorted(set(args.k))
    if 1 not in ks:
        ks = [1] + ks
    report = render(tasks, log_path, ks)

    if args.output:
        with open(args.output, "w") as fh:
            fh.write(report)
        print(f"wrote report to {args.output}")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
