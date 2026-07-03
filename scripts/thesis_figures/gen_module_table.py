"""Emit the Appendix B per-module pass-count longtable body as LaTeX."""

from pathlib import Path

from common import REPO, load_records, per_task_counts

ARMS = [
    ("Direct-t0", "realbench_direct_model"),
    ("Pipe-t0", "agentic_plan_legacy_realbench_plan_hallu_fix_t0"),
    ("Cache-off", "agentic_plan_legacy_realbench_oss20b_plan_hallu_no_rep_cache"),
    ("Pipe-120b", "agentic_plan_legacy_realbench_oss120b_plan_hallu_tool_call"),
    ("Router-2t", "Full-2T_router_oss20B_sync6_func4"),
]

cols = {}
tasks = None
for label, run in ARMS:
    counts = per_task_counts(load_records(run))
    cols[label] = counts
    if run == "agentic_plan_legacy_realbench_plan_hallu_fix_t0":
        tasks = sorted(counts.keys())


def fam(t):
    return 0 if t.startswith("aes") else (1 if t.split("/")[0].startswith("sd") else 2)


tasks = sorted(tasks, key=lambda t: (fam(t), t))

out = []
prev = None
for t in tasks:
    if prev is not None and fam(t) != prev:
        out.append("\\midrule")
    prev = fam(t)
    name = t.split("/")[-1].replace("_", "\\_")
    cells = []
    for label, _ in ARMS:
        n, c = cols[label].get(t, (0, 0))
        cells.append(f"{c}/{n}" if n else "--")
    out.append(f"\\texttt{{{name}}} & " + " & ".join(cells) + " \\\\")

dest = REPO / "paper" / "back" / "module_table_body.tex"
dest.write_text("\n".join(out) + "\n")
print(f"[saved] {dest} ({len(tasks)} rows)")
