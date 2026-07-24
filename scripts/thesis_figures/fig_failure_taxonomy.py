"""Fig 6.6: outcome taxonomy stacked bars across runs (incl. repair-budget arm)."""

import matplotlib.pyplot as plt
import numpy as np

from common import STATUS, despine, load_records, save

ARMS = [
    ("Pure model\n(t0, 60)", "realbench_direct_model"),
    ("Planning\n(t0, 60)", "agentic_plan_legacy_realbench_plan_hallu_fix_t0"),
    ("Planning 20B\n(60×20)", "agentic_plan_legacy_realbench_oss20b_plan_hallu_no_rep_cache"),
    ("Func-10/10\n(60×10)", "repair_spec_slice_oss20b_func_v2_time-err-off_10syn_10func"),
    ("Func-10/10\n120B (60×10)", "repair_spec_slice_oss120b_func_v2_time-err-off_10syn_10func"),
]

BUCKETS = [
    ("pass", "passes", STATUS["pass"]),
    ("func_fail", "compiles, fails testbench", STATUS["func_fail"]),
    ("syntax_fail", "syntax fail", STATUS["syntax_fail"]),
    ("not_generated", "not generated", STATUS["not_generated"]),
]


def taxonomy(recs):
    n = len(recs)
    ng = sum(1 for r in recs if not r.get("generated", True))
    ps = sum(1 for r in recs if r.get("passed"))
    syn_ok = sum(1 for r in recs if r.get("syntax"))
    func_fail = syn_ok - ps
    syn_fail = n - ng - syn_ok
    return {"pass": 100 * ps / n, "func_fail": 100 * func_fail / n,
            "syntax_fail": 100 * syn_fail / n, "not_generated": 100 * ng / n,
            "_compile_fail_share": 100 * func_fail / syn_ok if syn_ok else 0}


fig, ax = plt.subplots(figsize=(6.2, 2.9))
x = np.arange(len(ARMS))
bottoms = np.zeros(len(ARMS))
vals = {}
for label, run in ARMS:
    t = taxonomy(load_records(run))
    vals[label] = t
    print(label.replace("\n", " "), {k: round(v, 1) for k, v in t.items()})

for key, blabel, color in BUCKETS:
    heights = [vals[label][key] for label, _ in ARMS]
    bars = ax.bar(x, heights, 0.55, bottom=bottoms, color=color,
                  edgecolor="white", linewidth=1.2, label=blabel)
    for xi, (h, b) in enumerate(zip(heights, bottoms)):
        if h > 6:
            ax.text(xi, b + h / 2, f"{h:.0f}", ha="center", va="center",
                    fontsize=7.5, color="white" if key != "func_fail" else "#0b0b0b")
    bottoms += np.array(heights)

ax.set_xticks(x, [label for label, _ in ARMS], fontsize=8)
ax.set_ylabel("% of samples")
ax.set_ylim(0, 100)
ax.grid(axis="x", visible=False)
ax.legend(fontsize=7.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.22))
despine(ax)
fig.tight_layout()
save(fig, "fig_failure_taxonomy")
