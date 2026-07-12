"""Fig 6.2: unbiased pass@k curves (k=1..10) across runs on one axis.

Normalizing at k<=10 lets every 60x10 run (and the 60x20 runs, subsampled by
the unbiased estimator) be read on the same axis.
"""

import matplotlib.pyplot as plt

from common import (AQUA, BLUE, GREEN, INK, MUTED, RED, VIOLET, YELLOW,
                    despine, load_records, pass_at_k, per_task_counts, save)

RUNS = [
    ("Full-2T_router_oss20B_sync6_func4", "Router-2T", VIOLET, "-"),
    ("hybrid_oss20b_10syn_10func_rep_spec", "Hybrid", GREEN, "-"),
    ("repair_spec_slice_oss20b_func_v2_time-err-off_10syn_10func",
     "Func-10/10", YELLOW, "-"),
    ("agentic_plan_legacy_realbench_oss120b_plan_hallu_tool_call",
     "Pipe-120B", RED, "--"),
    ("agentic_plan_legacy_realbench_oss20b_plan_hallu_no_rep_cache",
     "Cache-off", BLUE, "-"),
]

ks = list(range(1, 11))

fig, ax = plt.subplots(figsize=(4.8, 3.1))
for run, label, color, ls in RUNS:
    counts = per_task_counts(load_records(run))
    curve = [100.0 * sum(pass_at_k(n, c, k) for n, c in counts.values())
             / len(counts) for k in ks]
    ax.plot(ks, curve, color=color, linewidth=1.9, linestyle=ls,
            marker="o", markersize=3.2, markevery=[0, 4, 9])
    ax.annotate(f"{label} {curve[-1]:.1f}", (10, curve[-1]),
                xytext=(5, 0), textcoords="offset points", va="center",
                fontsize=7.5, color=color)
    print(label, "pass@1", round(curve[0], 1), "pass@5", round(curve[4], 1),
          "pass@10", round(curve[-1], 1))

ax.set_xlabel("k (samples)")
ax.set_ylabel("pass@k (%)")
ax.set_xlim(1, 10)
ax.set_xticks([1, 3, 5, 7, 10])
ax.set_ylim(0, 60)
ax.grid(axis="x", visible=False)
despine(ax)
fig.subplots_adjust(right=0.74)
save(fig, "fig_passk_multi")
