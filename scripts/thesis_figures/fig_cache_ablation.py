"""Fig 6.3: repair-cache ablation (off/task/run): syntax, pass@1, pass@20."""

import matplotlib.pyplot as plt
import numpy as np

from common import (BLUE, VIOLET, AQUA, bar_value_labels, despine,
                    load_records, mean_pass_at_k, save)

ARMS = [("off", "agentic_plan_legacy_realbench_oss20b_plan_hallu_no_rep_cache"),
        ("task", "agentic_plan_legacy_realbench_oss20b_plan_hallu_task_rep_cache"),
        ("run", "agentic_plan_legacy_realbench_oss20b_plan_hallu")]

rows = []
for scope, run in ARMS:
    recs = load_records(run)
    n = len(recs)
    syn = 100.0 * sum(bool(r.get("syntax")) for r in recs) / n
    p1 = 100.0 * sum(bool(r.get("passed")) for r in recs) / n
    p20 = 100.0 * mean_pass_at_k(recs, 20)
    rows.append((scope, syn, p1, p20))
    print(scope, round(syn, 1), round(p1, 1), round(p20, 1))

metrics = [("Syntax rate", 1, BLUE), ("pass@1", 2, AQUA), ("pass@20", 3, VIOLET)]
x = np.arange(len(ARMS))
w = 0.26

fig, ax = plt.subplots(figsize=(4.6, 2.7))
for j, (label, idx, color) in enumerate(metrics):
    vals = [row[idx] for row in rows]
    bars = ax.bar(x + (j - 1) * w, vals, w, color=color,
                  edgecolor="white", linewidth=1.0, label=label)
    bar_value_labels(ax, bars)
ax.set_xticks(x, [f"cache {s}" for s, *_ in rows])
ax.set_ylabel("% ")
ax.set_ylim(0, 78)
ax.grid(axis="x", visible=False)
ax.legend(loc="upper left", ncol=3, fontsize=8, columnspacing=1.2)
despine(ax)
fig.tight_layout()
save(fig, "fig_cache_ablation")
