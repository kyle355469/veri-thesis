"""Fig 5.6 (bar variant): the routing arms as grouped bars - two panels that
each share one y axis: task counts (functional / syntax, of 60) and
per-sample pass rates (functional / syntax, %). One color per arm; the boxed
legend above the panels maps color to arm and carries each arm's total
compute, which the scatter version encodes on its x-axis."""

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from common import (AQUA, BASELINE, BLUE, INK2, RED, VIOLET, despine,
                    load_records, load_summary, mean_pass_at_k,
                    per_task_counts, save)

# arm -> (legend label, summary dir, color)
# Color follows the entity: RED stays Router-v2 (the arm the routing claim
# stands on), BLUE the pipeline-only arm, AQUA the v1 cascade (most samples
# direct-routed), VIOLET the Hybrid deployment. Every bar carries a direct
# value label, which the sub-3:1 aqua slot requires (the relief rule).
# The Hybrid point uses the 5+5 rerun so every arm carries 60x10 samples.
ARMS = [
    ("v1 cascade", "Full-2T_router_oss20B_sync6_func4", AQUA),
    ("Hybrid 5+5", "hybrid_oss20b_10syn_10func_rep_spec_sample_5", VIOLET),
    ("Router-v2", "Full-2T_router_oss20B_syn6_func4_s10_t02_v2", RED),
    ("Func-10/10",
     "repair_spec_slice_oss20b_func_v2_time-err-off_10syn_10func", BLUE),
]

pts = []
for label, d, color in ARMS:
    s = load_summary(d)
    # All values recomputed from wrap-cleaned records; summary.json totals
    # predate ref-wrap exclusion.
    recs = load_records(d)
    solved = sum(1 for n, c in per_task_counts(recs).values() if c > 0)
    syn_tasks = sum(1 for n, c in per_task_counts(recs, "syntax").values()
                    if c > 0)
    pass_rate = 100 * mean_pass_at_k(recs, 1)
    syn_rate = 100 * mean_pass_at_k(recs, 1, "syntax")
    if "cost" in s and s["cost"].get("total_wall_s"):
        wall = s["cost"]["total_wall_s"]
    else:
        wall = sum(float(r.get("wall_s") or 0) for r in recs)
        if not wall:
            wall = s.get("mean_wall_s", 0) * s.get("num_records", len(recs))
    pts.append((label, wall / 3600.0, solved, syn_tasks, pass_rate,
                syn_rate, color))
    print(label, round(wall / 3600.0, 1), "h |", solved, "solved |",
          syn_tasks, "syntax tasks |", f"pass@1 {pass_rate:.1f}% |",
          f"syntax@1 {syn_rate:.1f}%")

fig, (ax_tasks, ax_rates) = plt.subplots(
    1, 2, figsize=(7.2, 3.1), gridspec_kw={"wspace": 0.26})

PANELS = [
    (ax_tasks, "task counts", "tasks (of 60)", (2, 3), 65, 20, False),
    (ax_rates, "per-sample pass rates", "share of samples (%)", (4, 5), 92,
     30, True),
]

WIDTH = 0.2
OFFSETS = [-0.33, -0.11, 0.11, 0.33]

# Sideways label nudges in x data units, keyed by (column index, arm label);
# they separate the functional-rate labels of Router-v2 (26.5%) and
# Func-10/10 (27.5%), whose bar tops are 1 point apart.
LABEL_NUDGE = {
    (4, "Router-v2"): -0.05,
    (4, "Func-10/10"): 0.05,
}

for ax, title, ylabel, cols, ymax, step, pct in PANELS:
    for g, idx in enumerate(cols):
        for off, p in zip(OFFSETS, pts):
            val, color = p[idx], p[6]
            ax.bar(g + off, val, WIDTH, color=color, edgecolor="white",
                   linewidth=1.0, zorder=3)
            txt = f"{val:.1f}%" if pct else f"{val:.0f}"
            ax.text(g + off + LABEL_NUDGE.get((idx, p[0]), 0), val, txt,
                    ha="center", va="bottom", fontsize=6.5, color=INK2)
    ax.set_title(title, loc="left", fontsize=8.5, color=INK2)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, ymax)
    ax.yaxis.set_major_locator(MultipleLocator(step))
    ax.set_xlim(-0.6, 1.6)
    ax.set_xticks([0, 1], ["functional pass", "syntax pass"], fontsize=8)
    ax.grid(axis="x", visible=False)
    despine(ax)

handles = [mpatches.Patch(color=p[6], label=f"{p[0]} ({round(p[1], 1):.0f} h)")
           for p in pts]
fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=7.5,
           frameon=True, edgecolor=BASELINE, framealpha=1.0,
           bbox_to_anchor=(0.5, 1.02))
fig.subplots_adjust(top=0.80)

save(fig, "fig_routing_frontier_bars")
