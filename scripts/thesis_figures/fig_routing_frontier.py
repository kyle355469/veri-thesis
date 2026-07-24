"""Fig 5.6: routing frontier (gpt-oss-20B arms) - four graphs against total
compute: tasks solved / tasks compiling (top row) and per-sample functional /
syntax pass rates (bottom row). Arms are identified by marker style via a
shared legend, not by in-graph labels; each point is annotated with its
panel value."""

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from common import (BLUE, INK2, RED, despine, load_records,
                    load_summary, mean_pass_at_k, per_task_counts, save)

PURPLE = "#5141aa"

# arm -> (legend label, summary dir, marker, style, flow filter)
# style: None = plain blue, "highlight" = red, "hollow" = open marker (arm
# whose rule was fitted to the pre-audit solution set; excluded from the
# frontier), "reference" = purple (context point, not a routing arm).
# The Hybrid point uses the 5+5 rerun so every arm carries 60x10 samples
# and the task-count panels compare equal sample budgets. The Direct-only
# reference point is the 10+10 Hybrid run's direct flow taken alone (600
# unrepaired direct samples); a flow filter restricts its records, and its
# wall-clock is summed from those records rather than the run's two-flow
# cost total.
ARMS = [
    ("Direct-only", "hybrid_oss20b_10syn_10func_rep_spec",
     "v", "reference", "direct"),
    ("Hybrid (both flows, 5+5)", "hybrid_oss20b_10syn_10func_rep_spec_sample_5",
     "s", None, None),
    ("Planning-only Func-10/10",
     "repair_spec_slice_oss20b_func_v2_time-err-off_10syn_10func",
     "o", None, None),
    ("Router-v1 (direct rules)",
     "Full-2T_router_oss20B_sync6_func4", "D", "hollow", None),
    ("Router-v2 (planning rules)",
     "Full-2T_router_oss20B_syn6_func4_s10_t02_v2", "^", "highlight", None),
]

pts = []
for label, d, marker, style, flow in ARMS:
    s = load_summary(d)
    # All values recomputed from wrap-cleaned records; summary.json totals
    # predate ref-wrap exclusion.
    recs = load_records(d)
    if flow:
        recs = [r for r in recs if r.get("flow") == flow]
    solved = sum(1 for n, c in per_task_counts(recs).values() if c > 0)
    syn_tasks = sum(1 for n, c in per_task_counts(recs, "syntax").values()
                    if c > 0)
    pass_rate = 100 * mean_pass_at_k(recs, 1)
    syn_rate = 100 * mean_pass_at_k(recs, 1, "syntax")
    if flow:
        wall = sum(float(r.get("wall_s") or 0) for r in recs)
    elif "cost" in s and s["cost"].get("total_wall_s"):
        wall = s["cost"]["total_wall_s"]
    else:
        wall = sum(float(r.get("wall_s") or 0) for r in recs)
        if not wall:
            wall = s.get("mean_wall_s", 0) * s.get("num_records", len(recs))
    pts.append((label, wall / 3600.0, solved, syn_tasks, pass_rate,
                syn_rate, marker, style))
    print(label, round(wall / 3600.0, 1), "h |", solved, "solved |",
          syn_tasks, "syntax tasks |", f"pass@1 {pass_rate:.1f}% |",
          f"syntax@1 {syn_rate:.1f}%")


def draw_marker(ax, x, y, marker, style, label=None):
    if style == "hollow":
        return ax.scatter(x, y, s=50, marker=marker, facecolor="white",
                          edgecolor=BLUE, linewidth=1.4, zorder=3,
                          label=label)
    color = {"highlight": RED, "reference": PURPLE}.get(style, BLUE)
    size = 70 if style == "highlight" else 50
    return ax.scatter(x, y, s=size, marker=marker, color=color, zorder=3,
                      edgecolor="white", linewidth=1.2, label=label)


fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2),
                         gridspec_kw={"wspace": 0.28, "hspace": 0.34})
(ax_ft, ax_st), (ax_fr, ax_sr) = axes

PANELS = [
    (ax_ft, 2, "functional pass (tasks)", "tasks solved (of 60)", (5, 30), 5),
    (ax_st, 3, "syntax pass (tasks)", "tasks compiling (of 60)", (20, 60), 10),
    (ax_fr, 4, "functional pass rate", "passing samples (%)", (0, 30), 10),
    (ax_sr, 5, "syntax pass rate", "compiling samples (%)", (10, 90), 20),
]

# Value-label offsets in points, keyed by (column index, arm label); default
# is centered above the marker. The overrides dodge the tasks-solved panel,
# where Hybrid and Router-v2 sit at 21 only 30 h apart.
LABEL_OFFSET = {
    (2, "Hybrid (both flows, 5+5)"): (0, -14),
    (3, "Router-v2 (planning rules)"): (0, -15),
}

handles = []
for ax, idx, title, ylabel, ylim, step in PANELS:
    for p in pts:
        label, hours, marker, style = p[0], p[1], p[6], p[7]
        h = draw_marker(ax, hours, p[idx], marker, style,
                        label=label if ax is ax_ft else None)
        if ax is ax_ft:
            handles.append(h)
        val = f"{p[idx]:.0f}" if idx in (2, 3) else f"{p[idx]:.1f}%"
        ax.annotate(val, (hours, p[idx]),
                    textcoords="offset points",
                    xytext=LABEL_OFFSET.get((idx, label), (0, 6)),
                    ha="center", fontsize=7,
                    color={"highlight": RED,
                           "reference": PURPLE}.get(style, BLUE))
    ax.set_title(title, loc="left", fontsize=8.5, color=INK2)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_locator(MultipleLocator(step))
    ax.set_xlim(0, 500)
    despine(ax)

ax_fr.set_xlabel("total compute (hours)")
ax_sr.set_xlabel("total compute (hours)")

fig.legend(handles, [p[0] for p in pts], loc="upper center", ncol=2,
           fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, 1.0))
fig.subplots_adjust(top=0.84)

save(fig, "fig_routing_frontier")
