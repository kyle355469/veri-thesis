"""Fig 6.5: routing frontier - solved tasks vs total compute (sum of wall-s)."""

import matplotlib.pyplot as plt

from common import (BLUE, MUTED, RED, despine, load_records, load_summary,
                    per_task_counts, save)

# arm -> (label, summary dir, style, annotation offset, alignment)
# style: None = plain, "highlight" = red, "hollow" = open marker (arm whose
# rule was fitted to the pre-audit solution set; excluded from the frontier).
ARMS = [
    ("Hybrid (both flows, 10+10)", "hybrid_oss20b_10syn_10func_rep_spec",
     None, (-10, -4), "right"),
    ("Pipeline-only 10/10",
     "repair_spec_slice_oss20b_func_v2_time-err-off_10syn_10func",
     None, (-12, 4), "right"),
    ("T1 (all-pipeline base)", "T1_router_oss20B_sync6_func4",
     None, (0, -26), "center"),
    ("v1 cascade (pre-audit rule)", "Full-2T_router_oss20B_sync6_func4",
     "hollow", (0, 8), "center"),
    ("Router-v2 (re-derived rule)",
     "Full-2T_router_oss20B_syn6_func4_s10_t02_v2",
     "highlight", (10, -2), "left"),
]

pts = []
for label, d, style, off, ha in ARMS:
    s = load_summary(d)
    # Solved counts always recomputed from wrap-cleaned records; summary.json
    # totals predate ref-wrap exclusion.
    counts = per_task_counts(load_records(d))
    solved = sum(1 for n, c in counts.values() if c > 0)
    if "cost" in s and s["cost"].get("total_wall_s"):
        wall = s["cost"]["total_wall_s"]
    else:
        recs = load_records(d)
        wall = sum(float(r.get("wall_s") or 0) for r in recs)
        if not wall:
            wall = s.get("mean_wall_s", 0) * s.get("num_records", len(recs))
    pts.append((label, wall / 3600.0, solved, style, off, ha))
    print(label, round(wall / 3600.0, 1), "h,", solved, "solved")

fig, ax = plt.subplots(figsize=(4.8, 3.0))
for label, hours, solved, style, (dx, dy), ha in pts:
    if style == "hollow":
        ax.scatter(hours, solved, s=45, facecolor="white", edgecolor=BLUE,
                   linewidth=1.4, zorder=3)
    else:
        color = RED if style == "highlight" else BLUE
        size = 70 if style == "highlight" else 45
        ax.scatter(hours, solved, s=size, color=color, zorder=3,
                   edgecolor="white", linewidth=1.2)
    ax.annotate(label, (hours, solved), xytext=(dx, dy),
                textcoords="offset points", fontsize=7.5, ha=ha,
                color="#0b0b0b" if style == "highlight" else "#52514e")

# Pareto frontier (upper-left is better) over arms whose rule survives the
# audit; the hollow v1 point is shown but not claimed.
frontier = []
for p in sorted((p for p in pts if p[3] != "hollow"), key=lambda p: p[1]):
    if not frontier or p[2] > frontier[-1][2]:
        frontier.append(p)
ax.plot([p[1] for p in frontier], [p[2] for p in frontier],
        color=MUTED, linewidth=1.0, linestyle="--", zorder=1)

ax.set_xlabel("total compute (summed sample wall-clock, hours)")
ax.set_ylabel("tasks solved (of 60)")
ax.set_xlim(0, 500)
ax.set_ylim(15, 30)
despine(ax)
fig.tight_layout()
save(fig, "fig_routing_frontier")
