"""Fig 6.5: routing frontier - solved tasks vs total compute (sum of wall-s)."""

import matplotlib.pyplot as plt

from common import BLUE, MUTED, RED, despine, load_summary, save

# arm -> (label, summary dir, solved tasks, marker emphasis)
ARMS = [
    ("Hybrid (both flows, 10+10)", "hybrid_oss20b_10syn_10func_rep_spec", None),
    ("Pipeline-only 10/10", "repair_spec_slice_oss20b_func_v2_time-err-off_10syn_10func", None),
    ("T1 router (plan-probe)", "T1_router_oss20B_sync6_func4", None),
    ("Full cascade router", "Full-2T_router_oss20B_sync6_func4", "highlight"),
]

pts = []
for label, d, hl in ARMS:
    s = load_summary(d)
    from common import load_records, per_task_counts
    if "combined" in s:
        solved = s["combined"]["solved_tasks"]
    else:
        counts = per_task_counts(load_records(d))
        solved = sum(1 for n, c in counts.values() if c > 0)
    if "cost" in s and s["cost"].get("total_wall_s"):
        wall = s["cost"]["total_wall_s"]
    else:
        recs = load_records(d)
        wall = sum(float(r.get("wall_s") or 0) for r in recs)
        if not wall:
            wall = s.get("mean_wall_s", 0) * s.get("num_records", len(recs))
    pts.append((label, wall / 3600.0, solved, hl))
    print(label, round(wall / 3600.0, 1), "h,", solved, "solved")

fig, ax = plt.subplots(figsize=(4.8, 3.0))
for label, hours, solved, hl in pts:
    color = RED if hl else BLUE
    size = 70 if hl else 45
    ax.scatter(hours, solved, s=size, color=color, zorder=3,
               edgecolor="white", linewidth=1.2)
    dx, dy = (8, 4) if not hl else (8, -2)
    ax.annotate(label, (hours, solved), xytext=(dx, dy),
                textcoords="offset points", fontsize=7.5,
                color="#0b0b0b" if hl else "#52514e")

# Pareto frontier (upper-left is better)
frontier = []
for p in sorted(pts, key=lambda p: p[1]):
    if not frontier or p[2] > frontier[-1][2]:
        frontier.append(p)
ax.plot([p[1] for p in frontier], [p[2] for p in frontier],
        color=MUTED, linewidth=1.0, linestyle="--", zorder=1)

ax.set_xlabel("total compute (summed sample wall-clock, hours)")
ax.set_ylabel("tasks solved (of 60)")
ax.set_xlim(0, 500)
ax.set_ylim(15, 40)
despine(ax)
fig.tight_layout()
save(fig, "fig_routing_frontier")
