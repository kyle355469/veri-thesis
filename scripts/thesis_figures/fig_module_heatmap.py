"""Fig 6.8: per-module pass fraction across key runs (heatmap, sequential blue)."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from common import despine, load_records, per_task_counts, save

ARMS = [
    ("Pure model", "realbench_direct_model"),
    ("Pipe t0", "agentic_plan_legacy_realbench_plan_hallu_fix_t0"),
    ("Pipe 20B", "agentic_plan_legacy_realbench_oss20b_plan_hallu_no_rep_cache"),
    ("Pipe 120B", "agentic_plan_legacy_realbench_oss120b_plan_hallu_tool_call"),
    ("Func 10/10", "repair_spec_slice_oss20b_func_v2_time-err-off_10syn_10func"),
    ("Router 2T", "Full-2T_router_oss20B_sync6_func4"),
]

# sequential blue ramp (dataviz reference, steps 100->700), white for zero
CMAP = LinearSegmentedColormap.from_list("blues", [
    "#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
    "#256abf", "#184f95", "#0d366b"])

cols = {}
tasks = None
for label, run in ARMS:
    counts = per_task_counts(load_records(run))
    cols[label] = {t: c / n for t, (n, c) in counts.items()}
    if run == "agentic_plan_legacy_realbench_plan_hallu_fix_t0":
        tasks = sorted(counts.keys())

# order rows by family then by mean pass fraction (desc) for readability
def fam(t):
    return 0 if t.startswith("aes") else (1 if "sdc" in t.split("/")[0] else 2)

means = {t: np.mean([cols[l].get(t, np.nan) for l, _ in ARMS]) for t in tasks}
tasks = sorted(tasks, key=lambda t: (fam(t), -means[t]))

M = np.array([[cols[label].get(t, np.nan) for label, _ in ARMS] for t in tasks])
print("rows:", len(tasks), "cols:", len(ARMS))

fig, ax = plt.subplots(figsize=(4.6, 8.2))
im = ax.imshow(M, cmap=CMAP, vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(ARMS)), [l for l, _ in ARMS], rotation=35,
              ha="right", fontsize=7.5)
ax.set_yticks(range(len(tasks)),
              [t.split("/")[-1] for t in tasks], fontsize=5.8)
ax.grid(visible=False)
despine(ax, keep=())
# family separators
prev = None
for i, t in enumerate(tasks):
    if prev is not None and fam(t) != prev:
        ax.axhline(i - 0.5, color="#0b0b0b", linewidth=0.8)
    prev = fam(t)
cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
cbar.set_label("pass fraction (per task)", fontsize=8)
cbar.ax.tick_params(labelsize=7)
fig.tight_layout()
save(fig, "fig_module_heatmap")
