"""Fig 6.9 (grounding audit): reuse decisions by grounding outcome, two runs."""

import glob
import json

import matplotlib.pyplot as plt
import numpy as np

from common import REPO, STATUS, BLUE, VIOLET, despine, save

RUNS_ = {
    "Temp-0 run (60 plans)": "agentic_plan_legacy_realbench_plan_hallu_fix_t0",
    "20-sample run (1200 plans)": "agentic_plan_legacy_realbench_oss20b_plan_hallu",
}


def audit(run):
    tot = {"exact": 0, "remapped": 0, "dropped": 0}
    files = glob.glob(str(REPO / "runs" / run / "plans" / "**" / "agent_result.json"),
                      recursive=True)
    for f in files:
        try:
            g = json.load(open(f)).get("grounding") or {}
        except Exception:
            continue
        for k in tot:
            tot[k] += int(g.get(k) or 0)
    return tot, len(files)


fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.3))
for ax, (label, run) in zip(axes, RUNS_.items()):
    tot, nfiles = audit(run)
    print(label, tot, f"({nfiles} plan files)")
    cats = ["exact catalog id", "remapped near-miss", "dropped hallucination"]
    vals = [tot["exact"], tot["remapped"], tot["dropped"]]
    colors = [BLUE, VIOLET, STATUS["syntax_fail"]]
    bars = ax.bar(range(3), vals, 0.55, color=colors, edgecolor="white",
                  linewidth=1.0)
    for i, v in enumerate(vals):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=8,
                color="#52514e")
    ax.set_xticks(range(3), ["exact", "remapped", "dropped"], fontsize=8)
    ax.set_title(label, loc="left", fontsize=8.5)
    ax.set_ylim(0, max(vals) * 1.2 if max(vals) else 1)
    ax.grid(axis="x", visible=False)
    despine(ax)
axes[0].set_ylabel("reuse decisions")
fig.tight_layout()
save(fig, "fig_grounding_bar")
