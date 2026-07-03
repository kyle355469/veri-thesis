"""Fig 6.1: syntax/pass per family, pure model vs grounded pipeline (temp 0)."""

import matplotlib.pyplot as plt
import numpy as np

from common import (AQUA, BLUE, bar_value_labels, despine, family_of,
                    load_records, save)

FAMS = ["aes", "sdc", "e203", "all"]
FAM_LABEL = {"aes": "AES (6)", "sdc": "SDC (14)", "e203": "E203 (40)", "all": "All (60)"}
ARMS = [("realbench_direct_model", "Pure model", AQUA),
        ("agentic_plan_legacy_realbench_plan_hallu_fix_t0", "Grounded pipeline", BLUE)]


def rates(records, fam):
    sel = [r for r in records if fam == "all" or family_of(r) == fam]
    n = len(sel)
    syn = 100.0 * sum(bool(r.get("syntax")) for r in sel) / n
    ps = 100.0 * sum(bool(r.get("passed")) for r in sel) / n
    return syn, ps


fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.5), sharey=True)
x = np.arange(len(FAMS))
w = 0.36

data = {}
for run, label, color in ARMS:
    recs = load_records(run)
    data[label] = [rates(recs, f) for f in FAMS]
    print(label, {f: tuple(round(v, 1) for v in rates(recs, f)) for f in FAMS})

for ax, metric, idx in ((axes[0], "Syntax rate (%)", 0), (axes[1], "Pass rate (%)", 1)):
    for j, (run, label, color) in enumerate(ARMS):
        vals = [data[label][i][idx] for i in range(len(FAMS))]
        bars = ax.bar(x + (j - 0.5) * w, vals, w, color=color,
                      edgecolor="white", linewidth=1.0, label=label)
        bar_value_labels(ax, bars, fmt="{:.0f}")
    ax.set_xticks(x, [FAM_LABEL[f] for f in FAMS])
    ax.set_title(metric, loc="left")
    ax.set_ylim(0, 108)
    ax.grid(axis="x", visible=False)
    despine(ax)

axes[0].set_ylabel("% of tasks")
axes[1].legend(loc="upper left", fontsize=8)
fig.tight_layout()
save(fig, "fig_headline_family")
