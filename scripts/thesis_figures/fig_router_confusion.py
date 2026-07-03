"""Fig 6.10: router flow allocation vs oracle class (Full-2T, 20B and 120B)."""

import matplotlib.pyplot as plt
import numpy as np

from common import BLUE, AQUA, STATUS, despine, load_summary, save

ARMS = [("gpt-oss-20B", "Full-2T_router_oss20B_sync6_func4"),
        ("gpt-oss-120B", "Full-2T_router_oss120B_sync6_func4")]

CATS = [
    ("A→pipeline (correct)", "A->pipeline", BLUE),
    ("B→direct (correct)", "B->direct", AQUA),
    ("A→direct (misroute)", "A->direct", STATUS["syntax_fail"]),
    ("B→pipeline (misroute)", "B->pipeline", STATUS["func_fail"]),
]

fig, ax = plt.subplots(figsize=(5.2, 2.6))
x = np.arange(len(ARMS))
w = 0.2
for j, (clabel, key, color) in enumerate(CATS):
    vals = [load_summary(run)["routing_confusion"][key] for _, run in ARMS]
    bars = ax.bar(x + (j - 1.5) * w, vals, w, color=color,
                  edgecolor="white", linewidth=1.0, label=clabel)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 3,
                f"{b.get_height():.0f}", ha="center", fontsize=7.5,
                color="#52514e")
for label, run in ARMS:
    s = load_summary(run)
    print(label, s["routing_confusion"], "misroutes:", s["misroutes_total"])

ax.set_xticks(x, [l for l, _ in ARMS])
ax.set_ylabel("samples (of 600)")
ax.set_ylim(0, 300)
ax.grid(axis="x", visible=False)
ax.legend(fontsize=7.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.25))
despine(ax)
fig.tight_layout()
save(fig, "fig_router_confusion")
