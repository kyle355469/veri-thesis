"""Fig 6.2: unbiased pass@k curves per family (Cache-off, 60x20)."""

import matplotlib.pyplot as plt

from common import (FAMILY_COLORS, despine, family_of, load_records,
                    pass_at_k, per_task_counts, save)

RUN = "agentic_plan_legacy_realbench_oss20b_plan_hallu_no_rep_cache"
FAM_LABEL = {"aes": "AES", "sdc": "SDC", "e203": "E203", "all": "All 60 tasks"}

recs = load_records(RUN)
ks = list(range(1, 21))

fig, ax = plt.subplots(figsize=(4.6, 3.0))
# end-label vertical dodge (points): curves converge at k=20
DODGE = {"aes": 5, "e203": 0, "all": -6, "sdc": 0}
for fam in ["all", "e203", "sdc", "aes"]:
    sel = [r for r in recs if fam == "all" or family_of(r) == fam]
    counts = per_task_counts(sel)
    curve = [100.0 * sum(pass_at_k(n, c, k) for n, c in counts.values()) / len(counts)
             for k in ks]
    lw = 2.2 if fam == "all" else 1.6
    ax.plot(ks, curve, color=FAMILY_COLORS[fam], linewidth=lw,
            marker="o", markersize=3.2 if fam != "all" else 4.0,
            markevery=[0, 4, 9, 19], label=f"{FAM_LABEL[fam]} ({curve[-1]:.1f})")
    ax.annotate(f"{FAM_LABEL[fam]} {curve[-1]:.1f}", (20, curve[-1]),
                xytext=(5, DODGE[fam]), textcoords="offset points", va="center",
                fontsize=7.5, color=FAMILY_COLORS[fam] if fam != "all" else "#0b0b0b")
    print(fam, "pass@1", round(curve[0], 1), "pass@5", round(curve[4], 1),
          "pass@10", round(curve[9], 1), "pass@20", round(curve[-1], 1))
ax.legend(loc="lower right", fontsize=7.5)

ax.set_xlabel("k (samples)")
ax.set_ylabel("pass@k (%)")
ax.set_xlim(1, 20)
ax.set_xticks([1, 5, 10, 15, 20])
ax.set_ylim(0, 40)
ax.grid(axis="x", visible=False)
despine(ax)
fig.subplots_adjust(right=0.78)
save(fig, "fig_passk_curves")
