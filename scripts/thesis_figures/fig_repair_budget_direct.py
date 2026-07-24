"""Repair-budget distribution for the direct-model 10/10 arm.

Mirror of fig_repair_budget (pipeline Func-10/10) on the direct-model run
with the same 10/10 caps, so the two figures read side by side. Left: syntax
phase - records by syntax-repair attempts used, split by whether they ended
syntax-clean. Right: functional phase - same for testbench pass, over the
samples that compiled. Log-scale y keeps the fix-fast-or-never tail legible.
Not yet referenced by the paper (direct-model arms are held back).
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter

from common import INK2, STATUS, despine, load_records, save

RUN = "realbench_direct_model_s10_syn10_func10"
recs = [r for r in load_records(RUN) if r.get("generated", True)]

fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.6), sharey=True)

W = 0.42
YTICKS = [1, 3, 10, 30, 100, 300]


def style_log_axis(ax):
    ax.set_yscale("log")
    ax.set_ylim(0.7, 600)
    ax.yaxis.set_major_locator(FixedLocator(YTICKS))
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.grid(axis="x", visible=False)
    despine(ax)


def label_bar(ax, x, h):
    ax.annotate(str(h), (x, h), xytext=(0, 2), textcoords="offset points",
                ha="center", va="bottom", fontsize=7, color=INK2)


# --- syntax phase ---
att = np.array([int(r.get("direct_repair_attempts") or 0) for r in recs])
ok = np.array([bool(r.get("syntax")) for r in recs])
xs = np.arange(0, att.max() + 1)
succ = [int(((att == a) & ok).sum()) for a in xs]
fail = [int(((att == a) & ~ok).sum()) for a in xs]
print("syntax attempts hist (succ/fail):", list(zip(xs.tolist(), succ, fail)))

ax = axes[0]
ax.bar(xs - W / 2, succ, width=W, color=STATUS["pass"], edgecolor="white",
       linewidth=0.8, label="ends syntax-clean")
ax.bar(xs + W / 2, fail, width=W, color=STATUS["syntax_fail"],
       edgecolor="white", linewidth=0.8, label="still failing")
label_bar(ax, xs[-1] + W / 2, fail[-1])
ax.set_title("Syntax repair (direct flow)", loc="left")
ax.set_xlabel("syntax-repair attempts used")
ax.set_ylabel("samples (log)")
ax.set_xticks(xs)
ax.legend(fontsize=7.5, loc="upper right")
style_log_axis(ax)

# --- functional phase (only samples that entered it: compiled, ran testbench) ---
frecs = [r for r in recs if r.get("syntax")]
fatt = np.array([int(r.get("direct_functional_repair_attempts") or 0) for r in frecs])
fok = np.array([bool(r.get("function")) for r in frecs])
xs2 = np.arange(0, fatt.max() + 1)
fsucc = [int(((fatt == a) & fok).sum()) for a in xs2]
ffail = [int(((fatt == a) & ~fok).sum()) for a in xs2]
print("functional attempts hist (succ/fail):", list(zip(xs2.tolist(), fsucc, ffail)))
burned = int(((fatt == fatt.max()) & ~fok).sum())
print("samples burning full functional budget without converting:", burned)

ax = axes[1]
ax.bar(xs2 - W / 2, fsucc, width=W, color=STATUS["pass"], edgecolor="white",
       linewidth=0.8, label="passes testbench")
ax.bar(xs2 + W / 2, ffail, width=W, color=STATUS["func_fail"],
       edgecolor="white", linewidth=0.8, label="still mismatching")
label_bar(ax, xs2[-1] + W / 2, ffail[-1])
ax.set_title("Functional repair (compiling samples)", loc="left")
ax.set_xlabel("functional-repair attempts used")
ax.set_xticks(xs2)
ax.legend(fontsize=7.5, loc="upper center")
style_log_axis(ax)

fig.tight_layout()
save(fig, "fig_repair_budget_direct")
