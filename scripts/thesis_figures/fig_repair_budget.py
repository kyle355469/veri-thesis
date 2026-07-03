"""Fig 6.4: repair-budget calibration (Func-10/10 run).

Left: syntax phase - records by syntax-repair attempts used, split by whether
they ended syntax-clean. Right: functional phase - same for testbench pass.
Shows the fix-fast-or-never shape: successes concentrate at low attempt
indices while the cap bucket is almost entirely failures.
"""

import matplotlib.pyplot as plt
import numpy as np

from common import STATUS, despine, load_records, save

RUN = "repair_spec_slice_oss20b_func_v2_time-err-off_10syn_10func"
recs = [r for r in load_records(RUN) if r.get("generated", True)]

fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.6))

# --- syntax phase ---
att = np.array([int(r.get("legacy_repair_attempts") or 0) for r in recs])
ok = np.array([bool(r.get("syntax")) for r in recs])
xs = np.arange(0, att.max() + 1)
succ = [int(((att == a) & ok).sum()) for a in xs]
fail = [int(((att == a) & ~ok).sum()) for a in xs]
print("syntax attempts hist (succ/fail):", list(zip(xs.tolist(), succ, fail)))

ax = axes[0]
ax.bar(xs, succ, color=STATUS["pass"], edgecolor="white", linewidth=0.8,
       label="ends syntax-clean")
ax.bar(xs, fail, bottom=succ, color=STATUS["syntax_fail"], edgecolor="white",
       linewidth=0.8, label="still failing")
ax.set_title("Syntax repair", loc="left")
ax.set_xlabel("syntax-repair attempts used")
ax.set_ylabel("samples")
ax.set_xticks(xs)
ax.grid(axis="x", visible=False)
ax.legend(fontsize=7.5, loc="upper right")
despine(ax)

# --- functional phase (only samples that entered it: compiled, ran testbench) ---
frecs = [r for r in recs if r.get("syntax")]
fatt = np.array([int(r.get("legacy_functional_repair_attempts") or 0) for r in frecs])
fok = np.array([bool(r.get("function")) for r in frecs])
xs2 = np.arange(0, fatt.max() + 1)
fsucc = [int(((fatt == a) & fok).sum()) for a in xs2]
ffail = [int(((fatt == a) & ~fok).sum()) for a in xs2]
print("functional attempts hist (succ/fail):", list(zip(xs2.tolist(), fsucc, ffail)))
burned = int(((fatt == fatt.max()) & ~fok).sum())
print("samples burning full functional budget without converting:", burned)

ax = axes[1]
ax.bar(xs2, fsucc, color=STATUS["pass"], edgecolor="white", linewidth=0.8,
       label="passes testbench")
ax.bar(xs2, ffail, bottom=fsucc, color=STATUS["func_fail"], edgecolor="white",
       linewidth=0.8, label="still mismatching")
ax.set_title("Functional repair (compiling samples)", loc="left")
ax.set_xlabel("functional-repair attempts used")
ax.set_xticks(xs2)
ax.grid(axis="x", visible=False)
ax.legend(fontsize=7.5, loc="upper left")
despine(ax)

fig.tight_layout()
save(fig, "fig_repair_budget")
