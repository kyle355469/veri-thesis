"""Fig 6.7: Verilator error classes among syntax-failing samples (Cache-off)."""

import re
from collections import Counter

import matplotlib.pyplot as plt

from common import RED, despine, load_records, save

RUN = "agentic_plan_legacy_realbench_oss20b_plan_hallu_no_rep_cache"
CODE_RE = re.compile(r"%(?:Error|Warning)-([A-Z0-9]+)")

recs = [r for r in load_records(RUN)
        if r.get("generated", True) and not r.get("syntax")]

counts = Counter()
for r in recs:
    info = r.get("syntax_info") or ""
    codes = set(CODE_RE.findall(info))
    if codes:
        counts.update(codes)
    elif "%Error" in info or info:
        counts.update(["PARSE/other"])
    else:
        counts.update(["PARSE/other"])

# Fold bare %Error (no code) into PARSE/other, keep top classes
top = counts.most_common(9)
rest = sum(c for _, c in counts.items()) - sum(c for _, c in top)
labels = [k for k, _ in top] + (["misc"] if rest else [])
values = [c for _, c in top] + ([rest] if rest else [])
print(list(zip(labels, values)))

fig, ax = plt.subplots(figsize=(4.8, 2.9))
y = range(len(labels))[::-1]
bars = ax.barh(list(y), values, 0.6, color=RED, edgecolor="white", linewidth=1.0)
for yi, v in zip(y, values):
    ax.text(v + 3, yi, str(v), va="center", fontsize=7.5, color="#52514e")
ax.set_yticks(list(y), labels, fontsize=8)
ax.set_xlabel("syntax-failing samples containing the class (of "
              f"{len(recs)})")
ax.grid(axis="y", visible=False)
despine(ax)
fig.tight_layout()
save(fig, "fig_error_classes")
