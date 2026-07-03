"""Shared style, palette, and data helpers for thesis figures.

Every figure script reads only runs/**/{summary.json,records.jsonl} (plus plan
artifacts) and writes a vector PDF into paper/figures/. Numbers that also
appear in thesis tables are printed to stdout so they can be cross-checked
against runs/*/complete_analysis_report.md.
"""

from __future__ import annotations

import json
from math import comb
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "runs"
OUT = REPO / "paper" / "figures"

# Validated categorical palette (dataviz reference, light mode, white surface).
# Slots are assigned in fixed order; aqua/yellow are sub-3:1 on white, so any
# use of them carries direct value labels (the relief rule).
BLUE = "#2a78d6"
AQUA = "#1baf7a"
YELLOW = "#eda100"
GREEN = "#008300"
VIOLET = "#4a3aa7"
RED = "#e34948"

# Fixed entity colors reused across every figure (color follows the entity).
FAMILY_COLORS = {"aes": BLUE, "sdc": AQUA, "e203": YELLOW, "all": "#0b0b0b"}
FLOW_COLORS = {"pipeline": BLUE, "direct": AQUA}

# Status palette for outcome buckets (reserved, never reused for series).
STATUS = {
    "pass": "#0ca30c",
    "func_fail": "#fab219",
    "syntax_fail": "#d03b3b",
    "not_generated": "#898781",
}

INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

mpl.rcParams.update({
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 8.5,
    "font.family": "sans-serif",
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK2,
    "axes.titlesize": 9.5,
    "axes.titlecolor": INK,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2,
    "legend.frameon": False,
    "pdf.fonttype": 42,
})


def despine(ax, keep=("left", "bottom")):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.pdf"
    fig.savefig(path)
    print(f"[saved] {path}")


def load_records(run_dir: str, module_only: bool = True) -> list[dict]:
    recs = []
    with open(RUNS / run_dir / "records.jsonl") as fh:
        for line in fh:
            r = json.loads(line)
            if module_only and r.get("task_level") not in (None, "module"):
                continue
            recs.append(r)
    return recs


def load_summary(run_dir: str) -> dict:
    with open(RUNS / run_dir / "summary.json") as fh:
        return json.load(fh)


def family_of(record: dict) -> str:
    sysname = str(record.get("system") or record.get("family") or "")
    if "aes" in sysname:
        return "aes"
    if "sdc" in sysname or sysname.startswith("sd"):
        return "sdc"
    return "e203"


def task_key(record: dict) -> str:
    return f"{record.get('system')}/{record.get('task')}"


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al., 2021)."""
    if k > n:
        k = n
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def per_task_counts(records: list[dict], field: str = "passed") -> dict[str, tuple[int, int]]:
    """task -> (n_samples, n_success). Ungenerated records count as failures."""
    out: dict[str, list[int]] = {}
    for r in records:
        n, c = out.setdefault(task_key(r), [0, 0])
        out[task_key(r)][0] = n + 1
        out[task_key(r)][1] = c + int(bool(r.get(field)))
    return {t: (n, c) for t, (n, c) in out.items()}


def mean_pass_at_k(records: list[dict], k: int, field: str = "passed") -> float:
    counts = per_task_counts(records, field)
    vals = [pass_at_k(n, c, k) for n, c in counts.values()]
    return sum(vals) / len(vals)


def bar_value_labels(ax, bars, fmt="{:.1f}", dy=0.5, fontsize=7.5, color=INK2):
    for b in bars:
        ax.annotate(
            fmt.format(b.get_height()),
            (b.get_x() + b.get_width() / 2, b.get_height()),
            xytext=(0, dy + 1.5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color=color,
        )
