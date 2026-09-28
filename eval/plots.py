"""Paper figures (static, print-ready PDF + PNG).

  fig_pareto.pdf     accuracy vs. mean tokens: validation sweep cloud + test configs
  fig_tail_cost.pdf  p50 / p95 tokens per configuration (the long-tail claim)
  fig_over_read.pdf  share of tokens spent after the evidence was already read

    python eval/plots.py --table results/table_main.csv --grid results/sweep_grid.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# Validated categorical slots (dataviz reference palette, light mode), fixed order.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
METHOD = "gate_full"


def style() -> None:
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
        "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "figure.dpi": 200, "savefig.bbox": "tight", "pdf.fonttype": 42,
    })


def save(fig, out_dir: Path, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}.{ext}")
    plt.close(fig)


def pareto(table: pd.DataFrame, grid: pd.DataFrame | None, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    if grid is not None and len(grid):
        ax.scatter(grid.mean_tokens / 1e3, grid.acc * 100, s=9, color=AQUA, alpha=0.5,
                   linewidths=0, label="Threshold settings (val replay)")
    for _, r in table.iterrows():
        is_method = r.config == METHOD
        ax.scatter(r.tokens_mean / 1e3, r.acc_mean * 100, s=36 if is_method else 22,
                   color=ORANGE if is_method else BLUE, edgecolors="white", linewidths=1,
                   zorder=3)
        ax.annotate(r.config, (r.tokens_mean / 1e3, r.acc_mean * 100), fontsize=6.5,
                    color=INK, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("Mean tokens per question (thousands)")
    ax.set_ylabel("Accuracy (%)")
    lo, hi = ax.get_ylim()
    ax.set_ylim(max(0, lo), min(101, hi))
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    save(fig, out, "fig_pareto")


def tail_cost(table: pd.DataFrame, out: Path) -> None:
    t = table.sort_values("tokens_p95")
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    y = range(len(t))
    ax.barh([i + 0.2 for i in y], t.tokens_p50 / 1e3, height=0.36, color=BLUE, edgecolor="white", linewidth=1, label="p50")
    ax.barh([i - 0.2 for i in y], t.tokens_p95 / 1e3, height=0.36, color=ORANGE, edgecolor="white", linewidth=1, label="p95")
    ax.set_yticks(list(y), t.config)
    ax.set_xlabel("Tokens per question (thousands)")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    save(fig, out, "fig_tail_cost")


def over_read(table: pd.DataFrame, out: Path) -> None:
    t = table.dropna(subset=["over_read_mean"]).sort_values("over_read_mean")
    if t.empty:
        return
    fig, ax = plt.subplots(figsize=(3.4, 2.0))
    colors = [ORANGE if c == METHOD else BLUE for c in t.config]
    ax.barh(t.config, t.over_read_mean * 100, height=0.55, color=colors)
    for i, v in enumerate(t.over_read_mean * 100):
        ax.text(v + 1, i, f"{v:.0f}%", va="center", fontsize=6.5, color=INK)
    ax.set_xlabel("Tokens spent after evidence was already read (%)")
    ax.grid(axis="y", visible=False)
    save(fig, out, "fig_over_read")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="results/table_main.csv")
    ap.add_argument("--grid", default="results/sweep_grid.csv")
    ap.add_argument("--out", default="results/figures")
    args = ap.parse_args()
    style()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(args.table)
    grid = pd.read_csv(args.grid) if Path(args.grid).exists() else None
    pareto(table, grid, out)
    tail_cost(table, out)
    over_read(table, out)
    print(f"figures written to {out}/")


if __name__ == "__main__":
    main()
