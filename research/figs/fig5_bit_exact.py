#!/usr/bin/env python3
"""Figure 5: bit-exact rate per scheme.

X-axis: scheme
Y-axis: bit-exact rate (%)
Bar groups: (model, dataset)

Data source: research/exp_E4/E4_*.json
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import setup_style, save, load_E4_runs, SCHEME_ORDER

import numpy as np
import matplotlib.pyplot as plt


def main():
    setup_style()
    runs = load_E4_runs()
    if not runs:
        print("No E4 data yet"); return

    runs.sort(key=lambda r: (r["model"], r["dataset"]))
    nm = len(runs); ns = len(SCHEME_ORDER)
    x = np.arange(ns)
    w = 0.85 / nm

    fig, ax = plt.subplots(figsize=(7, 3.2))
    cmap = plt.cm.tab10
    for i, run in enumerate(runs):
        d = {s["scheme"]: s for s in run["schemes"]}
        ys = [d[s]["aggregate"]["bit_exact_rate"] * 100 if s in d else 0
              for s in SCHEME_ORDER]
        label = f"{run['model']} / {run['dataset']}"
        ax.bar(x + (i - (nm - 1) / 2) * w, ys, w, label=label,
               color=cmap(i), edgecolor="black", linewidth=0.4)

    ax.set_xticks(x); ax.set_xticklabels(SCHEME_ORDER, rotation=15, ha="right")
    ax.set_ylabel("bit-exact pairs  (%)")
    ax.set_title("Bit-exact rate across (problem, bs) pairs")
    ax.set_ylim(0, 105)
    ax.axhline(100, color="grey", linestyle=":", linewidth=0.6)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    save(fig, "fig5_bit_exact")


if __name__ == "__main__":
    main()
