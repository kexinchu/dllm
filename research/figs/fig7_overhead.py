#!/usr/bin/env python3
"""Figure 7: overhead profile, 2×2 panels.

Panels:  TTFT  /  TPOT  /  Throughput  /  Peak memory
X-axis (each panel): scheme
Y-axis: respective metric
Bar groups within each scheme: bs ∈ {1, 8, 32}
Subplots can be tagged with model name; if multiple models present, makes one
column per model (so the figure becomes 2 cols × 4 rows when there are 2 models).

Data source: research/exp_E7/E7_*.json
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import setup_style, save, load_E7_runs, SCHEME_ORDER

import numpy as np
import matplotlib.pyplot as plt


METRIC_DEFS = [
    ("ttft_ms",        "TTFT (ms)",            False),
    ("tpot_ms",        "TPOT (ms / token)",    False),
    ("throughput_tps", "Throughput (tok/s)",   False),
    ("peak_mem_gb",    "Peak memory (GB)",     False),
]


def main():
    setup_style()
    runs = load_E7_runs()
    if not runs:
        print("No E7 data yet"); return

    runs.sort(key=lambda r: r["model"])
    nm = len(runs)
    fig, axes = plt.subplots(4, nm, figsize=(3.2 * nm, 8), squeeze=False)

    for col, run in enumerate(runs):
        bs_set = sorted(set(r["bs"] for r in run["results"]))
        ns = len(SCHEME_ORDER); nb = len(bs_set)
        x = np.arange(ns); w = 0.8 / nb
        for row, (key, ylabel, log) in enumerate(METRIC_DEFS):
            ax = axes[row][col]
            cmap = plt.cm.tab20c
            for j, bs in enumerate(bs_set):
                ys = []
                for sc in SCHEME_ORDER:
                    rec = next((r for r in run["results"]
                                if r["scheme"] == sc and r["bs"] == bs), None)
                    ys.append(rec[key] if rec else 0)
                ax.bar(x + (j - (nb - 1) / 2) * w, ys, w,
                       label=f"bs={bs}", color=cmap(j * 4), edgecolor="black", linewidth=0.4)
            ax.set_ylabel(ylabel)
            ax.set_xticks(x); ax.set_xticklabels(SCHEME_ORDER, rotation=20, ha="right")
            ax.grid(alpha=0.3, axis="y")
            if log: ax.set_yscale("log")
            if row == 0: ax.set_title(run["model"])
            if row == 0 and col == 0: ax.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    save(fig, "fig7_overhead")


if __name__ == "__main__":
    main()
