#!/usr/bin/env python3
"""Figure 3: Avg_Std@top1_prob bar chart, log y-axis.

X-axis: scheme
Y-axis: Avg_Std@top1_prob  (log)
Bar groups: per model
Reference lines: 7.8e-3 (BF16 unit) and 1.2e-7 (FP32 unit)

Data source: research/exp_E3/E3_*.json
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import setup_style, save, load_E3_runs, SCHEME_ORDER, SCHEME_COLOR

import numpy as np
import matplotlib.pyplot as plt


# floor value for log scale (bit-exact = 0 → can't log)
ZERO_FLOOR = 1e-9


def main():
    setup_style()
    runs = load_E3_runs()
    if not runs:
        print("No E3 data yet"); return

    models = sorted(runs.keys())
    nm = len(models); ns = len(SCHEME_ORDER)
    x = np.arange(ns)
    w = 0.8 / nm

    fig, ax = plt.subplots(figsize=(6, 3.2))
    for i, m in enumerate(models):
        data = runs[m]
        d = {r["scheme"]: r for r in data}
        ys = [max(d[s]["avg_std_top1_prob"], ZERO_FLOOR) if s in d else ZERO_FLOOR
              for s in SCHEME_ORDER]
        ax.bar(x + (i - (nm - 1) / 2) * w, ys, w, label=m,
               edgecolor="black", linewidth=0.4)

    ax.axhline(7.8125e-3, color="grey", linestyle="--", linewidth=0.8)
    ax.text(ns - 0.5, 7.8125e-3 * 1.4, "BF16 step", fontsize=8, ha="right")
    ax.axhline(1.2e-7, color="grey", linestyle=":", linewidth=0.8)
    ax.text(ns - 0.5, 1.2e-7 * 1.4, "FP32 step", fontsize=8, ha="right")

    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(SCHEME_ORDER, rotation=15, ha="right")
    ax.set_ylabel("Avg_Std @ top-1 prob  (log)")
    ax.set_title("Probability-level stability across batch sizes")
    ax.set_ylim(ZERO_FLOOR / 5, 1.0)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3, axis="y", which="both")
    fig.tight_layout()
    save(fig, "fig3_avg_std")


if __name__ == "__main__":
    main()
