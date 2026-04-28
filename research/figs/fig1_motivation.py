#!/usr/bin/env python3
"""Figure 1: motivation. accuracy vs batch size, per scheme.

Subpanels: 2x2 = (model: Llama / DeepSeek) × (dataset: MATH500 / AIME25)
X-axis: batch size (1, 8, 16)
Y-axis: accuracy (%)
Lines: scheme

Data source: research/exp_E4/E4_*.json → schemes[*].aggregate.accuracy_by_bs
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import setup_style, save, load_E4_runs, SCHEME_ORDER, SCHEME_COLOR, SCHEME_MARKER

import matplotlib.pyplot as plt


def main():
    setup_style()
    runs = load_E4_runs()
    if not runs:
        print("No E4 data yet"); return

    # Index by (model, dataset)
    by_md = {(r["model"], r["dataset"]): r for r in runs}
    panels = sorted(by_md.keys())

    nrows = 2 if len(panels) > 2 else 1
    ncols = 2 if len(panels) > 1 else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.7 * nrows),
                             sharey=False)
    if not hasattr(axes, "flat"): axes = [axes]
    else: axes = axes.flat

    for ax, key in zip(axes, panels):
        run = by_md[key]
        bsl = run["bs_list"]
        for s in run["schemes"]:
            sc = s["scheme"]
            if sc not in SCHEME_ORDER: continue
            agg = s["aggregate"]["accuracy_by_bs"]
            ys = []
            for bs in bsl:
                k = bs if bs in agg else str(bs)
                ys.append(100.0 * agg[k])
            ax.plot(bsl, ys, marker=SCHEME_MARKER[sc], color=SCHEME_COLOR[sc], label=sc)
        ax.set_xlabel("batch size")
        ax.set_ylabel("accuracy (%)")
        ax.set_title(f"{run['model']} / {run['dataset']}  (N={run['n_problems']})")
        ax.set_xticks(bsl)
        ax.grid(alpha=0.3)

    # one shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(SCHEME_ORDER),
               bbox_to_anchor=(0.5, 1.04), frameon=False)
    fig.tight_layout()
    save(fig, "fig1_motivation_accuracy_vs_bs")


if __name__ == "__main__":
    main()
