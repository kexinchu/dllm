#!/usr/bin/env python3
"""Figure 4: CDF of Div_Index per scheme.

X-axis: token position (0 to gen_len)
Y-axis: cumulative fraction of (problem, bs) pairs with div_idx ≤ x
Lines:  scheme
Subpanels: 2x2 = model × dataset

Bit-exact = div_idx == gen_len → contributes only at the right edge.

Data source: research/exp_E4/E4_*.json
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import setup_style, save, load_E4_runs, SCHEME_ORDER, SCHEME_COLOR

import numpy as np
import matplotlib.pyplot as plt


def main():
    setup_style()
    runs = load_E4_runs()
    if not runs:
        print("No E4 data yet"); return

    by_md = {(r["model"], r["dataset"]): r for r in runs}
    panels = sorted(by_md.keys())
    nrows = 2 if len(panels) > 2 else 1
    ncols = 2 if len(panels) > 1 else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.8 * nrows),
                             sharey=True)
    if not hasattr(axes, "flat"): axes = [axes]
    else: axes = axes.flat

    for ax, key in zip(axes, panels):
        run = by_md[key]
        gl = run["gen_len"]
        for s in run["schemes"]:
            sc = s["scheme"]
            if sc not in SCHEME_ORDER: continue
            divs = []
            for q in s["per_problem"]:
                for bs in run["bs_list"][1:]:  # skip the reference bs=1
                    bs_key = bs if bs in q["per_bs"] else str(bs)
                    divs.append(q["per_bs"][bs_key]["div_idx"])
            divs = np.sort(divs)
            cdf = np.arange(1, len(divs) + 1) / len(divs)
            # extend the last horizontal segment to gen_len
            divs = np.concatenate([divs, [gl]])
            cdf  = np.concatenate([cdf,  [cdf[-1]]])
            ax.plot(divs, cdf, color=SCHEME_COLOR[sc], label=sc, drawstyle="steps-post")

        ax.set_xlim(0, gl)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("token position (Div_Index)")
        ax.set_ylabel("CDF over (problem, bs) pairs")
        ax.set_title(f"{run['model']} / {run['dataset']}")
        ax.grid(alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(SCHEME_ORDER),
               bbox_to_anchor=(0.5, 1.04), frameon=False)
    fig.tight_layout()
    save(fig, "fig4_div_index_cdf")


if __name__ == "__main__":
    main()
