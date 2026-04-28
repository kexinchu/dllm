#!/usr/bin/env python3
"""Figure 6: % problems diverging vs MATH-500 difficulty level.

X-axis: MATH level {1, 2, 3, 4, 5}
Y-axis: % problems with any divergence (across bs > 1)
Lines:  scheme  (BF16 — wiggly upward; DetermLLM+attn — flat at 0)
Subpanels: per model

Data source: research/exp_E4/E4_*math500.json + math500_cached.json (level field)
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import setup_style, save, load_E4_runs, SCHEME_ORDER, SCHEME_COLOR, SCHEME_MARKER

import matplotlib.pyplot as plt
from collections import defaultdict


CACHE = "/home/kec23008/docker-sys/dllm/research/math500_cached.json"


def main():
    setup_style()

    cache = json.load(open(CACHE))
    level_by_idx = {i: p.get("level") for i, p in enumerate(cache)}

    runs = [r for r in load_E4_runs() if r["dataset"] == "math500"]
    if not runs:
        print("No math500 E4 data yet"); return

    nm = len(runs)
    fig, axes = plt.subplots(1, nm, figsize=(3.6 * nm, 2.8), sharey=True)
    if nm == 1: axes = [axes]

    for ax, run in zip(axes, runs):
        bsl = run["bs_list"]
        for s in run["schemes"]:
            sc = s["scheme"]
            if sc not in SCHEME_ORDER: continue
            buckets = defaultdict(lambda: [0, 0])  # level -> [n_div, n]
            for q in s["per_problem"]:
                lvl = level_by_idx.get(q["idx"])
                if lvl is None: continue
                pbs = q["per_bs"]
                diverges = any(
                    not pbs[bs if bs in pbs else str(bs)]["bit_exact"]
                    for bs in bsl[1:]
                )
                buckets[lvl][1] += 1
                if diverges: buckets[lvl][0] += 1

            xs = sorted(buckets.keys())
            ys = [100.0 * buckets[l][0] / max(buckets[l][1], 1) for l in xs]
            ax.plot(xs, ys, marker=SCHEME_MARKER[sc], color=SCHEME_COLOR[sc], label=sc)

        ax.set_xlabel("MATH-500 level")
        ax.set_ylabel("% problems diverging")
        ax.set_title(run["model"])
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_ylim(0, 105)
        ax.grid(alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(SCHEME_ORDER),
               bbox_to_anchor=(0.5, 1.05), frameon=False)
    fig.tight_layout()
    save(fig, "fig6_difficulty")


if __name__ == "__main__":
    main()
