#!/usr/bin/env python3
"""Figure 2: top-1 / top-2 probability gap at divergence positions.

Panel A — CDF of (p1 - p2) on log-x, BF16 only, comparing model families.
Panel B — bar chart of top-5 token probs at one concrete divergence position.

Data source: research/exp_E2/E2*.json
"""
import sys, os, glob, json, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import setup_style, save, ROOT

import numpy as np
import matplotlib.pyplot as plt

BF16_STEP = 7.8125e-3   # mantissa unit for BF16 (eps_bf16)


def load_e2_runs():
    paths = sorted(glob.glob(os.path.join(ROOT, "exp_E2", "E2_*.json")))
    if not paths:
        p = os.path.join(ROOT, "exp_E2", "E2.json")
        if os.path.exists(p): paths = [p]
    return {os.path.basename(p).replace("E2_", "").replace(".json", ""): json.load(open(p))
            for p in paths}


def main():
    setup_style()
    runs = load_e2_runs()
    if not runs:
        print("No E2 data yet"); return

    fig = plt.figure(figsize=(8, 3))
    axA = fig.add_subplot(1, 2, 1)
    axB = fig.add_subplot(1, 2, 2)

    # ── Panel A: CDF of p1-p2 across runs ──
    cmap = plt.cm.tab10
    for i, (label, run) in enumerate(runs.items()):
        bf16 = run["schemes"].get("BF16")
        if not bf16: continue
        gaps = [e["ref_p1_minus_p2"] for e in bf16["per_problem"]
                if e.get("div_idx") is not None]
        if not gaps: continue
        gaps = np.sort(gaps)
        cdf = np.arange(1, len(gaps) + 1) / len(gaps)
        axA.plot(gaps, cdf, label=f"{run['model']}  (N_div={len(gaps)})", color=cmap(i))

    axA.axvline(BF16_STEP, color="k", linestyle="--", linewidth=1)
    axA.text(BF16_STEP * 1.1, 0.05, "BF16 step\n7.8e-3", fontsize=8)
    axA.set_xscale("log")
    axA.set_xlabel("top-1 − top-2 probability gap")
    axA.set_ylabel("cumulative fraction of\ndivergence positions")
    axA.set_title("(a) gap CDF at first divergence")
    axA.legend(loc="lower right", fontsize=8)
    axA.grid(alpha=0.3, which="both")

    # ── Panel B: top-5 example bar (pick smallest-gap example from any run) ──
    best = None
    for run in runs.values():
        bf16 = run["schemes"].get("BF16")
        if not bf16: continue
        for e in bf16["per_problem"]:
            if e.get("div_idx") is None: continue
            if best is None or e["ref_p1_minus_p2"] < best["ref_p1_minus_p2"]:
                best = e

    if best is not None:
        ref_probs = best["ref_top5_probs"]
        prt_probs = best["prt_top5_probs"]
        ref_tok   = [str(t).strip()[:8] or "·" for t in best["ref_top5_tokens"]]
        prt_tok   = [str(t).strip()[:8] or "·" for t in best["prt_top5_tokens"]]
        # use ref_top5 token strings as x labels
        x = np.arange(5)
        w = 0.4
        axB.bar(x - w/2, ref_probs, w, label="bs=1 (ref)",
                color="#3B83BD", edgecolor="black", linewidth=0.4)
        axB.bar(x + w/2, prt_probs, w, label=f"bs perturb",
                color="#D62728", edgecolor="black", linewidth=0.4)
        axB.set_xticks(x)
        axB.set_xticklabels([f"{r}\nvs\n{p}" for r, p in zip(ref_tok, prt_tok)],
                            fontsize=7)
        axB.set_ylabel("probability")
        axB.set_title(f"(b) example: gap = {best['ref_p1_minus_p2']:.4f}")
        axB.legend(fontsize=8)
        axB.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    save(fig, "fig2_prob_gap")


if __name__ == "__main__":
    main()
