"""Phase-1 figures from existing exp_matrix/ data.

Two figures for the paper, Yuan et al. 2025 visual style:

  FigA — Non-determinism rate bar chart.
         X axis: (model × dataset) cells (6 total).
         Bars: 3 methods (BF16, LayerCast, Hybrid).
         Y axis: Non-det rate = 1 − same_toks / n_pairs
                 (averaged over 1v8 + 8v32 pairs).

  FigB — Divergence-position histogram (DeepSeek-7B × MATH500).
         Overlayed step histograms for 3 methods.
         Shows *where* bs-variants begin to diverge; right-heavy = better.
"""
import json
import os
import statistics
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = "/home/kec23008/docker-sys/dllm/research/exp_matrix"
OUT  = "/home/kec23008/docker-sys/dllm/research/figs"
os.makedirs(OUT, exist_ok=True)

METHOD_ORDER  = ["bf16_baseline", "layercast", "dllm_hybrid"]
METHOD_LABELS = {"bf16_baseline": "BF16", "layercast": "LayerCast", "dllm_hybrid": "DetermLLM"}
METHOD_COLORS = {"bf16_baseline": "#888888", "layercast": "#1f77b4", "dllm_hybrid": "#d62728"}

MODEL_SHORT = {
    "DeepSeek-R1-Distill-Qwen-7B": "DeepSeek-7B",
    "Llama-3.1-8B-Instruct":       "Llama-8B",
    "Phi-4":                       "Phi-4",
}
DATASET_SHORT = {"math500": "MATH500", "gsm8k": "GSM8K"}


def load_summary():
    with open(os.path.join(ROOT, "summary.json")) as f:
        return json.load(f)


def load_run(tag):
    path = os.path.join(ROOT, f"{tag}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ─────────────────────────── Figure A ───────────────────────────
def fig_a_non_det_rate(summary):
    """Bar chart: non-det rate per (model, dataset) cell, 3 bars per cell."""
    # Average both pair directions (1v8, 8v32) for robustness.
    # nondet[(model, dataset, method)] = list of 1 - same_toks/n
    nondet = defaultdict(list)
    for row in summary["cross_bs"]:
        key = (row["model"], row["dataset"], row["method"])
        if row["n"] == 0:
            continue
        nondet[key].append(1.0 - row["same_toks"] / row["n"])

    models  = ["DeepSeek-R1-Distill-Qwen-7B", "Llama-3.1-8B-Instruct", "Phi-4"]
    dsets   = ["math500", "gsm8k"]
    cells   = [(m, d) for m in models for d in dsets]
    labels  = [f"{MODEL_SHORT[m]}\n{DATASET_SHORT[d]}" for m, d in cells]

    x       = np.arange(len(cells))
    width   = 0.26

    fig, ax = plt.subplots(figsize=(7.0, 2.9), dpi=150)
    for i, m in enumerate(METHOD_ORDER):
        vals = [statistics.mean(nondet[(mod, ds, m)]) if nondet[(mod, ds, m)] else 0
                for (mod, ds) in cells]
        ax.bar(x + (i - 1) * width, vals, width,
               label=METHOD_LABELS[m], color=METHOD_COLORS[m], edgecolor="black", linewidth=0.4)
        for xi, v in zip(x + (i - 1) * width, vals):
            ax.text(xi, v + 0.02, f"{v:.0%}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Non-determinism rate\n(1 − same-token fraction)", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.legend(loc="upper right", fontsize=8, ncol=3, frameon=False)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out = os.path.join(OUT, "fig_non_det_rate.pdf")
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"), dpi=200)
    plt.close(fig)
    print(f"saved {out}")


# ─────────────────────────── Figure B ───────────────────────────
def fig_b_div_hist(summary, model="DeepSeek-R1-Distill-Qwen-7B", dataset="math500"):
    """Divergence-position histogram for one (model, dataset) cell."""
    # Gather divergence positions from raw per-problem data of bs1/bs8 pair.
    # first_100_tokens list gives the compare window; when all 100 match
    # we treat div_pos = 100 (rightmost bin, "no divergence within window").
    # ``analyze_matrix.cross_bs_stats`` already exposes divs excluding -1
    # but we want all problems including identical ones mapped to 100.
    import sys
    sys.path.insert(0, "/home/kec23008/docker-sys/dllm/research")
    tag_templates = {
        "bf16_baseline": "ds7_math_bf16",
        "layercast":     "ds7_math_lc",
        "dllm_hybrid":   "ds7_math_hy",
    }
    if dataset == "gsm8k":
        tag_templates = {k: v.replace("math", "gsm") for k, v in tag_templates.items()}

    bins = np.linspace(0, 100, 11)   # 10 bins, width 10
    fig, ax = plt.subplots(figsize=(4.4, 2.8), dpi=150)

    linestyles = {"bf16_baseline": ":", "layercast": "--", "dllm_hybrid": "-"}
    for m in METHOD_ORDER:
        base = tag_templates[m]
        a = load_run(f"{base}_bs1"); b = load_run(f"{base}_bs8")
        c = load_run(f"{base}_bs32")
        if not a or not b or not c:
            continue

        divs = []
        # Aggregate both 1v8 and 8v32 pairs.
        for ra, rb in [(a, b), (b, c)]:
            pa = ra["per_problem"]; pb = rb["per_problem"]
            for i in range(min(len(pa), len(pb))):
                if pa[i]["token_hash"] == pb[i]["token_hash"]:
                    divs.append(100)
                else:
                    ta = pa[i].get("first_100_tokens") or pa[i].get("first_20_tokens", [])
                    tb = pb[i].get("first_100_tokens") or pb[i].get("first_20_tokens", [])
                    d = next((j for j, (x, y) in enumerate(zip(ta, tb)) if x != y),
                             min(len(ta), len(tb)))
                    divs.append(d)

        med = statistics.median(divs) if divs else 0
        ax.hist(divs, bins=bins, histtype="step", linewidth=1.7,
                linestyle=linestyles[m],
                label=f"{METHOD_LABELS[m]}  (median={med:.0f})",
                color=METHOD_COLORS[m])

    ax.set_xlabel("First-divergence token position\n(within first 100 decoded tokens)", fontsize=9)
    ax.set_ylabel("# problems", fontsize=9)
    ax.set_xlim(0, 100)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.grid(alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(f"{MODEL_SHORT[model]} · {DATASET_SHORT[dataset]} (bs-pairs 1↔8, 8↔32)",
                 fontsize=9)
    fig.tight_layout()
    mslug = MODEL_SHORT[model].lower().replace("-", "")
    dslug = DATASET_SHORT[dataset].lower()
    out = os.path.join(OUT, f"fig_div_hist_{mslug}_{dslug}.pdf")
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"), dpi=200)
    plt.close(fig)
    print(f"saved {out}")


# ─────────────────────────── Figure C ───────────────────────────
def fig_c_same_prefix_curve():
    """Per-model panel: fraction of problem pairs whose first-K tokens are
    identical, as a function of K. A right-persisting curve ≡ later divergence.

    Each panel aggregates bs-pair directions (1↔8, 8↔32) over all problems.
    """
    models = [
        ("DeepSeek-R1-Distill-Qwen-7B", "math500", "ds7_math_{m}"),
        ("Llama-3.1-8B-Instruct",       "math500", "llm8_math_{m}"),
        ("Phi-4",                       "math500", "phi4_math_{m}"),
    ]
    method_tag = {"bf16_baseline": "bf16", "layercast": "lc", "dllm_hybrid": "hy"}
    linestyles = {"bf16_baseline": ":", "layercast": "--", "dllm_hybrid": "-"}

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.6), dpi=150, sharey=True)
    Ks = list(range(1, 101))

    for ax, (model_full, dataset, tpl) in zip(axes, models):
        for m in METHOD_ORDER:
            base = tpl.format(m=method_tag[m])
            a = load_run(f"{base}_bs1"); b = load_run(f"{base}_bs8")
            c = load_run(f"{base}_bs32")
            if not a or not b or not c:
                continue

            # For each problem-pair, compute first-divergence position.
            first_divs = []
            for ra, rb in [(a, b), (b, c)]:
                pa = ra["per_problem"]; pb = rb["per_problem"]
                for i in range(min(len(pa), len(pb))):
                    if pa[i]["token_hash"] == pb[i]["token_hash"]:
                        first_divs.append(10**9)   # sequences match beyond window
                    else:
                        ta = pa[i].get("first_100_tokens") or pa[i].get("first_20_tokens", [])
                        tb = pb[i].get("first_100_tokens") or pb[i].get("first_20_tokens", [])
                        d = next((j for j, (x, y) in enumerate(zip(ta, tb)) if x != y),
                                 min(len(ta), len(tb)))
                        first_divs.append(d)
            n = len(first_divs)
            if n == 0:
                continue
            frac = [sum(1 for d in first_divs if d >= K) / n for K in Ks]
            ax.plot(Ks, frac, linewidth=1.8, linestyle=linestyles[m],
                    color=METHOD_COLORS[m], label=METHOD_LABELS[m])
        ax.set_title(f"{MODEL_SHORT[model_full]} · {DATASET_SHORT[dataset]}", fontsize=9)
        ax.set_xlabel("K (tokens)", fontsize=9)
        ax.grid(alpha=0.25, linestyle="--")
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("Prob. first-K tokens match\nacross batch size", fontsize=9)
    axes[0].legend(loc="lower left", fontsize=8, frameon=False)
    fig.tight_layout()
    out = os.path.join(OUT, "fig_same_prefix_curve.pdf")
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"), dpi=200)
    plt.close(fig)
    print(f"saved {out}")


def main():
    summary = load_summary()
    fig_a_non_det_rate(summary)
    fig_b_div_hist(summary, "DeepSeek-R1-Distill-Qwen-7B", "math500")
    fig_b_div_hist(summary, "Llama-3.1-8B-Instruct",       "math500")
    fig_c_same_prefix_curve()


if __name__ == "__main__":
    main()
