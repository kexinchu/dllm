"""Phase-3 final paper figures from n=50 matrix.

Three figures matching Yuan et al. 2025 visualization style:

  Fig 1  Same-first-K survival curve across models (headline).
  Fig 2  Div_Index distribution (log-scale hist + box) for DeepSeek-7B.
  Fig 3  Runtime-vs-Determinism Pareto scatter.

Expects exp_matrix_n50/*.json written by run_matrix_n50.sh.
"""
import json
import math
import os
import statistics
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = "/home/kec23008/docker-sys/dllm/research/exp_matrix_n50"
OUT  = "/home/kec23008/docker-sys/dllm/research/figs"
os.makedirs(OUT, exist_ok=True)

METHOD_ORDER  = ["bf16_baseline", "layercast", "dllm_hybrid"]
METHOD_LABELS = {"bf16_baseline": "BF16", "layercast": "LayerCast", "dllm_hybrid": "DetermLLM"}
METHOD_COLORS = {"bf16_baseline": "#888888", "layercast": "#1f77b4", "dllm_hybrid": "#d62728"}
METHOD_STYLES = {"bf16_baseline": ":", "layercast": "--", "dllm_hybrid": "-"}

MODEL_TAGS = [
    ("DeepSeek-R1-Distill-Qwen-7B", "DeepSeek-7B", "ds7_math_{m}"),
    ("Llama-3.1-8B-Instruct",       "Llama-8B",    "llm8_math_{m}"),
    ("Phi-4",                       "Phi-4",       "phi4_math_{m}"),
]
MODEL_SHORT = {full: short for full, short, _ in MODEL_TAGS}
METHOD_TAG = {"bf16_baseline": "bf16", "layercast": "lc", "dllm_hybrid": "hy"}


def load_run(tag):
    p = os.path.join(ROOT, f"{tag}.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def first_div(pa, pb):
    if pa["token_hash"] == pb["token_hash"]:
        return min(pa["output_len"], pb["output_len"])
    ta = pa.get("token_ids") or pa.get("first_100_tokens", [])
    tb = pb.get("token_ids") or pb.get("first_100_tokens", [])
    for j, (x, y) in enumerate(zip(ta, tb)):
        if x != y:
            return j
    return min(len(ta), len(tb))


def gather_divs(model_tpl):
    """Return dict[method] -> list of per-pair div positions (1v8, 8v32)."""
    out = {}
    for m in METHOD_ORDER:
        base = model_tpl.format(m=METHOD_TAG[m])
        a = load_run(f"{base}_bs1"); b = load_run(f"{base}_bs8")
        c = load_run(f"{base}_bs32")
        if not a or not b or not c:
            continue
        divs = []
        for ra, rb in [(a, b), (b, c)]:
            pa = ra["per_problem"]; pb = rb["per_problem"]
            n = min(len(pa), len(pb))
            for i in range(n):
                divs.append(first_div(pa[i], pb[i]))
        out[m] = divs
    return out


# ─────────────────────────── Figure 1 ───────────────────────────
def fig1_same_prefix_curve(max_k=500):
    """Headline: Prob(first-K tokens match) vs K, 3 model panels."""
    fig, axes = plt.subplots(1, 3, figsize=(9.8, 2.7), dpi=150, sharey=True)
    Ks = np.arange(1, max_k + 1)

    for ax, (model_full, model_short, tpl) in zip(axes, MODEL_TAGS):
        divs_by_m = gather_divs(tpl)
        for m in METHOD_ORDER:
            if m not in divs_by_m:
                continue
            divs = divs_by_m[m]
            frac = np.array([sum(1 for d in divs if d >= K) / len(divs) for K in Ks])
            ax.plot(Ks, frac, linewidth=1.8, linestyle=METHOD_STYLES[m],
                    color=METHOD_COLORS[m], label=METHOD_LABELS[m])
        ax.set_title(f"{model_short} · MATH500", fontsize=9)
        ax.set_xlabel("K (decoded tokens)", fontsize=9)
        ax.grid(alpha=0.25, linestyle="--")
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("P(first-K tokens identical\nacross batch sizes)", fontsize=9)
    axes[0].legend(loc="lower left", fontsize=8, frameon=False)
    fig.tight_layout()
    out = os.path.join(OUT, "fig1_same_prefix_n50.pdf")
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"), dpi=200)
    plt.close(fig)
    print(f"saved {out}")


# ─────────────────────────── Figure 2 ───────────────────────────
def fig2_div_distribution():
    """Div_Index frequency histogram, log-scale x, 3 methods overlaid per model.
    Matches Yuan et al. 2025 Fig. 5 style. One panel per model."""
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 2.8), dpi=150, sharey=False)
    # log bins shared across methods within a panel
    for ax, (model_full, model_short, tpl) in zip(axes, MODEL_TAGS):
        divs_by_m = gather_divs(tpl)
        if not divs_by_m:
            continue
        all_divs = [d for divs in divs_by_m.values() for d in divs if d > 0]
        if not all_divs:
            continue
        lo = max(1, min(all_divs))
        hi = max(all_divs)
        bins = np.logspace(np.log10(lo), np.log10(hi) + 0.05, 28)
        for m in METHOD_ORDER:
            if m not in divs_by_m:
                continue
            divs = [d for d in divs_by_m[m] if d > 0]
            med = int(statistics.median(divs))
            ax.hist(divs, bins=bins,
                    color=METHOD_COLORS[m], alpha=0.55, edgecolor="black",
                    linewidth=0.3,
                    label=f"{METHOD_LABELS[m]}  (med={med})")
        ax.set_xscale("log")
        ax.set_title(f"{model_short} · MATH500", fontsize=9)
        ax.set_xlabel("Div_Index (log scale)", fontsize=9)
        ax.grid(alpha=0.25, linestyle="--")
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="upper left", fontsize=7.5, frameon=False)
    axes[0].set_ylabel("Frequency", fontsize=9)
    fig.tight_layout()
    out = os.path.join(OUT, "fig2_div_index_hist_n50.pdf")
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"), dpi=200)
    plt.close(fig)
    print(f"saved {out}")


# ─────────────────────────── Figure 3 ───────────────────────────
def fig3_pareto():
    """Runtime vs determinism Pareto scatter.

    X = runtime relative to BF16 baseline (averaged across bs).
    Y = fraction of problems with bit-identical sequences across bs (same_all).
    One marker per (model, method); BF16 sits at (1.0, same_all_BF16).
    """
    fig, ax = plt.subplots(figsize=(5.2, 3.2), dpi=150)
    markers = {"DeepSeek-R1-Distill-Qwen-7B": "o",
               "Llama-3.1-8B-Instruct":       "s",
               "Phi-4":                       "^"}

    for model_full, model_short, tpl in MODEL_TAGS:
        # Baseline runtime reference = BF16 mean across bs.
        rt_bf16 = []
        for bs in (1, 8, 32):
            r = load_run(f"{tpl.format(m='bf16')}_bs{bs}")
            if r:
                rt_bf16.append(r["aggregate"]["total_runtime_seconds"])
        if not rt_bf16:
            continue
        rt_bf16_mean = statistics.mean(rt_bf16)

        for m in METHOD_ORDER:
            base = tpl.format(m=METHOD_TAG[m])
            rts = []
            for bs in (1, 8, 32):
                r = load_run(f"{base}_bs{bs}")
                if r:
                    rts.append(r["aggregate"]["total_runtime_seconds"])
            if not rts:
                continue
            x = statistics.mean(rts) / rt_bf16_mean

            # Determinism = fraction of problems where all 3 bs have identical hash.
            a = load_run(f"{base}_bs1"); b = load_run(f"{base}_bs8"); c = load_run(f"{base}_bs32")
            if not a or not b or not c:
                continue
            n = min(len(a["per_problem"]), len(b["per_problem"]), len(c["per_problem"]))
            same_all = sum(
                1 for i in range(n)
                if a["per_problem"][i]["token_hash"] == b["per_problem"][i]["token_hash"]
                == c["per_problem"][i]["token_hash"]
            ) / n

            ax.scatter(x, same_all, s=140, marker=markers[model_full],
                       facecolor=METHOD_COLORS[m], edgecolor="black", linewidth=0.8,
                       label=f"{model_short} · {METHOD_LABELS[m]}", zorder=3)

    ax.set_xlabel("Runtime relative to BF16 baseline", fontsize=9)
    ax.set_ylabel("Token-wise determinism\nP(all 3 bs produce identical tokens)", fontsize=9)
    ax.axvline(1.0, color="#888", linewidth=0.7, linestyle=":")
    ax.grid(alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7, frameon=False)
    fig.tight_layout()
    out = os.path.join(OUT, "fig3_pareto_n50.pdf")
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"), dpi=200)
    plt.close(fig)
    print(f"saved {out}")


def fig4_avg_std_top1_bars():
    """Avg_Std@top1_prob bar chart, Yuan et al. 2025 Fig. 4 style.
    X axis: 3 models. Grouped bars: BF16 / LayerCast / DetermLLM.
    Log y-scale so Phi-4 (20× smaller values) remains visible alongside
    DeepSeek."""
    # Reuse summary data: Avg_Std@top1 per (model, method).
    import statistics as _st
    data = {}   # model -> method -> avg_std_top1
    for model_full, _, tpl in MODEL_TAGS:
        data[model_full] = {}
        for m in METHOD_ORDER:
            base = tpl.format(m=METHOD_TAG[m])
            runs = [load_run(f"{base}_bs{bs}") for bs in (1, 8, 32)]
            if not all(runs):
                continue
            n = min(len(r["per_problem"]) for r in runs)
            stds = []
            for i in range(n):
                lps = [r["per_problem"][i].get("top1_logprobs", []) for r in runs]
                if not all(lps):
                    continue
                L = min(len(x) for x in lps)
                if L == 0:
                    continue
                stds.append(_st.mean(_st.pstdev([lps[k][j] for k in range(3)])
                                     for j in range(L)))
            data[model_full][m] = _st.mean(stds) if stds else 0.0

    fig, ax = plt.subplots(figsize=(5.6, 2.8), dpi=150)
    models = [m for m, _, _ in MODEL_TAGS]
    model_labels = [MODEL_SHORT[m] for m, _, _ in MODEL_TAGS]
    x = np.arange(len(models))
    width = 0.26
    for i, meth in enumerate(METHOD_ORDER):
        vals = [data[m].get(meth, 0) for m in models]
        bars = ax.bar(x + (i - 1) * width, vals, width,
                      label=METHOD_LABELS[meth],
                      color=METHOD_COLORS[meth],
                      edgecolor="black", linewidth=0.4)
        for xi, v in zip(x + (i - 1) * width, vals):
            ax.text(xi, v * 1.08, f"{v:.3f}", ha="center", va="bottom",
                    fontsize=6.5)

    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, fontsize=9)
    ax.set_ylabel("Avg_Std@top1_prob (nats)", fontsize=9)
    ax.set_yscale("log")
    ax.set_ylim(0.005, 0.5)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", fontsize=8, ncol=3, frameon=False)
    ax.set_title("MATH500, $bs\\in\\{1,8,32\\}$, $n\\!=\\!50$", fontsize=9)
    fig.tight_layout()
    out = os.path.join(OUT, "fig4_avgstd_top1_n50.pdf")
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"), dpi=200)
    plt.close(fig)
    print(f"saved {out}")


def main():
    fig1_same_prefix_curve()
    fig2_div_distribution()
    fig3_pareto()
    fig4_avg_std_top1_bars()


if __name__ == "__main__":
    main()
