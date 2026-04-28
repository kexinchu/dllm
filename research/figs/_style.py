"""Shared plotting style + data loaders for paper figures.

Every figure script imports from here. JSON loaders are centralized so the
schema can change once and all plots update.
"""
import os, json, glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/kec23008/docker-sys/dllm/research"
OUT  = os.path.join(ROOT, "figs")
os.makedirs(OUT, exist_ok=True)

# canonical scheme order + colors (consistent across all figures)
SCHEME_ORDER = ["BF16", "FP32flag", "LayerCast", "DetermLLM", "DetermLLM+attn"]
SCHEME_COLOR = {
    "BF16":           "#999999",
    "FP32flag":       "#E5B43A",
    "LayerCast":      "#3B83BD",
    "DetermLLM":      "#D62728",
    "DetermLLM+attn": "#2CA02C",
}
SCHEME_MARKER = {
    "BF16":           "o",
    "FP32flag":       "s",
    "LayerCast":      "^",
    "DetermLLM":      "D",
    "DetermLLM+attn": "*",
}


def setup_style():
    plt.rcParams.update({
        "font.size":       10,
        "axes.labelsize":  10,
        "axes.titlesize":  11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": 1.5,
        "lines.markersize": 5,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "figure.dpi":        110,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
    })


def save(fig, name):
    pdf = os.path.join(OUT, f"{name}.pdf")
    png = os.path.join(OUT, f"{name}.png")
    fig.savefig(pdf); fig.savefig(png)
    print(f"  saved: {pdf}")


# ── data loaders ──────────────────────────────────────────────────────────────
def load_E4_runs():
    """Return list of {'model','dataset','bs_list','gen_len','schemes':...}."""
    paths = sorted(glob.glob(os.path.join(ROOT, "exp_E4", "E4_*_eager.json")))
    if not paths:
        # Fallback to non-eager runs if eager not yet present (for partial dev plots)
        paths = sorted(glob.glob(os.path.join(ROOT, "exp_E4", "E4_*.json")))
    return [json.load(open(p)) for p in paths]


def load_E3_runs():
    paths = sorted(glob.glob(os.path.join(ROOT, "exp_E3", "E3_*.json")))
    return {os.path.basename(p).replace("E3_", "").replace(".json", ""): json.load(open(p))
            for p in paths}


def load_E7_runs():
    paths = sorted(glob.glob(os.path.join(ROOT, "exp_E7", "E7_*.json")))
    paths = [p for p in paths if "all" not in os.path.basename(p)]
    return [json.load(open(p)) for p in paths]


def load_E2_run():
    p = os.path.join(ROOT, "exp_E2", "E2.json")
    if not os.path.exists(p):
        # fallback to deepseek log
        for cand in ["E2_deepseek.json", "E2_deepseek_eager.json"]:
            p2 = os.path.join(ROOT, "exp_E2", cand)
            if os.path.exists(p2): return json.load(open(p2))
        return None
    return json.load(open(p))
