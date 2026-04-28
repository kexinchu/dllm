#!/usr/bin/env python3
"""
Reproduce LayerCast paper Fig 4: Avg_Std@top1_prob across 3 precision schemes.

Setup
-----
Model      : DeepSeek-R1-Distill-Qwen-7B  (reasoning, narrower top-1/top-2 gaps)
Dataset    : MATH500 first 5 problems     (quick validation)
Schemes (3): BF16 / FP32-all / SRP-FP32   (model-level vs no-precision-change vs site-fixed-plan)
Configs (5): bs ∈ {1, 2, 4, 8, 16}        (vary "system config" by batch size)
Decode     : greedy 256 tokens (normal inference, NOT teacher-forced)

Per (scheme, problem):
  for each bs in {1,2,4,8,16}:
    greedy decode 256 tokens, capture row-0's
      tokens[256]   - the actual greedy token sequence
      top1_prob[256]- prob of the token chosen at each step
  pre_div_idx = min over bs∈{2,4,8,16} of first_div_idx(tokens_bs vs tokens_bs1)
  for i in [0, pre_div_idx):    # positions where ALL 5 bs still pick same token
    std_pos[i] = std over 5 bs of top1_prob_bs[i]
  avg_std_problem = mean of std_pos[i] over i in [0, pre_div_idx)

Per scheme: Avg_Std@top1_prob = mean of avg_std_problem over 5 problems.

Outputs
-------
1. JSON with all per-problem data
2. Markdown summary table
3. Bar chart (paper Fig 4 style): X = scheme, Y = log Avg_Std@top1_prob
"""
import sys, os, time, json, gc, hashlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from research.methods import (
    method_BF16, method_SRP_FP32, method_FP32_all, ALL_SITES,
)


MODEL_PATH = "/home/kec23008/docker-sys/Models/DeepSeek-R1-Distill-Qwen-7B"
MATH500_CACHE = "/home/kec23008/docker-sys/dllm/research/math500_cached.json"
N_PROBLEMS = 5
MAX_NEW = 256
BS_LIST = [1, 2, 4, 8, 16]    # bs=1 is reference
SYSTEM_REASON = "Please reason step by step, and put your final answer within \\boxed{}."

OUT_DIR = "/home/kec23008/docker-sys/dllm/research/exp_validate"
OUT_JSON = os.path.join(OUT_DIR, "avg_std_results.json")
OUT_FIG  = os.path.join(OUT_DIR, "fig_avg_std.png")
OUT_MD   = os.path.join(OUT_DIR, "avg_std_summary.md")


# ─── prompt wrap (DeepSeek-R1 chat template) ─────────────────────────────────
def wrap_prompt(tok, problem):
    msgs = [{"role": "user", "content": f"{problem}\n\n{SYSTEM_REASON}"}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


# ─── manual greedy decode w/ per-token prob capture (row 0 only) ─────────────
def greedy_decode_capture(model, ids: torch.Tensor, max_new: int):
    """Greedy decode all rows, return row-0's tokens + per-step top-1 probs."""
    L_in = ids.shape[1]; bs = ids.shape[0]
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True)
        past = out.past_key_values
        logits = out.logits[:, -1]
        p = F.softmax(logits.float(), dim=-1)
        max_p, max_t = p.max(dim=-1)
        toks = [int(max_t[0].item())]
        prbs = [float(max_p[0].item())]
        cur = max_t.unsqueeze(1)
        for _ in range(max_new - 1):
            out = model(input_ids=cur, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[:, -1]
            p = F.softmax(logits.float(), dim=-1)
            max_p, max_t = p.max(dim=-1)
            toks.append(int(max_t[0].item()))
            prbs.append(float(max_p[0].item()))
            cur = max_t.unsqueeze(1)
    return toks, prbs


def first_div(ref_tokens, run_tokens, max_len=MAX_NEW):
    n = min(len(ref_tokens), len(run_tokens), max_len)
    for i in range(n):
        if ref_tokens[i] != run_tokens[i]:
            return i
    return max_len


# ─── load problems ───────────────────────────────────────────────────────────
def load_problems():
    with open(MATH500_CACHE) as f:
        data = json.load(f)
    return data[:N_PROBLEMS]


# ─── one (scheme, problem) cell ──────────────────────────────────────────────
def run_one(model, tok, scheme_name, scheme_cm_factory, problem, problem_idx, device):
    prompt_text = wrap_prompt(tok, problem["problem"])
    enc = tok(prompt_text, return_tensors="pt")["input_ids"].to(device)
    L_in = enc.shape[1]
    print(f"   problem {problem_idx} (input_len={L_in}, level={problem.get('level','?')})", flush=True)

    tokens_by_bs = {}
    probs_by_bs  = {}

    with scheme_cm_factory(model):
        for bs in BS_LIST:
            ids = enc.repeat(bs, 1).contiguous()
            torch.cuda.synchronize(); t0 = time.perf_counter()
            toks, prbs = greedy_decode_capture(model, ids, MAX_NEW)
            torch.cuda.synchronize(); t = time.perf_counter() - t0
            tokens_by_bs[bs] = toks
            probs_by_bs[bs]  = prbs
            print(f"     bs={bs:>2} ({t:5.1f}s)", flush=True)

    # first_div_idx vs bs=1 reference
    ref_toks = tokens_by_bs[1]
    first_div_idx = {bs: first_div(ref_toks, tokens_by_bs[bs]) for bs in BS_LIST if bs != 1}
    pre_div = min([MAX_NEW] + list(first_div_idx.values()))

    # std at every position for diagnostics; valid (pre-div) only used for Avg_Std
    prob_matrix = np.array([probs_by_bs[bs] for bs in BS_LIST])  # [5, 256]
    std_per_pos_full = prob_matrix.std(axis=0).tolist()           # [256]

    if pre_div > 0:
        avg_std_problem = float(np.mean(std_per_pos_full[:pre_div]))
        max_std_pre_div = float(np.max(std_per_pos_full[:pre_div]))
    else:
        avg_std_problem = None
        max_std_pre_div = None

    return {
        "problem_idx": problem_idx,
        "level": problem.get("level"),
        "input_len": L_in,
        "tokens_by_bs": {str(bs): tokens_by_bs[bs] for bs in BS_LIST},
        "probs_by_bs":  {str(bs): probs_by_bs[bs]  for bs in BS_LIST},
        "first_div_idx": {str(bs): first_div_idx[bs] for bs in first_div_idx},
        "pre_div_idx": pre_div,
        "std_per_pos": std_per_pos_full,
        "avg_std_problem": avg_std_problem,
        "max_std_pre_div": max_std_pre_div,
    }


# ─── one full scheme ─────────────────────────────────────────────────────────
def run_scheme(model, tok, scheme_name, scheme_cm_factory, problems, device,
               all_results):
    print(f"\n{'='*92}\n  SCHEME: {scheme_name}\n{'='*92}", flush=True)
    rows = []
    for i, prob in enumerate(problems):
        r = run_one(model, tok, scheme_name, scheme_cm_factory, prob, i, device)
        rows.append(r)
        with open(OUT_JSON, "w") as f:
            json.dump(all_results + [{"scheme": scheme_name, "problems": rows}], f)

    # aggregate
    valid = [r for r in rows if r["avg_std_problem"] is not None]
    avg_std_scheme = float(np.mean([r["avg_std_problem"] for r in valid])) if valid else None
    pre_divs = [r["pre_div_idx"] for r in rows]
    summary = {
        "scheme": scheme_name,
        "n_problems": len(rows),
        "Avg_Std_top1_prob": avg_std_scheme,
        "median_pre_div_idx": int(np.median(pre_divs)),
        "min_pre_div_idx": int(np.min(pre_divs)),
        "max_pre_div_idx": int(np.max(pre_divs)),
        "problems": rows,
    }
    print(f"   Avg_Std@top1_prob = {avg_std_scheme:.3e}" if avg_std_scheme is not None
          else "   Avg_Std@top1_prob = N/A (no pre-div positions)")
    print(f"   pre_div_idx: min={summary['min_pre_div_idx']}, "
          f"med={summary['median_pre_div_idx']}, max={summary['max_pre_div_idx']}")
    return summary


# ─── markdown + figure ───────────────────────────────────────────────────────
def write_markdown(summaries):
    lines = ["# Avg_Std@top1_prob validation — DeepSeek-R1-Distill-Qwen-7B / MATH500 (5 problems)",
             "",
             f"Configs: bs ∈ {BS_LIST}",
             f"Decode: greedy 256 tokens",
             "",
             "## Per scheme",
             "",
             "| Scheme | Avg_Std@top1_prob | min pre_div | median pre_div | max pre_div |",
             "|---|---|---|---|---|"]
    for s in summaries:
        astd = s["Avg_Std_top1_prob"]
        astd_s = f"{astd:.3e}" if astd is not None else "N/A"
        lines.append(f"| {s['scheme']} | {astd_s} | {s['min_pre_div_idx']} | "
                     f"{s['median_pre_div_idx']} | {s['max_pre_div_idx']} |")

    lines += ["", "## Per problem (per scheme)", ""]
    for s in summaries:
        lines.append(f"### {s['scheme']}")
        lines.append("")
        lines.append("| problem | level | input_len | pre_div | avg_std_problem | max_std_pre_div |")
        lines.append("|---|---|---|---|---|---|")
        for r in s["problems"]:
            astd = r["avg_std_problem"]
            mstd = r["max_std_pre_div"]
            astd_s = f"{astd:.3e}" if astd is not None else "N/A"
            mstd_s = f"{mstd:.3e}" if mstd is not None else "N/A"
            lines.append(f"| {r['problem_idx']} | {r['level']} | {r['input_len']} | "
                         f"{r['pre_div_idx']} | {astd_s} | {mstd_s} |")
        lines.append("")

    with open(OUT_MD, "w") as f:
        f.write("\n".join(lines))
    print(f"\nMarkdown summary: {OUT_MD}")


def write_figure(summaries):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    schemes = [s["scheme"] for s in summaries]
    values = [(s["Avg_Std_top1_prob"] or 1e-12) for s in summaries]
    floor = 1e-9   # for log scale visibility

    fig, ax = plt.subplots(figsize=(5, 3.2))
    colors = {"BF16":"#999999", "FP32-all":"#D62728", "SRP-FP32":"#2CA02C"}
    bar_colors = [colors.get(s, "#1f77b4") for s in schemes]
    bars = ax.bar(schemes, [max(v, floor) for v in values], color=bar_colors,
                  edgecolor="black", linewidth=0.5)
    ax.set_yscale("log")
    ax.set_ylim(floor / 5, 1.0)
    ax.set_ylabel("Avg_Std @ top-1 prob  (log)")
    ax.axhline(7.8125e-3, color="grey", linestyle="--", linewidth=0.7)
    ax.text(len(schemes)-0.5, 7.8125e-3*1.4, "BF16 step (7.8e-3)", fontsize=8, ha="right")
    ax.axhline(1.2e-7, color="grey", linestyle=":", linewidth=0.7)
    ax.text(len(schemes)-0.5, 1.2e-7*1.4, "FP32 step (1.2e-7)", fontsize=8, ha="right")
    ax.set_title("Avg_Std@top1_prob — DeepSeek-7B / MATH500 (5 problems, 5 bs configs)")
    for bar, val in zip(bars, values):
        if val > floor:
            ax.text(bar.get_x() + bar.get_width()/2, val * 1.3, f"{val:.2e}",
                    ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=150)
    print(f"Figure saved: {OUT_FIG}")


# ─── main ────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"GPU: {torch.cuda.get_device_name(0)}  free={torch.cuda.mem_get_info()[0]/1e9:.1f} GB", flush=True)
    print(f"Model: {MODEL_PATH}")
    print(f"N_PROBLEMS={N_PROBLEMS}  BS_LIST={BS_LIST}  MAX_NEW={MAX_NEW}")

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    problems = load_problems()
    print(f"Loaded {len(problems)} MATH500 problems")

    summaries = []

    # ── BF16 + SRP-FP32 share BF16 model load ──
    print("\nLoading BF16 model ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa")
    model.eval()
    device = next(model.parameters()).device

    summaries.append(run_scheme(model, tok, "BF16", method_BF16, problems, device, summaries))
    summaries.append(run_scheme(model, tok, "SRP-FP32",
                                lambda m: method_SRP_FP32(m, ALL_SITES),
                                problems, device, summaries))

    del model; torch.cuda.empty_cache(); gc.collect()

    # ── FP32-all needs separate load ──
    print("\nLoading FP32 model (~32 GB) ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.float32, device_map={"": 0}, attn_implementation="sdpa")
    model.eval()
    device = next(model.parameters()).device

    summaries.append(run_scheme(model, tok, "FP32-all", method_FP32_all, problems, device, summaries))
    del model; torch.cuda.empty_cache(); gc.collect()

    # final write
    with open(OUT_JSON, "w") as f:
        json.dump(summaries, f)

    # ── Final summary ──
    print(f"\n{'='*92}\n  FINAL SUMMARY\n{'='*92}")
    print(f"  {'Scheme':<12} {'Avg_Std@top1_prob':>20} {'min pre_div':>12} {'med pre_div':>12}")
    for s in summaries:
        astd = s["Avg_Std_top1_prob"]
        astd_s = f"{astd:.3e}" if astd is not None else "N/A"
        print(f"  {s['scheme']:<12} {astd_s:>20} {s['min_pre_div_idx']:>12} {s['median_pre_div_idx']:>12}")

    write_markdown(summaries)
    write_figure(summaries)

    print(f"\nSaved: {OUT_JSON}")


if __name__ == "__main__":
    main()
