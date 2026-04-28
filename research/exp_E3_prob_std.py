#!/usr/bin/env python3
"""
E3 — Avg_Std@top1_prob (plan.md §3 / Priority P0).

Per position: the probability of the greedy (top-1) token.
Across bs variants: how much does that probability vary?

  For each problem:
    prompt_ids = tokenize(prompt)                         # shape [1, L]
    for bs in BS_LIST:
      ids = prompt_ids.repeat(bs, 1)                      # [bs, L]
      logits = model(ids).logits[0]                       # row 0, [L, V]
      top1_prob[bs] = softmax(logits).max(dim=-1).values  # [L]
    stacked = stack(top1_prob[bs] for bs in BS_LIST)      # [n_bs, L]
    per_problem_std = stacked.std(dim=0).mean()           # scalar

  overall_Avg_Std = mean of per_problem_stds across problems

Prefill-only (no generation) → cheap (~0.5s / prefill).

Baseline schemes follow plan.md §0. FP32full skipped (reload cost).
"""
import sys, os, time, json, gc, argparse, statistics
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings; warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from research import determ_llm
from motivation.test_layercast_latency import apply_layercast, remove_layercast

MATH500_CACHE = "/home/kec23008/docker-sys/dllm/research/math500_cached.json"

MODELS = {
    "llama8b":    "/home/kec23008/docker-sys/Models/Llama-3.1-8B-Instruct",
    "deepseek7b": "/home/kec23008/docker-sys/Models/DeepSeek-R1-Distill-Qwen-7B",
}

N_PROBLEMS = 50
BS_LIST    = [1, 4, 8]           # bs=16 OOMs LayerCast on a 20 GB free GPU
MAX_INPUT_TOKENS = 256           # truncate long MATH500 prompts to keep memory bounded


def make_schemes(model):
    state = {}
    def bf16_enter():
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    def bf16_exit(): pass

    def fp32flag_enter():
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    def fp32flag_exit():
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True

    def lc_enter():
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        state["lc"] = apply_layercast(model)
    def lc_exit():
        remove_layercast(model, state.pop("lc"))

    def det_enter():
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        determ_llm.enable(backend="triton")
    def det_exit():
        determ_llm.disable()

    def det_attn_enter():
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        determ_llm.enable(backend="triton", attn=True)
    def det_attn_exit():
        determ_llm.disable()

    return [
        ("BF16",           bf16_enter,     bf16_exit),
        ("FP32flag",       fp32flag_enter, fp32flag_exit),
        ("LayerCast",      lc_enter,       lc_exit),
        ("DetermLLM",      det_enter,      det_exit),
        ("DetermLLM+attn", det_attn_enter, det_attn_exit),
    ]


def load_problems(n):
    with open(MATH500_CACHE) as f:
        return json.load(f)[:n]


def prefill_top1(model, ids):
    """Return top-1 prob at each position for row 0 of the batch. [L] on CPU."""
    with torch.no_grad():
        logits = model(input_ids=ids, use_cache=False).logits[0]  # [L, V]
    probs = F.softmax(logits.float(), dim=-1)
    return probs.max(dim=-1).values.detach().cpu()                # [L]


def run_scheme(model, tok, problems, device, scheme_name, enter_fn, exit_fn):
    print(f"\n-- {scheme_name} --", flush=True)
    enter_fn()
    per_problem_std = []
    per_problem_maxdiff = []
    try:
        t0 = time.perf_counter()
        skipped = 0
        for i, ex in enumerate(problems):
            prompt = ex["problem"]
            enc = tok(prompt, return_tensors="pt",
                      truncation=True, max_length=MAX_INPUT_TOKENS)["input_ids"].to(device)

            try:
                top1_by_bs = []
                for bs in BS_LIST:
                    ids = enc.repeat(bs, 1).contiguous()
                    top1 = prefill_top1(model, ids)
                    top1_by_bs.append(top1)
                stacked = torch.stack(top1_by_bs)                                # [n_bs, L]
                std_pos = stacked.std(dim=0)                                     # [L]
                max_diff = (stacked - stacked[0:1]).abs().max().item()
                per_problem_std.append(std_pos.mean().item())
                per_problem_maxdiff.append(max_diff)
            except torch.cuda.OutOfMemoryError:
                skipped += 1
                torch.cuda.empty_cache(); gc.collect()
                continue

            if (i + 1) % 10 == 0:
                elapsed = time.perf_counter() - t0
                print(f"   [{i+1:>3}/{len(problems)}] elapsed={elapsed:.0f}s  "
                      f"running avg_std={statistics.mean(per_problem_std):.3e}  "
                      f"skip={skipped}", flush=True)
    finally:
        exit_fn()
        torch.cuda.empty_cache(); gc.collect()

    return {
        "scheme":              scheme_name,
        "n_problems":          len(problems),
        "bs_list":             BS_LIST,
        "avg_std_top1_prob":   statistics.mean(per_problem_std),
        "median_std":          statistics.median(per_problem_std),
        "avg_max_diff":        statistics.mean(per_problem_maxdiff),
        "max_max_diff":        max(per_problem_maxdiff),
        "per_problem_std":     per_problem_std,
        "per_problem_maxdiff": per_problem_maxdiff,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS.keys()), default="llama8b")
    ap.add_argument("--n-problems", type=int, default=N_PROBLEMS)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model_path = MODELS[args.model]
    if args.out is None:
        args.out = f"/home/kec23008/docker-sys/dllm/research/exp_E3/E3_{args.model}.json"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    print(f"GPU: {torch.cuda.get_device_name(0)}  "
          f"free={torch.cuda.mem_get_info()[0] / 1e9:.1f} GB", flush=True)
    print(f"Loading {model_path} ...", flush=True)

    tok = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation="eager",  # consistent with E4
    )
    model.eval()
    device = next(model.parameters()).device

    problems = load_problems(args.n_problems)
    print(f"Loaded {len(problems)} MATH500 problems", flush=True)

    schemes = make_schemes(model)
    all_results = []
    for name, ein, eout in schemes:
        r = run_scheme(model, tok, problems, device, name, ein, eout)
        all_results.append(r)
        print(f"   [{name}] avg_std={r['avg_std_top1_prob']:.3e}  "
              f"median={r['median_std']:.3e}  "
              f"avg_maxdiff={r['avg_max_diff']:.3e}", flush=True)
        with open(args.out, "w") as f:
            json.dump(all_results, f, indent=2)

    print("\n" + "=" * 84)
    print(f"  E3 SUMMARY (Llama-8B, N={args.n_problems} MATH500, bs={BS_LIST})")
    print("=" * 84)
    bf16_std = next((r["avg_std_top1_prob"] for r in all_results if r["scheme"] == "BF16"), None)
    print(f"  {'Scheme':<18} {'Avg_Std@top1':>14} {'Median':>12} {'AvgMaxDiff':>14} {'vs BF16':>10}")
    for r in all_results:
        ratio = r["avg_std_top1_prob"] / bf16_std if bf16_std else 1.0
        print(f"  {r['scheme']:<18} {r['avg_std_top1_prob']:>14.3e} "
              f"{r['median_std']:>12.3e} {r['avg_max_diff']:>14.3e} {ratio:>9.3f}x")


if __name__ == "__main__":
    main()
