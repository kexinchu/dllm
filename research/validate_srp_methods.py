#!/usr/bin/env python3
"""
Method validation: BF16 vs FP32-all vs SRP-FP32.

Single fixed prompt, greedy decode 256 tokens.
- bs=1: 1 reference run
- bs ∈ {2, 4, 8, 16}: 100 runs each

Each run records row-0's 256 tokens + per-token top-1 probability.

Outputs:
- # unique 256-token sequences per scheme (across all 101 runs)
- # unique sequences per (scheme, bs)
- first_div_idx (vs bs=1 reference) distribution per scheme
"""
import sys, os, time, hashlib, json, argparse, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from research.methods import (
    method_BF16, method_SRP_FP32, method_FP32_all, ALL_SITES,
)


MODEL_PATH = "/home/kec23008/docker-sys/Models/Llama-3.1-8B-Instruct"
MAX_NEW   = 256
BS_LIST   = [2, 4, 8, 16]
RUNS_PER_BS = 100                # 4 × 25 = 100 perturbed runs per scheme

DEFAULT_PROMPT = (
    "Explain the concept of deterministic inference in large language models, "
    "covering hardware, software, and algorithmic factors."
)


# ─── manual greedy decode that records row-0 tokens + top-1 probs ──────────────
def manual_decode(model, ids: torch.Tensor, max_new: int):
    """Greedy decode all rows; return row-0's (tokens, probs) lists of length max_new."""
    L_in = ids.shape[1]
    bs = ids.shape[0]

    with torch.no_grad():
        out = model(input_ids=ids, use_cache=False)
        past = out.past_key_values
        logits = out.logits[:, -1]                 # [bs, V]
        p = F.softmax(logits.float(), dim=-1)      # FP32 for numerical stability of `max`
        max_p, max_t = p.max(dim=-1)
        tokens = [int(max_t[0].item())]
        probs  = [float(max_p[0].item())]
        cur = max_t.unsqueeze(1)                   # [bs, 1]

        for _ in range(max_new - 1):
            out = model(input_ids=cur, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[:, -1]
            p = F.softmax(logits.float(), dim=-1)
            max_p, max_t = p.max(dim=-1)
            tokens.append(int(max_t[0].item()))
            probs.append(float(max_p[0].item()))
            cur = max_t.unsqueeze(1)

    return tokens, probs


def first_div_idx(ref_tokens, run_tokens, max_len: int = MAX_NEW):
    n = min(len(ref_tokens), len(run_tokens), max_len)
    for i in range(n):
        if ref_tokens[i] != run_tokens[i]:
            return i
    return max_len  # identical (no divergence within max_len tokens)


# ─── one scheme run: 1 ref + 100 perturbed ─────────────────────────────────────
def run_scheme(model, tok, scheme_name, scheme_cm_factory, prompt, device,
               out_path, all_results):
    print(f"\n{'='*92}\n  SCHEME: {scheme_name}\n{'='*92}", flush=True)

    enc = tok(prompt, return_tensors="pt")["input_ids"].to(device)
    print(f"  prompt={prompt!r}\n  input_len={enc.shape[1]}", flush=True)

    with scheme_cm_factory(model):
        # ── reference run at bs=1 ──
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        ref_tokens, ref_probs = manual_decode(model, enc, MAX_NEW)
        torch.cuda.synchronize()
        t_ref = time.perf_counter() - t0
        ref_hash = hashlib.md5(str(ref_tokens).encode()).hexdigest()[:12]
        print(f"  bs= 1 ref ({t_ref:5.1f}s): hash={ref_hash}", flush=True)

        runs = [{"bs": 1, "run_idx": 0, "tokens": ref_tokens, "probs": ref_probs,
                 "hash": ref_hash, "first_div_idx": MAX_NEW, "wall_s": t_ref}]

        # ── perturbed runs ──
        for bs in BS_LIST:
            ids_bs = enc.repeat(bs, 1).contiguous()
            for r in range(RUNS_PER_BS / bs):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                tokens, probs = manual_decode(model, ids_bs, MAX_NEW)
                torch.cuda.synchronize()
                t = time.perf_counter() - t0
                h = hashlib.md5(str(tokens).encode()).hexdigest()[:12]
                fdi = first_div_idx(ref_tokens, tokens)
                runs.append({"bs": bs, "run_idx": r, "tokens": tokens, "probs": probs,
                             "hash": h, "first_div_idx": fdi, "wall_s": t})
                if r == 0 or (r + 1) % 10 == 0:
                    print(f"    bs={bs:>2} run={r+1:>2}/{RUNS_PER_BS} "
                          f"({t:5.1f}s) hash={h} fdi={fdi}", flush=True)

    result = {"scheme": scheme_name, "prompt": prompt, "ref_hash": ref_hash,
              "ref_tokens": ref_tokens, "ref_probs": ref_probs, "runs": runs}
    all_results.append(result)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print_one_scheme(result)
    return result


# ─── per-scheme summary ────────────────────────────────────────────────────────
def print_one_scheme(result):
    sch = result["scheme"]
    runs = result["runs"]
    by_bs = {}
    for r in runs:
        by_bs.setdefault(r["bs"], []).append(r)

    print(f"\n  --- {sch} aggregate ---")
    all_hashes = [r["hash"] for r in runs]
    print(f"  total unique outputs across {len(runs)} runs (incl. ref): {len(set(all_hashes))}")
    print(f"  {'bs':>3}  {'#runs':>5}  {'unique':>6}  {'match_ref':>9}  "
          f"{'fdi min':>7}  {'fdi mean':>8}  {'fdi max':>7}  {'fdi p50':>7}")
    for bs in sorted(by_bs):
        rs = by_bs[bs]
        hs = [r["hash"] for r in rs]
        n_match = sum(1 for r in rs if r["hash"] == result["ref_hash"])
        if bs == 1:
            print(f"  {bs:>3}  {len(rs):>5}  {len(set(hs)):>6}  {n_match:>9}  "
                  f"{'-':>7}  {'-':>8}  {'-':>7}  {'-':>7}")
        else:
            fdis = [r["first_div_idx"] for r in rs]
            print(f"  {bs:>3}  {len(rs):>5}  {len(set(hs)):>6}  {n_match:>9}  "
                  f"{min(fdis):>7}  {np.mean(fdis):>8.1f}  {max(fdis):>7}  "
                  f"{int(np.median(fdis)):>7}")


# ─── overall final summary ────────────────────────────────────────────────────
def print_final(all_results):
    print(f"\n{'='*92}\n  FINAL COMPARISON\n{'='*92}")
    print(f"  {'Scheme':<12}  {'#unique':>8}  {'#match_ref/100':>15}  "
          f"{'fdi mean':>9}  {'fdi median':>10}  {'fdi min':>8}")
    for r in all_results:
        runs = r["runs"]
        non_ref = [run for run in runs if run["bs"] != 1]
        all_hashes = [run["hash"] for run in runs]
        unique = len(set(all_hashes))
        match = sum(1 for run in non_ref if run["hash"] == r["ref_hash"])
        fdis = [run["first_div_idx"] for run in non_ref]
        if fdis:
            print(f"  {r['scheme']:<12}  {unique:>8}  {match:>4}/{len(non_ref):<10}  "
                  f"{np.mean(fdis):>9.1f}  {int(np.median(fdis)):>10}  {min(fdis):>8}")
        else:
            print(f"  {r['scheme']:<12}  {unique:>8}  -  -  -  -")


# ─── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/kec23008/docker-sys/dllm/research/exp_validate/results.json")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--model-path", default=MODEL_PATH)
    ap.add_argument("--skip-fp32-all", action="store_true",
                    help="skip the FP32-all scheme (saves ~35 min + 32 GB GPU mem)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    print(f"GPU: {torch.cuda.get_device_name(0)}  "
          f"free={torch.cuda.mem_get_info()[0]/1e9:.1f} GB")
    print(f"Model: {args.model_path}")
    print(f"Prompt: {args.prompt!r}")
    print(f"Schemes: BF16 + SRP-FP32" + ("" if args.skip_fp32_all else " + FP32-all"))
    print(f"BS list: {BS_LIST}, RUNS_PER_BS={RUNS_PER_BS}, MAX_NEW={MAX_NEW}")

    tok = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    all_results = []

    # ── BF16 + SRP-FP32 share one model load ──
    print("\nLoading model in BF16 ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation="sdpa")
    model.eval()
    device = next(model.parameters()).device

    run_scheme(model, tok, "BF16", method_BF16, args.prompt, device, args.out, all_results)
    run_scheme(model, tok, "SRP-FP32",
               lambda m: method_SRP_FP32(m, ALL_SITES),
               args.prompt, device, args.out, all_results)

    del model; torch.cuda.empty_cache(); gc.collect()

    # ── FP32-all needs separate model load ──
    if not args.skip_fp32_all:
        print("\nLoading model in FP32 (32 GB) ...", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, dtype=torch.float32, device_map={"": 0},
            attn_implementation="sdpa")
        model.eval()
        device = next(model.parameters()).device

        run_scheme(model, tok, "FP32-all", method_FP32_all,
                   args.prompt, device, args.out, all_results)
        del model; torch.cuda.empty_cache(); gc.collect()

    print_final(all_results)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
