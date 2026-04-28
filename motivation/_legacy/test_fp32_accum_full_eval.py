#!/usr/bin/env python3
"""
Full evaluation of FP32 Accumulation Only approach.

Key insight: HuggingFace model.generate() is already deterministic on single GPU
because kernel selection doesn't change for same-shape inputs.
The non-determinism problem occurs in serving engines (vLLM/SGLang) with
continuous batching, split-KV, and dynamic kernel selection.

This test simulates the effect by:
1. Comparing LOGITS (not tokens) across different batch compositions
   - Even if tokens match via argmax, logits can differ
2. Manually testing individual ops with different reduction orders
3. Measuring the "near-tie vulnerability" — how often logits are close enough
   that a small perturbation could flip the argmax

Test structure:
  Test 1: Logits-level batch invariance (BF16 vs FP32 accum)
  Test 2: Op-level non-determinism simulation (shuffle reduction order)
  Test 3: Performance comparison
  Test 4: Near-tie prevalence analysis
  Test 5: Split-KV simulation
"""
import sys
import os
import time
import json
import warnings
from collections import Counter

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from FP32.model_patcher import (
    fp32_accum_mode,
    apply_fp32_accum_all,
    restore_fp32_accum_all,
)

MODEL_PATH = "/home/kec23008/docker-sys/Models/Llama-3.1-8B-Instruct"
PROMPT = "What is deterministic inference in large language models?"
MAX_NEW_TOKENS = 32


def load_model():
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print(f"  Model loaded on {next(model.parameters()).device}")
    return model, tokenizer


def header(title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")


# ============================================================================
# Test 1: Logits-level batch invariance
# ============================================================================

def test_logits_batch_invariance(model, tokenizer):
    """
    Compare logits of same prompt at batch_size=1 vs batch_size=2/4/8/16.
    Even if argmax tokens are same, logits can differ due to:
    - Different GEMM kernel selection (split-K)
    - Different attention split-KV strategy
    - Padding interactions
    """
    header("TEST 1: LOGITS-LEVEL BATCH INVARIANCE")
    device = next(model.parameters()).device

    fillers = [
        "What is the capital of France?",
        "Explain quantum computing in detail.",
        "Write a haiku about rain and mountains.",
        "How does photosynthesis work step by step?",
        "What is 2 + 2 and why?",
        "Tell me a joke about programming.",
        "Describe the solar system in order.",
        "What is machine learning used for?",
        "How do airplanes fly and stay in the air?",
        "What is gravity and how does it work?",
        "Explain neural networks to a child.",
        "What is the speed of light in vacuum?",
        "How does DNA replication work exactly?",
        "What is entropy in thermodynamics?",
        "Explain the Turing test and its significance.",
    ]

    results = {}

    for mode_name, patch_kwargs in [
        ("Pure BF16", None),
        ("Linear-only FP32 accum", dict(patch_linear=True, patch_rmsnorm=False, patch_attention=False, patch_softmax=False)),
        ("Full FP32 accum", dict(patch_linear=True, patch_rmsnorm=True, patch_attention=True, patch_softmax=True)),
    ]:
        print(f"\n  --- {mode_name} ---")
        originals = None
        if patch_kwargs:
            originals = apply_fp32_accum_all(model, **patch_kwargs)

        try:
            # Reference: batch_size=1
            inputs_1 = tokenizer(PROMPT, return_tensors="pt", truncation=True).to(device)
            with torch.no_grad():
                logits_ref = model(**inputs_1).logits[0]  # [seq_len, vocab]

            batch_results = {}
            for bs in [2, 4, 8, 16]:
                prompts = [PROMPT] + fillers[:bs-1]
                inputs_b = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
                with torch.no_grad():
                    logits_batch = model(**inputs_b).logits[0]  # first sample

                # Compare up to the ref length (batch may have different length due to padding)
                min_len = min(logits_ref.shape[0], logits_batch.shape[0])
                diff = (logits_ref[:min_len].float() - logits_batch[:min_len].float()).abs()
                max_diff = diff.max().item()
                mean_diff = diff.mean().item()
                nonzero = (diff > 0).sum().item()
                total = diff.numel()

                # Check argmax match
                argmax_ref = logits_ref[:min_len].float().argmax(dim=-1)
                argmax_batch = logits_batch[:min_len].float().argmax(dim=-1)
                argmax_match = (argmax_ref == argmax_batch).all().item()

                print(f"    bs={bs:>2}: logits_max_diff={max_diff:.4e}, mean={mean_diff:.4e}, "
                      f"nonzero={nonzero}/{total} ({nonzero/total*100:.1f}%), "
                      f"argmax_match={'YES' if argmax_match else 'NO'}")

                batch_results[f"bs{bs}"] = {
                    "logits_max_diff": max_diff,
                    "logits_mean_diff": mean_diff,
                    "nonzero_ratio": nonzero / total,
                    "argmax_match": argmax_match,
                }

            results[mode_name] = batch_results

        finally:
            if originals:
                restore_fp32_accum_all(model, originals)

    return results


# ============================================================================
# Test 2: Op-level non-determinism simulation
# ============================================================================

def test_op_level_nondeterminism():
    """
    Simulate the non-determinism that occurs in serving engines:
    - GEMM with different split-K strategies
    - Attention with different split-KV
    - RMSNorm with different reduction chunk sizes

    For each op, run with shuffled reduction order and check if BF16 output differs.
    This is the core hypothesis: FP32 accum should make results invariant to order.
    """
    header("TEST 2: OP-LEVEL NON-DETERMINISM SIMULATION")
    device = torch.device("cuda")
    torch.manual_seed(42)

    results = {}

    # --- GEMM: simulate split-K with different K-splits ---
    print("\n  --- GEMM: Split-K simulation ---")
    M, K, N = 32, 4096, 4096
    A = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    B = torch.randn(K, N, device=device, dtype=torch.bfloat16)

    def gemm_split_k(a, b, num_splits, use_fp32_accum=False):
        """Simulate split-K GEMM: split along K, reduce partial results."""
        K = a.shape[1]
        chunk = K // num_splits
        partials = []
        for i in range(num_splits):
            start = i * chunk
            end = K if i == num_splits - 1 else (i + 1) * chunk
            if use_fp32_accum:
                p = (a[:, start:end].float() @ b[start:end, :].float())  # FP32 accum
                partials.append(p)
            else:
                p = a[:, start:end] @ b[start:end, :]  # BF16 accum
                partials.append(p)

        # Reduce partials (this is where order matters)
        if use_fp32_accum:
            result = sum(partials)  # FP32 sum
            return result.to(torch.bfloat16)
        else:
            result = partials[0]
            for p in partials[1:]:
                result = result + p  # BF16 additions
            return result

    # Compare different split-K values
    gemm_bf16_results = {}
    gemm_fp32_results = {}
    ref_bf16 = gemm_split_k(A, B, 1, use_fp32_accum=False)
    ref_fp32 = gemm_split_k(A, B, 1, use_fp32_accum=True)

    for splits in [2, 4, 8, 16, 32]:
        out_bf16 = gemm_split_k(A, B, splits, use_fp32_accum=False)
        out_fp32 = gemm_split_k(A, B, splits, use_fp32_accum=True)

        diff_bf16 = (out_bf16.float() - ref_bf16.float()).abs().max().item()
        diff_fp32 = (out_fp32.float() - ref_fp32.float()).abs().max().item()
        gemm_bf16_results[splits] = diff_bf16
        gemm_fp32_results[splits] = diff_fp32

        print(f"    splits={splits:>2}: BF16 max_diff={diff_bf16:.4e}, FP32_accum max_diff={diff_fp32:.4e}")

    results["gemm"] = {
        "bf16_diffs": gemm_bf16_results,
        "fp32_diffs": gemm_fp32_results,
    }

    # --- RMSNorm: simulate different chunk-based reductions ---
    print("\n  --- RMSNorm: Chunk reduction simulation ---")
    hidden = 4096
    x = torch.randn(32, hidden, device=device, dtype=torch.bfloat16)
    weight = torch.ones(hidden, device=device, dtype=torch.bfloat16)

    def rmsnorm_chunked(x, weight, num_chunks, use_fp32_accum=False):
        """RMSNorm with chunked variance reduction."""
        N = x.shape[-1]
        chunk_size = N // num_chunks

        if use_fp32_accum:
            x_fp32 = x.float()
            partials = []
            for i in range(num_chunks):
                start = i * chunk_size
                end = N if i == num_chunks - 1 else (i + 1) * chunk_size
                partials.append(x_fp32[:, start:end].pow(2).sum(-1))
            variance = sum(partials) / N
            rstd = torch.rsqrt(variance.unsqueeze(-1) + 1e-6)
            return (x * rstd.to(x.dtype)) * weight
        else:
            partials = []
            for i in range(num_chunks):
                start = i * chunk_size
                end = N if i == num_chunks - 1 else (i + 1) * chunk_size
                partials.append(x[:, start:end].float().pow(2).sum(-1).to(torch.bfloat16))
            variance = sum(partials).float() / N
            rstd = torch.rsqrt(variance.unsqueeze(-1) + 1e-6)
            return (x * rstd.to(x.dtype)) * weight

    ref_norm = rmsnorm_chunked(x, weight, 1, use_fp32_accum=False)
    norm_results = {}
    for chunks in [2, 4, 8, 16, 32]:
        out_bf16 = rmsnorm_chunked(x, weight, chunks, use_fp32_accum=False)
        out_fp32 = rmsnorm_chunked(x, weight, chunks, use_fp32_accum=True)
        diff_bf16 = (out_bf16.float() - ref_norm.float()).abs().max().item()
        ref_fp32 = rmsnorm_chunked(x, weight, 1, use_fp32_accum=True)
        diff_fp32 = (out_fp32.float() - ref_fp32.float()).abs().max().item()

        print(f"    chunks={chunks:>2}: BF16 max_diff={diff_bf16:.4e}, FP32_accum max_diff={diff_fp32:.4e}")
        norm_results[chunks] = {"bf16": diff_bf16, "fp32": diff_fp32}

    results["rmsnorm"] = norm_results

    # --- Attention: simulate split-KV ---
    print("\n  --- Attention: Split-KV simulation ---")
    B, H, S, D = 1, 8, 512, 64
    Q = torch.randn(B, H, S, D, device=device, dtype=torch.bfloat16)
    K = torch.randn(B, H, S, D, device=device, dtype=torch.bfloat16)
    V = torch.randn(B, H, S, D, device=device, dtype=torch.bfloat16)

    def attention_split_kv(Q, K, V, num_splits, use_fp32_accum=False):
        """Simulate split-KV attention: process KV in chunks, combine."""
        S_kv = K.shape[2]
        chunk = S_kv // num_splits
        scale = D ** -0.5

        # For single split, just do standard attention
        if num_splits == 1:
            if use_fp32_accum:
                scores = torch.matmul(Q.float(), K.float().transpose(-2, -1)) * scale
                weights = torch.softmax(scores, dim=-1)
                return torch.matmul(weights, V.float()).to(Q.dtype)
            else:
                scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
                weights = torch.softmax(scores.float(), dim=-1).to(Q.dtype)
                return torch.matmul(weights, V)

        # Multi-split: each split computes partial attention, then combine
        # This simulates FlashDecoding's split-KV reduction
        partial_outs = []
        partial_lse = []  # log-sum-exp for online softmax correction

        for i in range(num_splits):
            start = i * chunk
            end = S_kv if i == num_splits - 1 else (i + 1) * chunk
            k_chunk = K[:, :, start:end, :]
            v_chunk = V[:, :, start:end, :]

            if use_fp32_accum:
                scores = torch.matmul(Q.float(), k_chunk.float().transpose(-2, -1)) * scale
                max_scores = scores.max(dim=-1, keepdim=True).values
                exp_scores = torch.exp(scores - max_scores)
                sum_exp = exp_scores.sum(dim=-1, keepdim=True)
                weights = exp_scores / sum_exp
                out = torch.matmul(weights, v_chunk.float())
                lse = max_scores + torch.log(sum_exp)
                partial_outs.append(out)
                partial_lse.append(lse)
            else:
                scores = torch.matmul(Q, k_chunk.transpose(-2, -1)) * scale
                scores_f = scores.float()
                max_scores = scores_f.max(dim=-1, keepdim=True).values
                exp_scores = torch.exp(scores_f - max_scores)
                sum_exp = exp_scores.sum(dim=-1, keepdim=True)
                weights = (exp_scores / sum_exp).to(Q.dtype)
                out = torch.matmul(weights, v_chunk).float()
                lse = max_scores + torch.log(sum_exp)
                partial_outs.append(out)
                partial_lse.append(lse)

        # Combine using log-sum-exp correction (online softmax)
        combined_lse = partial_lse[0]
        combined_out = partial_outs[0]
        for i in range(1, num_splits):
            max_lse = torch.maximum(combined_lse, partial_lse[i])
            w1 = torch.exp(combined_lse - max_lse)
            w2 = torch.exp(partial_lse[i] - max_lse)
            combined_out = combined_out * w1 + partial_outs[i] * w2
            combined_lse = max_lse + torch.log(w1 + w2)

        combined_out = combined_out / torch.exp(combined_lse - combined_lse)  # normalize
        if use_fp32_accum:
            return combined_out.to(Q.dtype)
        else:
            return combined_out.to(Q.dtype)

    ref_attn = attention_split_kv(Q, K, V, 1, use_fp32_accum=False)
    attn_results = {}
    for splits in [2, 4, 8, 16]:
        out_bf16 = attention_split_kv(Q, K, V, splits, use_fp32_accum=False)
        ref_fp32 = attention_split_kv(Q, K, V, 1, use_fp32_accum=True)
        out_fp32 = attention_split_kv(Q, K, V, splits, use_fp32_accum=True)
        diff_bf16 = (out_bf16.float() - ref_attn.float()).abs().max().item()
        diff_fp32 = (out_fp32.float() - ref_fp32.float()).abs().max().item()

        print(f"    splits={splits:>2}: BF16 max_diff={diff_bf16:.4e}, FP32_accum max_diff={diff_fp32:.4e}")
        attn_results[splits] = {"bf16": diff_bf16, "fp32": diff_fp32}

    results["attention"] = attn_results

    return results


# ============================================================================
# Test 3: Performance
# ============================================================================

def test_performance(model, tokenizer, num_warmup=3, num_runs=20):
    header("TEST 3: PERFORMANCE COMPARISON")
    device = next(model.parameters()).device

    def measure(mode_name, apply_fn=None, restore_fn=None):
        originals = apply_fn(model) if apply_fn else None
        try:
            inputs = tokenizer(PROMPT, return_tensors="pt", truncation=True).to(device)
            # Warmup
            for _ in range(num_warmup):
                with torch.no_grad():
                    model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                   pad_token_id=tokenizer.pad_token_id)
            torch.cuda.synchronize()

            times = []
            for _ in range(num_runs):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                   pad_token_id=tokenizer.pad_token_id)
                torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1000)

            mean_ms = sum(times) / len(times)
            std_ms = (sum((t - mean_ms)**2 for t in times) / len(times)) ** 0.5
            print(f"  {mode_name:<40} {mean_ms:>8.2f} +/- {std_ms:.2f} ms")
            return {"mean_ms": mean_ms, "std_ms": std_ms}
        finally:
            if originals and restore_fn:
                restore_fn(model, originals)

    results = {}
    results["bf16"] = measure("Pure BF16")
    results["linear_only"] = measure(
        "Linear-only FP32 accum",
        lambda m: apply_fp32_accum_all(m, patch_linear=True, patch_rmsnorm=False, patch_attention=False, patch_softmax=False),
        restore_fp32_accum_all,
    )
    results["linear_rmsnorm"] = measure(
        "Linear + RMSNorm FP32 accum",
        lambda m: apply_fp32_accum_all(m, patch_linear=True, patch_rmsnorm=True, patch_attention=False, patch_softmax=True),
        restore_fp32_accum_all,
    )
    results["full"] = measure(
        "Full FP32 accum (incl. Attention)",
        lambda m: apply_fp32_accum_all(m, patch_linear=True, patch_rmsnorm=True, patch_attention=True, patch_softmax=True),
        restore_fp32_accum_all,
    )

    bf16_ms = results["bf16"]["mean_ms"]
    print(f"\n  Slowdown vs BF16:")
    for key, r in results.items():
        if key != "bf16":
            print(f"    {key:<35} {r['mean_ms']/bf16_ms:.2f}x")

    return results


# ============================================================================
# Test 4: Near-tie prevalence
# ============================================================================

def test_near_tie_prevalence(model, tokenizer):
    """
    Analyze how often the top-1 and top-2 logits are close enough
    that a small numerical perturbation could flip the argmax.
    """
    header("TEST 4: NEAR-TIE PREVALENCE ANALYSIS")
    device = next(model.parameters()).device

    prompts = [
        PROMPT,
        "Explain the theory of relativity.",
        "What is the meaning of life?",
        "How do computers process information?",
        "Describe photosynthesis step by step.",
    ]

    thresholds = [1e-1, 1e-2, 1e-3, 1e-4]

    for prompt_idx, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits[0]  # [seq_len, vocab]

        # For each position, compute gap between top-1 and top-2
        top2_vals, _ = logits.float().topk(2, dim=-1)
        gaps = (top2_vals[:, 0] - top2_vals[:, 1]).abs()

        if prompt_idx == 0:
            print(f"\n  Prompt: \"{prompt[:60]}...\"")
            print(f"  Sequence length: {logits.shape[0]} tokens")
            print(f"  Gap statistics (top1 - top2 logit):")
            print(f"    min={gaps.min().item():.4e}, max={gaps.max().item():.4e}, "
                  f"mean={gaps.mean().item():.4e}, median={gaps.median().item():.4e}")
            print(f"\n  Near-tie prevalence:")
            for tau in thresholds:
                ratio = (gaps < tau).float().mean().item()
                print(f"    P(gap < {tau:.0e}) = {ratio*100:.2f}%")

    # Aggregate across all prompts
    print(f"\n  Aggregated across {len(prompts)} prompts:")
    all_gaps = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits[0]
        top2_vals, _ = logits.float().topk(2, dim=-1)
        gaps = (top2_vals[:, 0] - top2_vals[:, 1]).abs()
        all_gaps.append(gaps)

    all_gaps = torch.cat(all_gaps)
    for tau in thresholds:
        ratio = (all_gaps < tau).float().mean().item()
        print(f"    P(gap < {tau:.0e}) = {ratio*100:.2f}%")

    return {
        "total_tokens": all_gaps.shape[0],
        "near_ties": {f"{tau:.0e}": (all_gaps < tau).float().mean().item() for tau in thresholds},
    }


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--perf-runs", type=int, default=20)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    print("=" * 80)
    print("  FP32 ACCUMULATION ONLY -- FULL EVALUATION")
    print("=" * 80)
    print(f"  Model: {MODEL_PATH}")
    print(f"  Prompt: \"{PROMPT}\"")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

    model, tokenizer = load_model()
    all_results = {}

    # Test 1: Logits-level batch invariance
    all_results["logits_bi"] = test_logits_batch_invariance(model, tokenizer)
    torch.cuda.empty_cache()

    # Test 2: Op-level non-determinism simulation
    all_results["op_nondeterminism"] = test_op_level_nondeterminism()
    torch.cuda.empty_cache()

    # Test 3: Performance
    all_results["performance"] = test_performance(model, tokenizer, num_runs=args.perf_runs)
    torch.cuda.empty_cache()

    # Test 4: Near-tie prevalence
    all_results["near_ties"] = test_near_tie_prevalence(model, tokenizer)

    # ======== SUMMARY ========
    header("FINAL SUMMARY")

    print("\n  1. LOGITS BATCH INVARIANCE (bs=1 vs bs=8, logits max_diff)")
    print(f"     {'Mode':<40} {'max_diff':<12} {'argmax OK'}")
    print("     " + "-" * 62)
    for mode_name, batch_res in all_results["logits_bi"].items():
        r = batch_res.get("bs8", batch_res.get("bs4", {}))
        print(f"     {mode_name:<40} {r.get('logits_max_diff', 'N/A'):<12.4e} "
              f"{'YES' if r.get('argmax_match', False) else 'NO'}")

    print("\n  2. SPLIT-K/KV SIMULATION (BF16 vs FP32 accum, max_diff across splits)")
    for op in ["gemm", "rmsnorm", "attention"]:
        op_data = all_results["op_nondeterminism"].get(op, {})
        if op == "gemm":
            bf16_max = max(op_data.get("bf16_diffs", {}).values(), default=0)
            fp32_max = max(op_data.get("fp32_diffs", {}).values(), default=0)
        else:
            bf16_max = max((v.get("bf16", 0) for v in op_data.values()), default=0)
            fp32_max = max((v.get("fp32", 0) for v in op_data.values()), default=0)
        print(f"     {op:<15} BF16 worst={bf16_max:.4e}   FP32_accum worst={fp32_max:.4e}   "
              f"improvement={bf16_max/fp32_max:.0f}x" if fp32_max > 0 else f"     {op:<15} FP32=0 (perfect)")

    print(f"\n  3. PERFORMANCE")
    bf16_ms = all_results["performance"]["bf16"]["mean_ms"]
    for key in ["bf16", "linear_only", "linear_rmsnorm", "full"]:
        r = all_results["performance"][key]
        print(f"     {key:<35} {r['mean_ms']:.2f} ms ({r['mean_ms']/bf16_ms:.2f}x)")

    print(f"\n  4. NEAR-TIE VULNERABILITY")
    nt = all_results["near_ties"]
    for tau, ratio in nt["near_ties"].items():
        print(f"     P(gap < {tau}) = {ratio*100:.2f}%")

    print(f"\n  CONCLUSION:")
    logits_bi = all_results["logits_bi"]
    bf16_diff = logits_bi["Pure BF16"].get("bs8", {}).get("logits_max_diff", 0)
    full_diff = logits_bi["Full FP32 accum"].get("bs8", {}).get("logits_max_diff", 0)
    if full_diff < bf16_diff:
        print(f"     FP32 accum reduces batch-variant logits diff: {bf16_diff:.4e} -> {full_diff:.4e}")
    elif full_diff == 0 and bf16_diff == 0:
        print(f"     Both BF16 and FP32 accum show zero logits diff in HF inference.")
        print(f"     Non-determinism manifests in serving engines (vLLM/SGLang) with")
        print(f"     continuous batching and split-KV. See Test 2 for simulation.")

    gemm_data = all_results["op_nondeterminism"].get("gemm", {})
    bf16_worst = max(gemm_data.get("bf16_diffs", {}).values(), default=0)
    fp32_worst = max(gemm_data.get("fp32_diffs", {}).values(), default=0)
    if bf16_worst > 0 and fp32_worst < bf16_worst:
        ratio = bf16_worst / max(fp32_worst, 1e-30)
        print(f"     Split-K GEMM: FP32 accum reduces error by {ratio:.0f}x vs BF16")

    nt_01 = nt["near_ties"].get("1e-01", 0)
    print(f"     Near-tie tokens (gap<0.1): {nt_01*100:.1f}% — these are vulnerable to flip")

    # Save
    out_path = args.output or os.path.join(os.path.dirname(__file__), "fp32_accum_full_eval_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
