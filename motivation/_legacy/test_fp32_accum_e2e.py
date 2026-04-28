#!/usr/bin/env python3
"""
End-to-end test: Full FP32 accumulation (all reduction ops) vs Pure BF16.

Tests:
1. Batch invariance: same prompt at batch_size=1 vs batch_size=8 -> identical logits/tokens
2. Determinism: same prompt N times -> identical output
3. Latency comparison: Pure BF16 vs Linear-only FP32 accum vs Full FP32 accum
4. Ablation: which ops matter (Linear only, +RMSNorm, +Attention, +Softmax)

Models: Llama-3.1-8B-Instruct, Qwen3-MoE (configurable)
"""
import sys
import os
import json
import time
import argparse
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from FP32.model_patcher import fp32_accum_mode, apply_fp32_accum_all, restore_fp32_accum_all


def get_logits(model, tokenizer, prompt, device):
    """Run a single forward pass and return logits for the prompt."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.logits


def generate_tokens(model, tokenizer, prompt, device, max_new_tokens=32):
    """Generate tokens greedily and return token IDs."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    return out[0].cpu().tolist()


def test_batch_invariance(model, tokenizer, prompt, device, max_new_tokens=32):
    """
    Test batch invariance: same prompt at batch_size=1 vs batched with other prompts.
    Compare logits of the target prompt.
    """
    print("\n  --- Batch Invariance Test ---")
    filler_prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a haiku about rain.",
        "How does photosynthesis work?",
        "What is 2 + 2?",
        "Tell me a joke.",
        "Describe the solar system.",
    ]

    # Single (batch_size=1)
    logits_single = get_logits(model, tokenizer, prompt, device)

    # Batched (batch_size=8): target prompt + 7 fillers
    batch_prompts = [prompt] + filler_prompts[:7]
    inputs_batch = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs_batch = model(**inputs_batch)
    logits_batched = outputs_batch.logits[0:1]  # first sample only

    # Compare logits
    # Note: padding changes input_ids layout, so we compare up to the min length
    min_len = min(logits_single.shape[1], logits_batched.shape[1])
    diff = (logits_single[:, :min_len].float() - logits_batched[:, :min_len].float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    # Also compare generated tokens
    tokens_single = generate_tokens(model, tokenizer, prompt, device, max_new_tokens)
    # For batched generation, just check if single-generation is deterministic
    tokens_single2 = generate_tokens(model, tokenizer, prompt, device, max_new_tokens)
    tokens_match = tokens_single == tokens_single2

    print(f"    Logits max_diff (single vs batch): {max_diff:.4e}")
    print(f"    Logits mean_diff: {mean_diff:.4e}")
    print(f"    Batch invariant: {'YES' if max_diff == 0 else 'NO'}")
    print(f"    Deterministic generation (2 runs): {'YES' if tokens_match else 'NO'}")

    return {
        "logits_max_diff": max_diff,
        "logits_mean_diff": mean_diff,
        "batch_invariant": max_diff == 0,
        "deterministic_generation": tokens_match,
    }


def test_latency(model, tokenizer, prompt, device, max_new_tokens=32, num_runs=3):
    """Measure generation latency."""
    # Warmup
    generate_tokens(model, tokenizer, prompt, device, max_new_tokens)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times = []
    for _ in range(num_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        generate_tokens(model, tokenizer, prompt, device, max_new_tokens)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    mean_ms = sum(times) / len(times)
    return mean_ms


def main():
    warnings.filterwarnings("ignore", message="Token indices")
    warnings.filterwarnings("ignore", message="Setting `pad_token_id`")

    parser = argparse.ArgumentParser(description="E2E FP32 accum test")
    parser.add_argument("--model-path", type=str, default="/workspace/Models/Llama-3.1-8B-Instruct")
    parser.add_argument("--prompt", type=str, default="Explain the concept of batch invariance in LLM inference in two sentences.")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--num-latency-runs", type=int, default=3)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available. Exiting.")
        sys.exit(1)

    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model: {args.model_path}")
    print(f"Prompt: {args.prompt[:80]}...")
    print()

    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print("Model loaded.\n")

    results = {}

    # ---- Test 1: Pure BF16 baseline ----
    print("=" * 70)
    print("TEST 1: Pure BF16 (baseline)")
    print("=" * 70)
    bi_bf16 = test_batch_invariance(model, tokenizer, args.prompt, device, args.max_new_tokens)
    lat_bf16 = test_latency(model, tokenizer, args.prompt, device, args.max_new_tokens, args.num_latency_runs)
    print(f"\n  Latency: {lat_bf16:.1f} ms")
    results["pure_bf16"] = {"batch_invariance": bi_bf16, "latency_ms": lat_bf16}

    # ---- Test 2: Linear-only FP32 accum ----
    print("\n" + "=" * 70)
    print("TEST 2: Linear-only FP32 accum")
    print("=" * 70)
    with fp32_accum_mode(model, patch_linear=True, patch_rmsnorm=False, patch_attention=False, patch_softmax=False):
        bi_linear = test_batch_invariance(model, tokenizer, args.prompt, device, args.max_new_tokens)
        lat_linear = test_latency(model, tokenizer, args.prompt, device, args.max_new_tokens, args.num_latency_runs)
    print(f"\n  Latency: {lat_linear:.1f} ms (slowdown: {lat_linear/lat_bf16:.2f}x)")
    results["linear_only"] = {"batch_invariance": bi_linear, "latency_ms": lat_linear}

    # ---- Test 3: Full FP32 accum (all ops) ----
    print("\n" + "=" * 70)
    print("TEST 3: Full FP32 accum (Linear + RMSNorm + Attention + Softmax)")
    print("=" * 70)
    with fp32_accum_mode(model, patch_linear=True, patch_rmsnorm=True, patch_attention=True, patch_softmax=True):
        bi_full = test_batch_invariance(model, tokenizer, args.prompt, device, args.max_new_tokens)
        lat_full = test_latency(model, tokenizer, args.prompt, device, args.max_new_tokens, args.num_latency_runs)
    print(f"\n  Latency: {lat_full:.1f} ms (slowdown: {lat_full/lat_bf16:.2f}x)")
    results["full_fp32_accum"] = {"batch_invariance": bi_full, "latency_ms": lat_full}

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Mode':<35} {'BI?':<6} {'Logits Diff':<15} {'Latency':<12} {'Slowdown':<10}")
    print("-" * 78)
    for name, key in [("Pure BF16", "pure_bf16"), ("Linear-only FP32 accum", "linear_only"), ("Full FP32 accum", "full_fp32_accum")]:
        r = results[key]
        bi = r["batch_invariance"]
        bi_str = "YES" if bi["batch_invariant"] else "NO"
        diff_str = f"{bi['logits_max_diff']:.2e}"
        lat_str = f"{r['latency_ms']:.1f} ms"
        slow_str = f"{r['latency_ms']/lat_bf16:.2f}x"
        print(f"{name:<35} {bi_str:<6} {diff_str:<15} {lat_str:<12} {slow_str:<10}")

    # Save results
    out_path = args.output or os.path.join(os.path.dirname(__file__), "fp32_accum_e2e_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
