#!/usr/bin/env python3
"""
Unit tests for FP32 accumulation ops: RMSNorm, Softmax, Attention.

For each op, tests:
1. Correctness: max_diff vs FP64 reference
2. Determinism: 20 runs with identical input -> std_mean == 0
3. Batch invariance: M=1 vs M=8 vs M=32, first-row output identical
4. dtype: input/output are BF16
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from FP32.rmsnorm_fp32_accum import rmsnorm_fp32_accum, rmsnorm_fp32_accum_pytorch
from FP32.softmax_fp32_accum import softmax_fp32_accum, softmax_fp32_accum_pytorch
from FP32.attention_fp32_accum import attention_fp32_accum, attention_fp32_accum_pytorch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_DETERMINISM_RUNS = 20


def _header(name: str):
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")


# ---------------------------------------------------------------------------
# RMSNorm tests
# ---------------------------------------------------------------------------

def test_rmsnorm():
    _header("RMSNorm FP32 Accum")
    torch.manual_seed(42)
    hidden = 4096
    weight = torch.randn(hidden, device=DEVICE, dtype=torch.bfloat16)
    eps = 1e-6

    # --- Correctness vs FP64 ---
    x = torch.randn(8, hidden, device=DEVICE, dtype=torch.bfloat16)
    ref_fp64 = x.double()
    var_fp64 = ref_fp64.pow(2).mean(-1, keepdim=True)
    ref_out = (ref_fp64 * torch.rsqrt(var_fp64 + eps) * weight.double()).to(torch.bfloat16)
    our_out = rmsnorm_fp32_accum(x, weight, eps)
    max_diff = (our_out.float() - ref_out.float()).abs().max().item()
    print(f"  Correctness vs FP64 ref: max_diff = {max_diff:.2e}")
    assert our_out.dtype == torch.bfloat16, f"Output dtype {our_out.dtype} != bfloat16"
    print(f"  Output dtype: {our_out.dtype} (OK)")

    # --- Determinism ---
    x_det = torch.randn(32, hidden, device=DEVICE, dtype=torch.bfloat16)
    outputs = [rmsnorm_fp32_accum(x_det, weight, eps) for _ in range(N_DETERMINISM_RUNS)]
    stack = torch.stack([o.float() for o in outputs], 0)
    std_mean = stack.std(0).mean().item()
    print(f"  Determinism ({N_DETERMINISM_RUNS} runs): std_mean = {std_mean:.2e} {'(OK)' if std_mean == 0 else '(FAIL)'}")

    # --- Batch invariance ---
    row = torch.randn(1, hidden, device=DEVICE, dtype=torch.bfloat16)
    ref_single = rmsnorm_fp32_accum(row, weight, eps)
    for M in (1, 8, 32):
        batch = torch.cat([row] + [torch.randn(1, hidden, device=DEVICE, dtype=torch.bfloat16) for _ in range(M - 1)], 0)
        out = rmsnorm_fp32_accum(batch, weight, eps)
        diff = (out[0:1].float() - ref_single.float()).abs().max().item()
        status = "OK" if diff == 0 else f"FAIL (diff={diff:.2e})"
        print(f"  Batch invariance M={M}: first-row max_diff = {diff:.2e} ({status})")

    return {"correctness_max_diff": max_diff, "determinism_std_mean": std_mean}


# ---------------------------------------------------------------------------
# Softmax tests
# ---------------------------------------------------------------------------

def test_softmax():
    _header("Softmax FP32 Accum")
    torch.manual_seed(42)
    N = 256

    # --- Correctness vs FP64 ---
    x = torch.randn(8, N, device=DEVICE, dtype=torch.bfloat16)
    ref_out = torch.softmax(x.double(), dim=-1).to(torch.bfloat16)
    our_out = softmax_fp32_accum(x, dim=-1)
    max_diff = (our_out.float() - ref_out.float()).abs().max().item()
    print(f"  Correctness vs FP64 ref: max_diff = {max_diff:.2e}")
    assert our_out.dtype == torch.bfloat16, f"Output dtype {our_out.dtype} != bfloat16"
    print(f"  Output dtype: {our_out.dtype} (OK)")

    # --- Determinism ---
    x_det = torch.randn(32, N, device=DEVICE, dtype=torch.bfloat16)
    outputs = [softmax_fp32_accum(x_det, dim=-1) for _ in range(N_DETERMINISM_RUNS)]
    stack = torch.stack([o.float() for o in outputs], 0)
    std_mean = stack.std(0).mean().item()
    print(f"  Determinism ({N_DETERMINISM_RUNS} runs): std_mean = {std_mean:.2e} {'(OK)' if std_mean == 0 else '(FAIL)'}")

    # --- Batch invariance ---
    row = torch.randn(1, N, device=DEVICE, dtype=torch.bfloat16)
    ref_single = softmax_fp32_accum(row, dim=-1)
    for M in (1, 8, 32):
        batch = torch.cat([row] + [torch.randn(1, N, device=DEVICE, dtype=torch.bfloat16) for _ in range(M - 1)], 0)
        out = softmax_fp32_accum(batch, dim=-1)
        diff = (out[0:1].float() - ref_single.float()).abs().max().item()
        status = "OK" if diff == 0 else f"FAIL (diff={diff:.2e})"
        print(f"  Batch invariance M={M}: first-row max_diff = {diff:.2e} ({status})")

    return {"correctness_max_diff": max_diff, "determinism_std_mean": std_mean}


# ---------------------------------------------------------------------------
# Attention tests
# ---------------------------------------------------------------------------

def test_attention():
    _header("Attention FP32 Accum")
    torch.manual_seed(42)
    B, H, S, D = 2, 8, 128, 64

    # --- Correctness vs FP64 ---
    Q = torch.randn(B, H, S, D, device=DEVICE, dtype=torch.bfloat16)
    K = torch.randn(B, H, S, D, device=DEVICE, dtype=torch.bfloat16)
    V = torch.randn(B, H, S, D, device=DEVICE, dtype=torch.bfloat16)

    # FP64 reference
    scale = D ** -0.5
    scores_fp64 = torch.matmul(Q.double(), K.double().transpose(-2, -1)) * scale
    mask = torch.triu(torch.ones(S, S, device=DEVICE, dtype=torch.bool), diagonal=1)
    scores_fp64.masked_fill_(mask, float('-inf'))
    weights_fp64 = torch.softmax(scores_fp64, dim=-1)
    ref_out = torch.matmul(weights_fp64, V.double()).to(torch.bfloat16)

    our_out = attention_fp32_accum(Q, K, V, is_causal=True)
    max_diff = (our_out.float() - ref_out.float()).abs().max().item()
    print(f"  Correctness vs FP64 ref (causal): max_diff = {max_diff:.2e}")
    assert our_out.dtype == torch.bfloat16, f"Output dtype {our_out.dtype} != bfloat16"
    print(f"  Output dtype: {our_out.dtype} (OK)")

    # --- Determinism ---
    outputs = [attention_fp32_accum(Q, K, V, is_causal=True) for _ in range(N_DETERMINISM_RUNS)]
    stack = torch.stack([o.float() for o in outputs], 0)
    std_mean = stack.std(0).mean().item()
    print(f"  Determinism ({N_DETERMINISM_RUNS} runs): std_mean = {std_mean:.2e} {'(OK)' if std_mean == 0 else '(FAIL)'}")

    # --- Batch invariance ---
    # Single sample
    q1 = torch.randn(1, H, S, D, device=DEVICE, dtype=torch.bfloat16)
    k1 = torch.randn(1, H, S, D, device=DEVICE, dtype=torch.bfloat16)
    v1 = torch.randn(1, H, S, D, device=DEVICE, dtype=torch.bfloat16)
    ref_single = attention_fp32_accum(q1, k1, v1, is_causal=True)

    for M in (1, 4, 8):
        # Pad with random other samples
        q_batch = torch.cat([q1] + [torch.randn(1, H, S, D, device=DEVICE, dtype=torch.bfloat16) for _ in range(M - 1)], 0)
        k_batch = torch.cat([k1] + [torch.randn(1, H, S, D, device=DEVICE, dtype=torch.bfloat16) for _ in range(M - 1)], 0)
        v_batch = torch.cat([v1] + [torch.randn(1, H, S, D, device=DEVICE, dtype=torch.bfloat16) for _ in range(M - 1)], 0)
        out = attention_fp32_accum(q_batch, k_batch, v_batch, is_causal=True)
        diff = (out[0:1].float() - ref_single.float()).abs().max().item()
        status = "OK" if diff == 0 else f"FAIL (diff={diff:.2e})"
        print(f"  Batch invariance M={M}: first-sample max_diff = {diff:.2e} ({status})")

    return {"correctness_max_diff": max_diff, "determinism_std_mean": std_mean}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    results = {}
    results["rmsnorm"] = test_rmsnorm()
    results["softmax"] = test_softmax()
    results["attention"] = test_attention()

    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    for op, res in results.items():
        print(f"  {op}: correctness={res['correctness_max_diff']:.2e}, determinism_std={res['determinism_std_mean']:.2e}")

    out_path = os.path.join(os.path.dirname(__file__), "ops_fp32_accum_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
