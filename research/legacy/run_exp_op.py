#!/usr/bin/env python3
"""
Comprehensive Op-Level Batch Variance Test Suite
=================================================
Tests how GPU numerical behavior varies with batch size and reduction strategy
for common deep learning operations on NVIDIA RTX A6000.

Experiments:
  1. GEMM Batch Variance (F.linear with varying M)
  2. RMSNorm Chunk Reduction Variance
  3. Softmax Chunk Reduction Variance
  4. Attention Split-KV Variance
  5. Run-to-Run Variance (determinism check)
"""

import json
import math
import time
import sys
import os

import torch
import torch.nn.functional as F

# ── Reproducibility ──────────────────────────────────────────────────────────
torch.manual_seed(42)
torch.cuda.manual_seed(42)
DEVICE = "cuda"

# ── Result containers ────────────────────────────────────────────────────────
results = {
    "metadata": {
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    },
    "gemm_batch_variance": [],
    "rmsnorm_chunk_variance": [],
    "softmax_chunk_variance": [],
    "attention_splitkv_variance": [],
    "run_to_run_variance": [],
}

print("=" * 80)
print("Op-Level Batch Variance Test Suite")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"PyTorch: {torch.__version__}  CUDA: {torch.version.cuda}")
print("=" * 80)

# ═════════════════════════════════════════════════════════════════════════════
# Experiment 1 — GEMM Batch Variance
# ═════════════════════════════════════════════════════════════════════════════
print("\n[Experiment 1] GEMM Batch Variance")
print("-" * 60)

gemm_shapes = {
    "llama_q_proj":   (4096, 4096),
    "llama_gate_proj": (4096, 11008),
    "moe_expert":     (2048, 5632),
}
M_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

for shape_name, (K, N) in gemm_shapes.items():
    print(f"\n  Shape: {shape_name}  K={K}, N={N}")

    # Fixed weight and target row
    torch.manual_seed(42)
    W = torch.randn(N, K, dtype=torch.bfloat16, device=DEVICE)
    target_row = torch.randn(1, K, dtype=torch.bfloat16, device=DEVICE)

    for reduced_prec_label, flag_val in [("default", True), ("no_reduced_prec", False)]:
        # Reference: M=1 (just the target row)
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = flag_val
        torch.cuda.synchronize()
        ref_out = F.linear(target_row, W)  # (1, N)
        torch.cuda.synchronize()

        for M in M_values:
            torch.manual_seed(42 + M)
            # Build batch: target row at position 0, rest random
            if M == 1:
                batch = target_row.clone()
            else:
                pad = torch.randn(M - 1, K, dtype=torch.bfloat16, device=DEVICE)
                batch = torch.cat([target_row, pad], dim=0)

            torch.cuda.synchronize()
            out = F.linear(batch, W)  # (M, N)
            torch.cuda.synchronize()

            target_out = out[0:1]
            diff = (target_out - ref_out).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()

            entry = {
                "shape": shape_name,
                "K": K, "N": N, "M": M,
                "mode": reduced_prec_label,
                "max_diff": max_diff,
                "mean_diff": mean_diff,
            }
            results["gemm_batch_variance"].append(entry)

            if max_diff > 0:
                print(f"    [{reduced_prec_label:>18s}] M={M:>5d}  max_diff={max_diff:.6e}  mean_diff={mean_diff:.6e}")

print("\n  GEMM Batch Variance complete.")


# ═════════════════════════════════════════════════════════════════════════════
# Experiment 2 — RMSNorm Chunk Reduction Variance
# ═════════════════════════════════════════════════════════════════════════════
print("\n[Experiment 2] RMSNorm Chunk Reduction Variance")
print("-" * 60)

hidden_dims = [2048, 4096]
num_chunks_list = [1, 2, 4, 8, 16, 32]
eps = 1e-6

for hidden_dim in hidden_dims:
    torch.manual_seed(42)
    x = torch.randn(1, hidden_dim, dtype=torch.bfloat16, device=DEVICE)
    weight = torch.ones(hidden_dim, dtype=torch.bfloat16, device=DEVICE)

    # Reference: full FP32 reduction
    x_fp32 = x.float()
    rms_ref = torch.sqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + eps)
    out_ref = (x_fp32 / rms_ref).bfloat16() * weight

    for num_chunks in num_chunks_list:
        chunk_size = hidden_dim // num_chunks

        # BF16 chunked accumulation
        sq_sum_bf16 = torch.zeros(1, 1, dtype=torch.bfloat16, device=DEVICE)
        for c in range(num_chunks):
            chunk = x[:, c * chunk_size : (c + 1) * chunk_size]
            sq_sum_bf16 += chunk.pow(2).sum(dim=-1, keepdim=True)
        rms_bf16 = torch.sqrt(sq_sum_bf16 / hidden_dim + eps)
        out_bf16 = (x / rms_bf16) * weight

        # FP32 chunked accumulation
        sq_sum_fp32 = torch.zeros(1, 1, dtype=torch.float32, device=DEVICE)
        for c in range(num_chunks):
            chunk = x[:, c * chunk_size : (c + 1) * chunk_size]
            sq_sum_fp32 += chunk.float().pow(2).sum(dim=-1, keepdim=True)
        rms_fp32 = torch.sqrt(sq_sum_fp32 / hidden_dim + eps)
        out_fp32 = (x.float() / rms_fp32).bfloat16() * weight

        diff_bf16 = (out_bf16.float() - out_ref.float()).abs()
        diff_fp32 = (out_fp32.float() - out_ref.float()).abs()

        entry = {
            "hidden_dim": hidden_dim,
            "num_chunks": num_chunks,
            "bf16_max_diff": diff_bf16.max().item(),
            "bf16_mean_diff": diff_bf16.mean().item(),
            "fp32_max_diff": diff_fp32.max().item(),
            "fp32_mean_diff": diff_fp32.mean().item(),
        }
        results["rmsnorm_chunk_variance"].append(entry)
        print(f"  hidden={hidden_dim}  chunks={num_chunks:>2d}  "
              f"BF16 max={diff_bf16.max().item():.6e}  "
              f"FP32 max={diff_fp32.max().item():.6e}")

print("\n  RMSNorm Chunk Reduction Variance complete.")


# ═════════════════════════════════════════════════════════════════════════════
# Experiment 3 — Softmax Chunk Reduction Variance
# ═════════════════════════════════════════════════════════════════════════════
print("\n[Experiment 3] Softmax Chunk Reduction Variance")
print("-" * 60)

vocab_sizes = [128256, 151936]
sm_num_chunks_list = [1, 2, 4, 8, 16]

for vocab_size in vocab_sizes:
    torch.manual_seed(42)
    logits = torch.randn(1, vocab_size, dtype=torch.bfloat16, device=DEVICE)

    # Reference: full FP32 softmax
    sm_ref = F.softmax(logits.float(), dim=-1)

    for num_chunks in sm_num_chunks_list:
        chunk_size = math.ceil(vocab_size / num_chunks)

        # ── BF16 chunked online softmax ──
        running_max_bf16 = torch.tensor(float("-inf"), dtype=torch.bfloat16, device=DEVICE)
        running_sum_bf16 = torch.zeros(1, dtype=torch.bfloat16, device=DEVICE)
        for c in range(num_chunks):
            start = c * chunk_size
            end = min((c + 1) * chunk_size, vocab_size)
            chunk = logits[0, start:end]
            chunk_max = chunk.max()
            new_max = torch.maximum(running_max_bf16, chunk_max)
            # Rescale old sum
            running_sum_bf16 = running_sum_bf16 * torch.exp(running_max_bf16 - new_max)
            running_sum_bf16 += torch.exp(chunk - new_max).sum()
            running_max_bf16 = new_max

        # Compute final softmax
        sm_bf16 = torch.exp(logits[0] - running_max_bf16) / running_sum_bf16
        sm_bf16 = sm_bf16.unsqueeze(0)

        # ── FP32 chunked online softmax ──
        running_max_fp32 = torch.tensor(float("-inf"), dtype=torch.float32, device=DEVICE)
        running_sum_fp32 = torch.zeros(1, dtype=torch.float32, device=DEVICE)
        logits_fp32 = logits.float()
        for c in range(num_chunks):
            start = c * chunk_size
            end = min((c + 1) * chunk_size, vocab_size)
            chunk = logits_fp32[0, start:end]
            chunk_max = chunk.max()
            new_max = torch.maximum(running_max_fp32, chunk_max)
            running_sum_fp32 = running_sum_fp32 * torch.exp(running_max_fp32 - new_max)
            running_sum_fp32 += torch.exp(chunk - new_max).sum()
            running_max_fp32 = new_max

        sm_fp32 = torch.exp(logits_fp32[0] - running_max_fp32) / running_sum_fp32
        sm_fp32 = sm_fp32.unsqueeze(0)

        diff_bf16 = (sm_bf16.float() - sm_ref).abs()
        diff_fp32 = (sm_fp32 - sm_ref).abs()

        # Also check KL divergence
        kl_bf16 = F.kl_div(sm_bf16.float().log().clamp(min=-100), sm_ref, reduction="batchmean").item()
        kl_fp32 = F.kl_div(sm_fp32.log().clamp(min=-100), sm_ref, reduction="batchmean").item()

        entry = {
            "vocab_size": vocab_size,
            "num_chunks": num_chunks,
            "bf16_max_diff": diff_bf16.max().item(),
            "bf16_mean_diff": diff_bf16.mean().item(),
            "bf16_kl_div": kl_bf16,
            "fp32_max_diff": diff_fp32.max().item(),
            "fp32_mean_diff": diff_fp32.mean().item(),
            "fp32_kl_div": kl_fp32,
        }
        results["softmax_chunk_variance"].append(entry)
        print(f"  vocab={vocab_size}  chunks={num_chunks:>2d}  "
              f"BF16 max={diff_bf16.max().item():.6e}  "
              f"FP32 max={diff_fp32.max().item():.6e}  "
              f"BF16 KL={kl_bf16:.6e}  FP32 KL={kl_fp32:.6e}")

print("\n  Softmax Chunk Reduction Variance complete.")


# ═════════════════════════════════════════════════════════════════════════════
# Experiment 4 — Attention Split-KV Variance
# ═════════════════════════════════════════════════════════════════════════════
print("\n[Experiment 4] Attention Split-KV Variance")
print("-" * 60)

B, H, D = 1, 8, 128
seq_lens = [128, 512, 1024, 2048]
splits_list = [1, 2, 4, 8, 16]

def split_kv_attention(Q, K, V, num_splits, dtype):
    """
    Compute attention with split-KV and online softmax combination.
    Q: (B, H, 1, D) — single query token
    K: (B, H, S, D)
    V: (B, H, S, D)
    """
    B, H, _, D_head = Q.shape
    S = K.shape[2]
    split_size = math.ceil(S / num_splits)

    if dtype == torch.float32:
        Q_c, K_c, V_c = Q.float(), K.float(), V.float()
    else:
        Q_c, K_c, V_c = Q, K, V

    # Running state for online softmax combination
    running_max = torch.full((B, H, 1, 1), float("-inf"), dtype=Q_c.dtype, device=DEVICE)
    running_sum = torch.zeros(B, H, 1, 1, dtype=Q_c.dtype, device=DEVICE)
    running_out = torch.zeros(B, H, 1, D, dtype=Q_c.dtype, device=DEVICE)

    scale = 1.0 / math.sqrt(D)

    for s in range(num_splits):
        start = s * split_size
        end = min((s + 1) * split_size, S)
        if start >= S:
            break

        K_chunk = K_c[:, :, start:end, :]
        V_chunk = V_c[:, :, start:end, :]

        # Compute attention scores for this chunk
        attn_scores = torch.matmul(Q_c, K_chunk.transpose(-2, -1)) * scale  # (B, H, 1, chunk)
        chunk_max = attn_scores.max(dim=-1, keepdim=True).values  # (B, H, 1, 1)

        # Online softmax correction
        new_max = torch.maximum(running_max, chunk_max)

        # Rescale running statistics
        correction_old = torch.exp(running_max - new_max)
        correction_new = torch.exp(chunk_max - new_max)

        # Chunk softmax (unnormalized)
        exp_scores = torch.exp(attn_scores - chunk_max) * correction_new  # (B, H, 1, chunk)
        chunk_sum = exp_scores.sum(dim=-1, keepdim=True)  # (B, H, 1, 1)

        # Combine
        running_out = running_out * (running_sum * correction_old) + torch.matmul(exp_scores, V_chunk)
        running_sum = running_sum * correction_old + chunk_sum
        running_max = new_max

        # Normalize running output
        running_out = running_out / running_sum

    return running_out


for seq_len in seq_lens:
    torch.manual_seed(42)
    Q = torch.randn(B, H, 1, D, dtype=torch.bfloat16, device=DEVICE)
    K_full = torch.randn(B, H, seq_len, D, dtype=torch.bfloat16, device=DEVICE)
    V_full = torch.randn(B, H, seq_len, D, dtype=torch.bfloat16, device=DEVICE)

    # Reference: full FP32 attention, no splits
    ref_out = split_kv_attention(Q, K_full, V_full, num_splits=1, dtype=torch.float32)

    for num_splits in splits_list:
        for attn_dtype_label, attn_dtype in [("bf16", torch.bfloat16), ("fp32", torch.float32)]:
            out = split_kv_attention(Q, K_full, V_full, num_splits=num_splits, dtype=attn_dtype)

            diff = (out.float() - ref_out.float()).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()

            entry = {
                "seq_len": seq_len,
                "num_splits": num_splits,
                "dtype": attn_dtype_label,
                "max_diff": max_diff,
                "mean_diff": mean_diff,
            }
            results["attention_splitkv_variance"].append(entry)

            if max_diff > 0:
                print(f"  seq={seq_len:>5d}  splits={num_splits:>2d}  {attn_dtype_label:>4s}  "
                      f"max_diff={max_diff:.6e}  mean_diff={mean_diff:.6e}")

print("\n  Attention Split-KV Variance complete.")


# ═════════════════════════════════════════════════════════════════════════════
# Experiment 5 — Run-to-Run Variance
# ═════════════════════════════════════════════════════════════════════════════
print("\n[Experiment 5] Run-to-Run Variance (100 runs)")
print("-" * 60)

r2r_shapes = [
    (32, 4096, 4096),
    (1, 4096, 4096),
    (128, 4096, 11008),
]
NUM_RUNS = 100

for M_r, K_r, N_r in r2r_shapes:
    for dtype_label, dtype in [("bf16", torch.bfloat16), ("fp32", torch.float32)]:
        torch.manual_seed(42)
        x = torch.randn(M_r, K_r, dtype=dtype, device=DEVICE)
        W = torch.randn(N_r, K_r, dtype=dtype, device=DEVICE)

        # Set cublas flag
        if dtype == torch.bfloat16:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True

        # First run as reference
        torch.cuda.synchronize()
        ref_out = F.linear(x, W)
        torch.cuda.synchronize()

        max_diffs = []
        num_mismatches = 0
        for run_i in range(NUM_RUNS):
            torch.cuda.synchronize()
            out = F.linear(x, W)
            torch.cuda.synchronize()
            d = (out - ref_out).abs().max().item()
            max_diffs.append(d)
            if d > 0:
                num_mismatches += 1

        entry = {
            "shape": f"[{M_r}, {K_r}, {N_r}]",
            "M": M_r, "K": K_r, "N": N_r,
            "dtype": dtype_label,
            "num_runs": NUM_RUNS,
            "num_mismatches": num_mismatches,
            "max_of_max_diffs": max(max_diffs),
            "mean_of_max_diffs": sum(max_diffs) / len(max_diffs),
        }
        results["run_to_run_variance"].append(entry)
        status = "IDENTICAL" if num_mismatches == 0 else f"DIFFERS ({num_mismatches}/{NUM_RUNS})"
        print(f"  shape=[{M_r:>4d},{K_r:>5d},{N_r:>5d}]  {dtype_label:>4s}  {status}  "
              f"max_diff={max(max_diffs):.6e}")

print("\n  Run-to-Run Variance complete.")


# ═════════════════════════════════════════════════════════════════════════════
# Save results
# ═════════════════════════════════════════════════════════════════════════════
out_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(out_dir, "exp_op.json")
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {json_path}")
print("=" * 80)
print("All experiments complete.")
