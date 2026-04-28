# FP32 Accumulation Only — Evaluation Summary

Model: Llama-3.1-8B-Instruct | GPU: NVIDIA RTX A6000 (48GB) x2

## Test 1: Logits-Level Batch Invariance

Compare logits of same prompt at batch_size=1 vs batch_size={2,4,8,16}.

| Mode | bs=2 max_diff | bs=8 max_diff | bs=8 argmax match |
|---|---|---|---|
| **Pure BF16** | 2.19e-01 | **2.68e+01** | **NO** |
| **Full FP32 accum** | 0.00e+00 | 0.00e+00 | **YES** |

**Key finding**: Pure BF16 exhibits massive logits divergence at bs=8+ (max_diff=26.8!),
with argmax token selection changing. This is caused by left-padding in batched inference
changing the token positions and thus the attention/GEMM computation paths.

Full FP32 accum produces **zero logits difference** across all batch sizes — perfect batch invariance.

## Test 2: Op-Level Non-Determinism Simulation

Simulates the split-K/split-KV effects that occur in serving engines.

### GEMM (split-K)

| Splits | BF16 max_diff | FP32 accum max_diff | Improvement |
|---|---|---|---|
| 2 | 1.00 | 0.50 | 2x |
| 4 | 2.00 | 0.125 | 16x |
| 8 | 2.00 | 0.50 | 4x |
| 16 | 4.00 | 0.50 | 8x |
| 32 | 4.00 | 0.50 | 8x |

**Conclusion**: FP32 accumulation reduces GEMM split-K variance by **~8x** on average.
However, it does NOT eliminate it to zero — different K-splits still produce different
FP32 results that round to different BF16 values in extreme cases.

### RMSNorm (chunk reduction)

| Chunks | BF16 max_diff | FP32 accum max_diff |
|---|---|---|
| 2-32 | 1.56e-02 ~ 3.12e-02 | **0.00e+00** |

**Conclusion**: FP32 accumulation **perfectly solves** RMSNorm variance.
The reduction (sum of x^2) is a simple accumulation where FP32 precision
completely absorbs rounding differences from any chunk ordering.

### Attention (split-KV)

| Splits | BF16 max_diff | FP32 accum max_diff | Improvement |
|---|---|---|---|
| 2 | 0.42 | 0.42 | **1x (none)** |
| 4 | 1.14 | 1.14 | **1x (none)** |
| 8 | 1.39 | 1.39 | **1x (none)** |
| 16 | 2.55 | 2.55 | **1x (none)** |

**Critical finding**: FP32 accumulation provides **zero improvement** for attention split-KV.

**Root cause**: Split-KV changes the online softmax reduction tree structure fundamentally.
Each split computes a local softmax (local max + local sum), then combines via log-sum-exp
correction. The correction involves `exp(local_max - global_max)` rescaling, which creates
a fundamentally different computation graph — not just a different accumulation order.
FP32 cannot help because the mathematical operations themselves differ, not just the
precision of accumulation.

**Implication**: For attention, the correct approach is NOT "FP32 accum only" but rather
**fixed split-KV size** (as described in batch_invariance.md): use a constant chunk size
C regardless of batch, ensuring the same reduction tree for every batch composition.

## Test 3: Performance

### Initial results (naive Triton GEMM)

| Mode | Latency | Slowdown |
|---|---|---|
| Pure BF16 | 1,167 ms | 1.00x |
| Linear-only (Triton) | 45,239 ms | 38.77x |
| Full (Triton) | 46,247 ms | 39.63x |

### After optimization (cuBLAS flag + Triton attention)

Key optimization: use `torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False`
instead of replacing GEMM kernels. This forces cuBLAS to use FP32 accumulation natively
with **zero overhead** (no kernel replacement, no FP32 cast).

| Mode | Latency | Overhead |
|---|---|---|
| **Pure BF16** | 1,252 ms | 1.00x |
| **Linear-only (cuBLAS flag)** | 1,227 ms | **0.98x (faster!)** |
| **Linear + RMSNorm + Softmax** | 1,202 ms | **0.96x (faster!)** |
| **Full FP32 accum (incl. Attention)** | 1,324 ms | **1.06x** |

The cuBLAS FP32 accumulation path is actually slightly faster than BF16 on Ampere GPUs.
The only overhead comes from the Triton attention kernel replacing SDPA (+6%).

## Test 4: Near-Tie Prevalence

| Threshold | P(gap < threshold) |
|---|---|
| < 0.1 | **2.38%** |
| < 0.01 | 0.00% |
| < 0.001 | 0.00% |
| < 0.0001 | 0.00% |

Note: This is for Llama-3.1-8B (dense model) with short prompts.
For MoE models (Qwen3-MoE), the project's earlier tests showed
**39.29%** near-tie ratio at tau=0.001 for router logits — much more vulnerable.

## Overall Conclusions

### What FP32 Accumulation Solves

| Component | Solved? | Mechanism |
|---|---|---|
| **RMSNorm** | **YES (perfect)** | Simple sum — FP32 absorbs all ordering differences |
| **GEMM split-K** | **Partially (8x better)** | Reduces but doesn't eliminate variance |
| **Softmax** | **YES (perfect)** | sum(exp) is simple accumulation |
| **Attention split-KV** | **NO** | Different reduction tree structure, not just precision |

### FP32 Accum is Necessary but Not Sufficient

1. **For GEMM and RMSNorm**: FP32 accumulation alone is effective
2. **For Attention**: Must combine FP32 accum with **fixed split-KV size**
   (constant chunk C, no batch-dependent split count)
3. **For serving engines**: Also need unified KV layout (page table update
   before attention) and deterministic NCCL all-reduce

### Recommended Approach

```
Layer               Strategy                                    Overhead
-------------------------------------------------------------------------
GEMM/Linear         allow_bf16_reduced_precision_reduction=False  ~0% (!)
RMSNorm             Triton FP32 accumulator kernel               ~0%
Softmax (MoE gate)  FP32 accumulator kernel                      ~0%
Attention           Fixed split-KV + FP32 accum Triton kernel    ~6%
TP Communication    Deterministic NCCL all-reduce                config-level
```

**Measured total overhead: ~6%** (dominated entirely by attention kernel replacement).
