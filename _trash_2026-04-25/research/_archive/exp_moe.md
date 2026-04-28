# DeepSeek-V2-Lite MoE Determinism Results

**Model**: DeepSeek-V2-Lite (BF16, 15.7B total, 64 experts, top-6)
**Hardware**: NVIDIA RTX A6000

## Generation Determinism (200 runs, continuous batching sim)

| Mode | Unique Outputs | Deterministic? |
|------|---:|:---:|
| BF16 (default) | 3 | No |
| FP32 accum | 3 | No |

## Logits Batch Variance

| bs | BF16 max_diff | FP32 max_diff | BF16 argmax | FP32 argmax |
|---:|---:|---:|:---:|:---:|
| 2 | 6.88e-01 | 6.72e-01 | FLIP | FLIP |
| 4 | 8.75e-01 | 7.81e-01 | FLIP | FLIP |
| 8 | **1.50e+00** | 6.56e-01 | FLIP | FLIP |
| 16 | 8.75e-01 | 7.50e-01 | FLIP | FLIP |

FP32 accum reduces max_diff (especially at bs=8: 1.50→0.66), but argmax still flips at all batch sizes.

## Hash Distribution Analysis

**BF16**: 120:40:40 (3 groups) — bs={1,2,4} share hash A, bs=8 has hash B, bs=16 has hash C.
**FP32**: 80:80:40 (3 groups) — different grouping, still 3 distinct outputs.

Each individual batch size is perfectly deterministic (1 unique / 40 runs).
Non-determinism is purely cross-batch-size: MoE expert routing is too sensitive to the residual 0.5 ULP GEMM difference.

## Latency

| BF16 | FP32 accum | Overhead |
|---:|---:|---:|
| 253.9 ms | 253.8 ms | 1.00x |

## Key Insight

For MoE models, the cuBLAS FP32 accum flag is **necessary but not sufficient**.
The 0.5 ULP residual GEMM variance (Theorem 1's rounding-boundary events) is enough to flip expert selection at near-tie positions, because MoE routing involves:
1. Gate linear (GEMM) — 0.5 ULP diff
2. Softmax — amplifies small logit differences
3. Top-k — hard threshold at near-tie boundary

Complete solution for MoE requires:
- FP32 accum (reduces GEMM variance)
- **AND** deterministic top-k with tie-breaking rules
- **AND** potentially wider margin in expert selection (e.g., temperature scaling)
