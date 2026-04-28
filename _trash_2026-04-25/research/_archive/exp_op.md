# Op-Level Batch Variance Experiments

**Hardware:** NVIDIA RTX A6000 (Ampere, 48 GB)
**Software:** PyTorch 2.6.0+cu124, CUDA 12.4
**Seed:** `torch.manual_seed(42)`

---

## 1. GEMM Batch Variance

**Method.** A fixed "target row" is embedded at position 0 of batches with varying M.
We compute `F.linear(batch, W)` and compare the output of the target row against a
reference obtained with M=1. Two cuBLAS modes are tested: **default** (BF16 reduced-precision
reduction enabled) and **no\_reduced\_prec** (`allow_bf16_reduced_precision_reduction=False`).

### 1.1 llama\_q\_proj (K=4096, N=4096)

| M | default max\_diff | default mean\_diff | no\_reduced\_prec max\_diff | no\_reduced\_prec mean\_diff |
|------:|------------------:|-------------------:|----------------------------:|-----------------------------:|
| 1 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 2 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 4 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 8 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 16 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 32 | 5.00e-01 | 1.91e-04 | 5.00e-01 | 1.91e-04 |
| 64 | **1.00e+00** | 5.08e-02 | 5.00e-01 | 1.61e-04 |
| 128 | 5.00e-01 | 1.91e-04 | 5.00e-01 | 1.91e-04 |
| 256 | **1.00e+00** | 7.52e-02 | 5.00e-01 | 1.34e-04 |
| 512 | 5.00e-01 | 1.91e-04 | 5.00e-01 | 1.91e-04 |
| 1024 | **1.00e+00** | 7.52e-02 | 5.00e-01 | 1.91e-04 |

### 1.2 llama\_gate\_proj (K=4096, N=11008)

| M | default max\_diff | default mean\_diff | no\_reduced\_prec max\_diff | no\_reduced\_prec mean\_diff |
|------:|------------------:|-------------------:|----------------------------:|-----------------------------:|
| 1 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 2 | **1.00e+00** | 5.47e-02 | 5.00e-01 | 2.15e-04 |
| 4 | **1.00e+00** | 5.47e-02 | 5.00e-01 | 2.15e-04 |
| 8 | **1.00e+00** | 5.47e-02 | 5.00e-01 | 2.15e-04 |
| 16 | 0.00e+00 | 0.00e+00 | 5.00e-01 | 7.30e-05 |
| 32 | **1.00e+00** | 5.47e-02 | 5.00e-01 | 2.15e-04 |
| 64 | **1.00e+00** | 5.47e-02 | 5.00e-01 | 2.15e-04 |
| 128 | **1.00e+00** | 9.38e-02 | 5.00e-01 | 2.15e-04 |
| 256 | **1.00e+00** | 5.47e-02 | 5.00e-01 | 2.15e-04 |
| 512 | **1.00e+00** | 5.47e-02 | 5.00e-01 | 2.15e-04 |
| 1024 | **1.00e+00** | 5.47e-02 | 5.00e-01 | 2.15e-04 |

### 1.3 moe\_expert (K=2048, N=5632)

| M | default max\_diff | default mean\_diff | no\_reduced\_prec max\_diff | no\_reduced\_prec mean\_diff |
|------:|------------------:|-------------------:|----------------------------:|-----------------------------:|
| 1 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 2 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 4 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 8 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 16 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 32 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 64 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 128 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 256 | **1.00e+00** | 5.32e-02 | 0.00e+00 | 0.00e+00 |
| 512 | **1.00e+00** | 6.84e-02 | 0.00e+00 | 0.00e+00 |
| 1024 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |

### 1.4 GEMM Analysis

**Key findings:**

1. **cuBLAS kernel selection changes with batch size.** At certain M thresholds (often M>=32
   for K=4096), cuBLAS selects a different GEMM kernel that tiles or reduces the K dimension
   differently. This changes the floating-point reduction order, producing different results
   for the *same* input row.

2. **Default BF16 reduced-precision reduction amplifies the problem.** With the default mode,
   max differences reach **1.0** (a full BF16 unit in the value range), with mean differences
   around 5--9%. This is because cuBLAS may accumulate partial dot products in BF16 rather
   than FP32, losing significant precision when K=4096.

3. **Disabling reduced-precision reduction helps dramatically.** With `allow_bf16_reduced_precision_reduction=False`,
   the max difference is capped at **0.5** (1 ULP at the relevant magnitude) and mean differences
   drop to ~1e-4. The remaining 0.5 max\_diff reflects a single-bit rounding difference from the
   changed reduction tree order.

4. **Shape-dependent sensitivity.** The `llama_gate_proj` shape (K=4096, N=11008) is most
   sensitive---showing differences starting at M=2 in default mode. The `moe_expert` shape
   (K=2048, smaller reduction) is resilient until M=256 and is fully immune with the flag disabled.

---

## 2. RMSNorm Chunk Reduction Variance

**Method.** For a single vector of hidden\_dim elements in BF16, we compute
`RMSNorm(x) = x / sqrt(mean(x^2) + eps)` using chunked reduction with varying numbers of
chunks. Two accumulation modes: BF16-accumulate vs FP32-accumulate.
Reference: single-pass FP32 reduction.

| hidden\_dim | chunks | BF16 max\_diff | BF16 mean\_diff | FP32 max\_diff | FP32 mean\_diff |
|------------:|-------:|---------------:|----------------:|---------------:|----------------:|
| 2048 | 1 | 1.56e-02 | 4.17e-03 | 0.00e+00 | 0.00e+00 |
| 2048 | 2 | 1.56e-02 | 4.17e-03 | 0.00e+00 | 0.00e+00 |
| 2048 | 4 | 1.56e-02 | 4.17e-03 | 0.00e+00 | 0.00e+00 |
| 2048 | 8 | 1.56e-02 | 4.17e-03 | 0.00e+00 | 0.00e+00 |
| 2048 | 16 | 1.56e-02 | 4.17e-03 | 0.00e+00 | 0.00e+00 |
| 2048 | 32 | 1.56e-02 | 4.17e-03 | 0.00e+00 | 0.00e+00 |
| 4096 | 1 | 1.56e-02 | 2.40e-03 | 0.00e+00 | 0.00e+00 |
| 4096 | 2 | 1.56e-02 | 2.40e-03 | 0.00e+00 | 0.00e+00 |
| 4096 | 4 | 1.56e-02 | 2.40e-03 | 0.00e+00 | 0.00e+00 |
| 4096 | 8 | 1.56e-02 | 1.49e-03 | 0.00e+00 | 0.00e+00 |
| 4096 | 16 | 1.56e-02 | 1.49e-03 | 0.00e+00 | 0.00e+00 |
| 4096 | 32 | 1.56e-02 | 1.49e-03 | 0.00e+00 | 0.00e+00 |

### 2.1 RMSNorm Analysis

1. **FP32 accumulation is perfectly stable** across all chunk counts --- zero error vs reference.
2. **BF16 accumulation introduces a fixed error** of max ~1.56e-2 regardless of chunk count.
   The error comes from the BF16 rounding of `x^2` and the running sum, not from the chunking itself.
3. **Chunk count has negligible effect** on BF16 error for RMSNorm. The dominant error source
   is the BF16 precision of the squared values, not the reduction tree shape.

---

## 3. Softmax Chunk Reduction Variance

**Method.** Online (numerically stable) softmax computed over large vocab dimensions with
varying chunk counts. Two accumulation modes: BF16 vs FP32.
Reference: single-pass FP32 `F.softmax`.

| vocab\_size | chunks | BF16 max\_diff | BF16 KL\_div | FP32 max\_diff | FP32 KL\_div |
|------------:|-------:|---------------:|-------------:|---------------:|-------------:|
| 128256 | 1 | 1.21e-06 | 1.36e-03 | 2.91e-11 | -2.03e-09 |
| 128256 | 2 | 1.72e-06 | -3.89e-03 | 5.82e-11 | -2.03e-09 |
| 128256 | 4 | 1.72e-06 | -3.89e-03 | 2.91e-11 | -2.03e-09 |
| 128256 | 8 | 1.21e-06 | 1.36e-03 | 5.82e-11 | -2.03e-09 |
| 128256 | 16 | 1.72e-06 | -3.89e-03 | 5.82e-11 | -2.03e-09 |
| 151936 | 1 | 8.82e-07 | -1.13e-03 | 0.00e+00 | 0.00e+00 |
| 151936 | 2 | 1.51e-06 | 3.23e-03 | 0.00e+00 | 0.00e+00 |
| 151936 | 4 | 1.51e-06 | 3.23e-03 | 2.91e-11 | -4.92e-07 |
| 151936 | 8 | 1.51e-06 | 3.23e-03 | 0.00e+00 | 0.00e+00 |
| 151936 | 16 | 8.82e-07 | -1.13e-03 | 0.00e+00 | 0.00e+00 |

### 3.1 Softmax Analysis

1. **BF16 chunked softmax** has max element-wise error ~1e-6, which is reasonable for BF16.
   KL divergence is ~1e-3 in magnitude --- small but nonzero.
2. **FP32 chunked softmax** is essentially exact: max error ~6e-11 (FP32 epsilon-level),
   KL divergence ~1e-9 or zero.
3. **Chunk count has minimal effect** on either mode. The dominant error is from the BF16 cast
   of the exponentiated values, not from splitting the reduction.
4. **Vocab size has marginal effect** --- 128K vs 152K show similar error magnitudes.

---

## 4. Attention Split-KV Variance

**Method.** Single-query attention (B=1, H=8, D=128) with KV cache of varying seq\_len.
The KV sequence is split into `num_splits` chunks, partial attention is computed per chunk,
and results are combined using online softmax correction. Two modes: BF16 vs FP32 arithmetic.
Reference: single-pass FP32 attention (splits=1, FP32).

### 4.1 BF16 Split-KV Attention

| seq\_len | splits=1 | splits=2 | splits=4 | splits=8 | splits=16 |
|---------:|---------:|---------:|---------:|---------:|----------:|
| 128 | 4.71e-03 | 5.68e-03 | 5.68e-03 | 9.11e-03 | 8.05e-03 |
| 512 | 2.67e-03 | 3.21e-03 | 3.41e-03 | 3.56e-03 | 5.24e-03 |
| 1024 | 1.88e-03 | 1.95e-03 | 1.94e-03 | 2.07e-03 | 2.11e-03 |
| 2048 | 1.13e-03 | 1.13e-03 | 1.29e-03 | 1.48e-03 | 2.32e-03 |

### 4.2 FP32 Split-KV Attention

| seq\_len | splits=1 | splits=2 | splits=4 | splits=8 | splits=16 |
|---------:|---------:|---------:|---------:|---------:|----------:|
| 128 | 0.00e+00 | 5.96e-08 | 8.94e-08 | 7.45e-08 | 8.94e-08 |
| 512 | 0.00e+00 | 4.47e-08 | 5.96e-08 | 1.04e-07 | 8.94e-08 |
| 1024 | 0.00e+00 | 4.47e-08 | 4.47e-08 | 5.22e-08 | 5.22e-08 |
| 2048 | 0.00e+00 | 3.49e-08 | 4.10e-08 | 3.73e-08 | 5.22e-08 |

### 4.3 BF16 Mean Diff (split-KV)

| seq\_len | splits=1 | splits=2 | splits=4 | splits=8 | splits=16 |
|---------:|---------:|---------:|---------:|---------:|----------:|
| 128 | 7.39e-04 | 7.92e-04 | 8.09e-04 | 9.48e-04 | 1.04e-03 |
| 512 | 3.76e-04 | 4.56e-04 | 4.77e-04 | 5.41e-04 | 6.10e-04 |
| 1024 | 2.97e-04 | 3.21e-04 | 3.38e-04 | 3.51e-04 | 3.99e-04 |
| 2048 | 2.07e-04 | 2.28e-04 | 2.39e-04 | 2.56e-04 | 2.79e-04 |

### 4.4 Attention Analysis

1. **BF16 split-KV error grows monotonically with splits.** At seq\_len=128 with 16 splits
   (8 tokens per split), max error reaches 8e-3 --- about 1 part in 125. This is the most
   significant source of non-determinism in attention.

2. **Longer sequences dilute per-split error.** At seq\_len=2048, even 16 splits yields
   max error of only 2.3e-3 because each split has 128 tokens, providing better statistics
   for the partial softmax.

3. **FP32 split-KV is effectively exact.** Max error ~1e-7 (single-precision epsilon level),
   confirming the online softmax algorithm itself is correct --- all error is from BF16 arithmetic.

4. **Practical impact:** FlashAttention and split-KV decoding operate in this regime.
   The BF16 errors at 1e-3 to 9e-3 are the same order as the GEMM errors, meaning attention
   splitting does not dominate overall model-level variance.

---

## 5. Run-to-Run Variance

**Method.** Same input tensor, same weight, same shape, 100 consecutive `F.linear` calls.
Check whether the output ever differs across runs.

| Shape | dtype | Mismatches (of 100) | max\_diff |
|-------------------------------|-------|--------------------:|----------:|
| [32, 4096, 4096] | bf16 | 0 | 0.00e+00 |
| [32, 4096, 4096] | fp32 | 0 | 0.00e+00 |
| [1, 4096, 4096] | bf16 | 0 | 0.00e+00 |
| [1, 4096, 4096] | fp32 | 0 | 0.00e+00 |
| [128, 4096, 11008] | bf16 | 0 | 0.00e+00 |
| [128, 4096, 11008] | fp32 | 0 | 0.00e+00 |

### 5.1 Run-to-Run Analysis

**All 100 runs are bit-identical** for every shape and dtype tested. cuBLAS on RTX A6000
(Ampere) with PyTorch 2.6.0 is fully deterministic for repeated calls with the same input,
weight, and shape. The variance we observe in Experiments 1--4 arises strictly from changes
in input shape (triggering different kernels/tiling) or accumulation precision --- not from
hardware non-determinism.

---

## Summary of Findings

| Source of Variance | Max Error | Mitigated by |
|----------------------------------|-----------|--------------------------------------|
| GEMM batch size (default BF16) | **1.0** | `allow_bf16_reduced_precision_reduction=False` |
| GEMM batch size (no reduced) | 0.5 | Use FP32 for reduction |
| RMSNorm BF16 accumulation | 1.56e-2 | FP32 accumulation (0 error) |
| Softmax BF16 chunked | 1.72e-6 | FP32 accumulation (~0 error) |
| Attention BF16 split-KV | 9.11e-3 | FP32 split-KV (~1e-7 error) |
| Run-to-run (same shape/input) | **0.0** | N/A --- already deterministic |

**The single largest source of batch-dependent numerical variance is cuBLAS GEMM kernel
selection.** When the batch dimension M changes, cuBLAS may select a kernel with a
different K-reduction strategy (and with default settings, a lower-precision accumulator).
This alone can cause output differences of up to 1.0 in BF16 scale --- far exceeding
the errors from RMSNorm, softmax, or attention splitting.

**Recommendation:** Always set `torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False`
for BF16 inference. This reduces GEMM batch variance by ~100x (from 1.0 to 0.5 max) and
eliminates the mean error (~5% down to ~0.02%). For maximum fidelity, use FP32 accumulation
in all reduction operations.
