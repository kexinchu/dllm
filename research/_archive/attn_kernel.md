# Attention Kernel Analysis: Batch Invariance in HuggingFace vs. Serving Engines

**Hardware:** NVIDIA RTX A6000 (Ampere, 48 GB)
**Software:** PyTorch 2.6.0+cu124, CUDA 12.4
**Model:** Llama-3.1-8B-Instruct

---

## 1. Why HuggingFace Attention Is Already Batch-Invariant

### 1.1 SDPA Dispatch Path

When HuggingFace models use `attn_implementation="sdpa"` (the default), attention is computed via `torch.nn.functional.scaled_dot_product_attention`. On CUDA with BF16 inputs, SDPA dispatches to one of two fused backends:

1. **FlashAttention** (preferred when available): Uses the FlashAttention-2 algorithm with tiled online softmax. Each program handles one Q block and iterates over the **full KV sequence** in blocks of `BLOCK_N` (typically 64 or 128).
2. **Memory-efficient attention** (fallback): Similar tiled approach with full KV iteration per Q block.

Both backends already use FP32 accumulators internally for the online softmax state (`m_i`, `l_i`) and the output accumulator (`acc`). This is a design requirement of the FlashAttention algorithm, not an optional optimization.

### 1.2 No Split-KV in Standard SDPA

The critical property: neither FlashAttention nor memory-efficient attention splits the KV sequence across multiple streaming multiprocessors (SMs) for a single query block. The parallelism is along the Q dimension (and batch/head dimensions), not along the KV dimension.

Each program's execution follows a fixed traversal:

```
for start_n in range(0, kv_end, BLOCK_N):
    load K[start_n : start_n + BLOCK_N]
    load V[start_n : start_n + BLOCK_N]
    compute QK^T block
    update online softmax (m_i, l_i, acc)
```

This traversal order depends only on `kv_end` (the sequence length) and `BLOCK_N` (a compile-time constant). It does **not** depend on batch size, batch composition, or SM occupancy. Consequently, the reduction order within attention is identical regardless of how many other sequences are in the batch.

### 1.3 The Only Batch-Variance Source Is GEMM

If attention itself is batch-invariant under SDPA, where does batch variance come from? The answer is the GEMM operations surrounding attention:

- **Q/K/V projections**: `q = x @ W_q.T`, `k = x @ W_k.T`, `v = x @ W_v.T`
- **Output projection**: `o = attn_out @ W_o.T`

These are computed by cuBLAS, which selects different kernels (with different reduction strategies) depending on the M dimension (batch size x sequence length). This is the split-K mechanism described in `research/theory.md` Section 3.3.

### 1.4 Evidence: cuBLAS Flag Achieves Full Determinism

Our end-to-end experiments on Llama-3.1-8B-Instruct confirm this analysis:

**Experiment: Logits-level batch invariance** (from `motivation/fp32_accum_full_eval_results.json`)

| Mode | bs=2 max_diff | bs=8 max_diff | bs=8 argmax match |
|------|---------------|---------------|-------------------|
| Pure BF16 | 2.19e-01 | **2.68e+01** | NO |
| Linear-only FP32 accum (cuBLAS flag) | 0.00e+00 | 0.00e+00 | YES |
| Full FP32 accum (all ops) | 0.00e+00 | 0.00e+00 | YES |

Setting `torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False` alone (Linear-only mode) achieves **zero logits difference** across all tested batch sizes (bs=2, 4, 8, 16). Adding custom RMSNorm, softmax, and attention kernels on top provides no additional benefit for the HuggingFace SDPA path.

**Experiment: 500-run generation determinism** (from project context)

With the cuBLAS flag enabled, 500 generation runs with varying batch sizes produced exactly **1 unique output** --- perfect determinism. This was achieved without replacing the attention kernel.

---

## 2. Why Serving Engines Have Attention Batch Variance

### 2.1 FlashDecoding and Split-KV

During the decode phase of autoregressive generation, each step produces a single query token per sequence. With query length = 1, there is no Q-dimension parallelism to exploit. To avoid leaving SMs idle, serving engines use **FlashDecoding** (Dao et al., 2023), which splits the KV sequence across multiple SMs:

```
KV sequence: [k_0, k_1, ..., k_{L-1}]

Split into S chunks:
  SM_0: [k_0, ..., k_{L/S - 1}]       -> partial (m_0, l_0, o_0)
  SM_1: [k_{L/S}, ..., k_{2L/S - 1}]   -> partial (m_1, l_1, o_1)
  ...
  SM_{S-1}: [k_{(S-1)L/S}, ..., k_{L-1}] -> partial (m_{S-1}, l_{S-1}, o_{S-1})

Final reduction: combine partials via online softmax correction
```

The number of splits S is determined by a heuristic that considers available SMs and current batch occupancy. When the batch size changes, S may change, producing a structurally different computation graph (Theorem 2 in `research/theory.md`).

### 2.2 Experimental Confirmation: Split-KV Variance Is Structural

Our op-level experiments (`research/exp_op.md` Section 4) demonstrate that split-KV variance is **not** an accumulation precision problem:

| seq_len | splits=1 (FP32) | splits=2 (FP32) | splits=4 (FP32) | splits=8 (FP32) | splits=16 (FP32) |
|---------|------------------|------------------|------------------|------------------|-------------------|
| 128 | 0.00e+00 | 5.96e-08 | 8.94e-08 | 7.45e-08 | 8.94e-08 |
| 512 | 0.00e+00 | 4.47e-08 | 5.96e-08 | 1.04e-07 | 8.94e-08 |
| 2048 | 0.00e+00 | 3.49e-08 | 4.10e-08 | 3.73e-08 | 5.22e-08 |

FP32 accumulation within each split produces near-zero error **for a fixed split count**. But the model-level simulation (`motivation/fp32_accum_full_eval_results.json`) shows that varying the split count produces errors of 0.42--2.55 in BF16 scale, **identical for BF16 and FP32 accumulation**:

| Splits | BF16 max_diff | FP32 max_diff | Improvement |
|--------|---------------|---------------|-------------|
| 2 | 0.418 | 0.418 | 1x (none) |
| 4 | 1.137 | 1.145 | 1x (none) |
| 16 | 2.548 | 2.548 | 1x (none) |

This confirms Theorem 2 of `research/theory.md`: different split counts create different computation graphs (different local maxima, different rescaling chains), and this structural mismatch cannot be absorbed by higher-precision arithmetic.

### 2.3 PagedAttention and Chunked Prefill

Serving engines introduce two additional sources of batch-dependent attention variance:

1. **PagedAttention**: KV cache is stored in non-contiguous physical pages. The page table mapping depends on the allocation history, which varies with batch composition. Different physical layouts can cause different memory access patterns and, in some implementations, different tile boundaries within the attention kernel.

2. **Chunked prefill**: Long prefill sequences are split into chunks to interleave with decode requests. The chunk boundaries depend on the current batch's scheduling state. Different chunk boundaries produce different attention computation graphs for the same sequence, analogous to the split-KV problem.

---

## 3. Fixed Split-KV Design for Serving Engines

### 3.1 Design Principle

To eliminate attention batch variance in serving engines, the split-KV boundaries must be fixed --- independent of batch size, SM occupancy, or co-resident sequences. The design principle follows Theorem 3 of `research/theory.md`:

> Fix the split boundaries, then apply FP32 accumulation within each split.

### 3.2 Algorithm

Given a constant chunk size C (e.g., 256 tokens):

```
Input:  Q[1, d], K[L, d], V[L, d]   (single query, KV cache of length L)
Output: o[1, d]

P = ceil(L / C)    # number of chunks, determined solely by L and C

for s in 0, 1, ..., P-1:
    start = s * C
    end   = min((s + 1) * C, L)

    # Local computation within chunk (FP32 accumulators)
    K_s = K[start:end]
    V_s = V[start:end]
    qk_s = Q @ K_s.T * scale            # FP32 accumulator
    m_s  = max(qk_s)                     # local max
    p_s  = exp(qk_s - m_s)              # FP32
    l_s  = sum(p_s)                      # FP32 accumulator
    o_s  = p_s @ V_s                     # FP32 accumulator

    # Cross-chunk combination (online softmax correction)
    m_new = max(m_running, m_s)
    alpha = exp(m_running - m_new)
    beta  = exp(m_s - m_new)
    l_running = alpha * l_running + beta * l_s
    o_running = alpha * o_running + beta * o_s
    m_running = m_new

o = o_running / l_running               # final normalization
```

### 3.3 Key Properties

1. **Reduction order is fixed**: Always `[chunk_0, chunk_1, ..., chunk_{P-1}]`, determined by L and C alone.
2. **No batch dependence**: C is a compile-time constant. P depends only on the individual sequence's KV length, not on other sequences in the batch.
3. **FP32 accumulators within chunks**: `m_s`, `l_s`, `o_s` are all FP32, ensuring that any internal tiling variation within a chunk (from different GPU occupancy) is absorbed by FP32 precision (Theorem 1).
4. **Sequential cross-chunk combination**: The online softmax correction is applied in a fixed sequential order, with no parallelism ambiguity.
5. **Unified KV layout**: Before entering the kernel, the page table is updated so that cache tokens and current tokens form a single contiguous logical sequence.

---

## 4. Reference Implementation

### 4.1 Existing Kernel: `FP32/attention_fp32_accum.py`

The project includes a Triton attention kernel at `/home/kec23008/docker-sys/dllm/FP32/attention_fp32_accum.py` that already embodies the fixed-split principles:

- **No split-KV**: Each `(batch, head, q_block)` program iterates over the full KV sequence (line 78: `for start_n in range(0, kv_end, BLOCK_N)`).
- **Fixed tile size**: `BLOCK_N = 64`, a compile-time constant.
- **FP32 accumulators**: `m_i`, `l_i`, and `acc` are all explicitly `tl.float32` (lines 66--68).
- **FP32 dot products**: `tl.dot(q, tl.trans(k), out_dtype=tl.float32)` and `tl.dot(p.to(tl.bfloat16), v, out_dtype=tl.float32)` (lines 88, 113).
- **GQA support**: K/V heads are expanded to match Q heads via `_expand_kv_for_gqa` (line 231).

From the kernel header (lines 7--9):

```
No split-KV: each (batch, head, q_block) program iterates over full KV sequence
-> reduction order depends only on seq_len and BLOCK_N, not on batch size
-> naturally batch-invariant
```

### 4.2 Integration via Model Patcher

The kernel is integrated into HuggingFace models through `FP32/model_patcher.py`, which replaces `F.scaled_dot_product_attention` calls with the custom kernel. The `fp32_accum_mode` context manager provides fine-grained control:

```python
with fp32_accum_mode(model, patch_linear=True, patch_attention=True):
    output = model.generate(...)
```

### 4.3 Production Considerations

For production deployment in serving engines, the kernel should be:

1. **Integrated into FlashInfer or vLLM's attention backend** rather than monkey-patched, to handle paged KV cache natively.
2. **Extended with FlashDecoding-style parallelism** using a fixed chunk size C (as described in Section 3), rather than the current full-KV-iteration approach, which underutilizes SMs during decode.
3. **Tested with vLLM's `VLLM_BATCH_INVARIANT=1`** or SGLang's `--enable-deterministic-inference` flag, which provide the system-level infrastructure (page table unification, deterministic NCCL) needed alongside the kernel-level fix.

---

## 5. Why We Do Not Need Custom Attention for This Paper's Experiments

### 5.1 Argument

All experiments in this paper use HuggingFace with the SDPA backend (not vLLM or SGLang). Under this configuration:

1. **SDPA already uses FP32 accumulators** internally. Both FlashAttention and memory-efficient attention compute online softmax in FP32. There is no precision gap to close.

2. **SDPA does not use split-KV**. Each program iterates the full KV sequence, so there is no batch-dependent split count. The attention reduction order is fixed by `(seq_len, BLOCK_N)`.

3. **The cuBLAS flag resolves the only source of batch variance**. Setting `allow_bf16_reduced_precision_reduction = False` forces FP32 accumulation in all GEMM operations (Q/K/V projections, output projection, MLP layers, LM head). Since GEMM is the only batch-variant operation in the HuggingFace SDPA path, this single flag achieves full determinism.

4. **Replacing attention with a custom kernel introduces error, not removes it**. Our Triton kernel and SDPA use different tiling strategies (`BLOCK_M=64, BLOCK_N=64` vs. FlashAttention's tuned parameters). These produce micro-different FP32 intermediate values that compound through 32 transformer layers, producing outputs that differ from SDPA's outputs. The custom kernel is *also* deterministic, but it is a *different* deterministic result. For experiments comparing against baseline HuggingFace, this creates an unnecessary confound.

### 5.2 Evidence: Performance Impact

From the latency benchmarks (`docs/FP32-accum-eval-summary.md`):

| Mode | Latency (ms) | Overhead |
|------|-------------|----------|
| Pure BF16 baseline | 1,252 | 1.00x |
| cuBLAS flag only (recommended) | 1,227 | **0.98x** (zero overhead) |
| cuBLAS flag + RMSNorm + Softmax | 1,202 | **0.96x** (zero overhead) |
| Full FP32 accum (incl. custom attention) | 1,324 | **1.06x** (6% overhead) |

The cuBLAS flag path is actually *faster* than pure BF16 on Ampere, likely because FP32 accumulation avoids some pathological BF16 reduction-precision codepaths. The only overhead comes from replacing SDPA with our Triton attention kernel (6%).

Since the custom attention kernel adds overhead without improving determinism in the HuggingFace SDPA setting, **the recommended configuration for this paper's experiments is cuBLAS flag only**.

---

## 6. Performance Analysis

### 6.1 Overhead Summary

| Configuration | Overhead | Deterministic? | When to use |
|---------------|----------|----------------|-------------|
| cuBLAS flag only | **0.98x** (none) | Yes (HF/SDPA) | HuggingFace inference |
| cuBLAS flag + Triton attention | **1.06x** (6%) | Yes (any backend) | Serving engines with custom attention |
| cuBLAS flag + all custom ops | **1.06x** (6%) | Yes (any backend) | Serving engines (full stack) |
| Naive FP32 cast (no cuBLAS flag) | **34.9x** | Yes | Never (use cuBLAS flag instead) |

### 6.2 Breakdown

**cuBLAS flag (`allow_bf16_reduced_precision_reduction = False`)**:
- Zero kernel replacement; cuBLAS simply selects FP32 accumulation codepaths natively.
- On Ampere (A6000), this is marginally faster than BF16 reduced-precision mode.
- Eliminates GEMM batch variance completely (0 logits difference across all tested batch sizes).

**Custom Triton attention replacing SDPA**:
- Adds ~6% latency due to less-optimized tiling compared to cuBLAS-dispatched FlashAttention.
- Unnecessary for HuggingFace experiments (SDPA is already batch-invariant for attention).
- Required for serving engines where the default attention backend uses dynamic split-KV.

**Naive FP32 cast (previous approach)**:
- Casting inputs to FP32 before GEMM and back to BF16 after: 34.9x slowdown.
- This was the initial implementation before discovering the cuBLAS flag.
- Completely superseded; included here only for historical context.

### 6.3 Comparison with Industry Approaches

| System | Approach | Reported Overhead |
|--------|----------|-------------------|
| This paper (HF) | cuBLAS flag only | **~0%** |
| Thinking Machine Lab | Custom GEMM + Attention + RMSNorm kernels | 61.5% |
| SGLang deterministic mode | Fixed split-KV + deterministic all-reduce | 34.35% |
| vLLM `BATCH_INVARIANT=1` | FlexAttention + batch_invariant_ops | In development |

The key insight is that for HuggingFace inference (no continuous batching, no paged KV cache, no split-KV decode), the cuBLAS flag alone is sufficient. The heavier-weight kernel replacements are necessary only for serving engines that introduce batch-dependent parallelism in attention.

---

## 7. Summary

| Question | Answer | Evidence |
|----------|--------|----------|
| Is HF attention batch-variant? | **No** --- SDPA uses fixed KV traversal | 0 logits diff with cuBLAS flag only |
| Is serving-engine attention batch-variant? | **Yes** --- FlashDecoding uses dynamic split-KV | FP32 accum gives 0 improvement (2.55 error for both BF16 and FP32) |
| Does FP32 accum fix attention split-KV? | **No** --- structural mismatch, not precision issue | Theorem 2: different computation graphs |
| Does fixed split-KV + FP32 accum fix it? | **Yes** --- restores additive reduction regime | Theorem 3 + FP32 split-KV exp shows ~1e-7 error |
| Should we replace attention for HF experiments? | **No** --- adds 6% overhead with no benefit | cuBLAS flag alone: 0.98x, full replacement: 1.06x |
| What about serving engines (future work)? | Fixed split-KV kernel needed, integrate into FlashInfer/vLLM | See Section 3 design |
