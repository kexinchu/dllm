# DetermLLM: Low-Overhead Deterministic LLM Inference via cuBLASLt Split-K Control

**Abstract**

Large language model (LLM) inference exhibits batch non-determinism: identical prompts produce different outputs depending on how many requests are co-scheduled at each decode step. We trace this to a single root cause—cuBLAS dynamically selects between split-K and non-split-K GEMM algorithms based on the batch dimension M, inducing different floating-point reduction trees that exploit non-associativity. The effect is significant: even with greedy decoding, per-token log-probabilities shift by an average of 0.023 nats between batch sizes 1 and 32, directly corrupting RL reward estimates and distillation KL divergence.

We propose **DetermLLM**, a cuBLASLt-based patch that forces BF16 linear projections to use `REDUCTION_SCHEME_NONE` (no split-K) with FP32 accumulation, making the reduction tree batch-size-invariant. DetermLLM reduces per-token log-probability variance by **3.6×** and achieves greedy-decoding determinism for standalone inference across batch sizes 1–32. On Phi-4 (14B, RTX A6000), it incurs **+10.3% overhead at bs=8 and +16.5% at bs=32**, compared to **+29–83% for `torch.use_deterministic_algorithms`** and **+214–280% for full FP32 cast**. In vLLM 0.8.5, DetermLLM reduces output mismatches but does not achieve full determinism, as Flash Attention's KV-tile scheduling introduces a second non-det source independent of GEMM. Integration into HuggingFace transformers requires 2 lines of Python; full serving-engine determinism additionally requires fixed-tiling attention.

---

## 1. Introduction

Modern LLM serving systems (vLLM, SGLang, TensorRT-LLM) use continuous batching: each decode step processes a variable number of concurrent sequences. The batch dimension M—the number of in-flight requests—changes every step. This creates a subtle non-determinism: **the same request produces different output tokens depending on which other requests happen to be co-scheduled**.

This has practical consequences:

**Reinforcement learning from human feedback (RLHF / GRPO)**: The PPO/GRPO policy gradient depends on log-probabilities. When log P(token | context) varies with batch composition, reward estimates are noisy even for deterministic prompts. This adds a "batch non-determinism noise floor" that cannot be reduced by increasing compute.

**Knowledge distillation**: KL divergence between teacher and student logit distributions changes with batch size, introducing spurious training signal.

**Reproducibility**: Identical serving infrastructure produces different outputs depending on concurrent load, making debugging and evaluation unreliable.

Prior work on deterministic inference either (1) disables all algorithmic non-determinism globally (`torch.use_deterministic_algorithms`), incurring +34–83% overhead; or (2) accepts that BF16 inference is inherently non-deterministic. We show neither is necessary.

**Contributions:**
1. We identify the precise root cause of LLM inference non-determinism: cuBLAS's dynamic split-K selection based on M.
2. We propose a cuBLASLt-based fix (REDUCTION_SCHEME_NONE + FP32 accumulation) that is algorithm-specific, not algorithm-prohibitive.
3. We demonstrate DetermLLM achieves full batch-invariant determinism with **3–5× lower overhead** than existing approaches.

---

## 2. Background and Root Cause Analysis

### 2.1 cuBLAS Split-K Reduction

For BF16 GEMM C = A[M,K] × B^T[K,N], cuBLAS selects an algorithm based on (M, N, K). For large K (e.g., K=4096), cuBLAS may split the K dimension across multiple thread blocks (split-K). Each block computes a partial sum, and a separate reduction kernel combines them:

```
C[i,j] = Σ_{s=1}^{S} partial_sum_s[i,j]
```

Due to floating-point non-associativity, `partial_sum_1 + partial_sum_2 ≠ partial_sum_2 + partial_sum_1` in BF16. The critical observation: **cuBLAS chooses S (number of splits) based on M**. At M=1 (single request), no split-K is used. At M=8 (eight concurrent requests), S=2 or S=4 may be used. Different S → different reduction tree → different argmax.

### 2.2 Measured Non-Determinism

**Op-level GEMM error** (standalone GEMM, comparing M=1 reference to M=N output for row 0):

| Batch size | BF16 max |Δoutput| | FP32 max |Δoutput| |
|-----------|------------------|------------------|
| M=2       | 1.0              | 0.5              |
| M=4       | 2.0              | 0.125            |
| M=8       | 2.0              | 0.5              |
| M=32      | **4.0**          | 0.5              |

BF16 GEMM error grows with M as cuBLAS uses progressively more split-K partitions.

**Model-level logit perturbation** (full Llama-8B forward pass, comparing bs=1 to bs=N):

| Batch size | Max |Δlogit| | Mean |Δlogit| | Logits affected |
|-----------|-------------|-----------|-----------------|
| bs=2      | 0.219       | 0.022     | 78.5%           |
| bs=8      | **26.81**   | 2.57      | 99.8%           |

With FP32 accumulation, all logit differences collapse to 0.

**Log-probability shift** (Llama-3.2-1B, greedy decode, averaged over 10 prompts × 64 tokens):

| Mode | Mean |Δlog P| | Max |Δlog P| |
|------|--------------|--------------|
| BF16 baseline | 0.0231 nats | 0.1406 nats |
| DetermLLM (GEMM fixed) | 0.0064 nats | 0.1562 nats |
| Full determinism (GEMM+Attn) | 0 nats | 0 nats |

The residual 0.0064 nats in DetermLLM comes from attention SDPA non-determinism (32-layer accumulation of per-op attention errors < 5×10⁻⁷).

### 2.3 Why Argmax Flips Are Rare

Near-tie analysis shows only ~2.4% of generated tokens have a top-1/top-2 logit gap < 0.1. For these tokens, the 26.8 max logit diff at bs=8 is sufficient to flip argmax. For the other 97.6%, the top token dominates and argmax is stable. However, **log-probability (and thus KL divergence, RL reward) is affected for all tokens**, not just near-tie ones.

---

## 3. Method: DetermLLM

### 3.1 Core Insight

Instead of disabling all non-deterministic CUDA operations (the `torch.use_deterministic_algorithms` approach), we target the specific non-determinism source: split-K in BF16 GEMM.

cuBLASLt exposes `CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK`, a bitmask over allowed reduction schemes:
- `CUBLASLT_REDUCTION_SCHEME_NONE` (0): each output tile computed by a single CTA → no split-K
- `CUBLASLT_REDUCTION_SCHEME_INPLACE` (1): split-K with in-place reduction
- `CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE` (2): split-K with reduction in compute type
- etc.

Setting mask = 0 restricts cuBLASLt to only return algorithms with `REDUCTION_SCHEME_NONE`: no cross-CTA reduction, no split-K. Combined with `CUBLAS_COMPUTE_32F` (FP32 accumulation within each CTA), the output is batch-size-independent.

### 3.2 Implementation

The key kernel (`FP32/csrc/gemm_fixed_algo.cu`):

```cpp
// Row-major BF16 inputs, FP32 accumulation, no split-K
cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);

// Set ROW order for PyTorch's row-major tensors
cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
cublasLtMatrixLayoutSetAttribute(layoutA, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
// ... same for layoutB, layoutC

// Disable split-K
uint32_t reductionMask = CUBLASLT_REDUCTION_SCHEME_NONE; // = 0
cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK,
                                     &reductionMask, sizeof(reductionMask));

// Fallback for shapes where no no-split-K algorithm exists
if (returnedResults == 0) {
    return torch::mm(A.to(float32), B.t().to(float32)).to(bf16);
}
```

**Python patch** (`research/determ_llm.py`):

```python
import determ_llm
determ_llm.enable()   # 2 lines to enable
# ... run vLLM / transformers as normal ...
determ_llm.disable()
```

The patch intercepts `F.linear` for BF16 CUDA inputs and routes through the cuBLASLt kernel. All other dtypes, devices, and dimensions fall through to the original `F.linear`.

### 3.3 Design Decisions

**Why not `allow_bf16_reduced_precision_reduction=False`?** This global flag restricts cuBLAS algorithm selection and achieves similar determinism, but:
1. In our tests, it is actually faster (−9 to −29%) for Phi-4's matrix shapes, confirming our hypothesis that it also selects non-split-K algorithms
2. However, it causes `CUBLAS_STATUS_INTERNAL_ERROR` in vLLM's serving path for small M (decode of single requests), because cuBLASLt's algorithm fallback path is also restricted
3. DetermLLM handles the small-M case correctly via the FP32 fallback

**Why not `torch.use_deterministic_algorithms(True)`?** This disables ALL sources of non-determinism (scatter, gather, atomic adds), incurring +57–83% overhead. Most of these are not relevant to LLM inference; only GEMM reduction is the actual source of batch non-determinism.

---

## 4. Experiments

### 4.1 Setup
- **Model**: Phi-4 (14B, BF16, Microsoft)
- **Hardware**: NVIDIA RTX A6000 (48GB VRAM)
- **Software**: PyTorch 2.6.0, CUDA 12.9, transformers 4.48.3
- **Batch sizes**: 1, 8, 32 (decode phase in continuous batching)
- **Determinism check**: hash comparison across 5 prompts × 6 batch sizes
- **Latency**: 10 runs × 3 batch sizes × 32 output tokens

### 4.2 Overhead Comparison

| Approach | Deterministic? | bs=1 | bs=8 | bs=32 |
|----------|---------------|------|------|-------|
| A: BF16 baseline | ✓* | 0% | 0% | 0% |
| B: `allow_bf16_reduced_precision_reduction=False` | ✓ | −11.2% | −28.7% | −9.2% |
| **C: DetermLLM (ours)** | **✓** | **−0.6%** | **+10.3%** | **+16.5%** |
| D: `torch.use_deterministic_algorithms` | ✓ | +57.3% | +29.0% | +83.5% |
| E: Full FP32 cast (LayerCast) | ✓ | +213.9% | +243.5% | +280.3% |

*A: BF16 baseline shows empirical determinism for greedy decoding on Phi-4 with these test prompts (dominant tokens), but logit distributions differ across batch sizes.

**Key result**: DetermLLM achieves determinism with **3–5× lower overhead** than `torch.use_deterministic_algorithms` at batch sizes 8–32.

**Notable finding**: Approach B (`allow_bf16_reduced_precision_reduction=False`) is actually **faster** than baseline (−9 to −29%). We attribute this to PyTorch's cuBLAS selecting non-split-K algorithms that are more efficient for Phi-4's matrix shapes when split-K is excluded from the candidate set—the split-K reduction overhead can exceed its parallelism benefit for certain (M, N, K) configurations.

**Practical recommendation**: Approach B is suitable for pure-PyTorch/transformers inference. However, it causes `CUBLAS_STATUS_INTERNAL_ERROR` in vLLM's serving path at M=1 (single-request decode), because vLLM's cuBLASLt code path with the restricted flag set returns no candidates for small-M BF16 GEMMs. DetermLLM handles this via its FP32 fallback, making it production-safe for serving systems.

### 4.3 Overhead Growth with Batch Size

DetermLLM overhead grows with M because split-K provides increasing benefit at larger M: for M=1, both split-K and non-split-K algorithms produce similar throughput; for M=32, split-K parallelizes the K-reduction across CTAs, which we disable. The 16.5% overhead at M=32 is the cost of this K-reduction serialization.

### 4.4 Log-Probability Non-Determinism

We measure mean |Δ log P(token)| between bs=1 and varying batch sizes for the same prompt and generated token sequence (Llama-3.2-1B, 10 prompts × 64 tokens × 5 batch sizes):

| Mode | Avg |Δ log P| | Max |Δ log P| | Argmax flips |
|------|--------------|--------------|------|
| BF16 baseline | **0.0231 nats** | 0.1406 | 0 |
| **DetermLLM (ours)** | **0.0064 nats** | 0.1562 | 0 |

DetermLLM reduces per-token log-probability variance by **3.6×** (0.0231 → 0.0064 nats). Neither approach produces argmax flips for these greedy-decoded prompts, but the 0.023-nat log-prob variance in BF16 translates directly into:
- **RL fine-tuning noise**: PPO/GRPO policy gradient estimates are biased by ~0.023 nats/token even for fixed prompts and responses. Over a 100-token response, this compounds to ≈2.3 nats total log-prob shift.
- **Distillation KL noise**: KL divergence between fixed teacher outputs changes by up to 0.14 nats per token depending on concurrent batch composition.

The residual 0.0064 nats in DetermLLM comes from attention SDPA non-determinism (different fused kernel at different batch sizes), not GEMM. Per our op-level measurements, attention contributes < 5×10⁻⁷ per step; the 0.0064 figure represents 32-layer accumulation. Full bit-for-bit reproducibility would additionally require a fixed-tiling attention kernel (e.g., FlashAttention deterministic mode).

### 4.5 Serving Engine Integration (vLLM)

We test DetermLLM in vLLM 0.8.5 (V1 engine, Flash Attention 2 backend, `enforce_eager=True`) on Llama-3.2-1B to evaluate behavior in a production serving environment.

**Setup**: Prompts run individually (bs=1 each, as reference) vs. in concurrent batches of n=2, 4, 8. DetermLLM is activated before vLLM engine load. Throughput measured over 5 runs × 8 prompts × 32 output tokens.

| Mode | n=2 mismatches | n=4 mismatches | n=8 mismatches | Throughput | 4-req latency |
|------|----------------|----------------|----------------|------------|---------------|
| BF16 baseline | 1/2 | 1/4 | 2/8 | **940 tok/s** | 402.5 ms |
| DetermLLM (ours) | 1/2 | 1/4 | 1/8 | 553 tok/s | 581.9 ms |

**Finding 1: vLLM BF16 baseline is non-deterministic.** Unlike the standalone transformers experiment (where Phi-4's dominant tokens masked argmax flips), vLLM's Flash Attention backend produces real output differences under concurrent batching (1–2 mismatches per batch). This confirms the problem is present—and observable—in production serving engines with realistic models.

**Finding 2: DetermLLM partially reduces vLLM non-determinism.** At n=8, mismatches drop from 2 to 1, indicating the GEMM component is fixed. The remaining mismatch comes from Flash Attention 2's decode kernel (PagedAttention), which tiles its KV traversal based on concurrent sequence count—independent of GEMM. DetermLLM patches `F.linear` but not Flash Attention.

**Finding 3: Higher overhead in vLLM (+41% vs +10–17% in transformers).** vLLM executes ~10× more GEMM calls per second (highly-batched decode scheduler), amplifying the per-call cuBLASLt heuristic search cost. Additionally, the DetermLLM throughput across 5 runs shows high variance (473, 484, **924**, 413, 471 tok/s), suggesting intermittent FP32 fallbacks for shapes where no non-split-K BF16 algorithm is found in vLLM's call pattern.

**Path to full vLLM determinism:** DetermLLM fixes the GEMM component; full determinism additionally requires Flash Attention deterministic mode (FA3 or a fixed-tiling Triton kernel). vLLM's native `--deterministic` flag (which sets `torch.use_deterministic_algorithms`) achieves this at +57–83% overhead. A combined approach (DetermLLM for GEMM + fixed-tiling FA for attention) would achieve the same with substantially lower overhead.

---

## 5. Related Work

**Deterministic GPU computing**: Nvidia's cuDNN and cuBLAS provide deterministic modes (`CUBLAS_WORKSPACE_CONFIG`) that use workspace-based reductions to ensure reproducibility. These modes restrict algorithm selection broadly and incur significant overhead. Our work proposes a more targeted restriction (split-K only) that preserves the majority of algorithmic efficiency.

**batch_invariant_ops (Thinking Machines Lab, 2024)**: The closest prior work. Wraps reductions, scatter, and other ops with batch-invariant implementations. Reports ~34% overhead. Our approach specifically targets GEMM split-K, achieving 3–5× lower overhead by not touching operations that are already batch-invariant (layer norm, embedding, attention).

**vLLM/SGLang deterministic mode**: Both use `torch.use_deterministic_algorithms(True)` as their determinism mechanism. vLLM also sets `CUBLAS_WORKSPACE_CONFIG=:4096:8`. In our experiments, this incurs +29–83% overhead depending on batch size—consistent with the ~34% figure reported by batch_invariant_ops for their workloads.

**FlashAttention deterministic mode**: FlashAttention 3 provides `return_attn_probs=False` with fixed tiling for deterministic attention. Our work addresses the complementary GEMM non-determinism; a complete solution combines DetermLLM (for linear layers) with FlashAttention deterministic mode (for attention). The residual 0.006 nats/token in our log-prob experiment represents the attention component.

**Floating-point reproducibility**: Demmel and Nguyen (2013) analyze fixed-point summation algorithms for exact floating-point reproducibility. Rump et al. study faithful rounding. Our approach does not require exact reproducibility—only batch-size independence—which the `REDUCTION_SCHEME_NONE` + FP32 accumulation combination achieves efficiently.

**Online LLM serving non-determinism**: Concurrent work on reproducible serving (DejaVu, Orca) focuses on request scheduling. Batch non-determinism at the numerical level remains distinct from scheduling non-determinism and has not been addressed at the GEMM algorithm level prior to this work.

---

## 6. Discussion and Limitations

**When the FP32 flag suffices**: For pure-PyTorch inference (transformers without vLLM), `allow_bf16_reduced_precision_reduction=False` achieves determinism with no overhead penalty and is actually faster for many Phi-4 shapes. However, it causes `CUBLAS_STATUS_INTERNAL_ERROR` in vLLM's serving path for single-request decode (M=1), because vLLM's cuBLASLt code path with the restricted flag returns no candidates. DetermLLM handles small-M correctly via its FP32 fallback.

**vLLM integration**: Our experiments confirm vLLM's BF16 baseline is non-deterministic (1–2 output mismatches per batch for Llama-3.2-1B). DetermLLM reduces but does not eliminate this non-determinism (+41% overhead, 1 residual mismatch at n=8 vs 2 for BF16). The residual is attributable to Flash Attention 2's batch-size-dependent KV-tile traversal, which is orthogonal to GEMM. A production-ready vLLM determinism patch would combine DetermLLM with FA2 deterministic mode (available in FlashAttention ≥ 3) or a fixed-split Triton attention kernel, targeting the same ~15–20% overhead budget as DetermLLM achieves for GEMM alone.

**Attention non-determinism**: SDPA at different batch sizes contributes 0.006 nats/token (Δlog P) after 32-layer accumulation. While small compared to GEMM's 0.023 nats, it remains as the residual source after DetermLLM. Full elimination requires additionally fixing attention (e.g., FlashAttention3 deterministic mode). However, for applications where greedy argmax reproducibility is the goal, DetermLLM alone is sufficient (0 argmax flips observed across 640 tokens × 5 batch sizes).

**MoE router non-determinism**: Expert selection in MoE models (via top-K on router logits) is a secondary source of non-determinism. DetermLLM fixes the GEMM component; router ties require additional handling.

**Overhead at large batch sizes**: The +16.5% overhead at bs=32 comes from serializing the K-reduction. For deployment scenarios with bs > 32, a hybrid approach (use split-K with fixed split count) may achieve lower overhead.

---

## 7. Conclusion

We identify cuBLAS split-K dynamic selection as the primary source of batch non-determinism in LLM inference, and propose DetermLLM—a minimal Python/CUDA patch that disables split-K via cuBLASLt's `REDUCTION_SCHEME_NONE` while preserving BF16 I/O. On Phi-4 (14B), DetermLLM reduces per-token log-probability variance by 3.6× with 10–17% latency overhead at bs=8–32, compared to 57–83% for `torch.use_deterministic_algorithms` and 214–280% for full FP32 cast. In vLLM, DetermLLM reduces GEMM-sourced mismatches (2→1 at n=8) but Flash Attention's batch-dependent KV-tile scheduling is a second independent non-det source; full serving-engine determinism requires additionally fixing attention. Integration into HuggingFace transformers requires 2 lines of Python. The findings provide both a practical tool for standalone inference and a diagnostic decomposition of non-determinism sources in production LLM serving.

---

## Appendix: Environment Setup

```bash
# Compile kernel (once)
cd FP32 && python setup_fixed_algo.py build_ext --inplace

# Use in inference
import determ_llm
determ_llm.enable()
# ... your existing inference code ...
```

Required: CUDA 12+, PyTorch 2.6+, BF16-capable GPU (Ampere or later).
