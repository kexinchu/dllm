# NeurIPS Paper Outline: Deterministic LLM Inference via FP32 Accumulation

## Title Candidates

- "Where Does Non-Determinism Hide? Dissecting and Fixing Batch-Variant LLM Inference"
- "FP32 Accumulation Is Not Enough: A Systematic Analysis of Non-Determinism in LLM Serving"
- "Toward Deterministic LLM Inference: Precision, Parallelism, and the Limits of FP32 Accumulation"

---

## 1. Introduction (1.5 pages)

**Hook**: LLM inference non-determinism is a growing concern for reproducibility,
safety auditing, RL-based post-training, and distillation pipelines.
Same prompt + same model can produce different outputs depending on
*which other requests happen to be in the same batch*.

**Problem statement**: Batch invariance — the requirement that a sample's
numerical trajectory is identical regardless of batch composition, batch size,
or position within the batch.

**Key insight (thesis)**: Non-determinism in LLM inference arises from *two distinct mechanisms*:
1. **Accumulation precision** — reduction order changes + low-precision (BF16) accumulation
   → different rounding → different outputs (affects GEMM, RMSNorm, Softmax)
2. **Computation graph divergence** — different parallelization strategies create
   *mathematically different computation graphs* (affects Attention split-KV)

FP32 accumulation solves (1) but NOT (2). A complete solution requires both
precision control AND structural invariance.

**Contributions**:
- Systematic taxonomy of non-determinism sources across all ops in transformer inference
- Quantitative analysis: which ops are affected, by how much, and which approach fixes each
- Discovery that cuBLAS's `allow_bf16_reduced_precision_reduction=False` achieves zero-overhead FP32 GEMM accumulation
- Proof that FP32 accumulation alone cannot solve attention split-KV non-determinism
- Complete "fixed-split + FP32 accum" recipe with measured overhead of ~6%

---

## 2. Background (1 page)

### 2.1 Floating-Point Non-Associativity
- IEEE 754, BF16 (7-bit mantissa) vs FP32 (23-bit)
- Example: 102 unique results from summing 8 numbers in different orders

### 2.2 GPU Kernel Selection and Dynamic Parallelism
- Split-K in GEMM
- Split-KV / FlashDecoding in attention
- Dynamic kernel selection based on problem shape

### 2.3 Continuous Batching in Serving Engines
- vLLM, SGLang continuous batching
- How batch composition changes kernel behavior

---

## 3. Taxonomy of Non-Determinism Sources (2 pages)

### 3.1 Per-Op Analysis

| Op | Reduction Dimension | Mechanism | Severity |
|---|---|---|---|
| GEMM | K (inner product) | Split-K changes # chunks | High |
| RMSNorm | hidden_dim (variance) | SM count changes chunk strategy | Medium |
| Softmax | seq/vocab dim (sum of exp) | Chunk-based reduction | Medium |
| Attention | head_dim (QK^T) + seq_len (attn@V) | Split-KV changes reduction tree | **Critical** |
| Embedding, RoPE, bias, residual | None | Per-element ops | None |

### 3.2 Two Classes of Non-Determinism

**Class 1: Accumulation Order** (GEMM, RMSNorm, Softmax)
- Same mathematical formula, different evaluation order
- FP32 accumulation can absorb differences

**Class 2: Computation Graph Divergence** (Attention split-KV)
- Different # of splits → different online softmax correction chains
- `exp(local_max - global_max)` rescaling creates fundamentally different computation
- FP32 cannot help because the *operations themselves* differ

### 3.3 The Near-Tie Amplification Chain (MoE)
- Small logits perturbation → softmax amplification → expert selection flip
- 39% of tokens in Qwen3-MoE have gap < 0.001

---

## 4. Method: FP32 Accumulation + Fixed Split (1.5 pages)

### 4.1 FP32 Accumulation for Class 1 Ops

**GEMM**: `torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False`
- Forces cuBLAS to use FP32 accumulation internally
- Zero overhead (measured 0.98x of native BF16)
- No kernel replacement needed

**RMSNorm**: Triton kernel with FP32 accumulator for `sum(x^2)`
- Data-parallel over batch (one program per row)
- Zero cross-row reduction → naturally batch-invariant

**Softmax**: FP32 accumulator for `max` and `sum(exp)`
- Same per-row parallelism

### 4.2 Fixed Split-KV for Class 2 (Attention)

**The failure of FP32-only**: Experimental proof that FP32 accum gives 0 improvement
for attention split-KV (Table X).

**Fixed split-KV size**: Constant chunk C (e.g., 256) regardless of batch size.
Combined with FP32 accumulators for the online softmax:
- `m_i` (running max): FP32
- `l_i` (running sum of exp): FP32
- `acc` (output accumulator): FP32

### 4.3 Integration: `fp32_accum_mode` Context Manager
- One-line patch for HuggingFace models
- Selective patching (Linear, RMSNorm, Attention, Softmax independently)

---

## 5. Experiments (3 pages)

### 5.1 Setup

| | Details |
|---|---|
| Models | Llama-3.1-8B-Instruct, Qwen3-30B-A3B (MoE) |
| Hardware | NVIDIA A6000 (48GB), A100 (80GB) |
| Frameworks | HuggingFace (validation), vLLM/SGLang (serving) |
| Metrics | Logits max_diff, argmax match rate, token-level determinism, latency |

### 5.2 Experiment 1: Op-Level Characterization

**Goal**: Quantify non-determinism for each op under different split strategies.

**Method**: For each op (GEMM, RMSNorm, Softmax, Attention), simulate split-K/KV
with splits={1,2,4,8,16,32}. Compare BF16 vs FP32 accumulation.

**Expected results table**:

| Op | BF16 max_diff (worst) | FP32 accum max_diff | Improvement |
|---|---|---|---|
| GEMM split-K | ~4.0 | ~0.5 | 8x |
| RMSNorm chunks | ~3.1e-2 | 0 | ∞ (perfect) |
| Softmax chunks | ~X | ~0 | ∞ |
| Attention split-KV | ~2.5 | ~2.5 | 1x (none) |

### 5.3 Experiment 2: End-to-End Batch Invariance

**Goal**: Measure batch invariance on real models.

**Method**:
- Same prompt at batch_size={1,2,4,8,16}
- Compare logits of target prompt (padding-aware alignment)
- Modes: Pure BF16, Linear-only FP32, Full FP32+fixed-split

**Ablation**: Contribution of each component

| Mode | Logits max_diff | Overhead |
|---|---|---|
| Pure BF16 | X | 1.00x |
| +Linear FP32 accum | Y | ~1.00x |
| +RMSNorm FP32 | Z | ~1.00x |
| +Attention fixed-split | W | ~1.06x |
| All combined | ~0 | ~1.06x |

### 5.4 Experiment 3: Determinism Under Repeated Inference

**Goal**: Same input N times (N=1000+), varying batch sizes. Token-level match rate.

**Method**: Run on both HuggingFace and vLLM/SGLang with continuous batching.

| Setting | Mode | Unique outputs / N runs |
|---|---|---|
| HF, batch=1 | BF16 | 1 (deterministic) |
| HF, varying batch | BF16 | K>1 (non-deterministic) |
| HF, varying batch | Full FP32 accum | 1 (deterministic) |
| vLLM, continuous batching | BF16 | K>1 |
| vLLM, continuous batching | Full FP32 accum | 1 |
| SGLang, continuous batching | BF16 | K>1 |
| SGLang, continuous batching | Full FP32 accum | 1 |

### 5.5 Experiment 4: Performance

**Goal**: Measure overhead of each component.

| Component | Strategy | Overhead (A6000) | Overhead (A100) |
|---|---|---|---|
| GEMM | cuBLAS flag | ~0% | ~0% |
| RMSNorm | Triton FP32 accum | ~0% | ~0% |
| Softmax | Triton FP32 accum | ~0% | ~0% |
| Attention | Triton fixed-split + FP32 | ~6% | ~X% |
| **Total** | | **~6%** | **~X%** |

Compare with existing approaches:
- vLLM `VLLM_BATCH_INVARIANT=1`: ~20-35% overhead
- SGLang `--enable-deterministic-inference`: ~34% overhead
- LayerCast (full FP32 compute): ~3.44x overhead

### 5.6 Experiment 5: Impact on Downstream Tasks

**Goal**: Show that non-determinism actually matters.

**Method**:
- MoE expert selection flip rate with/without FP32 accum
- RL post-training reward variance due to non-deterministic rollouts
- Distillation KL-divergence under different batch compositions

### 5.7 Experiment 6: Near-Tie Prevalence Across Models

| Model | Type | P(gap < 0.001) | P(gap < 0.01) |
|---|---|---|---|
| Llama-3.1-8B | Dense | X% | Y% |
| Qwen3-30B-A3B | MoE (router) | ~39% | Z% |
| Mixtral-8x7B | MoE (router) | A% | B% |
| DeepSeek-V2 | MoE (router) | C% | D% |

---

## 6. Related Work (1 page)

- Floating-point determinism in deep learning (PyTorch reproducibility docs)
- FlashAttention, FlashDecoding, FlashDecoding++
- PagedAttention, continuous batching (vLLM, SGLang)
- Batch-invariant ops (Thinking Machine Lab blog)
- LayerCast approach
- NCCL deterministic communication

---

## 7. Discussion and Limitations (0.5 page)

- FP32 accum is not bit-exact across GPU architectures (Ampere vs Hopper)
- Fixed split-KV reduces decode parallelism
- Does not cover TP/EP communication non-determinism (orthogonal problem)
- Quantized models (INT8/FP8 KV cache) not yet addressed

---

## 8. Conclusion (0.5 page)

FP32 accumulation alone is necessary but insufficient for deterministic LLM inference.
We identify two distinct classes of non-determinism and show that the combination of
(1) FP32 accumulation for reduction ops and (2) fixed-split attention achieves
batch invariance with only ~6% overhead, compared to 20-35% for existing approaches.

---

## TODO: Experiments to Run Before Submission

### Must-have (blocking)
- [ ] Fix RMSNorm Triton kernel padding bug (NaN at bs>=4 with left-padding)
- [ ] Fix Attention Triton kernel correctness (max_diff=15 vs SDPA)
- [ ] End-to-end batch invariance test with ALL patches working
- [ ] vLLM integration test (continuous batching scenario)
- [ ] A100 performance numbers
- [ ] Qwen3-MoE end-to-end (expert selection flip rate)

### Nice-to-have
- [ ] SGLang integration test
- [ ] RL post-training impact experiment
- [ ] More models (Mixtral, DeepSeek-V2)
- [ ] FP8/INT8 KV cache analysis
- [ ] Multi-GPU TP scenario
