# Comparison with Existing Deterministic Serving Methods

## Method Comparison Table

| Aspect | Ours (cuBLAS flag) | Thinking Machine Lab | SGLang Deterministic | vLLM Batch Invariant |
|---|---|---|---|---|
| **Scope** | HuggingFace / research inference | Custom serving engine | SGLang production serving | vLLM production serving |
| **GEMM fix** | cuBLAS flag: `allow_bf16_reduced_precision_reduction=False` | Custom kernel (disable split-K) | `batch_invariant_ops` override | `batch_invariant_ops` override |
| **GEMM overhead** | **0%** | ~20% | ~20% | ~20% |
| **RMSNorm** | Not needed (HF already FP32 internally) | Custom data-parallel kernel | Custom kernel | Custom kernel |
| **Attention** | Not needed (SDPA is already BI) | Fixed split + unified KV | Fixed split-KV (FlashInfer/FA3) | FlexAttention fixed tile config |
| **Attention overhead** | **0%** (SDPA has no split-KV) | ~10-20% decode | ~15-25% decode | ~10-20% decode |
| **KV-cache** | Standard (no paged attention) | Unified logical layout | Radix cache **disabled** | Paged + logical unification |
| **TP communication** | N/A (single/pipeline parallel) | Not addressed | Deterministic NCCL all-reduce | Disabled custom all-reduce |
| **Total overhead** | **0% (-1.4%)** | **61.5%** (initial) | **34.35%** | **20-35%** (est.) |
| **Dense model determinism** | YES (proven) | YES | YES | YES |
| **MoE determinism** | NO (routing flips) | Partial | Partial | Partial |
| **Theoretical framework** | YES (Theorems 1-3) | NO | NO | NO |
| **Code change** | 1 line | Full kernel rewrite | Engine integration | Engine integration |

## Analysis

### Why Our Method Has Zero Overhead

The SDPA backend in HuggingFace does not use FlashDecoding's split-KV parallelism. Each query block iterates over the full KV sequence in fixed-size chunks within a single program instance. This means:
1. Attention reduction order depends only on sequence length and block size, not batch composition (Theorem 2 does not apply)
2. RMSNorm in HuggingFace's LlamaRMSNorm already computes in FP32 internally
3. The only source of batch variance is cuBLAS GEMM kernel selection, which the flag eliminates at zero cost

### Why Serving Engines Need More

Production serving engines use architectural optimizations that introduce additional non-determinism:
1. **FlashDecoding (split-KV)**: Splits KV sequence across SMs for decode throughput. Different batch sizes lead to different split counts and different rescaling chains (Theorem 2)
2. **PagedAttention**: KV cache stored in non-contiguous pages. Different batch compositions lead to different page layouts and block boundaries
3. **Radix/Prefix caching**: Shared prefix cache changes based on active requests, altering the attention computation structure
4. **Tensor Parallelism**: All-reduce operations with potentially non-deterministic NCCL collectives

### The Theoretical Insight

| Category | Operation | Root Cause | Fix | Overhead |
|---|---|---|---|---|
| Additive reduction (Thm 1) | GEMM, RMSNorm, Softmax | FP non-associativity in sum | FP32 accumulator | ~0% |
| Structural mismatch (Thm 2) | Attention split-KV | Different computation graphs | Cannot fix with precision | N/A |
| Fixed structure + additive (Thm 3) | Attention with fixed splits | FP non-associativity within fixed graph | Fixed splits + FP32 accum | ~20-30% |

### Complementarity

Our cuBLAS flag finding is complementary to serving engine work: it should be the first step in any deterministic serving pipeline (zero cost, eliminates GEMM variance). The fixed split-KV designs in serving engines align with our Theorem 3. Our MoE analysis identifies a gap that no current system fully addresses.
