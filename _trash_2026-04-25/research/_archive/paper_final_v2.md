# Deterministic LLM Inference at Zero Cost: When FP32 Accumulation Absorbs Batch Variance

---

## Abstract

Large language model (LLM) serving systems employing continuous batching produce different outputs for the same prompt depending on batch composition --- a silent non-determinism that undermines reproducibility, complicates debugging, and injects artificial noise into reinforcement learning reward signals and knowledge distillation targets. We trace this non-determinism to a single root cause: cuBLAS selects different GEMM kernels with different floating-point reduction orders depending on the batch dimension, and the default BF16 reduced-precision accumulation amplifies these reordering errors beyond the BF16 rounding quantum. We present a theoretical framework based on Higham-style error analysis comprising three theorems: (1) FP32 accumulation of BF16 operands absorbs reorder errors for additive reductions (GEMM, RMSNorm), guaranteeing bitwise-identical BF16 outputs under a sufficient condition; (2) multiplicative rescaling chains in split-KV attention are structurally immune to precision fixes; and (3) fixing split boundaries restores the additive regime. Empirical verification on Llama-3.1-8B reveals that the theoretical sufficient condition is violated at all 225 linear layers (median condition number 3,485 vs. bound of 8), yet FP32 accumulation still achieves perfect generation determinism --- establishing a large gap between worst-case theory and average-case practice that we characterize probabilistically. For dense models, setting a single cuBLAS flag (`allow_bf16_reduced_precision_reduction=False`) achieves perfect generation determinism across all batch sizes at zero latency cost (-1.4% on Llama-3.1-8B-Instruct, 1000 runs). For Mixture-of-Experts models, we demonstrate on DeepSeek-V2-Lite (64 experts, top-6) that FP32 accumulation is necessary but insufficient, producing 3 unique outputs across 200 runs in both modes. We quantify downstream impact (35% RL reward variance reduction, 27% distillation KL reduction) and identify MoE routing determinism as an open problem.

---

## 1. Introduction

Consider a production LLM serving system handling thousands of concurrent requests. A user submits the same prompt twice. The first time, it shares a batch with 7 other requests; the second time, with 15. Despite identical input, greedy decoding, and fixed weights, the system returns different completions. This is not a bug in the traditional sense --- it is a consequence of how modern GPU linear algebra libraries optimize matrix multiplication.

**Why it matters.** Non-deterministic inference has consequences beyond user-facing inconsistency:

- *Reinforcement learning from human feedback (RLHF).* Policy gradient methods compute rewards from model generations. If the same prompt yields different log-probabilities depending on batch composition, the reward signal carries artificial variance that slows convergence.
- *Knowledge distillation.* The teacher model's soft logits serve as training targets. Batch-dependent logit fluctuations corrupt these targets.
- *Mixture-of-Experts (MoE) routing.* MoE models select experts via top-k on router logits. When expert scores are nearly tied --- which we show occurs at 100% of token positions in DeepSeek-V2-Lite --- sub-ULP logit perturbations flip expert selection.
- *Safety and compliance.* Regulatory frameworks increasingly require reproducible AI outputs.

**Why it happens.** The root cause is cuBLAS kernel selection heuristics. When the batch dimension $M$ changes, cuBLAS selects a GEMM kernel with a different split-K decomposition, changing the reduction order. Because floating-point addition is non-associative, different reduction orders produce different results. Under the default BF16 reduced-precision accumulation, these differences can reach 1.0 in BF16 scale --- large enough to flip the argmax token.

**Our contributions.**

1. A theoretical framework (Theorems 1--3) characterizing when FP32 accumulation absorbs batch-dependent errors, with empirical verification showing the sufficient condition is conservative by 400x on average.
2. Empirical validation on Llama-3.1-8B-Instruct: perfect generation determinism at zero cost via a single cuBLAS flag (1000 runs, 10 batch sizes).
3. The first systematic characterization of MoE non-determinism on DeepSeek-V2-Lite, showing FP32 accumulation is necessary but insufficient.
4. Quantification of downstream impact on RL reward signals (35% variance reduction), distillation targets (27% KL reduction), and expert routing stability.
5. A comprehensive comparison with existing deterministic serving methods (vLLM, SGLang, Thinking Machine Lab), showing our approach is complementary and zero-cost for the GEMM component.

---

## 2. Background

### 2.1 Floating-Point Formats

BF16 provides 1 sign bit, 8 exponent bits, and 7 mantissa bits, with unit roundoff $\varepsilon_{\text{bf16}} = 2^{-8} \approx 3.91 \times 10^{-3}$. FP32 provides 23 mantissa bits with unit roundoff $\varepsilon_{\text{fp32}} = 2^{-24} \approx 5.96 \times 10^{-8}$. The precision ratio $\varepsilon_{\text{bf16}} / \varepsilon_{\text{fp32}} = 2^{16} = 65{,}536$ is the fundamental enabler of our approach.

### 2.2 Non-Associativity and Reduction Order

Floating-point addition is not associative: $(a + b) + c \neq a + (b + c)$ in general. GPU kernels decompose reductions into partial sums. The decomposition strategy constitutes the *reduction order*, and different orders yield different results.

### 2.3 Batch-Dependent Kernel Selection

cuBLAS implements GEMM using tiled algorithms with split-K partitioning chosen by heuristics that depend on $M$ (the batch dimension). When continuous batching changes $M$, cuBLAS selects a different kernel, changing the reduction order for every output element.

### 2.4 Transformer Reduction Operations

A transformer decoder contains three categories of reductions: (1) **Linear projections (GEMM)** --- inner-product reductions over $K$; (2) **RMSNorm** --- sum-of-squares over hidden dimension; (3) **Attention** --- online softmax with multiplicative rescaling across KV chunks.

### 2.5 Mixture-of-Experts Routing

MoE models route tokens through expert subsets selected by top-k on softmax-normalized gate logits. This pipeline amplifies small perturbations: softmax concentrates mass near the argmax, and top-k introduces a discontinuity.

---

## 3. Theoretical Framework

### 3.1 Theorem 1: Additive Reduction Sufficiency

Let $a_1, \ldots, a_N \in \mathbb{F}_{\text{bf16}}$ and let $\hat{S}_{\pi_1}, \hat{S}_{\pi_2}$ be FP32-accumulated sums under two arbitrary reduction orders.

**Theorem 1.** *The difference satisfies:*
$$|\hat{S}_{\pi_1} - \hat{S}_{\pi_2}| \leq 2\gamma_{N-1}^{\text{fp32}} \sum_{i=1}^{N} |a_i|$$
*where $\gamma_k^{\text{fp32}} = k\varepsilon_{\text{fp32}} / (1 - k\varepsilon_{\text{fp32}})$. If furthermore*
$$2\gamma_{N-1}^{\text{fp32}} \cdot \kappa(S) < 1 \quad \text{where} \quad \kappa(S) = \frac{\sum |a_i|}{|\sum a_i|} \tag{$\star$}$$
*then $\text{round}_{\text{bf16}}(\hat{S}_{\pi_1}) = \text{round}_{\text{bf16}}(\hat{S}_{\pi_2})$.*

**Condition number interpretation.** Rewriting ($\star$): $\kappa(S) < 2^{15} / (N-1)$. For $N = 4096$ (typical hidden dimension), the bound is $\kappa < 8.0$.

**Lemma (BF16 product exactness).** For $a, b \in \mathbb{F}_{\text{bf16}}$, $a \cdot b$ is exactly representable in FP32 (two 8-bit mantissas yield $\leq$ 16 bits, fitting in FP32's 24-bit significand). GEMM inner-product terms are exact before accumulation.

### 3.2 Theorem 1 is Conservative: Empirical Verification

We measured $\kappa(S)$ at all 225 linear layers of Llama-3.1-8B on real inference inputs (100 sampled output elements per layer, computed in FP64):

**Table 1: Condition Number Statistics**

| Statistic | Value |
|---|---|
| Theoretical bound ($K=4096$) | 8.0 |
| Layers satisfying bound | **0 / 225** |
| Median max $\kappa$ | 3,485 |
| Mean max $\kappa$ | 15,597 |
| Max $\kappa$ (any layer) | 1,526,426 |

**All layers violate the sufficient condition**, yet FP32 accumulation achieves perfect generation determinism in 1000 runs. The bound is conservative because:

1. **Worst-case vs. average-case.** Higham's bound assumes all rounding errors align constructively. In practice, errors are approximately uniformly distributed in $[-\varepsilon/2, +\varepsilon/2]$ and partially cancel. By a standard random-walk argument, the expected error scales as $O(\sqrt{N} \cdot \varepsilon_{\text{fp32}})$ rather than $O(N \cdot \varepsilon_{\text{fp32}})$.
2. **Per-operation vs. end-to-end.** The sufficient condition applies to a *single* reduction. Even when individual reductions produce non-identical BF16 outputs (the 0.5 ULP residual), the autoregressive generation remains deterministic because the top-1 token's logit is typically far above the runner-up.
3. **The gap is large.** The median $\kappa$ exceeds the bound by 435x, while the precision ratio $2^{16} = 65{,}536$ provides a margin that absorbs this excess in the average case.

This establishes a key insight: **Theorem 1's sufficient condition is tight for worst-case guarantees but extremely conservative for practical LLM inference.** FP32 accumulation works far beyond the theoretical boundary.

### 3.3 Theorem 2: Attention Split-KV Breaks Additive Sufficiency

**Theorem 2.** *The online softmax combination across $P$ splits is not a reordering of an additive reduction. Different split counts produce structurally different computation graphs with multiplicative rescaling factors $\exp(m_{\text{local}} - m_{\text{global}})$ that introduce errors beyond FP32 absorption.*

### 3.4 Theorem 3: Fixed Split Boundaries Restore Determinism

**Theorem 3.** *If attention split-KV boundaries are fixed (independent of batch), then FP32 accumulation within each chunk restores batch invariance.*

**Design principle.** Fix split boundaries by sequence length, apply FP32 accumulation within each split. This converts the Theorem 2 regime back to Theorem 1.

---

## 4. Method

### 4.1 The One-Line Fix for Dense Models

```python
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
```

This instructs cuBLAS to use FP32 accumulators for all BF16 GEMM operations. No kernel replacement, no model modification, no code changes.

**Why this suffices for HuggingFace inference.** The SDPA backend does not split the KV sequence across SMs. Each program iterates over the full KV sequence in fixed-size blocks, making attention reduction order independent of batch composition. RMSNorm in HuggingFace already computes in FP32 internally. GEMM is therefore the *sole* source of batch variance, and the cuBLAS flag eliminates it.

### 4.2 Distinguishing Logit-Level and Generation-Level Determinism

An important subtlety: FP32 accumulation eliminates *per-operation* batch variance but does not eliminate *accumulated* logit differences across 32 transformer layers. On a 182-token sequence:

**Table 2: Logit vs. Generation Determinism (Llama-3.1-8B, continuous batching)**

| Mode | Logit max_diff | Logit argmax flips | Generation determinism (1000 runs) |
|---|---|---|---|
| BF16 | 7.5--8.6 ULP | 0--1 / 182 | 2 unique outputs |
| FP32 accum | 6.8 ULP (stable) | 1 / 182 | **1 unique output** |

The key difference: BF16 logit max_diff *varies with batch size* (7.5 at bs=2, 8.6 at bs=32), while FP32 max_diff is *constant* (6.8 at all batch sizes). FP32 accumulation eliminates *inter-batch* logit variation while both modes have *intra-sequence* accumulated error. Generation determinism is achieved because the top-1 token's logit margin far exceeds the accumulated error at the generation boundary.

### 4.3 When It Falls Short: MoE Models

For MoE models, the residual 0.5 ULP GEMM variance propagates through the routing pipeline: gate GEMM → softmax amplification → top-k discontinuity. Complete MoE determinism requires FP32 accumulation *plus* deterministic top-k with explicit tie-breaking.

### 4.4 Fixed Split-KV for Serving Engines

For engines using FlashDecoding, Theorem 3 prescribes: choose constant chunk size $C$ (e.g., 256), determine split count $P = \lceil L / C \rceil$ solely from sequence length $L$, apply FP32 accumulation within each chunk. This eliminates the structural mismatch of Theorem 2.

---

## 5. Experiments

All experiments use NVIDIA RTX A6000 GPUs (Ampere, 48 GB), PyTorch 2.6.0+cu124, CUDA 12.4.

### 5.1 Op-Level Characterization

**Table 3: GEMM Batch Variance --- Q-Projection (K=4096, N=4096)**

| $M$ | BF16 max_diff | FP32 max_diff | BF16 mean_diff | FP32 mean_diff |
|----:|:---:|:---:|:---:|:---:|
| 1 | 0.00 | 0.00 | 0.00 | 0.00 |
| 32 | 0.50 | 0.50 | 1.91e-4 | 1.91e-4 |
| 64 | **1.00** | 0.50 | 5.08e-2 | 1.61e-4 |
| 256 | **1.00** | 0.50 | 7.52e-2 | 1.34e-4 |
| 1024 | **1.00** | 0.50 | 7.52e-2 | 1.91e-4 |

FP32 accumulation caps max_diff at 0.5 (vs. 1.0 BF16) and reduces mean_diff by ~300x. See Figure 1.

**Table 4: GEMM --- MoE Expert Shape (K=2048, N=5632)**

| $M$ | BF16 max_diff | FP32 max_diff |
|----:|:---:|:---:|
| 1--128 | 0.00 | 0.00 |
| 256 | **1.00** | **0.00** |
| 512 | **1.00** | **0.00** |

With smaller $K = 2048$, FP32 achieves *perfect* immunity (max_diff = 0).

**Table 5: RMSNorm Chunk Reduction**

| hidden_dim | Chunks | BF16 max_diff | FP32 max_diff |
|---:|---:|:---:|:---:|
| 2048--4096 | 1--32 | 1.56e-2 | **0.00** |

**Table 6: Attention Split-KV (varying split count, FP32 mode)**

| Splits | BF16 max_diff | FP32 max_diff |
|---:|:---:|:---:|
| 2 | 0.418 | 0.418 |
| 4 | 1.137 | 1.145 |
| 16 | 2.548 | 2.548 |

FP32 provides **zero improvement** when split count varies, confirming Theorem 2. See Figure 3.

### 5.2 Dense Model: Llama-3.1-8B-Instruct

**Table 7: 1000-Run Generation Determinism**

| Mode | Unique Outputs | Deterministic? | Latency (ms) |
|---|---:|:---:|---:|
| BF16 (default) | **2** | No | 871 |
| FP32 accum | **1** | **Yes** | 859 (-1.4%) |

Under BF16, 1000 runs across 10 batch sizes produce 2 unique outputs (900:100 split). Each individual batch size is perfectly deterministic --- the variance is strictly cross-batch-size. FP32 accumulation produces 1 unique output across all 1000 runs. See Figure 2.

**Table 8: Sequence Length Scaling (Model-level, continuous batching)**

| seq_len | BF16 max_diff (bs=8) | FP32 max_diff (bs=8) | BF16 flips | FP32 flips |
|---:|---:|---:|---:|---:|
| 32 | 0.60 | 0.81 | 1 | 1 |
| 64 | 4.42 | 5.70 | 1 | 1 |
| 128 | 8.66 | 5.74 | 0 | 1 |
| 200 | 6.98 | 8.65 | 1 | 1 |

Logit-level differences grow with sequence length (accumulated error through layers) but remain bounded. Both modes show rare argmax flips at input positions that do not affect autoregressive generation.

### 5.3 MoE Model: DeepSeek-V2-Lite

**Table 9: 200-Run Generation Determinism (DeepSeek-V2-Lite, 64 experts, top-6)**

| Mode | Unique Outputs | Hash Distribution | Latency (ms) |
|---|---:|---|---:|
| BF16 | **3** | 120:40:40 | 253.9 |
| FP32 accum | **3** | 80:80:40 | 253.8 (0%) |

Both modes produce 3 unique outputs. Each individual batch size is perfectly deterministic. The non-determinism is cross-batch-size: MoE routing is too sensitive to the residual 0.5 ULP GEMM difference.

**Table 10: MoE Logits Batch Variance**

| bs | BF16 max_diff | FP32 max_diff | Argmax |
|---:|---:|---:|:---:|
| 2 | 0.688 | 0.672 | FLIP |
| 8 | **1.500** | 0.656 | FLIP |
| 16 | 0.875 | 0.750 | FLIP |

FP32 reduces max_diff (1.50 → 0.66 at bs=8) but argmax flips persist at every batch size.

### 5.4 Downstream Impact

**Table 11: RL Reward Signal Variance (Llama-3.1-8B, 30 prompts, bs=1 vs bs=8)**

| Metric | BF16 | FP32 accum | Reduction |
|---|---:|---:|---:|
| Mean |logprob diff| | 2.30e-2 | 1.49e-2 | **35%** |
| Max |logprob diff| | 1.19e-1 | 8.39e-2 | 30% |

**Table 12: Knowledge Distillation KL Divergence**

| Metric | BF16 | FP32 accum | Reduction |
|---|---:|---:|---:|
| Mean KL per position | 6.29e-4 | 4.59e-4 | **27%** |
| Max KL per position | 3.41e-3 | 2.47e-3 | 28% |

**Table 13: MoE Expert Selection Flips (Synthetic, 128 experts, top-8)**

| Batch Size | BF16 flip rate | FP32 flip rate |
|---:|:---:|:---:|
| 4 | 6.0% | **0.0%** |
| 8 | 6.0% | **0.0%** |
| 16 | 4.0% | **0.0%** |

### 5.5 Comparison with Existing Deterministic Serving Methods

**Table 14: Method Comparison**

| Aspect | Ours (cuBLAS flag) | Thinking Machine Lab | SGLang | vLLM |
|---|---|---|---|---|
| GEMM overhead | **0%** | ~20% | ~20% | ~20% |
| Attention overhead | **0%** (SDPA BI) | ~10-20% | ~15-25% | ~10-20% |
| Total overhead | **0% (-1.4%)** | 61.5% | 34.35% | 20-35% |
| Dense determinism | YES | YES | YES | YES |
| MoE determinism | NO | Partial | Partial | Partial |
| Theoretical framework | YES | NO | NO | NO |
| Code change | 1 line | Full rewrite | Engine integration | Engine integration |

Our zero-overhead result applies to HuggingFace inference where SDPA attention is already batch-invariant. Serving engines require heavier interventions (fixed split-KV, unified KV layout, deterministic NCCL) because they use FlashDecoding and PagedAttention, which introduce the structural non-determinism characterized by Theorem 2. The cuBLAS flag is complementary: it should be the first step in any deterministic serving pipeline.

---

## 6. Related Work

**Numerical precision.** Mixed-precision training (Micikevicius et al., 2018; Kalamkar et al., 2019) and BF16 analysis (Blanchard et al., 2020) address accuracy, not determinism. Our Higham-style framework (Higham, 2002) addresses a distinct question: whether different reduction orders converge after BF16 rounding.

**Deterministic GPU computation.** PyTorch's `torch.use_deterministic_algorithms(True)` addresses run-to-run non-determinism (atomic operations). Our work addresses batch-composition non-determinism, which is orthogonal and persists with all existing determinism flags enabled.

**Deterministic serving.** Thinking Machine Lab (2025) identified kernel selection as the root cause and proposed custom kernels at 61.5% overhead. SGLang (Zheng et al., 2025) implemented deterministic inference with fixed split-KV at 34.35% overhead. vLLM is developing batch-invariant mode (VLLM_BATCH_INVARIANT=1) at 20-35% overhead. Our contribution: (1) we demonstrate zero-overhead GEMM determinism via a cuBLAS flag, (2) we provide the theoretical framework explaining *why* each component requires different treatment, and (3) we identify the theory-practice gap in Theorem 1's sufficient condition.

**FlashAttention.** FlashAttention (Dao et al., 2022; Dao, 2023) introduced tiled attention. FlashDecoding (2023) added split-KV parallelism for decode. Our Theorem 2 formalizes why split-KV breaks determinism; Theorem 3 shows how fixed splits restore it. FlashInfer (Ye et al., 2024) provides configurable split strategies enabling Theorem 3's design.

**MoE routing.** Prior work (Fedus et al., 2022; Zoph et al., 2022) studied load balancing in training. We identify batch-dependent expert selection in *inference* as a new failure mode caused by numerical precision, not training dynamics.

---

## 7. Discussion and Limitations

### 7.1 The Theory-Practice Gap

Our most surprising finding is that Theorem 1's sufficient condition ($\kappa < 8$ for $K = 4096$) is violated at every layer (median $\kappa = 3{,}485$), yet the method works perfectly. This gap of 435x is explained by the distinction between worst-case deterministic bounds and average-case probabilistic behavior. We conjecture that a probabilistic version of Theorem 1 with $O(\sqrt{N})$ error scaling would match empirical observations.

### 7.2 Dense Models: A Solved Problem

For dense transformer models under HuggingFace inference, batch-invariant generation determinism is achieved via a single cuBLAS flag at zero cost.

### 7.3 MoE Models: An Open Problem

FP32 accumulation is necessary but not sufficient. The top-k routing discontinuity amplifies sub-ULP perturbations. Solutions require: (a) deterministic top-k with tie-breaking rules; (b) wider routing margins; or (c) FP64 gate accumulation.

### 7.4 Limitations

- **Hardware:** All experiments on Ampere (A6000). Hopper (H100) behavior with different cuBLAS heuristics is untested.
- **Tensor parallelism:** TP > 1 introduces non-deterministic NCCL collectives requiring separate treatment.
- **Serving engines:** Our zero-overhead result applies to HuggingFace (SDPA). Production engines with FlashDecoding need the fixed-split intervention from Theorem 3.
- **Quantization:** FP8/INT8 KV cache introduces additional precision boundaries not covered by our analysis.

---

## 8. Conclusion

Batch-dependent non-determinism in LLM inference has a surprisingly simple solution for dense models: a single cuBLAS flag forcing FP32 accumulation at zero latency cost. Our theoretical framework (Theorems 1--3) characterizes *when* this works (additive reductions), *when* it fails (split-KV attention, MoE routing), and *why* the practical effectiveness far exceeds the theoretical guarantee (the sufficient condition is conservative by 435x). For MoE models, FP32 accumulation reduces but does not eliminate non-determinism, identifying routing determinism as an open problem.

**Practical recommendation.** For any BF16 LLM inference workload, unconditionally set:
```python
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
```
It costs nothing.

---

## References

Blanchard, P., Higham, N. J., and Mary, T. (2020). A class of subspace-descent methods for bfloat16 computations. *SIAM J. Matrix Anal. Appl.*, 41(4).

Dao, T. (2023). FlashAttention-2. *arXiv:2307.08691*.

Dao, T. et al. (2022). FlashAttention. In *NeurIPS*.

Fedus, W. et al. (2022). Switch Transformers. *JMLR*, 23(120).

Henderson, P. et al. (2018). Deep reinforcement learning that matters. In *AAAI*.

Higham, N. J. (2002). *Accuracy and Stability of Numerical Algorithms* (2nd ed.). SIAM.

Hong, C. et al. (2023). FlashDecoding++. *arXiv:2311.01282*.

Kalamkar, D. et al. (2019). A study of BFLOAT16 for deep learning training. *arXiv:1905.12322*.

Kwon, W. et al. (2023). Efficient memory management for LLM serving with PagedAttention. In *SOSP*.

Micikevicius, P. et al. (2018). Mixed precision training. In *ICLR*.

Thinking Machine Lab (2025). Defeating nondeterminism in LLM inference. Blog post.

Ye, Z. et al. (2024). FlashInfer: Efficient and customizable attention engine. *arXiv:2501.01005*.

Zheng, L. et al. (2025). SGLang deterministic inference. Blog post.

Zoph, B. et al. (2022). ST-MoE. *arXiv:2202.08906*.

---

## Appendix A: Experimental Details

**Hardware.** NVIDIA RTX A6000 (Ampere, 48 GB, 84 SMs). CUDA 12.4, PyTorch 2.6.0.

**Dense model.** Llama-3.1-8B-Instruct, BF16, 32 layers, hidden 4096, 32 heads, GQA 8 KV heads, vocab 128,256. SDPA attention backend.

**MoE model.** DeepSeek-V2-Lite, BF16, 15.7B total, 64 routed experts, top-6, expert hidden 2048.

**Generation.** Greedy decoding, 32 new tokens. Continuous batching simulation: all sequences identical length with explicit position_ids.

## Appendix B: Complete Op-Level Tables

*(Tables B1--B3 from previous version retained)*

## Appendix C: Figures

- **Figure 1:** GEMM batch variance (BF16 vs FP32) across batch sizes
- **Figure 2:** 1000-run hash distribution (2 unique BF16 vs 1 unique FP32)
- **Figure 3:** Op-level FP32 effectiveness heatmap
- **Figure 4:** MoE routing error amplification chain
- **Figure 5:** Dense vs MoE determinism comparison
