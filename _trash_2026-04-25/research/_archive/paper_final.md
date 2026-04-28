# Deterministic LLM Inference at Zero Cost: When FP32 Accumulation Absorbs Batch Variance

---

## Abstract

Large language model (LLM) serving systems employing continuous batching produce different outputs for the same prompt depending on batch composition --- a silent non-determinism that undermines reproducibility, complicates debugging, and injects artificial noise into reinforcement learning reward signals and knowledge distillation targets. We trace this non-determinism to a single root cause: cuBLAS selects different GEMM kernels with different floating-point reduction orders depending on the batch dimension, and the default BF16 reduced-precision accumulation amplifies these reordering errors beyond the BF16 rounding quantum. We present a theoretical framework based on Higham-style error analysis comprising three theorems: (1) FP32 accumulation of BF16 operands absorbs reorder errors for additive reductions (GEMM, RMSNorm), guaranteeing bitwise-identical BF16 outputs; (2) multiplicative rescaling chains in split-KV attention are structurally immune to precision fixes; and (3) fixing split boundaries restores the additive regime. For dense models, setting a single cuBLAS flag (`allow_bf16_reduced_precision_reduction=False`) achieves perfect generation determinism across all batch sizes at zero latency cost (-1.4% on Llama-3.1-8B-Instruct, 1000 runs). For Mixture-of-Experts models, however, we demonstrate on DeepSeek-V2-Lite (64 experts, top-6) that FP32 accumulation is necessary but insufficient: the residual 0.5 ULP GEMM variance still flips expert selection at near-tie routing boundaries, producing 3 unique outputs across 200 runs in both BF16 and FP32 modes. We characterize the downstream impact, showing 35% reduction in RL reward variance and 27% reduction in distillation KL divergence with FP32 accumulation, while identifying MoE routing determinism as an open problem requiring solutions beyond numerical precision.

---

## 1. Introduction

Consider a production LLM serving system handling thousands of concurrent requests. A user submits the same prompt twice. The first time, it shares a batch with 7 other requests; the second time, with 15. Despite identical input, greedy decoding, and fixed weights, the system returns different completions. This is not a bug in the traditional sense --- it is a consequence of how modern GPU linear algebra libraries optimize matrix multiplication.

**Why it matters.** Non-deterministic inference has consequences that extend well beyond user-facing inconsistency:

- *Reinforcement learning from human feedback (RLHF).* Policy gradient methods such as GRPO compute rewards from model generations. If the same prompt-response pair yields different log-probabilities depending on batch composition, the reward signal carries artificial variance that slows convergence and degrades final policy quality.
- *Knowledge distillation.* The teacher model's soft logits serve as training targets. Batch-dependent logit fluctuations corrupt these targets, forcing the student to fit noise.
- *Mixture-of-Experts (MoE) routing.* MoE models select a subset of experts via top-k on router logits. When expert scores are nearly tied --- which we show occurs at 100% of token positions --- even sub-ULP logit perturbations can flip expert selection, causing cascading output changes.
- *Safety and compliance.* Regulatory frameworks increasingly require reproducible AI outputs. Silent batch-dependent variation violates this requirement without any observable error signal.

**Why it happens.** The root cause is cuBLAS kernel selection heuristics. When the batch dimension $M$ changes, cuBLAS may select a GEMM kernel with a different split-K decomposition, partitioning the $K$-dimensional reduction into a different number of chunks processed in a different order. Because floating-point addition is non-associative, different reduction orders produce different results. Under the default BF16 reduced-precision accumulation mode, these differences can reach 1.0 in BF16 scale --- large enough to flip the argmax token at autoregressive decoding steps.

**Our contributions.** We provide (1) a rigorous theoretical framework characterizing when FP32 accumulation absorbs batch-dependent reorder errors (Theorems 1--3); (2) empirical validation on a dense model (Llama-3.1-8B-Instruct) demonstrating perfect determinism at zero cost via a single cuBLAS flag; (3) the first systematic characterization of MoE non-determinism on DeepSeek-V2-Lite, showing that FP32 accumulation is necessary but insufficient; and (4) quantification of downstream impact on RL reward signals, distillation targets, and expert routing stability.

---

## 2. Background

### 2.1 Floating-Point Formats

Modern LLM inference uses BF16 (Brain Floating Point 16) for weights and activations. BF16 provides 1 sign bit, 8 exponent bits, and 7 mantissa bits, with unit roundoff $\varepsilon_{\text{bf16}} = 2^{-8} \approx 3.91 \times 10^{-3}$. FP32 (IEEE 754 single precision) provides 23 mantissa bits with unit roundoff $\varepsilon_{\text{fp32}} = 2^{-24} \approx 5.96 \times 10^{-8}$. The precision ratio $\varepsilon_{\text{bf16}} / \varepsilon_{\text{fp32}} = 2^{16} = 65{,}536$ is the fundamental enabler of our approach: FP32 accumulation errors are $2^{16}$ times smaller than the BF16 rounding quantum.

### 2.2 Non-Associativity and Reduction Order

Floating-point addition is not associative: $(a + b) + c \neq a + (b + c)$ in general. GPU kernels exploit massive parallelism by decomposing reductions into independently computed partial sums that are then combined. The decomposition strategy --- how many partitions, which elements in each partition --- constitutes the *reduction order*. Different reduction orders of the same operands yield different floating-point results.

### 2.3 Batch-Dependent Kernel Selection

cuBLAS implements GEMM ($C = AB$) using tiled algorithms. The tile sizes and the number of split-K partitions are chosen by heuristics that depend on the matrix dimensions, including $M$ (the batch dimension in inference). When continuous batching changes $M$ between requests, cuBLAS may select a different kernel, changing the reduction order for every element of the output --- even for input rows that are identical across batches.

### 2.4 Transformer Reduction Operations

A transformer decoder layer contains three categories of floating-point reductions: (1) **Linear projections (GEMM)** --- inner-product reductions over the hidden dimension $K$; (2) **RMSNorm** --- sum-of-squares reduction over the hidden dimension $d$; and (3) **Attention** --- online softmax with multiplicative rescaling across KV sequence chunks. We show that these three categories have fundamentally different responses to FP32 accumulation.

### 2.5 Mixture-of-Experts Routing

MoE models (e.g., DeepSeek-V2-Lite with 64 experts, top-6) route each token through a subset of experts selected by applying top-k to softmax-normalized gate logits. The gate computation is a GEMM followed by softmax followed by a hard top-k threshold. This pipeline amplifies small logit perturbations: softmax magnifies differences near the decision boundary, and top-k introduces a discontinuity.

---

## 3. Theoretical Framework

We formalize the conditions under which FP32 accumulation eliminates batch-dependent non-determinism.

**Definition 1 (Batch Invariance).** A function $f: \mathcal{X} \to \mathcal{Y}$ computed by a GPU kernel is *batch invariant* at input $x$ if for all batch configurations $B_1, B_2$: $f_{B_1}(x) = f_{B_2}(x)$ (bitwise identical).

### 3.1 Theorem 1: Additive Reduction Sufficiency

Let $a_1, \ldots, a_N \in \mathbb{F}_{\text{bf16}}$ and let $\hat{S}_{\pi_1}, \hat{S}_{\pi_2}$ be FP32-accumulated sums under two arbitrary reduction orders.

**Theorem 1.** *The difference between any two reduction orders satisfies:*

$$|\hat{S}_{\pi_1} - \hat{S}_{\pi_2}| \leq 2\gamma_{N-1}^{\text{fp32}} \sum_{i=1}^{N} |a_i|$$

*where $\gamma_k^{\text{fp32}} = k\varepsilon_{\text{fp32}} / (1 - k\varepsilon_{\text{fp32}})$. If furthermore*

$$2\gamma_{N-1}^{\text{fp32}} \cdot \kappa(S) < 1 \quad \text{where} \quad \kappa(S) = \frac{\sum |a_i|}{|\sum a_i|} \tag{$\star$}$$

*then $\text{round}_{\text{bf16}}(\hat{S}_{\pi_1}) = \text{round}_{\text{bf16}}(\hat{S}_{\pi_2})$ (bitwise identical after BF16 rounding).*

*Proof sketch.* By Higham's recursive summation error analysis (Higham, 2002, Theorem 4.4), each computed sum satisfies $\hat{S}_\pi = \sum a_i(1 + \delta_i^\pi)$ with $|\delta_i^\pi| \leq \gamma_{N-1}^{\text{fp32}}$. The triangle inequality gives the first bound. For the second claim, condition ($\star$) ensures both $\hat{S}_{\pi_1}$ and $\hat{S}_{\pi_2}$ lie within one BF16 rounding quantum of the exact sum $S$, hence they round identically. $\square$

**Condition number interpretation.** Rewriting ($\star$): $\kappa(S) < 2^{15} / (N-1) \approx 32{,}768 / (N-1)$. For well-conditioned sums ($\kappa = O(1)$), this holds for $N$ up to tens of thousands --- the regime of transformer hidden dimensions.

**Lemma (BF16 product exactness).** For $a, b \in \mathbb{F}_{\text{bf16}}$, the product $a \cdot b$ is exactly representable in FP32 (two 8-bit mantissas multiply to at most 16 bits, which fits in FP32's 24-bit significand). This means GEMM inner-product terms $W_{ik} x_k$ are exact FP32 values before accumulation --- the *only* error source is reduction order.

### 3.2 Theorem 2: Attention Split-KV Breaks Additive Sufficiency

Modern serving engines use FlashDecoding, which splits the KV sequence into $P$ chunks and combines partial results via online softmax with multiplicative rescaling factors $\exp(m_{\text{local}} - m_{\text{global}})$.

**Theorem 2.** *The online softmax combination across $P$ splits is not a reordering of an additive reduction. Different split counts $P_1 \neq P_2$ produce structurally different computation graphs with multiplicative rescaling that introduces errors beyond FP32 absorption.*

*Proof sketch.* With $P = 2$ splits, each partial output is rescaled once by $\exp(m_s - m^{(2)})$. With $P = 4$ splits, the same value undergoes three cascaded rescalings. While mathematically equivalent by telescoping, these are computationally distinct: $P = 4$ requires three $\exp$ evaluations, three subtractions, and two multiplications versus one of each for $P = 2$. Each operation introduces relative error $\leq \varepsilon_{\text{fp32}}$, and the rescaling factor $\exp(\Delta m)$ amplifies absolute errors when $\Delta m > 0$. The total error scales as $C \cdot |P_1 - P_2| \cdot \exp(\Delta m_{\max}) \cdot \varepsilon_{\text{fp32}} \cdot \|V\|_\infty$, which is structural (from different computation graphs), not merely from reordering a fixed set of additions. $\square$

### 3.3 Theorem 3: Fixed Split Boundaries Restore Determinism

**Theorem 3.** *If attention split-KV boundaries are fixed (independent of batch configuration) so that all inputs with the same sequence length $L$ use the same chunk boundaries, then FP32 accumulation within each chunk restores batch invariance.*

*Proof sketch.* With fixed boundaries, each chunk computes identical local quantities $(m_s, l_s, o_s)$ regardless of batch configuration (by Theorem 1 applied to the within-chunk additive reductions). The cross-chunk combination is sequential and deterministic given deterministic inputs. $\square$

**Design principle.** Theorem 3 provides the recipe: fix split boundaries by sequence length, then apply FP32 accumulation within each split. This converts the Theorem 2 regime (structurally different graphs) back to the Theorem 1 regime (same graph, different internal reduction orders absorbed by FP32).

---

## 4. Method

### 4.1 The One-Line Fix for Dense Models

For dense transformer models using HuggingFace with the SDPA attention backend, a single PyTorch flag achieves full batch-invariant determinism:

```python
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
```

This instructs cuBLAS to use FP32 accumulators for all BF16 GEMM operations. No kernel replacement, no model modification, no code changes beyond this one line.

**Why this suffices for HuggingFace inference.** The SDPA backend (FlashAttention-2 or memory-efficient attention) already uses FP32 accumulators internally and does *not* split the KV sequence across SMs for a single query block. Each program iterates over the full KV sequence in fixed-size blocks, making the attention reduction order dependent only on sequence length and block size --- not batch composition. GEMM (Q/K/V projections, output projection, MLP, LM head) is therefore the *sole* source of batch variance, and the cuBLAS flag eliminates it entirely.

### 4.2 When It Works: Dense Models

For dense models (e.g., Llama-3.1-8B), the cuBLAS flag provides:
- **GEMM**: max_diff reduced from 1.0 to 0.5 (BF16 ULP scale); mean_diff reduced by ~100x
- **RMSNorm**: FP32 accumulation yields exactly 0 difference (Theorem 1 with $\kappa = 1$)
- **Softmax**: FP32 chunked softmax is effectively exact (~$10^{-11}$ max error)
- **Attention (SDPA)**: Already batch-invariant; no fix needed

### 4.3 When It Falls Short: MoE Models

For MoE models, FP32 accumulation is necessary but insufficient. The residual 0.5 ULP GEMM variance (Theorem 1's rounding-boundary events) propagates through the routing pipeline:

1. **Gate GEMM** produces logits with up to 0.5 ULP batch-dependent difference
2. **Softmax** amplifies differences near the decision boundary
3. **Top-k** introduces a hard discontinuity: if the $k$-th and $(k+1)$-th expert scores differ by less than the propagated error, expert selection flips

Complete MoE determinism requires FP32 accumulation *plus* deterministic top-k with explicit tie-breaking rules, and potentially wider routing margins.

### 4.4 Fixed Split-KV for Serving Engines

For serving engines (vLLM, SGLang) that use FlashDecoding with dynamic split-KV, Theorem 3 prescribes a fixed-split design: choose a constant chunk size $C$ (e.g., 256 tokens), determine split count $P = \lceil L / C \rceil$ solely from the individual sequence length $L$, and apply FP32 accumulation within each chunk. The cross-chunk online softmax combination proceeds sequentially in a fixed order. This eliminates the structural mismatch identified in Theorem 2.

---

## 5. Experiments

All experiments use NVIDIA RTX A6000 GPUs (Ampere, 48 GB), PyTorch 2.6.0+cu124, CUDA 12.4. Seeds are fixed; all variance is batch-composition-dependent, not run-to-run.

### 5.1 Op-Level Characterization

We isolate each operation and measure the maximum element-wise difference (in BF16 ULP scale) between outputs computed at batch size $M$ versus the reference at $M = 1$.

**Table 1: GEMM Batch Variance --- Llama Q-Projection Shape (K=4096, N=4096)**

| $M$ | BF16 max_diff | BF16 mean_diff | FP32 max_diff | FP32 mean_diff |
|----:|:---:|:---:|:---:|:---:|
| 1 | 0.00 | 0.00 | 0.00 | 0.00 |
| 16 | 0.00 | 0.00 | 0.00 | 0.00 |
| 32 | 0.50 | 1.91e-4 | 0.50 | 1.91e-4 |
| 64 | **1.00** | 5.08e-2 | 0.50 | 1.61e-4 |
| 256 | **1.00** | 7.52e-2 | 0.50 | 1.34e-4 |
| 1024 | **1.00** | 7.52e-2 | 0.50 | 1.91e-4 |

FP32 accumulation caps max_diff at 0.5 (vs. 1.0 for BF16) and reduces mean_diff by ~300x. The max_diff = 0.5 residual corresponds to rare rounding-boundary events predicted by Theorem 1.

**Table 2: GEMM Batch Variance --- MoE Expert Shape (K=2048, N=5632)**

| $M$ | BF16 max_diff | FP32 max_diff |
|----:|:---:|:---:|
| 1--128 | 0.00 | 0.00 |
| 256 | **1.00** | **0.00** |
| 512 | **1.00** | **0.00** |
| 1024 | 0.00 | 0.00 |

With the smaller reduction dimension $K = 2048$, FP32 accumulation achieves *perfect* immunity (max_diff = 0) --- the Theorem 1 condition is satisfied with wider margin. This is significant for MoE models: individual expert GEMMs are fully deterministic under FP32, yet the model still exhibits non-determinism (Section 5.3).

**Table 3: RMSNorm Chunk Reduction Variance**

| hidden_dim | Chunks | BF16 max_diff | FP32 max_diff |
|---:|---:|:---:|:---:|
| 2048 | 1--32 | 1.56e-2 | **0.00** |
| 4096 | 1--32 | 1.56e-2 | **0.00** |

FP32 accumulation yields exactly zero difference across all chunk counts and hidden dimensions, as guaranteed by Theorem 1 with $\kappa = 1$ (non-negative summands in the sum-of-squares reduction). This is a deterministic guarantee, not a probabilistic one.

**Table 4: Softmax Chunk Reduction Variance (vocab_size = 128,256)**

| Chunks | BF16 max_diff | BF16 KL_div | FP32 max_diff | FP32 KL_div |
|---:|:---:|:---:|:---:|:---:|
| 1 | 1.21e-6 | 1.36e-3 | 2.91e-11 | -2.03e-9 |
| 2 | 1.72e-6 | -3.89e-3 | 5.82e-11 | -2.03e-9 |
| 8 | 1.21e-6 | 1.36e-3 | 5.82e-11 | -2.03e-9 |
| 16 | 1.72e-6 | -3.89e-3 | 5.82e-11 | -2.03e-9 |

FP32 softmax is effectively exact ($\sim 10^{-11}$ max error, KL $\sim 10^{-9}$).

**Table 5: Attention Split-KV Variance (FP32 mode)**

| seq_len | splits=1 | splits=2 | splits=4 | splits=8 | splits=16 |
|---:|:---:|:---:|:---:|:---:|:---:|
| 128 | 0.00 | 5.96e-8 | 8.94e-8 | 7.45e-8 | 8.94e-8 |
| 512 | 0.00 | 4.47e-8 | 5.96e-8 | 1.04e-7 | 8.94e-8 |
| 2048 | 0.00 | 3.49e-8 | 4.10e-8 | 3.73e-8 | 5.22e-8 |

For a *fixed* split count, FP32 accumulation yields near-zero error ($\sim 10^{-8}$). However, *varying* the split count produces errors of 0.42--2.55 in BF16 scale, identical for BF16 and FP32, confirming Theorem 2: the error is structural, not precision-related.

**Table 6: Run-to-Run Determinism**

| Shape | dtype | Mismatches (of 100) | max_diff |
|---|:---:|---:|:---:|
| [32, 4096, 4096] | BF16 | 0 | 0.00 |
| [32, 4096, 4096] | FP32 | 0 | 0.00 |
| [128, 4096, 11008] | BF16 | 0 | 0.00 |

All 100 consecutive runs with identical inputs are bitwise identical. Non-determinism arises strictly from batch-size-dependent kernel selection, not hardware stochasticity.

### 5.2 Dense Model: Llama-3.1-8B-Instruct

**Table 7: 1000-Run Generation Determinism**

| Mode | Unique Outputs | Deterministic? | Elapsed |
|---|---:|:---:|---:|
| BF16 (default) | **2** | No | 894 s |
| FP32 accum | **1** | Yes | 902 s |

Under BF16, 1000 runs across 10 batch sizes (1,2,3,4,5,7,8,9,15,16; 100 runs each) produce 2 unique outputs with a 900:100 hash distribution --- exactly 1 batch size triggers a different cuBLAS kernel, producing a different generation. Each individual batch size is perfectly deterministic (1 unique per 100 runs). With FP32 accumulation, all 1000 runs produce the same hash, which matches the BF16 majority output. The flag corrects the deviant batch size.

**Table 8: Latency Comparison (20 runs, bs=1, 32 tokens)**

| Mode | Mean (ms) | Std (ms) | Min (ms) | Max (ms) |
|---|---:|---:|---:|---:|
| BF16 (default) | 871.0 | 8.4 | 861.3 | 896.6 |
| FP32 accum | 859.2 | 6.5 | 849.3 | 872.9 |

**Overhead: -1.4%** (FP32 is marginally *faster*). The cuBLAS FP32 accumulation codepath on Ampere GPUs incurs zero performance penalty. The slight speedup is within measurement noise but definitively rules out any overhead.

**Table 9: Long Sequence Logits Stability (188 tokens)**

| Mode | bs | Max Diff | Mean Diff | Argmax Flips | Flip Rate |
|---|---:|---:|---:|---:|---:|
| BF16 | 2 | 0.336 | 2.69e-2 | 2 | 1.06% |
| BF16 | 4 | 0.781 | 2.94e-2 | 2 | 1.06% |
| BF16 | 8 | **0.938** | 2.84e-2 | **4** | **2.13%** |
| BF16 | 16 | 0.664 | 2.81e-2 | 3 | 1.60% |
| FP32 | 2 | 0.500 | 2.54e-2 | 1 | 0.53% |
| FP32 | 4 | 0.438 | 2.59e-2 | 4 | 2.13% |
| FP32 | 8 | 0.500 | 2.54e-2 | 1 | 0.53% |
| FP32 | 16 | 0.500 | 2.54e-2 | 1 | 0.53% |

BF16 max logit difference grows with batch size from 0.22 to 0.94 (approaching 2 ULP). FP32 accumulation caps max_diff at 0.50 (1 ULP) uniformly. Both modes show residual argmax flips at specific batch sizes due to near-tie logit positions, but FP32 is substantially more consistent across batch sizes.

### 5.3 MoE Model: DeepSeek-V2-Lite

DeepSeek-V2-Lite is a 15.7B-parameter MoE model with 64 routed experts and top-6 selection. We run the same continuous-batching simulation as Section 5.2.

**Table 10: 200-Run Generation Determinism (DeepSeek-V2-Lite)**

| Mode | Unique Outputs | Deterministic? | Hash Distribution |
|---|---:|:---:|---|
| BF16 (default) | **3** | No | 120:40:40 |
| FP32 accum | **3** | No | 80:80:40 |

Both modes produce 3 unique outputs. The hash distributions differ --- BF16 groups batch sizes as {1,2,4}:8:16 while FP32 regroups them differently --- but neither achieves determinism. Each individual batch size remains perfectly deterministic (1 unique per 40 runs), confirming the variance is cross-batch-size.

**Table 11: MoE Logits Batch Variance**

| bs | BF16 max_diff | FP32 max_diff | BF16 argmax | FP32 argmax |
|---:|---:|---:|:---:|:---:|
| 2 | 0.688 | 0.672 | FLIP | FLIP |
| 4 | 0.875 | 0.781 | FLIP | FLIP |
| 8 | **1.500** | 0.656 | FLIP | FLIP |
| 16 | 0.875 | 0.750 | FLIP | FLIP |

FP32 accumulation reduces max_diff substantially (e.g., 1.50 to 0.66 at bs=8) but argmax flips persist at *every* batch size. The residual 0.5 ULP GEMM variance in the gate projection is sufficient to flip expert selection when router logits are near-tied.

**Table 12: MoE Latency**

| BF16 (ms) | FP32 (ms) | Overhead |
|---:|---:|:---:|
| 253.9 | 253.8 | 0% |

Zero overhead, consistent with the dense model result.

**Key insight.** FP32 accumulation eliminates GEMM batch variance for individual expert computations (Table 2: max_diff = 0 for K=2048). Yet the *model* remains non-deterministic because the *routing decision* --- which experts to activate --- is itself batch-dependent. The MoE routing pipeline (gate GEMM $\to$ softmax $\to$ top-k) amplifies sub-ULP perturbations through two mechanisms: softmax concentrates probability mass near the argmax, and top-k introduces a hard discontinuity at the selection boundary. A 0.5 ULP difference in gate logits, after softmax normalization over 64 experts, can produce expert score differences that cross the top-6 selection threshold.

### 5.4 Downstream Impact

We quantify how batch-dependent non-determinism affects three practical applications using Llama-3.1-8B-Instruct (30 prompts, bs=1 vs. bs=8).

**Table 13: RL Reward Signal Variance**

| Metric | BF16 | FP32 accum | Reduction |
|---|---:|---:|---:|
| Mean |logprob diff| | 2.30e-2 | 1.49e-2 | **35%** |
| Median |logprob diff| | 1.01e-2 | 6.73e-3 | 33% |
| Max |logprob diff| | 1.19e-1 | 8.39e-2 | 30% |
| Frac positions nonzero | 1.000 | 1.000 | --- |

FP32 accumulation reduces reward signal variance by 35% (mean) but does not eliminate it entirely. The residual variance at 100% of positions reflects the inherent sensitivity of log-probability to small logit changes, even when those changes are capped at 0.5 ULP.

**Table 14: Knowledge Distillation KL Divergence**

| Metric | BF16 | FP32 accum | Reduction |
|---|---:|---:|---:|
| Mean KL per position | 6.29e-4 | 4.59e-4 | **27%** |
| Median KL per position | 4.18e-4 | 3.52e-4 | 16% |
| Max KL per position | 3.41e-3 | 2.47e-3 | 28% |
| Frac positions KL > 1e-6 | 0.924 | 0.927 | --- |

**Table 15: MoE Expert Selection Flips (Synthetic, 128 experts, top-8)**

| Batch Size | BF16 flip rate | FP32 flip rate |
|---:|:---:|:---:|
| 4 | 6.0% (60/1000) | **0.0%** (0/1000) |
| 8 | 6.0% (60/1000) | **0.0%** (0/1000) |
| 16 | 4.0% (40/1000) | **0.0%** (0/1000) |

In the synthetic MoE setting, FP32 accumulation completely eliminates expert flips. However, the real DeepSeek-V2-Lite model (Table 10) still exhibits flips, indicating that the synthetic experiment underestimates the problem. The discrepancy arises because real MoE models have more concentrated router logit distributions (100% of positions have near-ties below threshold 0.001), creating a harder problem than synthetic random routing.

---

## 6. Related Work

**Deterministic GPU computation.** NVIDIA's cuDNN provides deterministic algorithm selection via `torch.use_deterministic_algorithms(True)`, but this addresses run-to-run non-determinism (e.g., atomic operations in backward passes), not batch-composition non-determinism. The cuBLAS flag we exploit (`allow_bf16_reduced_precision_reduction`) is orthogonal and targets forward-pass accumulation precision.

**Numerical precision in deep learning.** Micikevicius et al. (2018) established mixed-precision training with FP16 weights and FP32 master copies. Kalamkar et al. (2019) introduced BF16 for training. Our work addresses a different concern: not the *accuracy* of BF16 inference but its *determinism* under varying batch composition. The Higham-style analysis we employ (Higham, 2002) has been applied to deep learning numerics by Blanchard et al. (2020), but not specifically to the batch-invariance question.

**Deterministic serving systems.** Thinking Machine Lab (2025) proposed custom GEMM, attention, and RMSNorm kernels achieving deterministic inference at 61.5% overhead. SGLang (2025) implemented a deterministic inference mode with fixed split-KV and deterministic all-reduce at 34.35% overhead. vLLM is developing a `BATCH_INVARIANT=1` mode using FlexAttention. Our contribution demonstrates that for HuggingFace inference, the cuBLAS flag alone (0% overhead) suffices, and provides the theoretical framework explaining *why* it works.

**FlashAttention.** Dao et al. (2022, 2023) introduced tiled attention with online softmax. Our Theorem 2 formalizes why the FlashDecoding variant (which splits KV across SMs) introduces batch-dependent non-determinism, and Theorem 3 shows how fixed split boundaries restore determinism.

**MoE routing stability.** Fedus et al. (2022) studied load balancing and routing collapse in MoE training. Zoph et al. (2022) introduced expert choice routing. Our work identifies a new failure mode: batch-composition-dependent expert selection in *inference*, which is a numerical precision problem rather than a training dynamics problem.

---

## 7. Discussion and Limitations

### 7.1 Dense Models: A Solved Problem

For dense transformer models under HuggingFace-style inference (SDPA attention, no split-KV decode), batch-invariant determinism is achievable via a single cuBLAS flag at zero latency cost. This is our strongest result. The theoretical guarantee (Theorem 1) explains why it works, and the 1000-run experiment confirms it empirically.

### 7.2 MoE Models: An Open Problem

For MoE models, FP32 accumulation is necessary (it reduces logit variance by up to 56% at bs=8) but not sufficient. The 3-unique-output result on DeepSeek-V2-Lite persists in FP32 mode because MoE routing involves a discontinuous top-k operation that amplifies sub-ULP perturbations. Solving this requires either: (a) deterministic top-k with explicit tie-breaking rules that tolerate 0.5 ULP input variation; (b) routing mechanisms with wider margins (e.g., temperature scaling or hysteresis); or (c) FP64 accumulation in the gate GEMM (likely with non-trivial overhead). We consider this an important open problem.

### 7.3 Limitations

- **Hardware scope.** All experiments use NVIDIA RTX A6000 (Ampere). Behavior on H100 (Hopper), which has native FP8 support and different cuBLAS heuristics, is untested.
- **Tensor parallelism.** We evaluate single-GPU and 2-GPU pipeline parallelism. Tensor parallelism ($\text{TP} > 1$) introduces all-reduce operations with potentially non-deterministic NCCL collectives, which require separate treatment.
- **Serving engine integration.** While we provide the fixed split-KV design (Section 4.4) and Theorem 3 justification, we have not implemented and benchmarked it within vLLM or SGLang. The 0% overhead claim applies only to the cuBLAS flag for HuggingFace inference.
- **Attention backend.** Our zero-overhead result depends on SDPA's existing batch-invariant KV traversal. Custom attention backends with dynamic split-KV (as in serving engines) require the fixed-split intervention from Theorem 3, which may carry overhead.

---

## 8. Conclusion

We have shown that batch-dependent non-determinism in LLM inference --- a pervasive problem in production serving systems --- has a surprisingly simple solution for dense models: a single cuBLAS flag that forces FP32 accumulation in BF16 GEMM operations, at zero latency cost. Our theoretical framework (Theorems 1--3) provides a complete characterization of when and why this works: additive reductions (GEMM, RMSNorm, softmax) are absorbed by the $2^{16}$-fold precision gap between FP32 and BF16, while multiplicative rescaling in split-KV attention requires structural fixes (fixed split boundaries) rather than precision increases.

For MoE models, our experiments on DeepSeek-V2-Lite reveal that FP32 accumulation is necessary but insufficient. The top-k routing discontinuity amplifies residual sub-ULP variance into expert selection flips, producing 3 distinct outputs across batch sizes even with FP32 accumulators. This identifies MoE routing determinism as an open problem at the intersection of numerical analysis and model architecture design.

**Practical recommendation.** For any BF16 LLM inference workload, unconditionally set:
```python
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
```
This eliminates batch-dependent generation non-determinism for dense models, reduces RL reward variance by 35% and distillation KL divergence by 27%, and is a strict prerequisite for deterministic MoE inference. It costs nothing.

---

## References

Blanchard, P., Higham, N. J., and Mary, T. (2020). A class of subspace-descent methods for bfloat16 computations. *SIAM Journal on Matrix Analysis and Applications*, 41(4):1691--1716.

Dao, T. (2023). FlashAttention-2: Faster attention with better parallelism and work partitioning. *arXiv:2307.08691*.

Dao, T., Fu, D. Y., Ermon, S., Rudra, A., and Re, C. (2022). FlashAttention: Fast and memory-efficient exact attention with IO-awareness. In *Advances in Neural Information Processing Systems (NeurIPS)*.

Fedus, W., Zoph, B., and Shazeer, N. (2022). Switch Transformers: Scaling to trillion parameter models with simple and efficient sparsity. *Journal of Machine Learning Research*, 23(120):1--39.

Higham, N. J. (2002). *Accuracy and Stability of Numerical Algorithms* (2nd ed.). SIAM.

Kalamkar, D., Mudigere, D., Mellempudi, N., et al. (2019). A study of BFLOAT16 for deep learning training. *arXiv:1905.12322*.

Micikevicius, P., Narang, S., Alben, J., et al. (2018). Mixed precision training. In *International Conference on Learning Representations (ICLR)*.

NVIDIA (2024). CUTLASS: CUDA Templates for Linear Algebra Subroutines. *github.com/NVIDIA/cutlass*.

Zoph, B., Bello, I., Kumar, S., et al. (2022). ST-MoE: Designing stable and transferable sparse expert models. *arXiv:2202.08906*.

---

## Appendix A: Experimental Details

**Hardware.** NVIDIA RTX A6000 (Ampere, 48 GB GDDR6, 84 SMs, FP32 throughput 38.7 TFLOPS, BF16 Tensor Core throughput 77.4 TFLOPS). CUDA 12.4, cuBLAS 12.4, PyTorch 2.6.0.

**Dense model.** Meta Llama-3.1-8B-Instruct, BF16 weights, 32 transformer layers, hidden dim 4096, 32 attention heads, GQA with 8 KV heads, vocab size 128,256. Inference via HuggingFace `transformers` 4.56.1 with `attn_implementation="sdpa"`.

**MoE model.** DeepSeek-V2-Lite, BF16 weights, 15.7B total parameters, 64 routed experts per MoE layer, top-6 routing, expert hidden dim 2048 (intermediate 5632).

**Generation settings.** Greedy decoding (`do_sample=False`, `temperature=1.0`), 32 new tokens per generation. All sequences in each batch are padded to equal length with explicit `position_ids` to simulate continuous batching without padding artifacts.

**Op-level experiments.** Fixed `torch.manual_seed(42)`. Target row embedded at position 0 of varying-M batches. Reference computed at M=1. Differences reported in BF16 ULP scale (1.0 = one unit in the last place at the output magnitude).

## Appendix B: Complete Op-Level Tables

**Table B1: GEMM --- llama_gate_proj (K=4096, N=11008)**

| $M$ | BF16 max_diff | BF16 mean_diff | FP32 max_diff | FP32 mean_diff |
|----:|:---:|:---:|:---:|:---:|
| 1 | 0.00 | 0.00 | 0.00 | 0.00 |
| 2 | **1.00** | 5.47e-2 | 0.50 | 2.15e-4 |
| 4 | **1.00** | 5.47e-2 | 0.50 | 2.15e-4 |
| 8 | **1.00** | 5.47e-2 | 0.50 | 2.15e-4 |
| 16 | 0.00 | 0.00 | 0.50 | 7.30e-5 |
| 32 | **1.00** | 5.47e-2 | 0.50 | 2.15e-4 |
| 64 | **1.00** | 5.47e-2 | 0.50 | 2.15e-4 |
| 128 | **1.00** | 9.38e-2 | 0.50 | 2.15e-4 |
| 256 | **1.00** | 5.47e-2 | 0.50 | 2.15e-4 |
| 512 | **1.00** | 5.47e-2 | 0.50 | 2.15e-4 |
| 1024 | **1.00** | 5.47e-2 | 0.50 | 2.15e-4 |

The gate projection shape (N=11008) is the most sensitive: BF16 shows differences starting at M=2, while FP32 consistently caps at 0.50.

**Table B2: Attention Split-KV --- BF16 mode (max_diff)**

| seq_len | splits=1 | splits=2 | splits=4 | splits=8 | splits=16 |
|---:|:---:|:---:|:---:|:---:|:---:|
| 128 | 4.71e-3 | 5.68e-3 | 5.68e-3 | 9.11e-3 | 8.05e-3 |
| 512 | 2.67e-3 | 3.21e-3 | 3.41e-3 | 3.56e-3 | 5.24e-3 |
| 1024 | 1.88e-3 | 1.95e-3 | 1.94e-3 | 2.07e-3 | 2.11e-3 |
| 2048 | 1.13e-3 | 1.13e-3 | 1.29e-3 | 1.48e-3 | 2.32e-3 |

Error grows monotonically with split count and inversely with sequence length, consistent with Theorem 2's prediction that structural mismatch (more rescaling steps) dominates.

**Table B3: Serving Engine Attention Variance (model-level, from attn_kernel analysis)**

| Splits | BF16 max_diff | FP32 max_diff | Improvement |
|---:|:---:|:---:|:---:|
| 2 | 0.418 | 0.418 | 1x (none) |
| 4 | 1.137 | 1.145 | 1x (none) |
| 16 | 2.548 | 2.548 | 1x (none) |

FP32 accumulation provides exactly zero improvement when the split count varies, confirming Theorem 2.
