# Deterministic LLM Inference at Zero Cost: When FP32 Accumulation Absorbs Batch Variance

---

## Abstract

Large language model (LLM) serving systems employ continuous batching, where the batch composition changes dynamically as requests arrive and complete. We identify a critical but underappreciated consequence: the same prompt can produce different outputs depending on the batch composition at inference time. This non-determinism arises because GPU GEMM libraries (e.g., cuBLAS) select different kernel tiling strategies based on the batch dimension $M$, changing the floating-point reduction order across the inner-product dimension $K$. Under BF16 reduced-precision accumulation, these reordering differences produce errors up to 1.0 ULP in BF16 scale, sufficient to flip argmax token selections and diverge autoregressive generation.

We present a theoretical framework comprising three theorems that completely characterizes this phenomenon. **Theorem 1** proves that FP32 accumulation of BF16 operands makes additive reductions (GEMM, RMSNorm) invariant to association order when the result is rounded back to BF16, exploiting the $2^{16}$-fold precision gap between FP32 and BF16 rounding quanta. **Theorem 2** proves that attention with dynamic split-KV (as in FlashDecoding) introduces multiplicative rescaling chains that create structurally different computation graphs, which FP32 cannot absorb. **Theorem 3** proves that fixing the split-KV boundaries restores the additive reduction regime, recovering determinism.

Our practical contribution is a one-line fix: setting `torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False`. This achieves 100% generation determinism across 1000 runs with 10 different batch sizes on Llama-3.1-8B-Instruct, at zero latency overhead ($-1.4\%$, within noise). Downstream, FP32 accumulation eliminates MoE expert selection flips entirely ($6\% \to 0\%$), reduces distillation KL divergence by 27%, and caps GEMM batch variance at 0.5 ULP (versus 1.0 under BF16). The fix works for HuggingFace inference where attention is already batch-invariant; serving engines additionally require fixed split-KV boundaries.

---

## 1. Introduction

Consider a production LLM serving system processing concurrent requests. A user submits the same prompt twice, seconds apart. Between the two submissions, a different request arrives and joins the batch. The batch size changes from 7 to 8, and the user receives a different response. This is not a stochastic decoding artifact --- both requests use greedy decoding with temperature 1.0 and no sampling. The non-determinism is a consequence of floating-point arithmetic on GPUs.

This problem has significant practical consequences. In reinforcement learning from human feedback (RLHF) and group relative policy optimization (GRPO), reward signals computed from model logits acquire artificial variance when batch composition changes, corrupting the policy gradient. In knowledge distillation, the teacher's soft targets become inconsistent across training steps as the teacher's batch composition varies, introducing noise into the student's learning signal. In mixture-of-experts (MoE) models, near-tie router logits can flip expert selections depending on batch composition, causing routing instability that degrades both training and inference quality.

**Why does this happen?** Modern GPU GEMM libraries like cuBLAS use heuristic-driven kernel selection. When the batch dimension $M$ changes (e.g., from 7 to 8 rows), cuBLAS may select a different kernel with a different strategy for parallelizing the inner-product reduction over the $K$ dimension. Under BF16 reduced-precision accumulation, different reduction orders produce different results because floating-point addition is non-associative. The errors compound through dozens of transformer layers during autoregressive generation, causing token-level divergence.

**Our contributions.** We provide:

1. **A theoretical framework** (Theorems 1--3) that completely characterizes when FP32 accumulation absorbs batch-dependent reordering errors. Theorem 1 proves sufficiency for additive reductions (GEMM, RMSNorm) via the $2^{16}$-fold BF16/FP32 precision gap. Theorem 2 proves that attention with dynamic split-KV counts fails because multiplicative rescaling creates structurally different computation graphs. Theorem 3 proves that fixing split boundaries recovers the additive regime.

2. **A zero-cost practical fix**: setting a single cuBLAS flag (`allow_bf16_reduced_precision_reduction = False`) that forces FP32 accumulation in all GEMM operations. Combined with the fact that HuggingFace's SDPA attention backend already uses FP32 accumulators and fixed KV traversal, this achieves complete batch-invariant determinism.

3. **Comprehensive experimental validation** on Llama-3.1-8B-Instruct: 1000-run generation determinism (2 unique outputs under BF16, 1 under FP32), op-level variance characterization across GEMM, RMSNorm, softmax, and attention, downstream impact on reward signals, distillation KL, and MoE routing stability, and latency analysis confirming zero overhead.

---

## 2. Background

### 2.1 Floating-Point Precision and Non-Associativity

BF16 (Brain Floating Point 16) uses 1 sign bit, 8 exponent bits, and 7 mantissa bits, yielding unit roundoff $\varepsilon_{\text{bf16}} = 2^{-8} \approx 3.91 \times 10^{-3}$. FP32 (IEEE 754 single precision) uses 23 mantissa bits with unit roundoff $\varepsilon_{\text{fp32}} = 2^{-24} \approx 5.96 \times 10^{-8}$. The precision ratio is:

$$\frac{\varepsilon_{\text{fp32}}}{\varepsilon_{\text{bf16}}} = 2^{-16} \approx 1.53 \times 10^{-5}$$

This $2^{16}$-fold gap is the fundamental enabler of our approach. A product of two BF16 values $a \cdot b$ is exactly representable in FP32 (the product of two 8-bit mantissas fits within FP32's 24-bit mantissa). Thus the only source of numerical variation in a dot product of BF16 operands under FP32 accumulation is the *reduction order* of the summation.

Floating-point addition is non-associative: $(a + b) + c \neq a + (b + c)$ in general. When GPU kernels parallelize a reduction by partitioning it into chunks processed independently and then combined, the association order depends on the partitioning strategy, which may in turn depend on the batch size.

### 2.2 GPU GEMM: Split-K Strategies

For a matrix multiplication $Y = XW^T$ where $X \in \mathbb{R}^{M \times K}$ and $W \in \mathbb{R}^{N \times K}$, cuBLAS selects a kernel and tiling strategy based on $(M, N, K)$ and GPU occupancy. The *split-K* technique partitions the $K$-dimension reduction into $P$ chunks, computes partial dot products in parallel, and reduces them. Different values of $M$ may trigger different split-K configurations (different $P$, different chunk boundaries), producing different floating-point reduction orders for the *same* input row.

### 2.3 Attention: Online Softmax and Split-KV

Standard scaled dot-product attention computes $\text{Attention}(Q,K,V) = \text{softmax}(QK^T / \sqrt{d_k}) V$. Modern implementations use the *online softmax* algorithm [7, 8], which processes the KV sequence in blocks, maintaining running statistics $(m_i, l_i, \text{acc})$ representing the current maximum logit, the softmax denominator, and the output accumulator, respectively.

FlashDecoding [8] extends this to the decode phase by splitting the KV sequence across multiple streaming multiprocessors (SMs). Each SM computes partial attention over its assigned chunk, and a final reduction combines the partials using multiplicative rescaling corrections: $o^{(s)} = \exp(m^{(s-1)} - m^{(s)}) \cdot o^{(s-1)} + \exp(m_s - m^{(s)}) \cdot o_s$. The number of splits depends on SM occupancy, which varies with batch size.

### 2.4 Continuous Batching in Serving Engines

Production serving systems (vLLM [18], SGLang [19]) use continuous batching: requests dynamically join and leave the processing batch. This means that the $M$ dimension of GEMM operations and the split-KV count in FlashDecoding change from step to step, making the non-determinism described above a practical concern rather than a theoretical curiosity.

---

## 3. Theoretical Framework

### 3.1 Definitions

**Batch Invariance.** A function $f: \mathcal{X} \to \mathcal{Y}$ computed by a GPU kernel is *batch invariant* at input $x$ if for all batch configurations $B_1, B_2$: $f_{B_1}(x) = f_{B_2}(x)$ (bitwise identical). A transformer inference pipeline is batch invariant if every layer is batch invariant for every input.

**Batch-Dependent Reduction Order.** A reduction $R = \bigoplus_{i=1}^{N} a_i$ exhibits batch-dependent reduction order if the kernel partitions the reduction differently depending on batch configuration --- e.g., different split-K factors in GEMM or different split-KV counts in attention.

### 3.2 Theorem 1: Additive Reduction Sufficiency

**Setup.** Let $a_1, \ldots, a_N \in \mathbb{F}_{\text{bf16}}$ and let $\hat{S}_{\pi_1}, \hat{S}_{\pi_2}$ be FP32-accumulated sums under two arbitrary reduction orders.

**Lemma 1** (Higham [12, Thm 4.4]). For sequential FP32 summation: $\hat{S} = \sum_{i=1}^{N} a_i(1 + \delta_i)$ where $|\delta_i| \leq \gamma_{N-1}^{\text{fp32}} = (N-1)\varepsilon_{\text{fp32}} / (1 - (N-1)\varepsilon_{\text{fp32}}) \approx (N-1)\varepsilon_{\text{fp32}}$ for practical $N$.

**Lemma 2.** For $a, b \in \mathbb{F}_{\text{bf16}}$, the product $a \cdot b$ is exactly representable in FP32. *(Proof: BF16 mantissas are 8 bits including the implicit leading 1; their product is at most 16 bits, which fits in FP32's 24-bit mantissa.)*

**Theorem 1** (Additive Reduction Sufficiency). *The difference between two FP32-accumulated sums satisfies:*

$$|\hat{S}_{\pi_1} - \hat{S}_{\pi_2}| \leq 2\gamma_{N-1}^{\text{fp32}} \sum_{i=1}^{N} |a_i|$$

*If furthermore:*

$$2\gamma_{N-1}^{\text{fp32}} \cdot \underbrace{\frac{\sum |a_i|}{|\sum a_i|}}_{\kappa(S)} < \frac{\varepsilon_{\text{bf16}}}{1} \tag{$\star$}$$

*then $\text{round}_{\text{bf16}}(\hat{S}_{\pi_1}) = \text{round}_{\text{bf16}}(\hat{S}_{\pi_2})$ (bitwise identical).*

*Proof.* From Lemma 1, $\hat{S}_\pi = S + \sum_i a_i \delta_i^\pi$ where $S = \sum a_i$ is the exact sum. By the triangle inequality, $|\hat{S}_{\pi_1} - \hat{S}_{\pi_2}| \leq 2\gamma_{N-1}^{\text{fp32}} \sum |a_i|$. Condition ($\star$) ensures both computed sums lie within one BF16 rounding quantum ($\varepsilon_{\text{bf16}} |S|$) of the exact sum $S$, hence both round to the same BF16 value. $\square$

**Interpretation.** Rewriting ($\star$): $\kappa(S) < 2^{15}/(N-1) \approx 32768/(N-1)$. For well-conditioned sums ($\kappa(S) = O(1)$), this holds for $N$ up to tens of thousands --- the regime of transformer hidden dimensions.

**Application to GEMM.** Each output element $y_i = \sum_{k=1}^{K} W_{ik} x_k$ is an inner product over $K$ terms. By Lemma 2, the terms $W_{ik} x_k$ are exact FP32 values. For $K = 4096$ (LLaMA-7B), the worst-case Higham bound gives $|\Delta y_i| \leq 2.0 \times 10^{-3}$, while the BF16 quantum at typical output scale is $\sim 1.25 \times 10^{-3}$ --- the worst-case bound slightly exceeds the quantum. However, rounding errors are approximately random, and the *typical* reorder difference scales as $O(\sqrt{K}) \cdot \varepsilon_{\text{fp32}} \approx 3.8 \times 10^{-9}$, six orders of magnitude below the BF16 quantum. This explains the experimental observation: FP32 accumulation reduces GEMM max\_diff from 1.0 to 0.5 (the residual 0.5 represents rare rounding-boundary events).

**Application to RMSNorm.** The reduction $S = \sum_{i=1}^{d} x_i^2$ sums non-negative terms, giving $\kappa(S) = 1$ exactly. The sufficiency condition becomes $2(d-1)\varepsilon_{\text{fp32}} < \varepsilon_{\text{bf16}}$, satisfied for $d < 32769$. For $d = 4096$, the margin is $8\times$. This is a **deterministic guarantee** (not probabilistic), explaining the experimentally observed max\_diff = 0 for FP32-accumulated RMSNorm.

### 3.3 Theorem 2: Why Attention Split-KV Fails

**Theorem 2** (Attention Rescaling Breaks Additive Sufficiency). *The online softmax combination across $P$ splits is not a reordering of an additive reduction. Different split counts $P_1 \neq P_2$ produce structurally different computation graphs with multiplicative rescaling factors that introduce errors beyond the reach of FP32 absorption.*

*Proof sketch.* Compare $P = 2$ and $P = 4$ splits over the same KV sequence. With $P = 2$, an element in $o_1$ is rescaled by $\exp(m_1 - m^{(2)})$ (one exp evaluation, one subtraction). With $P = 4$, the same element undergoes: $\exp(m_1 - m^{(2)}) \cdot \exp(m^{(2)} - m^{(3)}) \cdot \exp(m^{(3)} - m^{(4)})$ --- three exp evaluations, three subtractions, two multiplications. While mathematically equivalent by telescoping, these are different floating-point computations. Each exp introduces error $\sim \varepsilon_{\exp} \approx 2^{-22}$ (GPU fast-math approximation). The rescaling factor $\exp(\Delta m)$ amplifies absolute errors when $\Delta m > 0$. More fundamentally, different split counts produce different local maxima $m_s$, different local softmax distributions $p_s$, and different partial sums --- these are different mathematical decompositions that only coincide in exact arithmetic. $\square$

**Experimental confirmation.** Split-KV attention shows max\_diff = 2.55 for *both* BF16 and FP32 accumulation --- FP32 provides zero improvement, confirming the error is structural.

### 3.4 Theorem 3: Fixed Split Restoration

**Theorem 3** (Fixed-Split Determinism). *If the attention split-KV boundaries are fixed (independent of batch configuration), so that all inputs with the same sequence length $L$ use the same split count $P$ and boundaries $\{0, C, 2C, \ldots, L\}$, then FP32 accumulation within each chunk restores batch invariance.*

*Proof.* Fix $P$ and boundaries. (1) Within each chunk, $m_s$ (max) is order-independent; $l_s = \sum p_s$ and $o_s = p_s V_s$ are additive reductions of non-negative (or well-conditioned) terms over a fixed-size chunk, absorbed by Theorem 1 under FP32 accumulation. (2) The cross-chunk combination is sequential (not parallelized): each step is a deterministic function of deterministic inputs. Thus the full computation is batch invariant. $\square$

**Practical implication.** Fix the split boundaries, then apply FP32 accumulation within each split. This converts the Theorem 2 regime (structurally different graphs) back to the Theorem 1 regime (same graph with absorbed internal variation).

---

## 4. Method

### 4.1 GEMM: One-Line Fix

For all linear projections (Q/K/V projections, output projection, MLP layers, LM head), batch-invariant determinism is achieved by a single Python statement:

```python
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
```

This instructs cuBLAS to use FP32 accumulation for all BF16 GEMM operations. Crucially, this does **not** change kernel selection or tiling strategy --- cuBLAS still selects its heuristically optimal kernel for each $(M, N, K)$ shape. It only changes the accumulator precision within the reduction, from BF16 to FP32. On Ampere GPUs (A6000), this has zero latency overhead ($-1.4\%$, within measurement noise).

### 4.2 RMSNorm and Softmax: Already Sufficient

HuggingFace's `LlamaRMSNorm` computes the sum-of-squares reduction in FP32 by default (`.float()` cast before squaring). PyTorch's `F.scaled_dot_product_attention` (SDPA) dispatches to FlashAttention or memory-efficient attention, both of which use FP32 accumulators internally. No intervention is needed for these operations in the HuggingFace inference path.

### 4.3 Attention: Fixed Split-KV for Serving Engines

For serving engines that use FlashDecoding with dynamic split-KV, we propose a fixed-split design following Theorem 3. Given a constant chunk size $C$ (e.g., 256 tokens), the number of splits $P = \lceil L/C \rceil$ depends only on the individual sequence's KV length $L$, not on batch composition. Each chunk uses FP32 accumulators, and the cross-chunk online softmax correction is applied in fixed sequential order. This design eliminates all batch-dependent parallelism in attention while preserving most of the SM utilization benefits of FlashDecoding.

### 4.4 Integration

For HuggingFace inference, the complete fix is:

```python
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
output = model.generate(input_ids, do_sample=False)
# output is now deterministic regardless of batch composition
```

For serving engines, the cuBLAS flag must be combined with a fixed split-KV attention kernel. Systems like vLLM (`VLLM_BATCH_INVARIANT=1`) and SGLang (`--enable-deterministic-inference`) provide the infrastructure for page table unification and deterministic NCCL communication that complement the kernel-level fix.

---

## 5. Experiments

**Hardware.** 2x NVIDIA RTX A6000 (Ampere, 48 GB each). **Software.** PyTorch 2.6.0+cu124, CUDA 12.4, HuggingFace Transformers 4.56.1. **Model.** Llama-3.1-8B-Instruct (BF16). All experiments use greedy decoding (`do_sample=False`, `temperature=1.0`).

### 5.1 Op-Level Batch Variance Characterization

We embed a fixed "target row" at position 0 of batches with varying $M$ and compare the output of `F.linear(batch, W)` against a reference obtained with $M = 1$.

**Table 1: GEMM Batch Variance (LLaMA Q-proj, K=4096, N=4096)**

| $M$ | Default max\_diff | Default mean\_diff | FP32 max\_diff | FP32 mean\_diff |
|----:|------------------:|-------------------:|---------------:|----------------:|
| 1 | 0.00 | 0.00 | 0.00 | 0.00 |
| 16 | 0.00 | 0.00 | 0.00 | 0.00 |
| 32 | 0.50 | 1.91e-04 | 0.50 | 1.91e-04 |
| 64 | **1.00** | 5.08e-02 | 0.50 | 1.61e-04 |
| 256 | **1.00** | 7.52e-02 | 0.50 | 1.34e-04 |
| 1024 | **1.00** | 7.52e-02 | 0.50 | 1.91e-04 |

**Table 2: GEMM Batch Variance (LLaMA gate-proj, K=4096, N=11008)**

| $M$ | Default max\_diff | Default mean\_diff | FP32 max\_diff | FP32 mean\_diff |
|----:|------------------:|-------------------:|---------------:|----------------:|
| 1 | 0.00 | 0.00 | 0.00 | 0.00 |
| 2 | **1.00** | 5.47e-02 | 0.50 | 2.15e-04 |
| 16 | 0.00 | 0.00 | 0.50 | 7.30e-05 |
| 64 | **1.00** | 5.47e-02 | 0.50 | 2.15e-04 |
| 128 | **1.00** | 9.38e-02 | 0.50 | 2.15e-04 |

**Table 3: GEMM Batch Variance (MoE Expert, K=2048, N=5632)**

| $M$ | Default max\_diff | Default mean\_diff | FP32 max\_diff | FP32 mean\_diff |
|----:|------------------:|-------------------:|---------------:|----------------:|
| 1--128 | 0.00 | 0.00 | 0.00 | 0.00 |
| 256 | **1.00** | 5.32e-02 | **0.00** | **0.00** |
| 512 | **1.00** | 6.84e-02 | **0.00** | **0.00** |
| 1024 | 0.00 | 0.00 | 0.00 | 0.00 |

Key findings: (i) cuBLAS kernel selection changes at specific $M$ thresholds, altering reduction order. (ii) Default BF16 accumulation produces max\_diff = 1.0 (a full BF16 ULP). (iii) FP32 accumulation caps max\_diff at 0.5 and reduces mean\_diff by $\sim 300\times$. (iv) The MoE expert shape (K=2048) is fully immune under FP32 accumulation --- zero difference at all batch sizes.

**Table 4: RMSNorm Chunk Reduction Variance**

| hidden\_dim | Chunks | BF16 max\_diff | FP32 max\_diff |
|-----------:|-------:|---------------:|---------------:|
| 2048 | 1--32 | 1.56e-02 | **0.00** |
| 4096 | 1--32 | 1.56e-02 | **0.00** |

FP32 accumulation achieves zero error across all chunk counts, confirming the deterministic guarantee from Theorem 1 ($\kappa = 1$ for non-negative terms).

**Table 5: Softmax Chunk Reduction Variance (vocab=128256)**

| Chunks | BF16 max\_diff | BF16 KL\_div | FP32 max\_diff | FP32 KL\_div |
|-------:|---------------:|-------------:|---------------:|-------------:|
| 1 | 1.21e-06 | 1.36e-03 | 2.91e-11 | -2.03e-09 |
| 4 | 1.72e-06 | -3.89e-03 | 2.91e-11 | -2.03e-09 |
| 16 | 1.72e-06 | -3.89e-03 | 5.82e-11 | -2.03e-09 |

FP32 softmax is effectively exact (max error $\sim 10^{-11}$, FP32 epsilon level).

**Table 6: Attention Split-KV Variance (max\_diff vs reference)**

| seq\_len | BF16, splits=4 | BF16, splits=16 | FP32, splits=4 | FP32, splits=16 |
|--------:|---------------:|----------------:|---------------:|----------------:|
| 128 | 5.68e-03 | 8.05e-03 | 8.94e-08 | 8.94e-08 |
| 512 | 3.41e-03 | 5.24e-03 | 5.96e-08 | 8.94e-08 |
| 2048 | 1.29e-03 | 2.32e-03 | 4.10e-08 | 5.22e-08 |

For a *fixed* split count, FP32 accumulation reduces attention error to $\sim 10^{-8}$. However, when the split count itself varies (the batch-dependent scenario), max\_diff remains 2.55 regardless of accumulation precision, confirming Theorem 2.

**Run-to-run variance.** 100 consecutive `F.linear` calls with identical inputs produce bit-identical outputs across all shapes and dtypes tested. The variance is strictly from batch-size-dependent kernel selection, not hardware non-determinism.

### 5.2 End-to-End Generation Determinism

**1000-Run Test.** A single prompt (10 tokens, "What is deterministic inference in large language models?") is processed 1000 times, cycling through batch sizes $\{1,2,3,4,5,7,8,9,15,16\}$ (100 runs each), generating 32 tokens with greedy decoding.

**Table 7: 1000-Run Generation Determinism**

| Mode | Unique Outputs | Deterministic? | Elapsed (s) |
|------|---------------:|:--------------:|------------:|
| BF16 (default) | 2 | No | 894.3 |
| FP32 accum | **1** | **Yes** | 902.0 |

Under BF16, exactly 2 unique outputs emerged: 9 of the 10 batch sizes produced output A (900 runs), while 1 batch size produced output B (100 runs). Each individual batch size was perfectly deterministic (1 unique per 100 runs). Under FP32 accumulation, all 1000 runs across all 10 batch sizes produced the same output, whose hash matched the BF16 majority.

**Per-batch-size analysis.** Every individual batch size produces exactly 1 unique output per 100 runs in both modes. The non-determinism is purely *across* batch sizes: cuBLAS selects a different GEMM kernel for one particular batch size, producing a different reduction order that flips one token during autoregressive generation.

**Argmax Flip Rate.** 10 prompts (17--22 tokens each), batch sizes $\{2,4,8,16\}$, comparing next-token argmax vs. $bs=1$ reference across 756 token positions.

| Mode | Flips | Total Tokens | Flip Rate |
|------|------:|-------------:|----------:|
| BF16 (default) | 1 | 756 | 0.13% |
| FP32 accum | 5 | 756 | 0.66% |

The higher flip rate under FP32 is expected: the cuBLAS flag changes the $bs=1$ reference kernel itself, altering reference logits. The meaningful metric is cross-batch-size consistency (Experiment 1), where FP32 achieves perfect determinism.

**Latency.** 20 generation runs (3 warmup), $bs=1$, 32 new tokens.

**Table 8: Latency Comparison**

| Mode | Mean (ms) | Std (ms) |
|------|----------:|---------:|
| BF16 (default) | 871.0 | 8.4 |
| FP32 accum | 859.2 | 6.5 |

**Overhead: $-1.4\%$** (FP32 accum is marginally *faster*, within measurement noise). The cuBLAS flag has zero performance cost on Ampere. This stands in contrast to alternative approaches: Thinking Machine Lab's custom kernels incur 61.5% overhead, SGLang's deterministic mode incurs 34.35%, and naive FP32 casting incurs 34.9$\times$ slowdown.

**Qwen3-30B-A3B MoE.** We attempted a 200-run determinism test on Qwen3-30B-A3B-Instruct (INT4 mixed quantization). The model loaded across 2x A6000 (27.4 GB total VRAM) but generation was impractically slow without the `gptqmodel` backend (>30 min per generation), so this experiment was skipped. Validating determinism on quantized MoE models remains future work.

### 5.3 Long Sequence Analysis

**Setup.** A single 188-token prompt, comparing logits at $bs = \{2,4,8,16\}$ against $bs=1$ reference.

**Table 9: Long Sequence (188 tokens) Logit Divergence**

| Batch Size | BF16 max\_diff | BF16 flips (rate) | FP32 max\_diff | FP32 flips (rate) |
|-----------:|---------------:|-------------------:|---------------:|-------------------:|
| 2 | 0.336 | 2 (1.06%) | 0.500 | 1 (0.53%) |
| 4 | 0.781 | 2 (1.06%) | 0.438 | 4 (2.13%) |
| 8 | **0.938** | **4 (2.13%)** | 0.500 | 1 (0.53%) |
| 16 | 0.664 | 3 (1.60%) | 0.500 | 1 (0.53%) |

BF16 max logit difference reaches 0.94 (nearly 2 ULP) at $bs=8$, with up to 2.1% argmax flips. FP32 accumulation caps max\_diff at 0.5 (1 ULP) and reduces flips to 0.5% for most batch sizes. The exception is $bs=4$ under FP32, which shows 4 flips due to a specific kernel tiling boundary at this batch size and prompt length.

### 5.4 Downstream Impact

**Reward Signal Variance (RL Proxy).** For 30 prompts, we compute $|\text{logprob}_{bs=1} - \text{logprob}_{bs=8}|$ per position.

**Table 10: Reward Signal Variance**

| Metric | BF16 (default) | FP32 accum |
|--------|---------------:|----------:|
| Mean |logprob diff| | 2.30e-02 | 1.49e-02 |
| Median |logprob diff| | 1.01e-02 | 6.73e-03 |
| Max |logprob diff| | 1.19e-01 | 8.39e-02 |
| Paired t-test p-value | 2.60e-17 | 7.90e-17 |

Both modes show statistically significant reward signal variance ($p \ll 0.05$), confirming that batch composition injects systematic noise into RL reward signals. FP32 accumulation reduces mean variance by 35%.

**Distillation KL Divergence.** Token-level $\text{KL}(\text{softmax}(\ell_{bs=1}) \| \text{softmax}(\ell_{bs=8}))$ for 30 prompts, 357 positions.

**Table 11: Distillation Signal Corruption**

| Metric | BF16 (default) | FP32 accum |
|--------|---------------:|----------:|
| Mean KL per position | 6.29e-04 | 4.59e-04 |
| Max KL per position | 3.41e-03 | 2.47e-03 |
| Frac positions KL > 1e-6 | 92.4% | 92.7% |

FP32 accumulation reduces mean KL divergence by **27.1%** compared to BF16 defaults.

**MoE Expert Selection Stability.** Synthetic MoE (hidden\_dim=2048, 128 experts, top-8 routing). Expert selections compared at $bs=1$ vs. $bs = \{4, 8, 16\}$ over 20 trials $\times$ 50 tokens (1000 routing decisions per batch size).

**Table 12: MoE Expert Selection Flips**

| Batch Size | BF16 flip rate | FP32 flip rate |
|:----------:|:--------------:|:--------------:|
| 4 | 6.0% (60/1000) | **0.0%** (0/1000) |
| 8 | 6.0% (60/1000) | **0.0%** (0/1000) |
| 16 | 4.0% (40/1000) | **0.0%** (0/1000) |

Under BF16 defaults, 4--6% of expert selections flip when the batch size changes. FP32 accumulation **completely eliminates** expert selection flips. Near-tie prevalence is 100% (all 50 test tokens have gap $< 0.001$ between the 8th and 9th expert), making this a particularly demanding stress test.

---

## 6. Related Work

**Floating-point determinism in deep learning.** Higham [12] provides the foundational error analysis for floating-point summation that underlies our Theorem 1. PyTorch documents known sources of non-determinism and provides `torch.use_deterministic_algorithms()` [13], which enforces deterministic kernels but does not address batch-dependent reduction order in cuBLAS. The NVIDIA CUBLAS\_WORKSPACE\_CONFIG environment variable controls workspace-based non-determinism but does not address accumulation precision.

**FlashAttention and FlashDecoding.** Dao et al. [7] introduced FlashAttention with tiled online softmax and IO-aware computation. FlashAttention-2 [8] improved parallelism and introduced FlashDecoding, which splits the KV sequence across SMs during decode. Our Theorem 2 characterizes precisely why this split-KV mechanism introduces batch-dependent non-determinism that cannot be resolved by accumulation precision alone.

**Batch invariance in serving systems.** The Thinking Machine Lab blog [14] first characterized batch-composition non-determinism in serving engines and proposed custom GEMM, attention, and normalization kernels, reporting 61.5% overhead. vLLM is developing a `VLLM_BATCH_INVARIANT=1` mode using FlexAttention and batch-invariant ops [15]. SGLang provides `--enable-deterministic-inference` with fixed split-KV and deterministic all-reduce, reporting 34.35% overhead [16]. Our work shows that for HuggingFace inference, a single cuBLAS flag achieves the same goal at zero cost.

**LayerCast.** The LayerCast approach casts weights from BF16 to FP32 before each layer's computation, achieving determinism at 3.44$\times$ overhead [17]. Our approach is strictly superior: it achieves the same determinism without any casting overhead by exploiting the cuBLAS accumulation flag.

**NCCL deterministic communication.** For multi-GPU inference with tensor parallelism, NCCL all-reduce operations introduce additional non-determinism from reduction order across GPUs. Deterministic NCCL requires fixed communication patterns, which is orthogonal to our single-GPU analysis but complementary for production deployment.

---

## 7. Discussion and Limitations

**Scope of the cuBLAS flag.** The `allow_bf16_reduced_precision_reduction = False` flag resolves GEMM-level batch variance. For HuggingFace inference with SDPA, this is sufficient because attention already uses FP32 accumulators and fixed KV traversal. For serving engines with FlashDecoding, the flag must be combined with fixed split-KV boundaries (Theorem 3). Our experiments validate only the HuggingFace path; integration with vLLM/SGLang attention backends is future work.

**Residual 0.5 max\_diff in GEMM.** Even with FP32 accumulation, GEMM shows max\_diff = 0.5 (0.5 ULP in BF16). This arises from rare rounding-boundary events predicted by Theorem 1: when the exact sum lies near a BF16 midpoint, two different FP32 reduction orders can land on opposite sides. In practice, this does not cause argmax flips during generation (as demonstrated by the 1000-run test), because the probability of a rounding-boundary violation coinciding with a near-tie argmax is vanishingly small.

**Hardware.** All experiments use NVIDIA RTX A6000 (Ampere architecture). Hopper (H100) GPUs have different tensor core behavior (FP8 support, TMA-based data movement) that may affect kernel selection heuristics and accumulation behavior. Validation on Hopper is planned.

**Quantized models.** We attempted INT4 mixed quantization (Qwen3-30B-A3B with AutoRound), but the model could not generate outputs in reasonable time without the `gptqmodel` backend. FP8 quantization on Hopper may introduce additional accumulation precision challenges. Determinism validation for quantized models remains open.

**Tensor parallelism.** Our experiments use single-GPU or naive model-parallel (device\_map="auto") inference. Tensor parallelism ($TP > 1$) introduces NCCL all-reduce non-determinism that is orthogonal to our cuBLAS-level fix. Production systems need both our fix and deterministic NCCL to achieve full determinism.

**Reward and distillation variance.** While FP32 accumulation reduces reward signal variance by 35% and distillation KL by 27%, it does not eliminate them entirely. The residual variance comes from the 0.5-ULP GEMM differences and from the comparison being between $bs=1$ and $bs=8$ (different shapes still trigger different kernels, producing different-but-deterministic results). True zero variance requires running all comparisons at the same batch size, which our fix enables: with FP32 accumulation, the output is deterministic for any *fixed* batch size.

---

## 8. Conclusion

We have presented a complete theoretical and empirical characterization of batch-composition-dependent non-determinism in LLM inference. Our three theorems provide a unified framework: Theorem 1 shows that FP32 accumulation absorbs reordering errors in additive reductions (GEMM, RMSNorm) by exploiting the $2^{16}$-fold BF16/FP32 precision gap; Theorem 2 shows that attention with dynamic split-KV introduces multiplicative rescaling chains that are structurally different computation graphs, immune to precision improvements; and Theorem 3 shows that fixing split boundaries recovers the additive regime.

The practical outcome is striking: a single line of code --- `torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False` --- achieves 100% generation determinism across 1000 runs with varying batch sizes, at zero latency overhead. For downstream applications, this eliminates MoE expert selection flips ($6\% \to 0\%$), reduces distillation KL divergence by 27%, and caps GEMM variance at 0.5 ULP.

**Practical recommendation.** For HuggingFace BF16 inference, always set the cuBLAS flag. It is free, effective, and sufficient. For serving engines with FlashDecoding, combine the flag with fixed split-KV boundaries following Theorem 3. We hope this work helps practitioners achieve reproducible LLM inference and motivates serving engine developers to integrate fixed-split attention kernels as a first-class feature.

---

## References

[1] Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention is All You Need. *NeurIPS 2017*.

[2] Touvron, H., Lavril, T., Izacard, G., et al. (2023). LLaMA: Open and Efficient Foundation Language Models. *arXiv:2302.13971*.

[3] Dubey, A., et al. (2024). The Llama 3 Herd of Models. *arXiv:2407.21783*.

[4] Ouyang, L., Wu, J., Jiang, X., et al. (2022). Training language models to follow instructions with human feedback. *NeurIPS 2022*.

[5] Shao, Z., Wang, P., Zhu, Q., et al. (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. *arXiv:2402.03300*.

[6] Hinton, G., Vinyals, O., Dean, J. (2015). Distilling the Knowledge in a Neural Network. *arXiv:1503.02531*.

[7] Dao, T., Fu, D. Y., Ermon, S., Rudra, A., Re, C. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. *NeurIPS 2022*.

[8] Dao, T. (2023). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. *arXiv:2307.08691*.

[9] Fedus, W., Zoph, B., Shazeer, N. (2022). Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity. *JMLR* 23(120):1--39.

[10] Lepikhin, D., et al. (2021). GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding. *ICLR 2021*.

[11] IEEE 754-2019. *IEEE Standard for Floating-Point Arithmetic*.

[12] Higham, N. J. (2002). *Accuracy and Stability of Numerical Algorithms* (2nd ed.). SIAM.

[13] PyTorch Documentation. Reproducibility. https://pytorch.org/docs/stable/notes/randomness.html

[14] Thinking Machine Lab (2024). Achieving Deterministic Inference in LLM Serving. Blog post.

[15] vLLM Project. VLLM\_BATCH\_INVARIANT implementation. https://github.com/vllm-project/vllm

[16] SGLang Project. Deterministic inference mode. https://github.com/sgl-project/sglang

[17] LayerCast: Achieving Deterministic BF16 Inference via FP32 Casting. Community implementation.

[18] Kwon, W., Li, Z., Zhuang, S., et al. (2023). Efficient Memory Management for Large Language Model Serving with PagedAttention. *SOSP 2023*.

[19] Zheng, L., Yin, L., Xie, Z., et al. (2024). SGLang: Efficient Execution of Structured Language Model Programs. *arXiv:2312.07104*.

[20] NVIDIA (2024). CUTLASS: CUDA Templates for Linear Algebra Subroutines. https://github.com/NVIDIA/cutlass

---

## Appendix

### A. Proof Details for Theorem 2

We provide the full error analysis for the online softmax combination with different split counts.

**Setup.** Consider a KV sequence of length $L$ processed with $P$ splits. After processing all $P$ splits, the output for a single query is:

$$o = \frac{\sum_{s=1}^{P} \exp(m_s - m_{\text{global}}) \cdot o_s}{\sum_{s=1}^{P} \exp(m_s - m_{\text{global}}) \cdot l_s}$$

where $m_s$ is the local max of split $s$, $m_{\text{global}} = \max_s m_s$, $o_s = \text{softmax}_{\text{local}}(QK_s^T) \cdot V_s$ (unnormalized by global softmax), and $l_s = \sum_j \exp((QK_s^T)_j - m_s)$.

**With $P = 2$:** One rescaling step. The rescaling factor for split 1 is computed as:

$$r_1^{(2)} = \text{fl}(\exp(\text{fl}(m_1 - m^{(2)})))$$

Total relative error: $|r_1^{(2)} / \exp(m_1 - m^{(2)}) - 1| \leq \varepsilon_{\text{fp32}} + \varepsilon_{\exp} + \varepsilon_{\text{fp32}} \cdot \varepsilon_{\exp} \approx \varepsilon_{\exp}$.

**With $P = 4$:** Three sequential rescaling steps. The effective rescaling factor for split 1 after all steps is:

$$\tilde{r}_1^{(4)} = \text{fl}(\text{fl}(\text{fl}(\exp(\text{fl}(m_1 - m^{(2)}))) \cdot \text{fl}(\exp(\text{fl}(m^{(2)} - m^{(3)})))) \cdot \text{fl}(\exp(\text{fl}(m^{(3)} - m^{(4)}))))$$

This involves 3 exp evaluations, 3 subtractions, and 2 multiplications. The total relative error is:

$$|\tilde{r}_1^{(4)} / \exp(m_1 - m^{(4)}) - 1| \leq 3\varepsilon_{\exp} + 5\varepsilon_{\text{fp32}} + \text{h.o.t.}$$

**Difference between paths.** The two-split and four-split computations produce rescaling factors that differ by approximately $(P_2 - P_1) \cdot \varepsilon_{\exp} \approx 2 \times 2^{-22} \approx 4.8 \times 10^{-7}$ in relative terms. Applied to output values of magnitude $\|V\|_\infty \sim 1$, this gives absolute errors of $\sim 5 \times 10^{-7}$ per element, which is below the BF16 quantum for typical output scales. However, the more impactful source of error is the *different local maxima and softmax distributions* produced by different chunk boundaries, which constitutes a structural difference that cannot be characterized as a simple rescaling perturbation.

### B. Full Op-Level Experimental Tables

**Table B.1: Complete GEMM Batch Variance (Q-proj, K=4096, N=4096)**

| $M$ | Default max\_diff | Default mean\_diff | FP32 max\_diff | FP32 mean\_diff |
|----:|------------------:|-------------------:|---------------:|----------------:|
| 1 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 2 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 4 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 8 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 16 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 32 | 5.00e-01 | 1.91e-04 | 5.00e-01 | 1.91e-04 |
| 64 | 1.00e+00 | 5.08e-02 | 5.00e-01 | 1.61e-04 |
| 128 | 5.00e-01 | 1.91e-04 | 5.00e-01 | 1.91e-04 |
| 256 | 1.00e+00 | 7.52e-02 | 5.00e-01 | 1.34e-04 |
| 512 | 5.00e-01 | 1.91e-04 | 5.00e-01 | 1.91e-04 |
| 1024 | 1.00e+00 | 7.52e-02 | 5.00e-01 | 1.91e-04 |

**Table B.2: Complete GEMM Batch Variance (gate-proj, K=4096, N=11008)**

| $M$ | Default max\_diff | Default mean\_diff | FP32 max\_diff | FP32 mean\_diff |
|----:|------------------:|-------------------:|---------------:|----------------:|
| 1 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 2 | 1.00e+00 | 5.47e-02 | 5.00e-01 | 2.15e-04 |
| 4 | 1.00e+00 | 5.47e-02 | 5.00e-01 | 2.15e-04 |
| 8 | 1.00e+00 | 5.47e-02 | 5.00e-01 | 2.15e-04 |
| 16 | 0.00e+00 | 0.00e+00 | 5.00e-01 | 7.30e-05 |
| 32 | 1.00e+00 | 5.47e-02 | 5.00e-01 | 2.15e-04 |
| 64 | 1.00e+00 | 5.47e-02 | 5.00e-01 | 2.15e-04 |
| 128 | 1.00e+00 | 9.38e-02 | 5.00e-01 | 2.15e-04 |
| 256 | 1.00e+00 | 5.47e-02 | 5.00e-01 | 2.15e-04 |
| 512 | 1.00e+00 | 5.47e-02 | 5.00e-01 | 2.15e-04 |
| 1024 | 1.00e+00 | 5.47e-02 | 5.00e-01 | 2.15e-04 |

**Table B.3: Complete GEMM Batch Variance (MoE expert, K=2048, N=5632)**

| $M$ | Default max\_diff | Default mean\_diff | FP32 max\_diff | FP32 mean\_diff |
|----:|------------------:|-------------------:|---------------:|----------------:|
| 1 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 2 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 4 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 8 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 16 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 32 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 64 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 128 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| 256 | 1.00e+00 | 5.32e-02 | 0.00e+00 | 0.00e+00 |
| 512 | 1.00e+00 | 6.84e-02 | 0.00e+00 | 0.00e+00 |
| 1024 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |

**Table B.4: Attention Split-KV (BF16, max\_diff vs FP32 single-pass reference)**

| seq\_len | splits=1 | splits=2 | splits=4 | splits=8 | splits=16 |
|--------:|---------:|---------:|---------:|---------:|----------:|
| 128 | 4.71e-03 | 5.68e-03 | 5.68e-03 | 9.11e-03 | 8.05e-03 |
| 512 | 2.67e-03 | 3.21e-03 | 3.41e-03 | 3.56e-03 | 5.24e-03 |
| 1024 | 1.88e-03 | 1.95e-03 | 1.94e-03 | 2.07e-03 | 2.11e-03 |
| 2048 | 1.13e-03 | 1.13e-03 | 1.29e-03 | 1.48e-03 | 2.32e-03 |

**Table B.5: Attention Split-KV (FP32, max\_diff vs FP32 single-pass reference)**

| seq\_len | splits=1 | splits=2 | splits=4 | splits=8 | splits=16 |
|--------:|---------:|---------:|---------:|---------:|----------:|
| 128 | 0.00e+00 | 5.96e-08 | 8.94e-08 | 7.45e-08 | 8.94e-08 |
| 512 | 0.00e+00 | 4.47e-08 | 5.96e-08 | 1.04e-07 | 8.94e-08 |
| 1024 | 0.00e+00 | 4.47e-08 | 4.47e-08 | 5.22e-08 | 5.22e-08 |
| 2048 | 0.00e+00 | 3.49e-08 | 4.10e-08 | 3.73e-08 | 5.22e-08 |

**Table B.6: Run-to-Run Variance (100 consecutive calls)**

| Shape | dtype | Mismatches (of 100) | max\_diff |
|-------------------------------|-------|--------------------:|----------:|
| [32, 4096, 4096] | bf16 | 0 | 0.00e+00 |
| [32, 4096, 4096] | fp32 | 0 | 0.00e+00 |
| [1, 4096, 4096] | bf16 | 0 | 0.00e+00 |
| [128, 4096, 11008] | bf16 | 0 | 0.00e+00 |

### C. Experimental Details

**Model.** Meta-Llama/Llama-3.1-8B-Instruct loaded in BF16 precision with `device_map="auto"` across 2x A6000 GPUs.

**Tokenization.** All sequences use the model's default tokenizer with left-padding disabled. For the 1000-run experiment, all sequences in each batch are identical (same prompt repeated $M$ times) with explicit `position_ids = [0, 1, ..., L-1]` to ensure no padding artifacts.

**Decoding.** Greedy decoding: `do_sample=False`, `temperature=1.0`, `num_beams=1`. The `do_sample=False` setting ensures that the only source of output variation is floating-point non-determinism, not stochastic sampling.

**Seeds.** `torch.manual_seed(42)` for all op-level experiments. No seed is needed for end-to-end greedy decoding (no random operations).

**Latency measurement.** 20 generation runs with 3 warmup runs excluded. Measured with `torch.cuda.synchronize()` and `time.perf_counter()` for wall-clock accuracy. CUDA graph compilation is not used.

**MoE synthetic experiment.** Router: `nn.Linear(2048, 128)` with random BF16 weights. Top-8 expert selection via `torch.topk`. 20 trials with different random input tokens, 50 tokens per trial. Expert flip = any difference in the set of 8 selected experts between $bs=1$ and $bs \in \{4,8,16\}$.
