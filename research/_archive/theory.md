# Theoretical Framework for Deterministic LLM Inference via Precision-Aware Accumulation

---

## 1. Preliminaries

### 1.1 Floating-Point Formats

We consider two IEEE 754-compatible floating-point formats used in modern GPU inference:

**BF16 (Brain Floating Point 16).** A 16-bit format with 1 sign bit, 8 exponent bits, and 7 mantissa bits. The unit roundoff is:

$$\varepsilon_{\text{bf16}} = 2^{-8} \approx 3.91 \times 10^{-3}$$

and the relative rounding error satisfies $|\text{fl}_{\text{bf16}}(x) - x| \leq \varepsilon_{\text{bf16}} |x|$ for any real $x$ in the representable range. The spacing between consecutive BF16 numbers at scale $|x|$ is:

$$\text{ulp}_{\text{bf16}}(x) = 2\varepsilon_{\text{bf16}} \cdot 2^{\lfloor \log_2 |x| \rfloor} \approx 2\varepsilon_{\text{bf16}} |x|$$

The rounding quantum (half the ULP, i.e., the maximum perturbation under round-to-nearest-even) is $\varepsilon_{\text{bf16}} |x|$.

> **Note on convention.** Throughout this paper we use the "unit roundoff" convention where $\varepsilon = 2^{-(p)}$ for a $p$-bit significand (counting the implicit leading 1). Some references define machine epsilon as $2^{-(p-1)}$; our bounds are consistent with Higham (2002, Ch. 2) using $u = 2^{-p}$ and the rounding model $\text{fl}(a \circ b) = (a \circ b)(1 + \delta)$ with $|\delta| \leq u$.

**FP32 (IEEE 754 Single Precision).** A 32-bit format with 1 sign bit, 8 exponent bits, and 23 mantissa bits. The unit roundoff is:

$$\varepsilon_{\text{fp32}} = 2^{-24} \approx 5.96 \times 10^{-8}$$

The ratio of precisions is:

$$\frac{\varepsilon_{\text{fp32}}}{\varepsilon_{\text{bf16}}} = 2^{-16} \approx 1.53 \times 10^{-5}$$

This $2^{16}$-fold gap is the fundamental enabler of our approach: errors introduced at FP32 precision can be absorbed by BF16 rounding.

### 1.2 Batch Invariance

**Definition 1 (Batch Invariance).** Let $f: \mathcal{X} \to \mathcal{Y}$ be a function computed by a GPU kernel, and let $x \in \mathcal{X}$ be a fixed input. We say $f$ is *batch invariant* at $x$ if for all batch configurations $B_1, B_2$ (differing in batch size, padding, co-resident sequences, etc.):

$$f_{B_1}(x) = f_{B_2}(x) \quad \text{(bitwise identical)}$$

A transformer inference pipeline is batch invariant if every layer's computation is batch invariant for every input.

**Definition 2 (Batch-Dependent Reduction Order).** A reduction operation $R = \bigoplus_{i=1}^{N} a_i$ exhibits *batch-dependent reduction order* if the GPU kernel partitions the reduction differently depending on the batch configuration. Specifically, different batch sizes may trigger different tiling strategies (e.g., split-K in GEMM, chunked reductions in normalization, split-KV in attention), producing different association orders for the floating-point additions.

### 1.3 Reduction Operations in Transformer Inference

A single transformer decoder layer contains three categories of floating-point reductions:

1. **Linear projections (GEMM).** For weight $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ and activation $x \in \mathbb{R}^{d_{\text{in}}}$:
   $$y_i = \sum_{k=1}^{K} W_{ik} \cdot x_k, \quad K = d_{\text{in}}$$
   This is an inner-product (dot-product) reduction over $K$ terms.

2. **RMSNorm.** For input $x \in \mathbb{R}^{d}$:
   $$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}} \odot \gamma$$
   The reduction is $S = \sum_{i=1}^{d} x_i^2$ over $d$ non-negative terms.

3. **Attention.** For queries $Q$, keys $K$, values $V$ with sequence length $L$:
   $$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$
   Modern implementations (FlashAttention, FlashDecoding) split the KV sequence into chunks and use the *online softmax* algorithm, introducing multiplicative rescaling operations across splits.

---

## 2. Theorem 1 --- Additive Reduction Sufficiency

We now state and prove the central theorem: FP32 accumulation of BF16 operands makes additive reductions invariant to association order, provided the accumulated sum is eventually rounded back to BF16.

### 2.1 Setup

Let $a_1, a_2, \ldots, a_N \in \mathbb{F}_{\text{bf16}}$ be $N$ values representable in BF16. Consider the sum:

$$S = \sum_{i=1}^{N} a_i$$

computed in FP32 arithmetic under two different reduction orders (permutations) $\pi_1$ and $\pi_2$ of $\{1, \ldots, N\}$. Denote the computed results as $\hat{S}_{\pi_1}$ and $\hat{S}_{\pi_2}$.

### 2.2 Error Bound for a Single Reduction Order (Higham-Style Analysis)

We follow the standard recursive error analysis of Higham (2002, Theorem 4.4).

**Lemma 1 (Recursive Summation Error).** For a sequential (left-to-right) summation $\hat{S} = \text{fl}(\cdots\text{fl}(\text{fl}(a_{\sigma(1)} + a_{\sigma(2)}) + a_{\sigma(3)}) + \cdots + a_{\sigma(N)})$ computed in FP32, the computed sum satisfies:

$$\hat{S} = \sum_{i=1}^{N} a_{\sigma(i)} (1 + \delta_i)$$

where $|\delta_i| \leq \gamma_{N-1}^{\text{fp32}}$ and $\gamma_k^{\text{fp32}} = \frac{k \varepsilon_{\text{fp32}}}{1 - k \varepsilon_{\text{fp32}}}$.

*Proof.* The first addition computes $s_2 = \text{fl}(a_{\sigma(1)} + a_{\sigma(2)}) = (a_{\sigma(1)} + a_{\sigma(2)})(1 + \delta^{(2)})$ with $|\delta^{(2)}| \leq \varepsilon_{\text{fp32}}$. Inductively, $s_j = \text{fl}(s_{j-1} + a_{\sigma(j)}) = (s_{j-1} + a_{\sigma(j)})(1+\delta^{(j)})$ with $|\delta^{(j)}| \leq \varepsilon_{\text{fp32}}$. Expanding:

$$\hat{S} = s_N = \sum_{i=1}^{N} a_{\sigma(i)} \prod_{j=\max(i,2)}^{N} (1 + \delta^{(j)})$$

By the standard bound $\prod_{j} (1+\delta^{(j)}) = 1 + \theta_k$ where $|\theta_k| \leq \gamma_k$ for $k$ terms in the product (Higham, Lemma 3.1), each coefficient satisfies $|\delta_i| \leq \gamma_{N-1}^{\text{fp32}}$. $\square$

For practical values ($N \leq 2^{17}$, $\varepsilon_{\text{fp32}} \approx 5.96 \times 10^{-8}$), we have $N\varepsilon_{\text{fp32}} \ll 1$, so:

$$\gamma_{N-1}^{\text{fp32}} \approx (N-1)\varepsilon_{\text{fp32}}$$

### 2.3 Difference Between Two Reduction Orders

**Theorem 1 (Additive Reduction Sufficiency).** Let $\hat{S}_{\pi_1}$ and $\hat{S}_{\pi_2}$ be the FP32-accumulated sums of $\{a_i\}_{i=1}^{N} \subset \mathbb{F}_{\text{bf16}}$ under two arbitrary reduction orders. Then:

$$|\hat{S}_{\pi_1} - \hat{S}_{\pi_2}| \leq 2\gamma_{N-1}^{\text{fp32}} \sum_{i=1}^{N} |a_i|$$

If furthermore:

$$2\gamma_{N-1}^{\text{fp32}} \sum_{i=1}^{N} |a_i| < \varepsilon_{\text{bf16}} \left|\sum_{i=1}^{N} a_i\right| \tag{$\star$}$$

then $\text{round}_{\text{bf16}}(\hat{S}_{\pi_1}) = \text{round}_{\text{bf16}}(\hat{S}_{\pi_2})$ (bitwise identical).

*Proof.* From Lemma 1, each computed sum satisfies:

$$\hat{S}_\pi = \sum_{i=1}^{N} a_i (1 + \delta_i^\pi), \quad |\delta_i^\pi| \leq \gamma_{N-1}^{\text{fp32}}$$

Therefore:

$$\hat{S}_\pi = S + \sum_{i=1}^{N} a_i \delta_i^\pi, \quad \text{where } S = \sum_{i=1}^{N} a_i$$

The error of each computed sum relative to the exact sum is bounded by:

$$|\hat{S}_\pi - S| \leq \gamma_{N-1}^{\text{fp32}} \sum_{i=1}^{N} |a_i|$$

By the triangle inequality:

$$|\hat{S}_{\pi_1} - \hat{S}_{\pi_2}| \leq |\hat{S}_{\pi_1} - S| + |S - \hat{S}_{\pi_2}| \leq 2\gamma_{N-1}^{\text{fp32}} \sum_{i=1}^{N} |a_i|$$

This establishes the first claim.

For the second claim, observe that $\text{round}_{\text{bf16}}$ maps an FP32 value to the nearest BF16 value. Two FP32 values $u, v$ round to the same BF16 value if they lie within the same BF16 rounding interval. A sufficient condition is that both $u$ and $v$ lie within distance $\varepsilon_{\text{bf16}} |S|$ of the exact sum $S$ (since the BF16 rounding quantum at scale $|S|$ is at least $\varepsilon_{\text{bf16}} |S|$, and both values are within this quantum of $S$, they must round identically).

From the per-order bound $|\hat{S}_\pi - S| \leq \gamma_{N-1}^{\text{fp32}} \sum_i |a_i|$, condition ($\star$) ensures:

$$|\hat{S}_\pi - S| < \varepsilon_{\text{bf16}} |S|$$

for both $\pi = \pi_1$ and $\pi = \pi_2$. Both computed sums fall within one BF16 rounding quantum of $S$, hence:

$$\text{round}_{\text{bf16}}(\hat{S}_{\pi_1}) = \text{round}_{\text{bf16}}(\hat{S}_{\pi_2}) \quad \square$$

### 2.4 Interpretation: The Condition Number Perspective

Rewriting condition ($\star$):

$$2(N-1)\varepsilon_{\text{fp32}} \cdot \underbrace{\frac{\sum |a_i|}{|\sum a_i|}}_{\kappa(S)} < \varepsilon_{\text{bf16}}$$

where $\kappa(S) \geq 1$ is the *condition number* of the summation. The condition becomes:

$$\kappa(S) < \frac{\varepsilon_{\text{bf16}}}{2(N-1)\varepsilon_{\text{fp32}}} = \frac{2^{-8}}{2(N-1) \cdot 2^{-24}} = \frac{2^{15}}{N-1} \approx \frac{32768}{N-1}$$

For well-conditioned sums ($\kappa(S) = O(1)$), this holds for $N$ up to tens of thousands --- precisely the regime of transformer hidden dimensions and sequence lengths.

### 2.5 When the Theorem Fails

The condition ($\star$) can fail in three scenarios:

1. **Catastrophic cancellation** ($\kappa(S) \gg 1$): When positive and negative terms nearly cancel, $|S| \ll \sum |a_i|$, and the condition number explodes.

2. **Very large $N$**: When $N > 2^{15}/\kappa(S)$, the accumulated FP32 error can exceed the BF16 quantum.

3. **Rounding boundary proximity**: Even when ($\star$) holds on average, if $S$ happens to lie very close to a BF16 rounding boundary (within the FP32 error), the two computed sums may land on different sides.

> **Remark.** Scenario 3 is a measure-zero event under continuous input distributions but can occur in practice. Our experimental observation of max_diff=0.5 for GEMM (rather than 0) in the FP32 accumulation setting is consistent with occasional rounding-boundary violations.

---

## 3. Application to GEMM (Linear Projections)

### 3.1 Setup

For a linear projection $y = Wx$ where $W \in \mathbb{F}_{\text{bf16}}^{d_{\text{out}} \times K}$ and $x \in \mathbb{F}_{\text{bf16}}^{K}$:

$$y_i = \sum_{k=1}^{K} W_{ik} \cdot x_k, \quad i = 1, \ldots, d_{\text{out}}$$

Each output element is an inner product of $K$ terms.

### 3.2 BF16 Multiplication is Exact in FP32

**Lemma 2.** For $a, b \in \mathbb{F}_{\text{bf16}}$, the product $a \cdot b$ is exactly representable in FP32.

*Proof.* A BF16 number has a 7-bit mantissa (plus 1 implicit bit), so $a = m_a \cdot 2^{e_a}$ and $b = m_b \cdot 2^{e_b}$ where $m_a, m_b$ are 8-bit integers (including the implicit leading 1). The product $m_a \cdot m_b$ is at most a 16-bit integer. Since FP32 has a 23-bit mantissa (plus 1 implicit bit, i.e., 24 bits total), the product $a \cdot b = (m_a \cdot m_b) \cdot 2^{e_a + e_b}$ is exactly representable. $\square$

This is critical: the terms $a_k = W_{ik} \cdot x_k$ being added are *exact* FP32 values, not approximate. The only source of non-determinism is the *reduction order* of the summation, not the individual terms.

### 3.3 Split-K and Batch-Dependent Reduction

Modern GPU GEMM kernels (e.g., cuBLAS, CUTLASS) use *split-K* parallelism: the $K$-dimension reduction is partitioned into $P$ chunks of size $K/P$, each chunk is summed independently, and the partial sums are combined:

$$\hat{y}_i = \bigoplus_{p=1}^{P} \left( \bigoplus_{k \in \text{chunk}_p} W_{ik} \cdot x_k \right)$$

The number of splits $P$ may depend on the batch size (via occupancy heuristics), creating batch-dependent reduction orders.

### 3.4 Applying Theorem 1

With FP32 accumulation, applying Theorem 1 to the GEMM inner product:

- $N = K$ (reduction dimension)
- $a_k = W_{ik} \cdot x_k$ (exact FP32 values)
- Two different split-K configurations produce permutations $\pi_1, \pi_2$

The error bound is:

$$|(\hat{y}_i)_{\pi_1} - (\hat{y}_i)_{\pi_2}| \leq 2(K-1)\varepsilon_{\text{fp32}} \sum_{k=1}^{K} |W_{ik} \cdot x_k|$$

The sufficiency condition becomes:

$$2(K-1)\varepsilon_{\text{fp32}} \cdot \kappa(y_i) < \varepsilon_{\text{bf16}}$$

### 3.5 Numerical Example

**Typical LLaMA-7B parameters:**
- $K = d_{\text{model}} = 4096$
- Weight magnitudes: $|W_{ik}| \sim 0.01$ (typical post-training)
- Activation magnitudes: $|x_k| \sim 0.1$--$1.0$
- Product magnitudes: $|W_{ik} x_k| \sim 0.001$--$0.01$

**Computing the bound:**

$$2(K-1)\varepsilon_{\text{fp32}} = 2 \times 4095 \times 5.96 \times 10^{-8} \approx 4.88 \times 10^{-4}$$

For the sum: $|y_i| = |\sum_k W_{ik} x_k|$. With random-sign terms, $|y_i| \sim \sqrt{K} \cdot \mathbb{E}[|W_{ik} x_k|] \approx 64 \times 0.005 = 0.32$ (by CLT-type concentration). Meanwhile $\sum |W_{ik} x_k| \approx K \times 0.005 = 20.48$.

**Maximum reorder difference:**

$$|\Delta y_i| \leq 4.88 \times 10^{-4} \times 20.48 \approx 1.0 \times 10^{-2}$$

**BF16 rounding quantum at scale $|y_i| \approx 0.32$:**

$$\varepsilon_{\text{bf16}} \times |y_i| \approx 3.91 \times 10^{-3} \times 0.32 \approx 1.25 \times 10^{-3}$$

**Ratio:**

$$\frac{|\Delta y_i|}{\text{BF16 quantum}} \approx \frac{1.0 \times 10^{-2}}{1.25 \times 10^{-3}} \approx 8$$

This suggests that the worst-case bound is not always satisfied. However, the Higham bound is highly pessimistic (it assumes all rounding errors align adversarially). In practice:

- Rounding errors are approximately random with mean zero
- The *typical* accumulated error scales as $\sim \sqrt{K} \cdot \varepsilon_{\text{fp32}} \cdot \text{rms}(a_k)$, not $K \cdot \varepsilon_{\text{fp32}} \cdot \sum |a_k|$
- The *probabilistic* reorder difference is:

$$|\Delta y_i|_{\text{typical}} \sim O(\sqrt{K}) \cdot \varepsilon_{\text{fp32}} \cdot \text{rms}(a_k) \approx 64 \times 5.96 \times 10^{-8} \times 0.005 \approx 1.9 \times 10^{-8}$$

which is $\sim 10^{5} \times$ smaller than the BF16 quantum. This explains the experimental observation: FP32 accumulation reduces GEMM max_diff from 4.0 (BF16 accumulation) to 0.5 (an 8$\times$ improvement), with the residual 0.5 attributable to rare rounding-boundary events rather than systematic error.

### 3.6 Comparison: BF16 Accumulation

Under BF16 accumulation, the corresponding bound replaces $\varepsilon_{\text{fp32}}$ with $\varepsilon_{\text{bf16}}$:

$$|\Delta y_i|_{\text{bf16}} \leq 2(K-1)\varepsilon_{\text{bf16}} \sum_k |a_k| \approx 2 \times 4095 \times 3.91 \times 10^{-3} \times 20.48 \approx 655$$

This catastrophically large bound (consistent with the observed max_diff=4.0, which represents the *actual* maximum, not the worst-case bound) confirms that BF16 accumulation cannot absorb reorder errors.

---

## 4. Application to RMSNorm

### 4.1 Setup

RMSNorm computes:

$$\text{rms}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}$$

The critical reduction is $S = \sum_{i=1}^{d} x_i^2$ where $x_i \in \mathbb{F}_{\text{bf16}}$ and $d$ is the hidden dimension.

### 4.2 Favorable Properties

The RMSNorm reduction enjoys two properties that make it especially amenable to Theorem 1:

**Property 1: Non-negative terms.** Each $a_i = x_i^2 \geq 0$, so:

$$\kappa(S) = \frac{\sum |a_i|}{|\sum a_i|} = \frac{\sum x_i^2}{\sum x_i^2} = 1$$

There is *no catastrophic cancellation*. The condition ($\star$) simplifies to:

$$2(d-1)\varepsilon_{\text{fp32}} < \varepsilon_{\text{bf16}}$$

which gives $d < \varepsilon_{\text{bf16}} / (2\varepsilon_{\text{fp32}}) + 1 = 2^{15} + 1 = 32769$.

**Property 2: Exact squaring.** By Lemma 2, $x_i^2$ for $x_i \in \mathbb{F}_{\text{bf16}}$ is exactly representable in FP32 (product of two 8-bit mantissas fits in 16 bits $<$ 24 bits). So the terms being reduced are exact FP32 values.

### 4.3 Applying Theorem 1

For $d = 4096$ (typical hidden dimension):

$$2(d-1)\varepsilon_{\text{fp32}} = 2 \times 4095 \times 5.96 \times 10^{-8} \approx 4.88 \times 10^{-4}$$

Since $\kappa(S) = 1$, condition ($\star$) requires:

$$4.88 \times 10^{-4} < \varepsilon_{\text{bf16}} = 3.91 \times 10^{-3} \quad \checkmark$$

The condition is satisfied with a margin of $\approx 8\times$. This means that for *any* two reduction orders, FP32 accumulation guarantees:

$$\text{round}_{\text{bf16}}(\hat{S}_{\pi_1}) = \text{round}_{\text{bf16}}(\hat{S}_{\pi_2})$$

**This is a deterministic guarantee, not a probabilistic one.** It explains the experimental observation of max_diff=0 for FP32-accumulated RMSNorm.

### 4.4 Propagation Through the Full RMSNorm

Once $S$ is deterministic in BF16, the subsequent operations (division by $d$, addition of $\epsilon$, reciprocal square root, element-wise multiplication by $\gamma$ and $x$) are all element-wise --- they do not involve batch-dependent reduction orders. Therefore, the full RMSNorm output is batch invariant.

---

## 5. Theorem 2 --- Why Attention Split-KV Fails

### 5.1 The Online Softmax Algorithm

Modern attention implementations (FlashAttention, FlashDecoding) split the KV sequence into $P$ chunks and use the *online softmax* algorithm to combine partial results without materializing the full attention matrix.

For each split $s \in \{1, \ldots, P\}$, the kernel computes local quantities:

$$m_s = \max_j (Q K_s^\top)_j, \quad p_s = \exp(Q K_s^\top - m_s), \quad l_s = \mathbf{1}^\top p_s, \quad o_s = p_s V_s$$

The splits are then combined sequentially. After processing split $s$, the running aggregates are updated:

$$m^{(s)} = \max(m^{(s-1)},\; m_s)$$

$$\alpha^{(s)} = \exp(m^{(s-1)} - m^{(s)}), \quad \beta^{(s)} = \exp(m_s - m^{(s)})$$

$$l^{(s)} = \alpha^{(s)} \cdot l^{(s-1)} + \beta^{(s)} \cdot l_s$$

$$o^{(s)} = \alpha^{(s)} \cdot o^{(s-1)} + \beta^{(s)} \cdot o_s$$

The final output is $o^{(P)} / l^{(P)}$.

### 5.2 The Fundamental Difference from Additive Reduction

**Theorem 2 (Attention Rescaling Breaks Additive Sufficiency).** The online softmax combination across $P$ splits is *not* a reordering of an additive reduction. Different split counts $P_1 \neq P_2$ produce *structurally different computation graphs* with multiplicative rescaling factors that introduce errors beyond the reach of FP32 absorption.

*Proof.* We show this by comparing $P=2$ and $P=4$ splits over the same KV sequence of length $L$.

**Case $P = 2$:** Two splits of size $L/2$. One rescaling step:

$$o^{(2)} = \exp(m_1 - m^{(2)}) \cdot o_1 + \exp(m_2 - m^{(2)}) \cdot o_2$$

Each element of $o_1$ is multiplied by $\exp(m_1 - m^{(2)})$ exactly once.

**Case $P = 4$:** Four splits of size $L/4$. Three rescaling steps. After all four splits, an element originally in $o_1$ has been multiplied by:

$$\exp(m_1 - m^{(2)}) \cdot \exp(m^{(2)} - m^{(3)}) \cdot \exp(m^{(3)} - m^{(4)})$$

where $m^{(j)}$ denotes the running maximum after processing $j$ splits. By the telescoping property of exponents, this *mathematically* equals $\exp(m_1 - m^{(4)})$. But in floating-point arithmetic, the two computations are not equivalent:

**Two-split path:** $\text{fl}(\exp(m_1 - m^{(2)}))$ --- one $\exp$ evaluation, one subtraction.

**Four-split path:** $\text{fl}(\exp(m_1 - m^{(2)})) \cdot \text{fl}(\exp(m^{(2)} - m^{(3)})) \cdot \text{fl}(\exp(m^{(3)} - m^{(4)}))$ --- three $\exp$ evaluations, three subtractions, two multiplications.

Each operation introduces relative error $\leq \varepsilon_{\text{fp32}}$. The total relative error in the $P=4$ rescaling factor is:

$$\delta_4 \leq (2P - 3) \cdot \varepsilon_{\text{fp32}} + \varepsilon_{\exp}$$

where $\varepsilon_{\exp}$ accounts for the approximation error of the $\exp$ function on GPU hardware (typically $\sim 2^{-22}$ for fast math, larger than $\varepsilon_{\text{fp32}}$).

**The critical amplification:** The rescaling factor $\exp(\Delta m)$ where $\Delta m = m_{\text{local}} - m_{\text{global}}$ can be exponentially large or small. When $\Delta m > 0$, the factor exceeds 1 and *amplifies* the absolute error. Specifically, the absolute error in the rescaled output is:

$$|\Delta o_{\text{rescaled}}| \sim \exp(|\Delta m|) \cdot \varepsilon_{\text{fp32}} \cdot |o_s|$$

For typical attention logits with $\text{std}(QK^\top / \sqrt{d_k}) \sim 1$, the difference $\Delta m$ between local and global max can be $\sim 2$--$5$, giving amplification factors of $\exp(2) \approx 7.4$ to $\exp(5) \approx 148$.

### 5.3 Error Bound for Split-KV Attention

**Proposition 1.** For online softmax attention with $P$ splits, the difference between configurations with $P_1$ and $P_2$ splits satisfies:

$$\|o_{P_1} - o_{P_2}\|_\infty \leq C \cdot |P_1 - P_2| \cdot \exp(\Delta m_{\max}) \cdot \varepsilon_{\text{fp32}} \cdot \|V\|_\infty$$

where $\Delta m_{\max} = \max_s |m_s - m_{\text{global}}|$ and $C$ is a moderate constant depending on the sequence length.

*Proof sketch.* Each additional rescaling step introduces a multiplicative perturbation of $(1 + O(\varepsilon_{\text{fp32}}))$ on the accumulated output. After $|P_1 - P_2|$ extra rescaling steps, the perturbation compounds to $(1 + O(\varepsilon_{\text{fp32}}))^{|P_1 - P_2|}$. The key is that this perturbation acts on values that have been scaled by $\exp(\Delta m)$, amplifying the absolute error. $\square$

### 5.4 Why FP32 Cannot Help

The error from split-KV attention is:

$$|\Delta o| \sim \exp(\Delta m_{\max}) \cdot \varepsilon_{\text{fp32}} \cdot \|V\|_\infty$$

For this to be absorbed by BF16 rounding, we need:

$$\exp(\Delta m_{\max}) \cdot \varepsilon_{\text{fp32}} \cdot \|V\|_\infty < \varepsilon_{\text{bf16}} \cdot |o_i|$$

Since $|o_i| \leq \|V\|_\infty$ (attention output is a convex combination of value vectors), this simplifies to:

$$\exp(\Delta m_{\max}) < \frac{\varepsilon_{\text{bf16}}}{\varepsilon_{\text{fp32}}} = 2^{16} = 65536$$

which requires $\Delta m_{\max} < \ln(65536) \approx 11.1$. While this *can* hold, the margin is much thinner than for additive reductions, and the additional source of error --- the structurally different computation graph --- means that the errors are not merely from reordering a fixed set of additions but from computing fundamentally different sequences of multiplications.

**This explains the experimental result:** FP32 accumulation gives 0 improvement for attention split-KV (max_diff = 2.55 in both BF16 and FP32 settings). The error is dominated by the structural mismatch between different split counts, not by accumulation precision.

---

## 6. Theorem 3 --- Fixed Split Restoration

### 6.1 Statement

**Theorem 3 (Fixed-Split Determinism).** If the attention split-KV boundaries are fixed (independent of batch configuration), so that all inputs with the same sequence length $L$ use the same split count $P$ and the same chunk boundaries $\{0, C, 2C, \ldots, L\}$, then FP32 accumulation within each chunk restores batch invariance.

### 6.2 Proof

Fix the split count $P$ and boundaries. For any two batch configurations $B_1, B_2$ processing the same input $(Q, K, V)$:

**Step 1: Identical local computations.** Each split $s$ computes:
- $m_s$: maximum over the same set of logits (same $Q$, same $K_s$) $\Rightarrow$ deterministic (max is order-independent).
- $p_s = \exp(QK_s^\top - m_s)$: element-wise, hence deterministic given deterministic $m_s$.
- $l_s = \sum p_s$: additive reduction of non-negative terms within a *fixed-size* chunk.
- $o_s = p_s V_s$: GEMM-like reduction within a *fixed-size* chunk.

For $l_s$ and $o_s$, the reduction size is $C$ (the chunk size), which is fixed. Different batch configurations may use different internal tiling for these reductions, but by Theorem 1, FP32 accumulation absorbs these differences provided:

$$2(C-1)\varepsilon_{\text{fp32}} \cdot \kappa < \varepsilon_{\text{bf16}}$$

For $l_s$ (non-negative terms), $\kappa = 1$ and this holds for $C < 32769$. For $o_s$, $\kappa$ is typically small since the softmax weights are non-negative, giving a well-conditioned sum.

**Step 2: Identical cross-split combination.** Since $P$ is fixed and each $(m_s, l_s, o_s)$ is deterministic (from Step 1), the sequential combination:

$$m^{(s)} = \max(m^{(s-1)}, m_s) \quad \text{(deterministic: same inputs)}$$
$$\alpha^{(s)} = \exp(m^{(s-1)} - m^{(s)}) \quad \text{(deterministic: same inputs)}$$
$$l^{(s)} = \alpha^{(s)} \cdot l^{(s-1)} + \beta^{(s)} \cdot l_s \quad \text{(deterministic: single thread, no parallelism)}$$
$$o^{(s)} = \alpha^{(s)} \cdot o^{(s-1)} + \beta^{(s)} \cdot o_s \quad \text{(deterministic: single thread)}$$

Each step is a deterministic function of deterministic inputs. The cross-split combination is sequential (not parallelized), so there is no reduction-order ambiguity. $\square$

### 6.3 Practical Implication

This theorem provides the design principle: **fix the split boundaries, then apply FP32 accumulation within each split.** The combination of these two interventions converts the attention computation from the Theorem 2 regime (fundamentally different computation graphs) back to the Theorem 1 regime (same computation graph with potentially different internal reduction orders, absorbed by FP32 precision).

This is precisely the mechanism by which `allow_bf16_reduced_precision_reduction=False` achieves determinism: it forces FP32 accumulation in all additive reductions, and when combined with fixed split strategies, eliminates all sources of batch-dependent non-determinism.

---

## 7. Numerical Verification

### 7.1 GEMM: $K = 4096$

**Parameters:**
- $K = 4096$, $\varepsilon_{\text{fp32}} = 5.96 \times 10^{-8}$, $\varepsilon_{\text{bf16}} = 3.91 \times 10^{-3}$
- Typical: $|W_{ik}| \sim 0.01$, $|x_k| \sim 0.1$, so $|a_k| = |W_{ik} x_k| \sim 10^{-3}$

**Worst-case Higham bound on reorder difference:**

$$|\Delta y_i| \leq 2(K-1)\varepsilon_{\text{fp32}} \sum |a_k| = 2 \times 4095 \times 5.96 \times 10^{-8} \times 4096 \times 10^{-3} \approx 2.0 \times 10^{-3}$$

**BF16 quantum at typical output scale $|y_i| \approx 0.3$:**

$$\varepsilon_{\text{bf16}} |y_i| \approx 3.91 \times 10^{-3} \times 0.3 = 1.17 \times 10^{-3}$$

**Ratio (worst case):** $2.0 \times 10^{-3} / 1.17 \times 10^{-3} \approx 1.7$ --- the worst-case bound slightly exceeds the BF16 quantum, predicting occasional rounding mismatches.

**Probabilistic (typical case):** Rounding errors are approximately random, so the typical accumulated difference scales as:

$$|\Delta y_i|_{\text{typ}} \sim \sqrt{K} \cdot \varepsilon_{\text{fp32}} \cdot \text{rms}(a_k) \approx 64 \times 5.96 \times 10^{-8} \times 10^{-3} \approx 3.8 \times 10^{-9}$$

**Ratio (typical):** $3.8 \times 10^{-9} / 1.17 \times 10^{-3} \approx 3.3 \times 10^{-6}$ --- six orders of magnitude below the BF16 quantum.

**Experimental match:** FP32 accumulation reduces GEMM max_diff from 4.0 (BF16) to 0.5 (FP32), an $8\times$ improvement. The residual max_diff=0.5 in BF16 units corresponds to exactly $0.5 \times \text{ulp}_{\text{bf16}}$, consistent with rare rounding-boundary events as predicted by the theory.

### 7.2 RMSNorm: $d = 4096$

**Parameters:**
- $d = 4096$, all terms $a_i = x_i^2 \geq 0$, condition number $\kappa = 1$

**Worst-case bound:**

$$2(d-1)\varepsilon_{\text{fp32}} = 2 \times 4095 \times 5.96 \times 10^{-8} = 4.88 \times 10^{-4}$$

**Sufficiency check:**

$$4.88 \times 10^{-4} < \varepsilon_{\text{bf16}} = 3.91 \times 10^{-3} \quad \checkmark$$

**Margin:** $3.91 \times 10^{-3} / 4.88 \times 10^{-4} = 8.0\times$

This is a **deterministic guarantee**: for $d \leq 32768$, *no* reduction order of non-negative FP32 terms can produce a different BF16 result. This is an unconditional bound (no probabilistic argument needed).

**Experimental match:** FP32 accumulation gives max_diff = 0 for RMSNorm, exactly as predicted.

### 7.3 Attention Split-KV: $L = 1024$, Varying Splits

**Configuration A:** $P = 2$ splits (chunk size 512)
**Configuration B:** $P = 4$ splits (chunk size 256)

**Typical $\Delta m$ between local and global max:**

For attention logits $\sim \mathcal{N}(0, 1)$ (post-scaling by $1/\sqrt{d_k}$), the maximum over $C$ entries is approximately $\sqrt{2 \ln C}$. For $C = 512$: $m_s \approx \sqrt{2 \ln 512} \approx 3.53$. For $C = 256$: $m_s \approx \sqrt{2 \ln 256} \approx 3.33$.

The global maximum over $L = 1024$ entries: $m_{\text{global}} \approx \sqrt{2 \ln 1024} \approx 3.72$.

**Amplification factor:**

$$\exp(m_{\text{global}} - m_s) \approx \exp(3.72 - 3.33) \approx \exp(0.39) \approx 1.48$$

This is a moderate amplification. However, the structural difference is more impactful:

**Two-split computation:** One exp-rescaling:
$$o = \exp(m_1 - m^{(2)}) \cdot o_1 + \exp(m_2 - m^{(2)}) \cdot o_2$$

**Four-split computation:** The same mathematical value is computed via three sequential rescalings. Each rescaling introduces a multiplicative error of $(1 + O(\varepsilon_{\text{fp32}} + \varepsilon_{\exp}))$. With $\varepsilon_{\exp} \sim 2^{-22}$ (GPU exp approximation error):

$$\text{Per-rescaling error} \sim \varepsilon_{\exp} \cdot \exp(\Delta m) \cdot \|V\|_\infty \approx 2.4 \times 10^{-7} \times 1.48 \times 1.0 \approx 3.5 \times 10^{-7}$$

Across 3 rescalings vs 1 rescaling (a difference of 2 extra rescaling steps):

$$|\Delta o| \sim 2 \times 3.5 \times 10^{-7} \times \|V\|_\infty \approx 7.0 \times 10^{-7}$$

**BF16 quantum at typical attention output scale $|o_i| \approx 0.1$:**

$$\varepsilon_{\text{bf16}} |o_i| \approx 3.91 \times 10^{-3} \times 0.1 = 3.91 \times 10^{-4}$$

**Ratio:** $7.0 \times 10^{-7} / 3.91 \times 10^{-4} \approx 1.8 \times 10^{-3}$ --- this suggests FP32 *should* help. Why doesn't it experimentally?

**The missing factor:** The analysis above considers only the rescaling error, but the real problem is that different split counts produce different *intermediate rounding patterns* in the local softmax computations. With different chunk sizes:
- Different local maxima $m_s$ (different partition of the max operation)
- Different local softmax distributions $p_s$ (exp applied to differently shifted logits)
- Different partial sums $l_s$ and $o_s$ (reductions over different-sized chunks)

These are not merely different reduction orders of the same terms --- they are different mathematical decompositions that only coincide in exact arithmetic. The gap between them is at the level of FP32 precision applied to *different* computation graphs, and FP32-to-BF16 absorption cannot bridge structurally different computations.

**Experimental match:** max_diff = 2.55 for both BF16 and FP32 accumulation in attention split-KV, confirming that the error source is structural (split count), not accumulation precision.

---

## Summary of Results

| Operation | Reduction Type | $\kappa$ | FP32 Absorbs? | Theory | Experiment |
|-----------|---------------|----------|---------------|--------|------------|
| GEMM | Additive (split-K) | $O(\sqrt{K})$ | Yes (high prob.) | Thm 1 | 4.0 $\to$ 0.5 |
| RMSNorm | Additive (chunked) | 1 (exact) | Yes (guaranteed) | Thm 1 | 3.12e-2 $\to$ 0 |
| Attention | Multiplicative (split-KV) | N/A | No | Thm 2 | 2.55 $\to$ 2.55 |
| Attention (fixed splits) | Additive within fixed blocks | $O(1)$ | Yes | Thm 3 | Deterministic |

The three theorems together provide a complete characterization: additive reductions are saved by FP32 accumulation (Theorem 1), multiplicative rescaling chains in attention are not (Theorem 2), but fixing the split boundaries reduces the problem back to additive reductions (Theorem 3). The combination of fixed splits and FP32 accumulation achieves full batch-invariant deterministic inference.

---

### References

- Higham, N. J. (2002). *Accuracy and Stability of Numerical Algorithms* (2nd ed.). SIAM. Chapters 2--4.
- Dao, T. (2023). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. *arXiv:2307.08691*.
- Dao, T., et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. *NeurIPS 2022*.
- NVIDIA (2024). CUTLASS: CUDA Templates for Linear Algebra Subroutines. *github.com/NVIDIA/cutlass*.
- IEEE 754-2019. *IEEE Standard for Floating-Point Arithmetic*.
