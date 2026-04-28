# Experiment E: Downstream Impact of Non-Determinism

Measures how batch-composition non-determinism in BF16 matmul affects
practical ML applications: RL reward signals, knowledge distillation, and MoE routing.

- **Model**: Llama-3.1-8B-Instruct
- **GPU**: NVIDIA RTX A6000
- **Precision**: BF16 weights, comparing BF16 vs FP32 accumulation
- **Batch comparison**: bs=1 (reference) vs bs=8 (continuous batching sim)

---
## Experiment 1: Reward Signal Variance (RL Proxy)

**Motivation**: In RLHF/GRPO, the reward for a generated sequence depends on model logits.
If logits change with batch composition, the reward signal has artificial noise.

**Method**: For 30 prompts, compute log-probability of each position's top-1 prediction
at bs=1 vs bs=8. Measure |logprob_bs1 - logprob_bs8| per position.

| Metric | BF16 (default) | FP32 accum |
|--------|----------------|------------|
| Mean \|logprob diff\| | 2.301480e-02 | 1.494682e-02 |
| Median \|logprob diff\| | 1.010048e-02 | 6.726265e-03 |
| Max \|logprob diff\| | 1.194340e-01 | 8.393586e-02 |
| Frac positions nonzero | 1.0000 | 1.0000 |
| Paired t-test (t-stat) | 18.0482 | 17.3108 |
| Paired t-test (p-value) | 2.6026e-17 | 7.8991e-17 |
| Total positions | 357 | 357 |

**Finding**: The reward signal difference is statistically significant (p < 0.05) under BF16 defaults.
This means batch composition injects systematic noise into RL reward signals.

---
## Experiment 2: Distillation Signal Corruption

**Motivation**: In knowledge distillation, the teacher's soft logits are the training signal.
If these change with batch composition, the student learns inconsistent targets.

**Method**: For 30 prompts, compute token-level KL(softmax(logits_bs1/T) || softmax(logits_bs8/T))
with T=1.

| Metric | BF16 (default) | FP32 accum |
|--------|----------------|------------|
| Mean KL per position | 6.291992e-04 | 4.587692e-04 |
| Median KL per position | 4.180758e-04 | 3.520655e-04 |
| Max KL per position | 3.407942e-03 | 2.471533e-03 |
| Frac positions KL > 1e-6 | 0.9244 | 0.9272 |
| Total positions | 357 | 357 |

**Finding**: FP32 accumulation reduces mean KL divergence by 27.1% compared to BF16 defaults.

---
## Experiment 3: MoE Near-Tie Analysis (Synthetic)

**Motivation**: MoE models select experts via top-k on router logits. Near-tie logits
(small gap between k-th and (k+1)-th) are vulnerable to expert selection flips.

**Method**: Synthetic MoE with hidden_dim=2048, 128 experts, top-8 routing.
Compare expert selections at bs=1 vs bs={4,8,16} over 20 trials x 50 tokens each.

### Near-Tie Prevalence (bs=1 reference)

| Threshold (tau) | BF16 fraction | FP32 fraction |
|-----------------|---------------|---------------|
| gap < 0.1 | 1.0000 (50/50) | 1.0000 (50/50) |
| gap < 0.01 | 1.0000 (50/50) | 1.0000 (50/50) |
| gap < 0.001 | 1.0000 (50/50) | 1.0000 (50/50) |

### Expert Selection Flip Rate

| Batch Size | BF16 flip rate | FP32 flip rate |
|------------|----------------|----------------|
| bs=4 | 0.0600 (60/1000) | 0.0000 (0/1000) |
| bs=8 | 0.0600 (60/1000) | 0.0000 (0/1000) |
| bs=16 | 0.0400 (40/1000) | 0.0000 (0/1000) |

---
## Summary

| Application | BF16 Impact | FP32 Accum Impact | Practical Concern |
|-------------|------------|-------------------|-------------------|
| RL Reward Signal | 100.0% positions affected | 100.0% positions affected | High / High |
| KD Soft Targets | 92.4% positions KL>1e-6 | 92.7% positions KL>1e-6 | High / High |
| MoE Expert Routing | 6.0% expert flips (bs=8) | 0.0% expert flips (bs=8) | High / Low |
