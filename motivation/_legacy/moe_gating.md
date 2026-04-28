# MoE Gating Determinism Motivation Tests

## 根因分析
- Gate Input: 100% 相同 (0.0000000000e+00)
- Raw Router Logits: 出现差异(6.2500000000e-02) — 根因
- Softmax Probabilities: 差异被放大 (1.1757239699e-03)
- Top-k Expert Indices: 3.1% 的 expert 选择改变
## 结论：
- 根因：Gate computation (Linear layer) 的 matmul 操作不是 batch-invariant
- 影响链：Raw Logits → Softmax → Top-k Selection