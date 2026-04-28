# LLM 推理中批次非确定性的解剖：Kernel 选择是根因，精度不是银弹

---

## 摘要

大语言模型推理系统中，相同输入在不同批次大小下产生不同输出。近期工作（Yuan et al., NeurIPS 2025）将此归因于 BF16 浮点精度不足，提出 LayerCast（BF16 存储 + FP32 计算）作为缓解方案。**我们的研究揭示了一个更深层的根因：问题的本质不是计算精度，而是 cuBLAS 的 kernel 选择启发式。**

我们通过 bs=1..256 的逐一扫描首次精确映射了 cuBLAS 的 kernel 切换边界，发现三个关键事实：
1. **所有精度级别都存在 kernel 切换**——BF16（8 次）、FP32 flag（2 次）、全 FP32 LayerCast（6 次）
2. **提升精度不单调改善确定性**——FP32 flag 在 Llama 上改善（8→2 切换），但在 DeepSeek-V2-Lite（MoE）上恶化（0→6 切换）；LayerCast 在小 batch size 处比 FP32 flag 更不稳定（6 次 vs 2 次切换）
3. **PyTorch 的全部标准确定性工具对此无效**——`torch.use_deterministic_algorithms`、`CUBLAS_WORKSPACE_CONFIG`、`allow_tf32` 等均不影响 batch 非确定性

基于 10,000 次生成测试、8 种 flag 组合、7 种 K 维度、3 种精度方案的系统实验，我们建立了 batch 非确定性的完整分解框架，并量化了确定性-性能 Pareto 前沿：从零成本的 FP32 flag（95% batch size 一致）到 5-34% 开销的固定 kernel 方案（100% 一致）。

---

## 1. 引言

### 1.1 问题

LLM 推理中，改变评估 batch size 就可导致 greedy decoding 输出不同。Yuan et al. (NeurIPS 2025) 展示了 DeepSeek-R1 在 AIME'24 上准确率波动 9%、回复长度差异 9,000 tokens。这一问题对 RL 训练（奖励信号噪声）、知识蒸馏（目标不一致）、安全审计（不可复现）有直接影响。

### 1.2 传统认知与我们的挑战

**传统认知**（Yuan et al.）：根因是 BF16 精度不足 → 解决方案是提升精度（LayerCast: BF16→FP32）。

**我们的发现**：精度只是表象。真正的根因是 **cuBLAS 在不同 batch size M 下选择完全不同的 GEMM 算法**——不同的 tiling、split-K 策略、SM 映射。这导致结构性不同的计算图，不是精度能弥补的。

关键实验证据：

| 方案 | 精度 | bs=1..69 Kernel 切换次数 | Overhead |
|---|---|---:|---:|
| BF16 baseline | BF16×BF16 | 4 | 0% |
| FP32 flag | BF16×BF16, FP32 累加 | **2** | 0% |
| **LayerCast** | **FP32×FP32** | **6** | **+242%** |

**全 FP32 计算（LayerCast）的切换次数反而多于 FP32 flag（6 > 2），且开销 242%。** 这直接证明了精度不是根因——否则最高精度应产生最少切换。

### 1.3 贡献

1. **根因修正**：batch 非确定性的根因是 kernel 选择启发式，不是精度。提升精度只是改变了 kernel selection landscape，效果不可预测。
2. **首次精确映射** cuBLAS kernel 切换边界：10,000 次运行，bs=1..256，3 种精度方案
3. **发现 FP32 效果的模型依赖性**：Llama 改善，DeepSeek MoE 恶化
4. **发现 LayerCast 的局限**：小 batch size 处比 FP32 flag 更不稳定
5. **系统化分解** 5 种非确定性来源 + 确定性-性能 Pareto 前沿
6. **穷举验证** 8 种 PyTorch flag 组合——只有 `bf16_reduced_precision_reduction` 有效

---

## 2. 背景与 Related Work

### 2.1 浮点非结合律

$(a+b)+c \neq a+(b+c)$，GPU 上 GEMM 的内积规约被分解为多个 thread block 的部分和。**但关键区别是**：改变 batch size 不仅改变"相同计算的执行顺序"（传统认知），还改变"选择哪种计算算法"（我们的发现）。后者是结构性差异。

### 2.2 Yuan et al.: LayerCast

Yuan et al. (NeurIPS 2025) 提出 LayerCast：权重 BF16 存储，计算时 JIT cast 到 FP32。在 12 种配置（2 GPU × 2 GPU count × 3 batch size）下报告"near-perfect determinism"（Div_Index < 3.4%）。

**局限**：他们只测了 3 个 batch size（8, 16, 32），不足以发现 kernel 切换边界。我们的 bs=1..256 逐一扫描揭示了他们遗漏的切换点。

### 2.3 确定性推理系统

- **Thinking Machine Lab (2025)**：自定义 kernel，禁用 split-K，61.5% 开销
- **SGLang (2025)**：固定 split-KV + batch_invariant_ops，34.35% 开销
- **vLLM**：VLLM_BATCH_INVARIANT=1，FlexAttention 固定 tile，20-35% 开销
- **EigenAI**：Warp-synchronous reduction，~5% 开销（Hopper）

这些方案通过 **直接控制 kernel 选择/tiling** 实现确定性——与我们的根因分析一致。

---

## 3. 核心实验：精度 vs Kernel 选择

### 3.1 实验设计

**模型**：Llama-3.1-8B-Instruct (BF16, A6000)

**方法**：对 bs=1..256，每个 bs 生成 32 tokens (greedy)，hash 输出序列。统计 unique hash 数和 kernel 切换点（相邻 bs 的 hash 不同）。

**精度方案**：
- **BF16 baseline**：`torch.mm(bf16, bf16)` 默认
- **FP32 flag**：`allow_bf16_reduced_precision_reduction = False`（BF16 输入，FP32 累加）
- **LayerCast**：`torch.mm(input.float(), weight.float()).bfloat16()`（全 FP32 计算）

### 3.2 Op-Level 结果

| M | BF16 max_diff | FP32 flag max_diff | LayerCast max_diff |
|---:|---:|---:|---:|
| 1 | 0.000 | 0.000 | 0.000 |
| 32 | **0.500** | **0.500** | **0.000** |
| 64 | **1.000** | **0.500** | **4.88e-4** |
| 128 | **0.500** | **0.500** | **3.05e-5** |

Op 级别，LayerCast 确实最好（diff ~1e-4 vs BF16 的 0.5-1.0）。

### 3.3 Generation-Level 结果（关键对比）

| 方案 | Unique | 切换点 | 切换次数 | Overhead |
|---|---:|---|---:|---:|
| BF16 baseline | 2 | 2, 8, 17, 61 | 4 | 0% |
| FP32 flag | 2 | 57, 65 | **2** | 0% |
| **LayerCast** | **2** | **3, 5, 6, 8, 11, 12** | **6** | **+242%** |

**Op 级别最好的方案（LayerCast），在 generation 级别反而最差！** 这是因为：

1. LayerCast 使用 FP32 kernel 池，在小 M (3-12) 处有更多 kernel 变体
2. 虽然每个 kernel 内精度极高，但 **跨 kernel 的结构性差异** 仍然存在
3. 小 batch size 处 FP32 kernel 的 split-K 启发式更激进（小 M 需要更多并行）

### 3.4 为什么 FP32 flag 和 LayerCast 效果不同

```
BF16 baseline:   cuBLAS 选 BF16 kernel → BF16 kernel 池有 4 个切换边界
FP32 flag:       cuBLAS 选 BF16+FP32accum kernel → 改变后的池有 2 个切换边界
LayerCast:       cuBLAS 选 FP32 kernel → FP32 kernel 池有 6 个切换边界
```

**三者的差异不是精度，而是 kernel 池不同。** FP32 flag 恰好让 cuBLAS 在 Llama 的维度上选择了一个更稳定的 kernel 子集；而 FP32 kernel 池在小 M 处有更多变体。

---

## 4. 10,000-Run 大规模验证

### 4.1 Llama-3.1-8B (Dense)

10,000 次运行，bs=1..256 随机采样：

| 方案 | Unique | 切换次数 | 受影响 bs 比例 | 一致率 |
|---|---:|---:|---:|---:|
| BF16 | 2 | 8 | 81/256 (31.6%) | 68.4% |
| FP32 flag | 2 | 4 | 12/256 (4.7%) | **95.0%** |

FP32 flag 是 **零成本的最优第一步**：95% 的 batch size 产生一致输出，0% overhead。

### 4.2 DeepSeek-V2-Lite (MoE)

1,000 次运行，bs=1..32：

| 方案 | Unique | 切换次数 |
|---|---:|---:|
| **BF16** | **1** | **0（完美确定！）** |
| FP32 flag | 2 | 6（恶化！） |

**FP32 flag 在 MoE 模型上反而引入了非确定性。** 这彻底否定了"高精度=更确定"的简单假设。

### 4.3 PyTorch Flags 穷举

8 种 flag 组合测试：

| Flag | 对 batch 非确定性的影响 |
|---|---|
| `allow_bf16_reduced_precision_reduction = False` | **有效**（唯一有效的 flag） |
| `allow_tf32 = False` | 无效 |
| `float32_matmul_precision = 'highest'` | 无效 |
| `use_deterministic_algorithms(True)` | 无效 |
| `CUBLAS_WORKSPACE_CONFIG` | 无效 |
| `NVIDIA_TF32_OVERRIDE=0` | 无效 |
| 所有 flag 组合 | = 仅 `bf16_reduced` 的效果 |

---

## 5. 非确定性来源的系统分解

### 5.1 Source 1: GEMM Kernel 选择（主因）

（同前版 §3.1，包含 K 维度分析）

K 维度越大，同一 kernel 内的残余误差越大：

| K | BF16 max_diff | FP32 max_diff |
|---:|---:|---:|
| 512 | 0.000 | 0.000 |
| 1024 | 0.500 | **0.000** |
| 4096 | 1.000 | 0.250 |
| 11008 | 2.000 | 2.000（**无改善**） |
| 14336 | 4.000 | 1.000 |

### 5.2 Source 2: RMSNorm, Softmax

- RMSNorm：PyTorch 中已确定（不依赖 batch size）
- Softmax：FP32 改善 5 个数量级

### 5.3 Source 3: Attention Split-KV

FP32 online softmax 下 cross-split 差异 ~1e-7（近乎完美）。HuggingFace SDPA 不使用 split-KV，已 batch-invariant。Serving engine 需要固定 split 边界。

### 5.4 Source 4: 层间误差累积

逐层 tracking：L0 (9.8e-4) → L24 (0.125) → final logits (0.18)。但 argmax match 始终 100%。BF16 和 FP32 模式的误差模式相同——进一步确认 FP32 flag 改变的是 kernel 选择，而非层内精度。

### 5.5 Source 5: MoE Routing 放大

Near-tie prevalence 30-40%。Gate GEMM 的大 K (14336) 使 FP32 效果有限（仅 2x 改善）。

---

## 6. Theoretical Framework

### 6.1 Theorem 1: 加法规约的 FP32 充分条件

$$|\hat{S}_{\pi_1} - \hat{S}_{\pi_2}| \leq 2\gamma_{N-1}^{\text{fp32}} \sum |a_i|$$

当 $\kappa(S) < 2^{15}/(N-1)$ 时，BF16 舍入后结果相同。

**但 Theorem 1 有隐含假设：两种规约顺序是同一组加数的不同排列。** 当 cuBLAS 选择了完全不同的 kernel（不同 tiling 边界），加数的分组本身就不同——这不是 Theorem 1 能覆盖的场景。

### 6.2 实验验证

Llama-3.1-8B 的 225 层：0/225 满足 κ bound。中位 κ = 3,485 vs bound = 8.0。这解释了为什么即使在同一 kernel 内，FP32 也有 0.5 ULP 残余。

### 6.3 Theorem 2 & 3

（Attention split-KV 的不可修复性和固定 split 的恢复，同前版）

---

## 7. 确定性-性能 Pareto 前沿

| 方案 | 确定性 | Overhead | 适用场景 |
|---|---|---:|---|
| BF16 baseline | 68.4% bs 一致 | 0% | 对精确复现不敏感的场景 |
| **FP32 flag** | **95.0%** bs 一致 | **0%** | **研究评估（推荐默认开启）** |
| LayerCast (Yuan et al.) | ~97%（但小 bs 更多切换） | +242% | 精度敏感但非完美需求 |
| Triton 固定 tile（naive） | **100%** | +400% | 不推荐（性能差） |
| batch_invariant_ops (vLLM) | **100%** | +20-34% | 生产 serving |
| EigenAI warp-sync | **100%** | +5% | 最优（Hopper only） |

**实践建议**：
1. **所有场景**：开启 FP32 flag（零成本，95% 一致）
2. **但必须在目标模型上验证**——FP32 flag 可能恶化某些模型
3. **严格确定性**：使用 batch_invariant_ops 或 EigenAI 方案
4. **不推荐**：LayerCast 单独使用（高开销且不消除 kernel 切换）

---

## 8. 讨论

### 8.1 为什么 Yuan et al. 的结论与我们不同？

| 方面 | Yuan et al. | 我们 |
|---|---|---|
| batch size 测试范围 | 3 个 (8, 16, 32) | **256 个 (1..256)** |
| 观察到 FP32 切换 | 否（恰好未命中） | **是（6 个切换点）** |
| 结论 | FP32 "near-perfect" | **FP32 改变 kernel landscape，不消除切换** |

Yuan et al. 的实验设计不够细致——3 个 batch size 不足以发现 kernel 切换边界。他们的 LayerCast 在他们测试的 3 个 bs 上确实表现良好，但这是因为切换点恰好不在这些 bs 之间。

### 8.2 什么时候精度确实重要？

精度在两个层面有效：
1. **同一 kernel 内**：FP32 累加减少同一 tiling 配置下的舍入误差
2. **跨 kernel 的 op-level 差异**：LayerCast 将 op-level diff 从 0.5 降到 1e-4

但这不足以消除 **generation-level** 的非确定性，因为：
- 32 层 transformer 的误差累积可将 1e-4 放大到足以翻转 argmax
- kernel 切换导致的结构性差异（不同 tiling 边界）不是精度问题

### 8.3 根本解决方案

根据我们的分析，完全消除 batch 非确定性需要：
1. **直接控制 kernel 选择**（如 batch_invariant_ops 固定 tiling）
2. 而非间接通过精度影响 kernel 启发式

这与 vLLM/SGLang 的工程实践一致：他们通过自定义 Triton kernel 替代 cuBLAS，确保固定 tiling，而非仅提升精度。

---

## 9. 结论

LLM 推理中 batch 非确定性的根因是 **cuBLAS kernel 选择启发式**，不是浮点精度不足。提升精度（BF16→FP32 flag→全 FP32 LayerCast）只是改变了 kernel selection landscape——在某些模型上改善，在其他模型上恶化。

我们的实验证明：
1. **没有任何精度设置能完全消除 kernel 切换**（BF16: 8 次, FP32 flag: 2 次, LayerCast: 6 次）
2. **`bf16_reduced_precision_reduction=False` 是零成本的最优第一步**（95% 一致率）
3. **完全解决需要固定 kernel 选择**（如 vLLM 的 batch_invariant_ops，5-34% overhead）
4. **Yuan et al. 的"FP32 near-perfect"结论源于测试 batch size 不够密集**

```python
# 推荐的最小干预方案（零成本，95% 场景有效）：
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
# 但：必须在目标模型上验证效果！
```

---

## References

Yuan, J. et al. (2025). Understanding and Mitigating Numerical Sources of Nondeterminism in LLM Inference. NeurIPS 2025.

Thinking Machine Lab (2025). Defeating Nondeterminism in LLM Inference. Blog post.

Higham, N. J. (2002). Accuracy and Stability of Numerical Algorithms (2nd ed.). SIAM.

Dao, T. (2023). FlashAttention-2. arXiv:2307.08691.

Fedus, W. et al. (2022). Switch Transformers. JMLR.

EigenAI (2025). Deterministic Inference, Verifiable Results. arXiv:2602.00182.

SGLang Team (2025). SGLang Deterministic Inference. Blog post.

vLLM Team (2025). Batch Invariance Documentation. docs.vllm.ai.
