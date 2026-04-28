# LLM 推理中批次非确定性的解剖：cuBLAS Kernel 选择才是真正的根因

---

## 摘要

大语言模型（LLM）推理系统中，相同输入在不同批次大小下产生不同输出，这一现象被普遍归因于浮点非结合律和有限精度。近期工作（Yuan et al., NeurIPS 2025）提出 LayerCast（BF16 存储 + FP32 计算）作为缓解方案。然而，我们的研究揭示了一个更深层的根因：**问题的本质不是计算精度，而是 cuBLAS 的 kernel 选择启发式**。当批次维度 M 改变时，cuBLAS 会选择完全不同的 GEMM 算法（不同的 tiling、split-K 策略），产生结构性不同的计算图——这不是精度能修复的。

我们通过精细扫描 bs=1..256 首次精确映射了 cuBLAS 的 kernel 切换边界，发现：(1) BF16 默认模式下有 **8 个切换点**，31.6% 的 batch size 产生不同输出；(2) FP32 accumulation flag 将切换点减少到 **4 个**（4.7% batch size 受影响），但不能完全消除；(3) **FP32 flag 的效果是模型依赖的**——在 Llama-3.1-8B 上改善（8→4 切换），但在 DeepSeek-V2-Lite（MoE）上反而恶化（0→6 切换）。这一发现挑战了"提高精度即可解决非确定性"的传统认知。

我们进一步系统分解了 LLM 推理中的 5 种非确定性来源（GEMM kernel 选择、reduction ops、attention split-KV、层间误差累积、MoE routing 放大），对每种来源分析 FP32 accumulation 的效果边界，并提出针对性的解决方案。实验覆盖 10,000 次生成测试、7 种 K 维度、多种模型架构。

---

## 1. 引言

### 1.1 问题背景

LLM 推理的可复现性问题日益受到关注。Yuan et al. (NeurIPS 2025) 展示了在 BF16 精度下，仅改变评估 batch size 就可导致 DeepSeek-R1-Distill-Qwen-7B 在 AIME'24 上 **准确率波动 9%、回复长度差异 9,000 tokens**。他们将根因归结为"浮点非结合律在有限精度下的放大"，并提出 LayerCast（BF16 存储 + FP32 计算）作为缓解方案。

然而，这一归因虽然正确但不够深入。**提高精度（BF16→FP32）并不总是改善确定性**——我们发现在某些模型上，FP32 反而引入了更多的非确定性。这表明根因不仅是精度问题。

### 1.2 我们的发现

通过对 cuBLAS 行为的精细分析，我们揭示了一个更深层的机制：

**核心发现 1：cuBLAS Kernel 切换边界**
cuBLAS 在不同批次大小下选择不同的 GEMM 算法。我们通过 bs=1..256 的逐一扫描，首次精确映射了这些切换边界：

| 模式 | 切换次数 | 切换点 | 受影响 bs 比例 |
|---|---:|---|---:|
| BF16 默认 | **8** | 2, 8, 17, 61, 83, 100, 129, 143 | 31.6% |
| FP32 accum | **4** | 57, 65, 79, 83 | 4.7% |

FP32 flag 减少了切换次数并收窄了受影响的 batch size 窗口，但没有消除切换。

**核心发现 2：FP32 效果是模型依赖的**

| 模型 | BF16 切换次数 | FP32 切换次数 | FP32 效果 |
|---|---:|---:|---|
| Llama-3.1-8B (dense) | 8 | 4 | 改善 |
| DeepSeek-V2-Lite (MoE) | **0** | **6** | **恶化** |

DeepSeek-V2-Lite 在 BF16 下 bs=1..32 完全确定（0 切换），但开启 FP32 flag 后反而出现 6 个切换点。这表明 FP32 flag 不是简单地"提升精度"，而是**改变了 cuBLAS 的 kernel 选择启发式**，在某些矩阵维度下引入了新的切换边界。

**核心发现 3：K 维度决定 FP32 有效性**

| K (规约维度) | BF16 max_diff | FP32 max_diff | FP32 改善 |
|---:|---:|---:|---|
| 512 | 0.000 | 0.000 | 无需改善 |
| 1024 | 0.500 | **0.000** | 完全消除 |
| 4096 | 1.000 | 0.250 | 4x 改善 |
| 11008 | 2.000 | 2.000 | **无改善** |
| 14336 | 4.000 | 1.000 | 4x 改善 |

K=11008（Llama gate_proj 的维度）时 FP32 几乎无效。这与 MoE 模型中 gate projection 的维度直接相关。

### 1.3 与 Yuan et al. 的关系

Yuan et al. 的工作关注"数值精度对可复现性的影响"，提出 LayerCast 作为方案。我们的工作更深一层：

1. **根因不同**：不只是精度问题，而是 kernel 选择问题。LayerCast 通过全 FP32 计算间接避免了某些 kernel 切换，但并非对所有模型/维度都有效。
2. **方向互补**：我们的 kernel 切换边界分析解释了为什么 LayerCast 在某些场景有效（FP32 kernel landscape 更稳定）而在其他场景效果有限。
3. **贡献维度不同**：他们关注"用什么精度"，我们关注"为什么不同精度会产生不同 kernel 选择"以及如何直接控制 kernel 选择。

### 1.4 贡献

1. **首次精确映射** cuBLAS kernel 切换边界（bs=1..256 逐一扫描，BF16 vs FP32）
2. **发现 FP32 flag 的效果是模型依赖的**：对某些模型恶化而非改善——挑战传统"高精度=更确定"的假设
3. **系统化分解** 5 种非确定性来源，逐一分析 FP32 的有效边界
4. **理论框架**（Theorems 1-3）解释 FP32 何时有效、何时失效
5. **K 维度依赖性分析**：解释为什么不同层/不同模型对 FP32 的响应不同
6. **10,000 次运行的大规模验证**（vs Yuan et al. 的 12 配置 × 单次运行）

---

## 2. 背景

### 2.1 浮点非结合律与 LLM 推理

（与 Yuan et al. §2 类似，但更聚焦于 kernel 选择机制）

浮点加法不满足结合律：$(a+b)+c \neq a+(b+c)$。在 GPU 上，GEMM 的内积规约被分解为多个 thread block 的部分和，分解策略（tiling、split-K）由 cuBLAS 启发式决定。**关键区别**：改变 batch size M 不仅改变"相同计算的执行顺序"，还可能改变"选择哪种计算算法"——后者是结构性差异，不是精度能弥补的。

### 2.2 cuBLAS Kernel 选择启发式

cuBLAS 根据矩阵维度 (M, N, K)、GPU 架构、可用 SM 数等因素选择最优 kernel：
- 小 M：可能使用 split-K 来增加并行度
- 大 M：output-tile 并行已足够，关闭 split-K
- 关键：`allow_bf16_reduced_precision_reduction` flag 改变的不只是累加精度，还改变了 cuBLAS 的启发式搜索空间

### 2.3 与 Yuan et al. 的 LayerCast 的关系

LayerCast 将权重存储为 BF16，计算时 just-in-time 上转为 FP32。这确保了 **FP32 × FP32 → FP32** 的计算路径，避免了 BF16 的累加误差。但 LayerCast 有 34% 的内存节省（vs 纯 FP32）同时引入了计算开销。更重要的是，即使在 FP32 下，**cuBLAS 仍然会在不同 M 下选择不同 kernel**——只是 FP32 kernel 的 landscape 不同于 BF16。

---

## 3. 非确定性来源的系统分解

我们将 LLM 推理中的 batch 非确定性分解为 5 个独立来源，逐一分析。

### 3.1 Source 1: GEMM Kernel 选择

**Observation 1: cuBLAS 在不同 batch size 下选择不同 kernel algorithm**

通过 Llama-3.1-8B 的 10,000 次生成测试（bs=1..256 随机采样），我们精确映射了 kernel 切换边界：

| 模式 | Unique 输出 | 切换点 | 10,000-run 分布 |
|---|---:|---|---|
| BF16 | 2 | 2, 8, 17, 61, 83, 100, 129, 143 | 6838 : 3162 |
| FP32 | 2 | 57, 65, 79, 83 | 9499 : 501 |

每个单独的 batch size 在同一模式下是完美确定的——方差纯粹来自跨 batch size 的 kernel 切换。

**Observation 2: FP32 改变了 kernel 选择 landscape，不只是精度**

| 测试 | BF16 行为 | FP32 行为 |
|---|---|---|
| Llama (dense) | 8 切换，bs=1 即切换 | 4 切换，bs=1-56 稳定 |
| DeepSeek (MoE) | **0 切换，完美确定** | **6 切换，引入非确定** |

**这是本文最核心的发现**：FP32 flag 的本质不是"提升精度"，而是"改变 cuBLAS 的 kernel 选择启发式"。改变后的 landscape 在某些模型上更稳定（Llama），在另一些上反而更不稳定（DeepSeek）。

**Observation 3: K 维度决定 FP32 有效性**

（表格同 §1.2）

K 维度越大，同一 kernel 内的 FP32 累加残余误差越大。K=11008 时 FP32 几乎无改善——这是 Llama 的 gate_proj 维度，也解释了为什么即使 FP32 在 op 级别有残余误差。

**Observation 4: PyTorch 标准确定性工具完全无效**

| 方案 | 切换次数 | 效果 |
|---|---:|---|
| Baseline (BF16) | 6 | — |
| `torch.use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG` | **6** | 零效果 |
| FP32 flag | 4 | 部分改善 |
| FP32 + deterministic algo | 4 | = FP32 alone |

`torch.use_deterministic_algorithms` 只解决 backward pass 的 atomic 非确定性，对 forward pass 的 kernel 选择无效。

### 3.2 Source 2: Reduction Ops (RMSNorm, Softmax)

**Observation 5: RMSNorm 在 PyTorch 中已确定**

HuggingFace 的 LlamaRMSNorm 在 FP32 下计算 sum-of-squares，不同 batch size 不影响（每个 token 独立处理）。BF16 的 chunk 顺序差异也为 0——因为 PyTorch 的 `.mean()` 实现不依赖 batch size。

**Observation 6: Softmax 的 FP32 改善达 5 个数量级**

| 模式 | Chunked vs reference max_diff |
|---|---|
| BF16 | ~1e-6 |
| FP32 | ~1e-11 |

### 3.3 Source 3: Attention Split-KV

**Observation 7: FP32 online softmax 下 cross-split 差异极小 (~1e-7)**

| seq_kv | 1-split vs 32-split (FP32) |
|---:|---|
| 256 | 1.04e-7 |
| 1024 | 5.96e-8 |
| 4096 | 2.98e-8 |

当 online softmax 全程在 FP32 下计算时，不同 split 数的差异仅为 ~1e-7——远小于 BF16 rounding quantum。真实 FlashDecoding 的问题来自 BF16 中间值的 cast 和 tiling 边界，而非 FP32 precision 不足。

**推论**：对 HuggingFace SDPA（不使用 split-KV），attention 已是 batch-invariant。对 serving engine（使用 FlashDecoding），固定 split 边界 + FP32 中间计算可恢复确定性。

### 3.4 Source 4: 层间误差累积

**Observation 8: 逐层 hidden state diff 呈非单调增长**

| 层 | max_diff (bs=1 vs bs=8) |
|---|---|
| L0 | 9.77e-4 |
| L8 | 3.12e-2 |
| L16 | 6.25e-2 |
| L24 | 1.25e-1 |
| L31 | 3.12e-2 |
| Final logits | 1.80e-1 |

误差从 L0 的 ~1e-3 增长到 L24 的 0.125，然后在后续层略微回落。但 **argmax match 始终 100%**——logit 级差异（0.18）不足以翻转 greedy token。

**关键发现**：BF16 和 FP32 模式的逐层误差模式完全相同。这进一步确认 FP32 flag 改变的是 kernel 选择（第一层的差异不同），而非每层的精度。

### 3.5 Source 5: MoE Routing 放大

**Observation 9: Near-tie prevalence 高达 30-40%**

| MoE 配置 | Near-tie (τ < 0.001) |
|---|---|
| 64 experts, top-6 | 30% |
| 128 experts, top-8 | 40% |
| 8 experts, top-2 | 1% |

**Observation 10: Gate GEMM 的大 K 维度使 FP32 效果有限**

Real gate-proxy GEMM (K=14336, N=4096):
- BF16 max_diff = 0.031
- FP32 max_diff = 0.016（仅 2x 改善）

MoE 模型的 gate projection 通常有大 K 维度，使 FP32 的改善非常有限。加上 30-40% 的 near-tie prevalence，微小的 GEMM 残余误差就足以翻转 expert 选择。

---

## 4. 理论框架

### 4.1 Theorem 1: 加法规约的 FP32 充分条件

（内容同前版，包含 κ bound 和 BF16 product exactness lemma）

### 4.2 Theorem 1 的理论-实践鸿沟

在 Llama-3.1-8B 的 225 个线性层上测量 κ(S)：

- 理论 bound: κ < 8.0 (K=4096)
- 实际: 中位 κ = 3,485，最大 κ = 1,526,426
- **0/225 层满足充分条件**

但实际中 FP32 在大多数 batch size 下仍然有效（95% 的 bs 产生一致输出）。差距来自：
1. Worst-case vs average-case（误差随机分布，部分抵消）
2. Per-op vs end-to-end（单个 op 的 0.5 ULP 残余不足以翻转 argmax）

**但这个"有效"只是统计意义上的**——仍有 4.7% 的 batch size 触发 kernel 切换，导致不同输出。Theorem 1 的 bound 虽然保守，但其核心局限是正确的：FP32 不能保证 determinism。

### 4.3 Theorem 2: Attention Split-KV 的结构不可修复性

（与前版相同）

### 4.4 Theorem 3: 固定 Split 恢复确定性

（与前版相同）

---

## 5. 讨论：为什么 FP32 不是银弹

### 5.1 FP32 Flag 的双重效应

`allow_bf16_reduced_precision_reduction = False` 同时做了两件事：
1. **精度效应**：BF16 乘积在 FP32 中累加，减少舍入误差
2. **Kernel 选择效应**：改变 cuBLAS 启发式的搜索空间，可能选择完全不同的算法

效应 2 是 dominant factor。证据：
- M4 实验（层间累积）显示 BF16 和 FP32 的误差模式完全相同——精度效应微弱
- E2 实验（DeepSeek）显示 FP32 反而引入新切换——kernel 选择效应 dominant

### 5.2 与 LayerCast 的互补关系

Yuan et al. 的 LayerCast 通过全 FP32 计算路径间接改变了 kernel landscape。我们的分析解释了为什么 LayerCast 在某些模型上接近完美确定（FP32 kernel landscape 恰好更稳定），而在其他场景仍有残余非确定性（Div_Index 的 3.4% 发散率）。

完全解决需要直接控制 kernel 选择——这超出了精度优化的范畴，需要 serving engine 级别的干预（如 vLLM 的 batch_invariant_ops 或 SGLang 的 deterministic inference mode）。

### 5.3 实践建议

基于我们的分析，对不同场景的推荐：

| 场景 | 推荐方案 | 预期效果 | 开销 |
|---|---|---|---|
| **研究评估（单 GPU）** | FP32 flag | 消除 95% batch variance (Llama) | 0% |
| **研究评估（需要完美）** | LayerCast (Yuan et al.) | ~97% 确定 | ~34% 内存节省 vs FP32 |
| **生产 serving (dense)** | batch_invariant_ops (vLLM/SGLang) | ~100% | 20-35% |
| **生产 serving (MoE)** | 开放问题 | 部分 | 未知 |

**关键警告**：FP32 flag 不保证改善。在部署前必须针对目标模型验证效果。

---

## 6. 实验

### 6.1 实验设置

- **Hardware**: NVIDIA RTX A6000 (Ampere, 48 GB) × 2
- **Software**: PyTorch 2.6.0+cu124, CUDA 12.4
- **Dense 模型**: Llama-3.1-8B-Instruct (BF16, 32 layers, hidden 4096)
- **MoE 模型**: DeepSeek-V2-Lite (BF16, 15.7B, 64 experts, top-6)

### 6.2 10,000-Run Generation Determinism (Llama)

1000 次运行 × 10 种 bs (前版实验) vs 10,000 次 × 256 种 bs (新实验)：

| 模式 | Unique 输出 | 切换次数 | 受影响 bs | 10,000-run 分布 |
|---|---:|---:|---:|---|
| BF16 | 2 | 8 | 81/256 (31.6%) | 6838 : 3162 |
| FP32 | 2 | 4 | 12/256 (4.7%) | 9499 : 501 |

FP32 将受影响的 batch size 从 31.6% 缩减到 4.7%（6.7x 改善），但无法完全消除。

### 6.3 MoE Model: DeepSeek-V2-Lite

| 模式 | Unique 输出 | 切换次数 | 说明 |
|---|---:|---:|---|
| BF16 | **1** | **0** | 完美确定！ |
| FP32 | **2** | **6** | 恶化！ |

这是全文最具颠覆性的结果。BF16 在 DeepSeek-V2-Lite 的 bs=1..32 范围内完全确定，而 FP32 flag 引入了 6 个 kernel 切换点。

### 6.4 GEMM K-Dimension Sweep

7 种 K 维度 × 11 种 batch size × 2 种模式 = 154 组实验。关键趋势：
- K ≤ 1024: FP32 完全消除差异
- K = 2048-4096: FP32 改善 4-8x
- K = 8192-14336: FP32 改善 2-4x
- K = 11008: FP32 几乎无改善

### 6.5 PyTorch 确定性工具评估

4 种方案 × 256 种 bs = 综合评估：

| 方案 | 切换次数 | 开销 |
|---|---:|---:|
| Baseline (BF16) | 8 | 0% |
| FP32 flag only | 4 | -0.7% |
| `deterministic_algorithms` + `WORKSPACE_CONFIG` | **8** | +3.3% |
| FP32 + deterministic | 4 | +1.1% |

`torch.use_deterministic_algorithms` 对 batch 非确定性完全无效。

### 6.6 Reduction Ops

- RMSNorm: 已确定（PyTorch 实现不依赖 batch size）
- Softmax: FP32 改善 5 个数量级（~1e-6 → ~1e-11）

### 6.7 逐层误差累积

- L0: 9.8e-4 → L24: 0.125 → Final: 0.18
- Argmax match: 100%（所有 bs、两种模式）
- BF16 和 FP32 模式的误差模式相同

---

## 7. 相关工作

**数值精度与可复现性**：Yuan et al. (NeurIPS 2025) 首次系统研究了数值精度对 LLM 推理可复现性的影响，提出 LayerCast 作为缓解方案。我们的工作在此基础上深入一层，揭示根因是 kernel 选择而非精度本身。

**确定性推理系统**：Thinking Machine Lab (2025) 提出自定义 kernel 方案（61.5% 开销），SGLang 实现了确定性推理模式（34.35% 开销），vLLM 正在开发 VLLM_BATCH_INVARIANT=1。这些工作通过 kernel 级控制实现确定性，与我们的根因分析一致。

**FlashAttention 与 Split-KV**：FlashDecoding 的 split-KV 并行引入了额外的非确定性来源。我们的 Theorem 2 形式化了这一问题，Theorem 3 给出了固定 split 的修复方案。

**浮点误差分析**：Higham (2002) 的递归求和误差界是我们 Theorem 1 的理论基础。我们的 κ 验证实验揭示了 worst-case bound 与 average-case 实践之间的 435x 鸿沟。

---

## 8. 结论

LLM 推理中的 batch 非确定性，其根因不是简单的浮点精度不足，而是 **cuBLAS kernel 选择启发式在不同 batch size 下选择结构性不同的计算算法**。FP32 accumulation 通过改变 kernel selection landscape 在某些模型上减少了切换次数（Llama: 8→4），但在其他模型上反而恶化（DeepSeek: 0→6）。

这一发现有几个重要启示：
1. **"高精度 = 更确定"是不准确的**。精度影响的是同一 kernel 内的累加误差，但跨 kernel 的切换是结构性问题。
2. **没有 universal 的一行修复**。需要针对目标模型和 deployment 场景选择合适的方案。
3. **完全解决需要 kernel 级控制**。这意味着需要 serving engine 级别的干预，如固定 GEMM algorithm、固定 attention split 边界、deterministic all-reduce。

**实践建议**：
- 对于研究评估：使用 FP32 flag 作为轻量级第一步（但需验证效果）
- 对于需要严格确定性的场景：使用 batch_invariant_ops (vLLM/SGLang) 或 LayerCast
- **始终在目标模型上验证**，不要假设 FP32 一定改善

---

## 附录

### A. 10,000-Run 实验的完整 Kernel 切换边界图

（图：x 轴 bs=1..256，y 轴 hash 值，BF16 和 FP32 两条线）

### B. GEMM K-Dimension 完整数据表

（7 K × 11 M × 2 mode 的完整矩阵）

### C. DeepSeek-V2-Lite 的 per-bs Hash 映射

（32 个 bs 在 BF16 和 FP32 下的 hash 值）
