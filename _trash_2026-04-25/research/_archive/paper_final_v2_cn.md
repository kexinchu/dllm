# 零成本的确定性LLM推理：FP32累加何时能吸收批次方差

---

## 摘要

采用连续批处理（continuous batching）的大语言模型（LLM）推理系统，会因批次组成不同而对同一输入产生不同输出——这种隐性的非确定性破坏了可复现性，增加了调试难度，并向强化学习奖励信号和知识蒸馏目标注入了人为噪声。我们将这一非确定性追溯到单一根因：cuBLAS 根据批次维度选择不同的 GEMM kernel，导致浮点规约顺序不同，而默认的 BF16 低精度累加模式会将这些重排误差放大到超过 BF16 舍入量子。

我们提出了基于 Higham 风格误差分析的理论框架，包含三个定理：
1. 对于加法规约（GEMM、RMSNorm），FP32 累加在充分条件下保证 BF16 输出逐位一致
2. Split-KV attention 中的乘法缩放链在结构上无法通过精度修复
3. 固定 split 边界可恢复加法规约体制

在 Llama-3.1-8B 上的实验验证发现，理论充分条件在所有 225 个线性层均被违反（中位条件数 3,485 vs 理论界 8.0），但 FP32 累加仍然实现了完美的生成确定性——揭示了最坏情况理论与平均情况实践之间的巨大鸿沟。对于 Dense 模型，设置一个 cuBLAS flag（`allow_bf16_reduced_precision_reduction=False`）即可在所有批次大小上实现完美的生成确定性，延迟开销为零（-1.4%，1000 次运行）。对于 MoE 模型，我们在 DeepSeek-V2-Lite（64 experts, top-6）上证明 FP32 累加是必要但不充分的，两种模式均产生 3 种不同输出。我们量化了下游影响（RL 奖励方差减少 35%，蒸馏 KL 散度减少 27%），并指出 MoE routing 确定性仍是开放问题。

---

## 1. 引言

设想一个处理数千并发请求的生产 LLM 推理系统。用户提交同一个 prompt 两次。第一次与 7 个其他请求共享一个 batch；第二次与 15 个。尽管输入相同、greedy decoding、权重固定，系统却返回了不同的续写。这不是传统意义上的 bug——而是现代 GPU 线性代数库优化矩阵乘法的必然后果。

### 为什么重要

- **RLHF**：策略梯度方法依赖模型生成计算奖励。如果同一 prompt 在不同批次下产生不同的 log-probability，奖励信号就包含了与算法无关的人为方差，拖慢收敛。
- **知识蒸馏**：教师模型的 soft logits 是训练目标。批次依赖的 logit 波动会污染这些目标。
- **MoE 路由**：MoE 模型通过 top-k 选择 expert。当 expert 分数接近——我们证明在 DeepSeek-V2-Lite 中 100% 的 token 位置都存在这种情况——亚 ULP 的 logit 扰动就足以翻转 expert 选择。
- **安全合规**：监管框架日益要求 AI 输出可复现。

### 为什么会发生

根因是 cuBLAS 的 kernel 选择启发式。当批次维度 $M$ 改变时，cuBLAS 会选择不同的 GEMM kernel，采用不同的 split-K 分解，改变规约顺序。由于浮点加法不满足结合律，不同的规约顺序产生不同结果。在默认的 BF16 低精度累加模式下，这些差异可达 BF16 量级的 1.0——足以翻转 argmax token。

### 我们的贡献

1. **理论框架**（定理 1-3）：刻画 FP32 累加何时能吸收批次依赖误差，实验验证表明充分条件平均保守 400 倍
2. **实验验证**：Llama-3.1-8B-Instruct 上通过单个 cuBLAS flag 实现完美生成确定性，零成本（1000 次运行，10 种批次大小）
3. **MoE 非确定性的首次系统刻画**：DeepSeek-V2-Lite 上证明 FP32 累加必要但不充分
4. **下游影响量化**：RL 奖励方差减少 35%，蒸馏 KL 减少 27%，expert 路由稳定性分析
5. **与现有方案的对比**：与 vLLM、SGLang、Thinking Machine Lab 的确定性方案进行系统比较

---

## 2. 背景

### 2.1 浮点格式

BF16：1 位符号 + 8 位指数 + 7 位尾数，单位舍入 $\varepsilon_{\text{bf16}} = 2^{-8} \approx 3.91 \times 10^{-3}$。FP32：23 位尾数，$\varepsilon_{\text{fp32}} = 2^{-24} \approx 5.96 \times 10^{-8}$。精度比 $\varepsilon_{\text{bf16}} / \varepsilon_{\text{fp32}} = 2^{16} = 65{,}536$，这是我们方法的根本支撑。

### 2.2 非结合性与规约顺序

浮点加法不满足结合律：$(a + b) + c \neq a + (b + c)$。GPU kernel 将规约分解为独立计算的部分和。分解策略即"规约顺序"，不同顺序产生不同结果。

### 2.3 批次依赖的 Kernel 选择

cuBLAS 使用带 split-K 分区的分块算法实现 GEMM，分区策略由启发式决定，依赖于 $M$（批次维度）。连续批处理改变 $M$ 时，cuBLAS 选择不同 kernel，改变每个输出元素的规约顺序。

### 2.4 Transformer 中的规约操作

Transformer decoder 包含三类规约：
1. **线性投影（GEMM）**：在 $K$ 维上的内积规约
2. **RMSNorm**：在隐藏维度上的平方和规约
3. **Attention**：跨 KV chunk 的 online softmax 乘法缩放

### 2.5 MoE 路由

MoE 模型通过 softmax 归一化的 gate logits 做 top-k 选择 expert 子集。该流水线放大小扰动：softmax 在 argmax 附近集中概率质量，top-k 引入硬阈值不连续性。

---

## 3. 理论框架

### 3.1 定理 1：加法规约的充分条件

设 $a_1, \ldots, a_N \in \mathbb{F}_{\text{bf16}}$，$\hat{S}_{\pi_1}, \hat{S}_{\pi_2}$ 为任意两种规约顺序下的 FP32 累加和。

**定理 1.** 差异满足：
$$|\hat{S}_{\pi_1} - \hat{S}_{\pi_2}| \leq 2\gamma_{N-1}^{\text{fp32}} \sum_{i=1}^{N} |a_i|$$

其中 $\gamma_k^{\text{fp32}} = k\varepsilon_{\text{fp32}} / (1 - k\varepsilon_{\text{fp32}})$。若进一步满足：
$$2\gamma_{N-1}^{\text{fp32}} \cdot \kappa(S) < 1 \quad \text{其中} \quad \kappa(S) = \frac{\sum |a_i|}{|\sum a_i|}$$

则 $\text{round}_{\text{bf16}}(\hat{S}_{\pi_1}) = \text{round}_{\text{bf16}}(\hat{S}_{\pi_2})$（BF16 舍入后逐位一致）。

**条件数解读**：改写为 $\kappa(S) < 2^{15} / (N-1)$。对 $N = 4096$（典型隐藏维度），界为 $\kappa < 8.0$。

**引理（BF16 乘积精确性）**：$a, b \in \mathbb{F}_{\text{bf16}}$ 的乘积 $a \cdot b$ 在 FP32 中精确可表示（两个 8 位尾数相乘 $\leq$ 16 位，FP32 有 24 位有效数字）。因此 GEMM 内积的各项在累加前是精确的——唯一的误差来源是规约顺序。

### 3.2 定理 1 是保守的：实验验证

我们在 Llama-3.1-8B 的所有 225 个线性层上测量了真实推理输入的 $\kappa(S)$（每层采样 100 个输出元素，FP64 计算）：

**表 1：条件数统计**

| 统计量 | 值 |
|---|---|
| 理论界（$K=4096$） | 8.0 |
| 满足理论界的层数 | **0 / 225** |
| 中位数 max $\kappa$ | 3,485 |
| 平均 max $\kappa$ | 15,597 |
| 最大 $\kappa$（任意层） | 1,526,426 |

**所有层都违反了充分条件**，但 FP32 累加在 1000 次运行中仍实现了完美的生成确定性。原因：

1. **最坏情况 vs 平均情况**：Higham 界假设所有舍入误差同向累积。实际中误差近似均匀分布在 $[-\varepsilon/2, +\varepsilon/2]$ 中，大量正负抵消。根据随机游走论证，期望误差为 $O(\sqrt{N} \cdot \varepsilon_{\text{fp32}})$ 而非 $O(N \cdot \varepsilon_{\text{fp32}})$。
2. **单操作 vs 端到端**：充分条件针对单次规约。即使单次规约产生 0.5 ULP 的 BF16 差异，自回归生成仍可确定——因为 top-1 token 的 logit 通常远高于第二名。
3. **间距很大**：中位 $\kappa$ 超界 435 倍，而精度比 $2^{16} = 65{,}536$ 在平均情况下提供了足够的吸收余量。

**核心洞察：定理 1 的充分条件对最坏情况是紧的，但对实际 LLM 推理极为保守。FP32 累加的实际有效范围远超理论边界。**

### 3.3 定理 2：Attention Split-KV 打破加法充分性

**定理 2.** 跨 $P$ 个 split 的 online softmax 合并不是加法规约的重排。不同的 split 数产生结构性不同的计算图，其中的乘法缩放因子 $\exp(m_{\text{local}} - m_{\text{global}})$ 引入的误差超出了 FP32 的吸收能力。

**直觉**：$P=2$ 时每个部分结果被缩放一次；$P=4$ 时需要三次级联缩放。虽然数学上等价（可收缩），但计算上不同——每次 $\exp$、减法、乘法都引入误差，且缩放因子 $\exp(\Delta m)$ 在 $\Delta m > 0$ 时放大绝对误差。

### 3.4 定理 3：固定 Split 边界恢复确定性

**定理 3.** 若 attention split-KV 边界固定（与批次无关），则每个 chunk 内的 FP32 累加可恢复批次不变性。

**设计原则**：按序列长度固定 split 边界，每个 split 内应用 FP32 累加。这将定理 2 的情形（结构不同的计算图）转化回定理 1 的情形（相同计算图，不同内部规约顺序）。

---

## 4. 方法

### 4.1 Dense 模型的一行修复

```python
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
```

指示 cuBLAS 对所有 BF16 GEMM 使用 FP32 累加器。无需替换 kernel、修改模型或更改任何代码。

**为什么对 HuggingFace 推理足够**：
- SDPA 后端不进行跨 SM 的 KV 序列拆分。每个程序在固定大小的块中遍历完整 KV 序列，attention 的规约顺序仅取决于序列长度和块大小，与批次组成无关
- HuggingFace 的 LlamaRMSNorm 内部已在 FP32 下计算
- 因此 GEMM 是唯一的批次方差来源，cuBLAS flag 完全消除它

### 4.2 区分 Logit 级和 Generation 级确定性

重要区分：FP32 累加消除了**每个操作**的批次方差，但不消除 32 层 transformer 的**累积** logit 差异。在 182-token 序列上：

**表 2：Logit 级 vs Generation 级确定性（Llama-3.1-8B, continuous batching）**

| 模式 | Logit max_diff | Logit argmax 翻转 | 生成确定性（1000次运行） |
|---|---|---|---|
| BF16 | 7.5--8.6 ULP（随 bs 变化） | 0--1 / 182 | 2 种不同输出 |
| FP32 accum | 6.8 ULP（**跨所有 bs 稳定**） | 1 / 182 | **1 种输出（完美确定）** |

关键区别：BF16 的 logit max_diff 随批次大小变化（bs=2 时 7.5，bs=32 时 8.6），而 FP32 的 max_diff 在所有批次大小上恒定（6.8）。FP32 消除了**批次间**的 logit 变化，虽然两种模式都有**序列内**的累积误差。生成确定性得以实现，因为 top-1 token 的 logit 边际远大于累积误差。

### 4.3 何时不够：MoE 模型

对 MoE 模型，残余的 0.5 ULP GEMM 方差通过路由流水线传播：gate GEMM → softmax 放大 → top-k 不连续性。完全的 MoE 确定性需要 FP32 累加加上具有显式 tie-breaking 规则的确定性 top-k。

### 4.4 面向 Serving Engine 的固定 Split-KV

对使用 FlashDecoding 的引擎，定理 3 给出方案：选择固定 chunk 大小 $C$（如 256），split 数 $P = \lceil L / C \rceil$ 仅由序列长度 $L$ 决定，每个 chunk 内应用 FP32 累加。

---

## 5. 实验

所有实验使用 NVIDIA RTX A6000 (Ampere, 48 GB)，PyTorch 2.6.0+cu124，CUDA 12.4。

### 5.1 算子级刻画

**表 3：GEMM 批次方差 — Q-Projection (K=4096, N=4096)**

| $M$ | BF16 max_diff | FP32 max_diff | BF16 mean_diff | FP32 mean_diff |
|----:|:---:|:---:|:---:|:---:|
| 1 | 0.00 | 0.00 | 0.00 | 0.00 |
| 32 | 0.50 | 0.50 | 1.91e-4 | 1.91e-4 |
| 64 | **1.00** | 0.50 | 5.08e-2 | 1.61e-4 |
| 256 | **1.00** | 0.50 | 7.52e-2 | 1.34e-4 |
| 1024 | **1.00** | 0.50 | 7.52e-2 | 1.91e-4 |

FP32 累加将 max_diff 限制在 0.5（vs BF16 的 1.0），mean_diff 降低约 300 倍。见图 1。

**表 4：GEMM — MoE Expert 形状 (K=2048, N=5632)**

| $M$ | BF16 max_diff | FP32 max_diff |
|----:|:---:|:---:|
| 1--128 | 0.00 | 0.00 |
| 256 | **1.00** | **0.00** |
| 512 | **1.00** | **0.00** |

较小的 $K = 2048$ 下，FP32 实现完美免疫（max_diff = 0）。

**表 5：RMSNorm 分块规约**

| 隐藏维度 | 分块数 | BF16 max_diff | FP32 max_diff |
|---:|---:|:---:|:---:|
| 2048--4096 | 1--32 | 1.56e-2 | **0.00** |

**表 6：Attention Split-KV（变化 split 数，FP32 模式）**

| Split 数 | BF16 max_diff | FP32 max_diff |
|---:|:---:|:---:|
| 2 | 0.418 | 0.418 |
| 4 | 1.137 | 1.145 |
| 16 | 2.548 | 2.548 |

改变 split 数时 FP32 **完全无改善**，验证定理 2。见图 3。

### 5.2 Dense 模型：Llama-3.1-8B-Instruct

**表 7：1000 次运行生成确定性**

| 模式 | 不同输出数 | 确定性? | 延迟 (ms) |
|---|---:|:---:|---:|
| BF16（默认） | **2** | 否 | 871 |
| FP32 累加 | **1** | **是** | 859 (-1.4%) |

BF16 下 1000 次运行（10 种批次大小各 100 次）产生 2 种不同输出（900:100 分布）。每个单独批次大小完美确定——方差纯粹是跨批次大小的。FP32 累加在全部 1000 次运行中产生 1 种输出。见图 2。

**表 8：序列长度缩放（模型级，continuous batching）**

| 序列长度 | BF16 max_diff (bs=8) | FP32 max_diff (bs=8) | BF16 翻转 | FP32 翻转 |
|---:|---:|---:|---:|---:|
| 32 | 0.60 | 0.81 | 1 | 1 |
| 64 | 4.42 | 5.70 | 1 | 1 |
| 128 | 8.66 | 5.74 | 0 | 1 |
| 200 | 6.98 | 8.65 | 1 | 1 |

Logit 级差异随序列长度增长（层间累积误差）但保持有界。两种模式都在输入位置出现罕见的 argmax 翻转，不影响自回归生成。

### 5.3 MoE 模型：DeepSeek-V2-Lite

**表 9：200 次运行生成确定性（DeepSeek-V2-Lite, 64 experts, top-6）**

| 模式 | 不同输出数 | Hash 分布 | 延迟 (ms) |
|---|---:|---|---:|
| BF16 | **3** | 120:40:40 | 253.9 |
| FP32 累加 | **3** | 80:80:40 | 253.8 (0%) |

两种模式都产生 3 种不同输出。每个单独批次大小完美确定。非确定性是跨批次大小的：MoE routing 对残余的 0.5 ULP GEMM 差异过于敏感。

**表 10：MoE Logits 批次方差**

| bs | BF16 max_diff | FP32 max_diff | Argmax |
|---:|---:|---:|:---:|
| 2 | 0.688 | 0.672 | 翻转 |
| 8 | **1.500** | 0.656 | 翻转 |
| 16 | 0.875 | 0.750 | 翻转 |

FP32 降低了 max_diff（bs=8 时 1.50 → 0.66），但 argmax 在所有批次大小都翻转。

### 5.4 下游影响

**表 11：RL 奖励信号方差（Llama-3.1-8B, 30 prompts, bs=1 vs bs=8）**

| 指标 | BF16 | FP32 累加 | 减少幅度 |
|---|---:|---:|---:|
| 平均 |logprob diff| | 2.30e-2 | 1.49e-2 | **35%** |
| 最大 |logprob diff| | 1.19e-1 | 8.39e-2 | 30% |

**表 12：知识蒸馏 KL 散度**

| 指标 | BF16 | FP32 累加 | 减少幅度 |
|---|---:|---:|---:|
| 平均 KL | 6.29e-4 | 4.59e-4 | **27%** |
| 最大 KL | 3.41e-3 | 2.47e-3 | 28% |

**表 13：MoE Expert 选择翻转（合成实验，128 experts, top-8）**

| 批次大小 | BF16 翻转率 | FP32 翻转率 |
|---:|:---:|:---:|
| 4 | 6.0% | **0.0%** |
| 8 | 6.0% | **0.0%** |
| 16 | 4.0% | **0.0%** |

### 5.5 与现有确定性推理方案对比

**表 14：方法对比**

| 方面 | 我们（cuBLAS flag） | Thinking Machine Lab | SGLang | vLLM |
|---|---|---|---|---|
| GEMM 开销 | **0%** | ~20% | ~20% | ~20% |
| Attention 开销 | **0%**（SDPA 已 BI） | ~10-20% | ~15-25% | ~10-20% |
| 总开销 | **0% (-1.4%)** | 61.5% | 34.35% | 20-35% |
| Dense 确定性 | 是 | 是 | 是 | 是 |
| MoE 确定性 | 否 | 部分 | 部分 | 部分 |
| 理论框架 | **有** | 无 | 无 | 无 |
| 代码改动 | **1 行** | 全部重写 | 引擎集成 | 引擎集成 |

我们的零开销结果适用于 HuggingFace 推理（SDPA attention 已具备批次不变性）。Serving engine 需要更重的干预（固定 split-KV、统一 KV 布局、确定性 NCCL），因为它们使用 FlashDecoding 和 PagedAttention，引入了定理 2 所刻画的结构性非确定性。cuBLAS flag 是互补的：应作为任何确定性推理流水线的第一步。

---

## 6. 相关工作

**数值精度**：混合精度训练（Micikevicius et al., 2018; Kalamkar et al., 2019）和 BF16 分析（Blanchard et al., 2020）关注精度而非确定性。我们基于 Higham (2002) 的框架回答不同的问题：不同规约顺序是否在 BF16 舍入后收敛。

**确定性 GPU 计算**：PyTorch 的 `torch.use_deterministic_algorithms(True)` 解决运行间非确定性（原子操作）。我们解决的是批次组成非确定性，二者正交，即使开启所有已有确定性 flag 后仍然存在。

**确定性推理系统**：Thinking Machine Lab (2025) 提出自定义 kernel 方案，开销 61.5%。SGLang 实现了固定 split-KV 的确定性推理，开销 34.35%。vLLM 正在开发 VLLM_BATCH_INVARIANT=1 模式，估计开销 20-35%。我们的贡献：(1) 通过 cuBLAS flag 实现零开销 GEMM 确定性；(2) 提供理论框架解释为什么每个组件需要不同处理；(3) 识别定理 1 充分条件的理论-实践鸿沟。

**FlashAttention**：FlashAttention (Dao et al., 2022; Dao, 2023) 引入分块 attention。FlashDecoding (2023) 增加了 decode 阶段的 split-KV 并行。我们的定理 2 形式化了 split-KV 打破确定性的原因；定理 3 证明固定 split 可恢复。FlashInfer (Ye et al., 2024) 提供可配置的 split 策略，支持定理 3 的设计。

**MoE 路由**：先前工作（Fedus et al., 2022; Zoph et al., 2022）研究训练中的负载均衡和路由崩溃。我们识别了一种新的失败模式：推理中的批次依赖 expert 选择，这是数值精度问题而非训练动力学问题。

---

## 7. 讨论与局限

### 7.1 理论-实践鸿沟

最令人意外的发现：定理 1 的充分条件（$K=4096$ 时 $\kappa < 8$）在每一层都被违反（中位 $\kappa = 3{,}485$），但方法完美有效。这 435 倍的鸿沟来自最坏情况确定性界与平均情况概率行为的区别。我们推测定理 1 的概率版本（$O(\sqrt{N})$ 误差缩放）将匹配实验观察。

### 7.2 Dense 模型：已解决的问题

在 HuggingFace 推理下，Dense transformer 的批次不变生成确定性通过单个 cuBLAS flag 以零成本实现。

### 7.3 MoE 模型：开放问题

FP32 累加是必要但不充分的。top-k 路由的不连续性放大了亚 ULP 扰动。解决方案需要：(a) 具有 tie-breaking 规则的确定性 top-k；(b) 更宽的路由边际（如温度缩放）；(c) gate GEMM 使用 FP64 累加。

### 7.4 局限性

- **硬件**：所有实验在 Ampere (A6000) 上。Hopper (H100) 的不同 cuBLAS 启发式可能有不同行为
- **张量并行**：TP > 1 引入非确定性 NCCL 集合通信，需要单独处理
- **Serving engine**：零开销结果适用于 HuggingFace (SDPA)。使用 FlashDecoding 的生产引擎需要定理 3 的固定 split 干预
- **量化**：FP8/INT8 KV cache 引入额外精度边界，不在我们的分析范围内

---

## 8. 结论

LLM 推理中的批次依赖非确定性，在 Dense 模型上有一个出人意料的简单解决方案：单个 cuBLAS flag 强制 FP32 累加，延迟开销为零。我们的理论框架（定理 1-3）刻画了它何时有效（加法规约）、何时失效（split-KV attention、MoE 路由）、以及为什么实际效果远超理论保证（充分条件保守 435 倍）。对 MoE 模型，FP32 累加减少但不消除非确定性，routing 确定性仍是开放问题。

**实践建议**：对任何 BF16 LLM 推理负载，无条件设置：
```python
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
```
没有任何成本。

---

## 附录 A：实验细节

**硬件**：NVIDIA RTX A6000 (Ampere, 48 GB, 84 SMs)。CUDA 12.4，PyTorch 2.6.0。

**Dense 模型**：Llama-3.1-8B-Instruct, BF16, 32 层, hidden 4096, 32 heads, GQA 8 KV heads, vocab 128,256。SDPA attention 后端。

**MoE 模型**：DeepSeek-V2-Lite, BF16, 15.7B 参数, 64 routed experts, top-6, expert hidden 2048。

**生成设置**：Greedy decoding, 32 new tokens。Continuous batching 模拟：所有序列等长，显式 position_ids。

## 附录 C：图表

- **图 1**：GEMM 批次方差（BF16 vs FP32）随批次大小变化
- **图 2**：1000 次运行 hash 分布（BF16 2 种 vs FP32 1 种）
- **图 3**：算子级 FP32 有效性热力图
- **图 4**：MoE 路由误差放大链
- **图 5**：Dense vs MoE 确定性对比
