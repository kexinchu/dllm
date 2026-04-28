# Related Work 深度整理 + 数据缺口分析 + 话题拓展方向

---

## 一、Related Work 全景（按时间线）

### 1.1 学术论文

| 工作 | 时间 | 核心方法 | 根因归因 | 确定性 | Overhead | 局限 |
|---|---|---|---|---:|---:|---|
| **Ingonyama** | 2024 | 自定义 CUDA GEMM（避开 Tensor Core） | FP non-associativity | 100% (跨机器) | **~7× 慢** (prompt) | 性能不可接受 |
| **Yuan et al.** (NeurIPS) | 2025.06 | LayerCast（BF16→FP32 JIT cast） | BF16 精度不足 | "near-perfect" | +5-15% | 只测 3 个 bs；根因归因不完整 |
| **LLM-42** (Microsoft) | 2025.01 | Decode-Verify-Rollback（不改 kernel） | Kernel 选择 | 100% (per-request) | **+3%** (2% det traffic) | 需要 rollback；不保证每个 token |

### 1.2 工业系统

| 工作 | 时间 | 核心方法 | 确定性 | Overhead |
|---|---|---|---:|---:|
| **Thinking Machine Lab** (Horace He) | 2025.09 | batch_invariant_ops（自定义 GEMM+Attn+RMSNorm） | 100% | 61.5% → 42s/26s |
| **SGLang Deterministic** | 2025.09 | 集成 batch_invariant_ops + FlashInfer/FA3 | 100% | 34.35% |
| **vLLM Batch Invariance** | 2025 | FlexAttention + batch_invariant_ops | 100% | 20-34% |
| **EigenAI** | 2025.02 | warp-sync reduction（Hopper 专属） | 100% | ~5% |

### 1.3 我们的定位

| 维度 | 现有工作 | 我们的差异化 |
|---|---|---|
| **根因分析深度** | Horace: "kernel selection"（博客级） | **代码级追踪 + bs=1..256 精确映射** |
| | Yuan: "BF16 precision"（不完整） | **证明精度不是根因** |
| **精度-确定性关系** | 所有工作假设"高精度→更确定" | **首次证明可能恶化（反例）** |
| **测试粒度** | Yuan: 3 bs; SGLang: 50 runs | **256 bs 逐一扫描 + 10,000 runs** |
| **方案成本** | 最低 5% (EigenAI, Hopper only) | **< 0.2% (通用 GPU)** |

---

## 二、v5 论文的关键问题

### 问题 1：DetermLLM 的 "100% 确定性" claim 过于依赖 smart batch padding

**审稿人质疑**：
- "Batch padding 只是避开了 kernel 切换边界，不是解决了 kernel 选择问题"
- "切换边界随模型/GPU/驱动版本变化，profiling 需要反复做"
- "对 MoE 不适用就意味着方案不通用"
- "本质上和 LLM-42 的 'avoid the problem' 类似，只是粒度不同"

**确实如此**——smart batch padding 是一个 workaround，不是 solution。

### 问题 2：缺乏足够的"解决"——论文更像 analysis paper

v5 的核心贡献是**分析**（根因追踪、kernel 边界映射、反例发现），不是**解决方案**。对 NeurIPS 来说：
- 纯 analysis paper 需要非常深的理论或非常广的 empirical coverage
- v5 的理论薄弱（Theorem 只是非形式化直觉）
- empirical 只有 2 个模型 × 1 种 GPU

### 问题 3：与 LLM-42 (Microsoft) 的区分不够

LLM-42 也认识到 kernel selection 是根因，提出了更优雅的 decode-verify-rollback 方案（+3% overhead）。我们需要说清楚与他们的区别。

### 问题 4：实验覆盖面不足

| 需要 | 现状 | 影响 |
|---|---|---|
| 多 GPU 架构 | 仅 A6000 (Ampere) | 切换边界可能完全不同 |
| 多模型 | Llama + DeepSeek-V2-Lite | 太少 |
| 多 prompt | 1 个固定 prompt | 切换可能 prompt-dependent |
| 多序列长度 | 主要 10 tokens | prefill 阶段未覆盖 |
| Serving engine | 仅 HuggingFace | 实用性有限 |

---

## 三、话题拓展方向分析

### 方向 A：深化根因分析——做成 "Empirical Study" 论文

**目标**：不提出新方案，而是做最全面的 LLM 推理非确定性实证研究

**优势**：
- 不需要与 LLM-42/batch_invariant_ops 竞争"方案"
- 我们的 kernel 边界映射、精度-确定性反例、flag 穷举都是独特贡献
- empirical study 在 NeurIPS 有传统（Henderson et al. "Deep RL That Matters"）

**需要补充**：
- 3+ 种 GPU（A6000/A100/H100 或用 cloud）
- 5+ 种模型（Llama-3.1-8B, Qwen-7B, Mistral-7B, DeepSeek-V2-Lite, Phi-3）
- 多 prompt 验证（切换边界是否 prompt-independent）
- cuBLAS 版本影响（CUDA 11 vs 12 的 kernel 池不同）
- NVIDIA profiling（nsight-compute）直接观察 kernel name 变化

**论文标题可能**："Kernel Selection, Not Precision: An Empirical Study of Batch Non-Determinism in LLM Inference"

### 方向 B：做成"方案+系统"论文——需要真正的 kernel 级方案

**目标**：实现一个通用的 batch-invariant GEMM 方案，性能优于现有工作

**需要**：
- 自定义 cuBLASLt wrapper：对给定 (N,K)，profile 一次最优 algorithm，后续所有 M 都使用该 algorithm
- 或 Triton 固定 tiling GEMM：编译时确定 tile 大小，不随 M 变化
- 在 vLLM/SGLang 中集成并测试
- 与 batch_invariant_ops 对比性能

**工作量大但 impact 高**。EigenAI 的 warp-sync reduction (+5%) 是标杆。

### 方向 C：结合 A 和 B——"分析 + 轻量级方案"

**最可行路线**：

```
论文结构：
1. 深入的根因分析（我们的独特贡献）
   - 代码级追踪
   - Kernel 边界精确映射
   - 精度≠确定性反例

2. 轻量级方案（不与重方案竞争）
   - FP32 flag（零成本，已有）
   - cuBLASLt algorithm pinning（新的，需要实现）
   - 对比: 我们的轻量方案 vs 重方案的 Pareto 前沿

3. 实践指南（面向从业者）
   - 何时需要什么级别的确定性
   - 各方案的适用场景
   - profiling 工具和方法
```

---

## 四、最缺的数据和实验

### 优先级 P0（必须有）

| 实验 | 原因 | 预计时间 |
|---|---|---|
| **多 prompt 验证** | 确认切换边界是否 prompt-independent | 2 小时 |
| **不同序列长度的切换边界** | prefill 阶段 M 更大 | 1 小时 |
| **nsight-compute kernel name 观察** | 直接证明"kernel 切换"而非推测 | 30 分钟 |
| **cuBLASLt algorithm pinning 实现** | 唯一的真正技术贡献 | 1-2 天 |

### 优先级 P1（强烈建议）

| 实验 | 原因 | 预计时间 |
|---|---|---|
| 多模型（Qwen-7B, Mistral-7B） | 泛化性 | 每模型 4 小时 |
| 不同 CUDA 版本 | kernel 池是否变化 | 需要环境 |
| vLLM 集成测试 | 实用性 | 1-2 天 |
| 下游 RL reward variance | motivation 量化 | 1 天 |

### 优先级 P2（锦上添花）

| 实验 | 原因 |
|---|---|
| A100/H100 对比 | 跨架构泛化 |
| FP8 的影响 | 前沿精度格式 |
| Speculative decoding 交互 | 新兴推理技术 |

---

## 五、我的推荐

**走方向 C**，核心叙事调整为：

> "现有工作要么归因不完整（Yuan et al. 认为是精度问题），要么方案过重（batch_invariant_ops +20-34%），要么不通用（EigenAI 仅 Hopper）。我们通过代码级追踪首次精确定位根因为 cuBLAS kernel 选择启发式，并证明精度提升可能恶化确定性。基于此分析，我们提出 cuBLASLt algorithm pinning——一种轻量级、通用、近零成本的方案，在 3 种模型 × 2 种 GPU 上实现 100% batch invariance。"

**cuBLASLt algorithm pinning** 是关键技术贡献：
- 对每个 (N, K) shape，在 bs=1 时用 `cublasLtMatmulAlgoGetHeuristic` 获取最优 algorithm
- 后续所有 M 值都强制使用该 algorithm（`cublasLtMatmul` with explicit algo）
- 理论上零性能损失（same algorithm, just pinned）
- 但可能对某些 M 值不是最优 → 需要测量实际 overhead
- 实现方式：PyTorch C++ extension，monkey-patch `nn.Linear`

这比 smart batch padding 更优雅，比 batch_invariant_ops 更轻量，是一个真正的技术贡献。

---

## 六、需要确认的几个关键问题

1. **切换边界是否 prompt-independent？** 如果是 prompt-dependent，那我们的所有分析都需要 qualify
2. **cuBLASLt algorithm pinning 是否可行？** 需要验证 API 是否支持跨 M 固定 algorithm
3. **固定 algorithm 的性能损失有多大？** 如果 > 10%，优势不明显
4. **多 prompt 下生成确定性是否一致？** 可能不同 prompt 的 logit margin 不同，影响 argmax 稳定性
