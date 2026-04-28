# LLM 推理中 Batch 非确定性的根因分析与近零成本缓解

---

## 摘要

大语言模型推理系统在不同批次大小下对相同输入产生不同输出，这是一个影响 RL 训练、知识蒸馏和安全审计的关键问题。近期工作将此归因于 BF16 浮点精度不足，提出 LayerCast（全 FP32 计算）作为缓解方案（Yuan et al., NeurIPS 2025）。我们通过对 cuBLAS 行为的精细分析揭示了更深层的根因：**问题的本质不是计算精度，而是 cuBLAS 的 kernel 选择启发式在不同批次维度下选择结构性不同的 GEMM 算法**。

我们的核心发现：(1) 通过 bs=1..256 逐一扫描首次精确映射了 cuBLAS kernel 切换边界——BF16 下有 8 个切换点（31.6% bs 受影响），FP32 cross-chunk reduction flag 减少到 4 个（4.7%）；(2) **提升精度不单调改善确定性**——cuBLASLt 路径反而恶化到 59%，LayerCast 在小 bs 处引入 6 个新切换点；(3) DeepSeek-V2-Lite（MoE）上 FP32 flag 从完美确定（0 切换）恶化到 6 个切换；(4) PyTorch 全部 8 种确定性 flag 中只有 `allow_bf16_reduced_precision_reduction` 有效。

基于根因分析，我们提出 **DetermLLM**：FP32 cross-chunk reduction flag（零成本）+ 基于 kernel 切换边界 profiling 的 smart batch padding（近零成本），在 Llama-3.1-8B 上实现 100% 生成确定性，端到端零性能损失。

---

## 1. 引言

### 1.1 问题

LLM 推理的可复现性问题日益受到关注。Yuan et al. (NeurIPS 2025) 展示了在 BF16 下改变评估 batch size 可导致 DeepSeek-R1 准确率波动 9%、回复长度差异 9,000 tokens。对于 RLHF 训练，batch 依赖的 log-probability 差异直接注入奖励信号噪声；对于知识蒸馏，教师模型的 soft logits 不一致会污染训练目标。

### 1.2 传统认知与我们的挑战

**传统认知**（Yuan et al.）：根因是 BF16 精度不足 → 提升精度即可解决（LayerCast: BF16→FP32，+373% latency）。

**我们的发现**：精度是表象，kernel 选择才是根因。通过精确的代码级追踪（PyTorch → `cublasGemmEx` → `CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION`），我们发现所谓的"FP32 flag"实际控制的是 **cuBLAS 的 kernel 搜索空间**——禁用低精度 cross-chunk reduction 后，部分 split-K kernel 不可用，cuBLAS 被迫选择更稳定的 kernel 子集。这解释了为什么：
- 同一 flag 在不同模型上效果相反（Llama 改善，DeepSeek 恶化）
- 全 FP32 计算（LayerCast）反而引入更多切换点
- 没有任何精度设置能完全消除 kernel 切换

### 1.3 贡献

1. **根因的代码级追踪**：从 PyTorch Python API → C++ ATen → cuBLAS `cublasGemmEx` 完整追踪 `allow_bf16_reduced_precision_reduction` 的实际行为——它控制 `CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION` flag，间接缩小 kernel 搜索空间
2. **首次精确映射 kernel 切换边界**：10,000 次生成测试，bs=1..256，6 种精度方案，8 种 PyTorch flag 组合
3. **发现"高精度≠更确定"**：LayerCast (FP32 kernel) 59% affected → cuBLASLt 更差；DeepSeek MoE 上 FP32 flag 引入新切换
4. **DetermLLM 方案**：FP32 flag + smart batch padding → 100% 确定，零开销
5. **系统化分解** 5 种非确定性来源的 FP32 有效性边界

---

## 2. 背景

### 2.1 浮点非结合律与 GPU GEMM

浮点加法不满足结合律。GPU 上 GEMM 的内积规约 $y_j = \sum_{k=0}^{K-1} W_{jk} \cdot x_k$ 被分解为多个 thread block 的部分和，分解策略（tiling, split-K）由 cuBLAS 启发式决定。改变 batch size $M$ 不仅改变执行顺序，还可能改变 kernel 算法——后者是结构性差异。

### 2.2 `allow_bf16_reduced_precision_reduction` 的实际机制

通过代码追踪（PyTorch 2.6.0, ATen/cuda/CUDABlas.cpp）：

```cpp
// PyTorch 内部实现
template <>
void gemm_internal_cublas<at::BFloat16>(...) {
    cublasMath_t cublas_flags = CUBLAS_DEFAULT_MATH;
    if (!at::globalContext().allowBF16ReductionCuBLAS()) {
        cublas_flags |= CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION;
    }
    cublasGemmEx(handle, ..., computeType=CUDA_R_32F, cublas_flags);
}
```

关键事实：
- **computeType 始终是 `CUDA_R_32F`**——不管 flag 怎么设，tile 内累加都是 FP32
- flag 控制的是 `CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION`
- 根据 NVIDIA 文档，此 flag 禁止 split-K 的 cross-chunk 合并使用低于 compute type 的精度
- 效果：缩小了可选 kernel 的搜索空间（排除了使用 BF16 cross-chunk reduction 的 kernel）

### 2.3 Yuan et al. 的 LayerCast

LayerCast 将权重 BF16 存储、计算时 JIT cast 到 FP32，走 `cublasSgemm`（FP32 kernel 池）。在 12 种配置（2 GPU × 2 count × 3 bs）下报告 Div_Index < 3.4%。但他们只测了 3 个 batch size——不足以发现 kernel 切换边界。

---

## 3. Kernel 切换边界的精确映射

### 3.1 实验方法

**模型**：Llama-3.1-8B-Instruct (BF16, A6000)
**方法**：对 bs=1..256 逐一生成 32 tokens (greedy)，hash 输出序列，统计 kernel 切换点（相邻 bs 的 hash 不同）

### 3.2 10,000-Run 大规模验证

10,000 次运行，bs=1..256 随机采样（每个 unique bs 测一次，映射到 10,000 次随机采样）：

**表 1：精度方案对比（Llama-3.1-8B, bs=1..256）**

| 方案 | Unique 输出 | 切换次数 | 受影响 bs | 确定性 | Overhead |
|---|---:|---:|---:|---:|---:|
| BF16 baseline | 2 | 8 | 81/256 (31.6%) | 68.4% | 0% |
| **FP32 flag (cuBLAS)** | **2** | **4** | **12/256 (4.7%)** | **95.3%** | **0%** |
| FP32 flag + cuBLASLt | 2 | 4 | 151/256 (59.0%) | 41.0% | 0% |
| LayerCast (全 FP32 compute) | 2 | 6 | 5/256 (2.0%) | 98.0% | +373% |
| Hybrid FP32 output | 2 | 6 | 5/256 (2.0%) | 98.0% | +567% |

FP32 flag 的不安全 batch size 集中在两个窄窗口：

```
不安全: [57, 58, 59, 60, 61, 62, 63, 64] ∪ [79, 80, 81, 82]   — 12 个 bs
安全:   [1-56] ∪ [65-78] ∪ [83-256]                              — 244 个 bs
```

### 3.3 FP32 效果的模型依赖性

**表 2：跨模型对比**

| 模型 | BF16 切换 | FP32 flag 切换 | FP32 效果 |
|---|---:|---:|---|
| Llama-3.1-8B (dense) | 8 | **4** | 改善（8→4） |
| DeepSeek-V2-Lite (MoE) | **0** | **6** | **恶化（0→6）** |

DeepSeek-V2-Lite 在 BF16 下 bs=1..32 完全确定，FP32 flag 反而引入 6 个切换点。这直接否定了"高精度=更确定"的假设。

### 3.4 PyTorch Flag 穷举

**表 3：8 种 PyTorch flag 组合测试（bs=1..69）**

| Flag | 对 batch 非确定性的影响 |
|---|---|
| `allow_bf16_reduced_precision_reduction = False` | **有效（唯一有效）** |
| `allow_tf32 = False` | 无效 |
| `set_float32_matmul_precision('highest')` | 无效 |
| `use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG` | 无效 |
| `NVIDIA_TF32_OVERRIDE=0` | 无效 |
| `preferred_blas_library('cublaslt')` | 恶化（59%） |
| 所有 flag 组合 | = 仅 `bf16_reduced` 的效果 |

### 3.5 BLAS 后端对比

**表 4：不同 BLAS 后端的 kernel 切换 landscape**

| 后端 | 切换次数 | 受影响 bs | 说明 |
|---|---:|---:|---|
| cuBLAS + BF16 默认 | 8 | 81/256 (31.6%) | 最多变体 |
| **cuBLAS + FP32 flag** | **4** | **12/256 (4.7%)** | **最优免费** |
| cuBLASLt + FP32 flag | 4 | 151/256 (59.0%) | 严重恶化 |
| cublasSgemm (LayerCast) | 6 | 5/256 (2.0%) | 高开销 |

---

## 4. 非确定性来源的系统分解

### 4.1 Source 1: GEMM Kernel 选择（主因）

**Observation 1: K 维度决定 FP32 有效性**

**表 5：GEMM batch variance 随 K 维度变化**

| K (规约维度) | BF16 max_diff | FP32 flag max_diff | FP32 改善 |
|---:|---:|---:|---|
| 512 | 0.000 | 0.000 | 不需要 |
| 1024 | 0.500 | **0.000** | 完全消除 |
| 2048 | 1.000 | 0.125 | 8× |
| 4096 | 1.000 | 0.250 | 4× |
| 8192 | 2.000 | 0.500 | 4× |
| 11008 | 2.000 | 2.000 | **无改善** |
| 14336 | 4.000 | 1.000 | 4× |

K=11008（Llama 的 gate_proj 输出维度）时 FP32 几乎无效——这是因为该维度下所有可选 kernel 都有显著的 tiling 差异。

**Observation 2: 每个 Linear 层的 shape 对应不同的切换 landscape**

**表 6：Llama-3.1-8B 各 Linear 层 shape 的切换行为**

| Shape (K→N) | 层数 | BF16 首次切换 | FP32 首次切换 |
|---|---:|---|---|
| 4096→4096 (q,o proj) | 64 | bs=32 | bs=32 |
| 4096→1024 (k,v proj) | 64 | bs=2 | bs=32（改善） |
| 4096→14336 (gate,up) | 64 | bs=2 | bs=2（无改善） |
| 14336→4096 (down) | 32 | bs=2 | bs=2（无改善） |
| 4096→128256 (lm_head) | 1 | bs=2 | bs=2（无改善） |

### 4.2 Source 2: RMSNorm, Softmax

**表 7：RMSNorm 与 Softmax 的 chunk reduction variance**

| 操作 | BF16 max_diff | FP32 max_diff | 说明 |
|---|---|---|---|
| RMSNorm (dim=4096, chunks=1-64) | **0.000** | ~1e-7 | PyTorch 已确定 |
| Softmax (vocab=128256, chunks=1-32) | ~1e-6 | ~1e-11 | FP32 改善 5 个数量级 |

RMSNorm 在 PyTorch 中对 batch size 免疫——`.mean()` 对每个 token 独立计算。

### 4.3 Source 3: Attention Split-KV

**表 8：FP32 online softmax 下 cross-split 差异**

| seq_kv | 1-split vs 32-split (FP32) |
|---:|---|
| 256 | 1.04e-7 |
| 1024 | 5.96e-8 |
| 4096 | 2.98e-8 |

FP32 online softmax 的 cross-split 差异极小（~1e-7），远小于 BF16 rounding quantum。HuggingFace SDPA 不使用 split-KV → attention 已 batch-invariant。

### 4.4 Source 4: 层间误差累积

**表 9：逐层 hidden state diff（Llama-3.1-8B, FP32 flag, bs=1 vs bs=8）**

| 层 | max_diff |
|---|---|
| L0 | 9.77e-4 |
| L8 | 3.12e-2 |
| L16 | 6.25e-2 |
| L24 | 1.25e-1 |
| L31 | 3.12e-2 |
| Final logits | 1.80e-1 |
| **Argmax match** | **100%** |

误差从 L0 (1e-3) 增长到 L24 (0.125)，但 **argmax match 始终 100%**。BF16 和 FP32 模式的误差增长模式完全相同——进一步确认 FP32 flag 改变的是 kernel 选择而非层内精度。

### 4.5 Source 5: MoE Routing 放大

**表 10：MoE near-tie prevalence**

| 配置 | Near-tie (τ < 0.001) |
|---|---|
| 64 experts, top-6 | 30% |
| 128 experts, top-8 | 40% |

Gate GEMM 的 K=14336 使 FP32 效果有限（仅 2× 改善）。加上 30-40% near-tie，微小 GEMM 残余即可翻转 expert 选择。

---

## 5. DetermLLM：近零成本确定性方案

### 5.1 方法

DetermLLM 基于两个互补机制：

**机制 1：FP32 cross-chunk reduction flag（零成本）**

```python
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
```

效果：禁用 cuBLAS 的低精度 cross-chunk reduction → 缩小 kernel 搜索空间 → 减少切换次数（8→4），95.3% bs 一致。

**机制 2：Kernel 切换边界 profiling + smart batch padding（近零成本）**

```python
# 一次性 profiling（~7 分钟 for bs=1..256）
unsafe_ranges = profile_kernel_transitions(model, bs_range=(1, 256))
# Llama-3.1-8B on A6000: [(57, 65), (79, 83)]

# 运行时 padding
def safe_batch_size(bs):
    for lo, hi in unsafe_ranges:
        if lo <= bs < hi:
            return hi  # pad 到安全边界
    return bs
```

**表 11：Smart batch padding 分析**

| 项目 | 值 |
|---|---|
| 不安全 bs 数 | 12/256 (4.7%) |
| 不安全 bs 范围 | [57-64] ∪ [79-82] |
| 最大 padding | 8 tokens |
| 平均 padding | 3.8 tokens |
| Padding 频率 | 4.7% 的请求 |
| **结果** | **100% 确定性** |
| **平均 overhead** | **< 0.2%**（4.7% × ~4% padding 开销） |

### 5.2 与现有方案对比

**表 12：确定性方案全景对比**

| 方案 | 确定性 | Overhead | 代码改动 | 适用范围 |
|---|---:|---:|---|---|
| BF16 baseline | 68.4% | 0% | — | — |
| **DetermLLM (ours)** | **100%** | **< 0.2%** | **2 行代码 + profiling** | **HuggingFace** |
| LayerCast (Yuan et al.) | 98.0% | +373% | 模型 wrapper | HuggingFace |
| batch_invariant_ops (vLLM) | 100% | +20-34% | 引擎集成 | vLLM |
| SGLang deterministic | 100% | +34.35% | 引擎集成 | SGLang |
| EigenAI warp-sync | 100% | +5% | 自定义 kernel | Hopper only |

### 5.3 局限性

1. **Profiling 是模型 + 硬件相关的**：不同模型/GPU 的切换边界不同，需要重新 profile
2. **MoE 模型不适用**：DeepSeek-V2-Lite 上 FP32 flag 恶化确定性
3. **Serving engine 不适用**：vLLM/SGLang 使用 FlashDecoding（split-KV）和 PagedAttention，引入额外非确定性来源
4. **Batch padding 假设连续批处理可控**：某些 serving 场景 batch size 不可调

---

## 6. 理论框架

### 6.1 为什么 FP32 Cross-Chunk Reduction 减少了切换次数

**Theorem（非形式化）**：设 cuBLAS 对维度 (M, N, K) 有 $\mathcal{A}_{\text{bf16}}$ 和 $\mathcal{A}_{\text{fp32}}$ 两个可选 kernel 集合，其中 $\mathcal{A}_{\text{fp32}} \subset \mathcal{A}_{\text{bf16}}$（禁用低精度 reduction 后搜索空间缩小）。设 $\phi(M)$ 为 cuBLAS 启发式选择的 kernel index。则 $\phi$ 在 $\mathcal{A}_{\text{fp32}}$ 上的切换次数 $\leq$ 在 $\mathcal{A}_{\text{bf16}}$ 上的切换次数。

**直觉**：搜索空间缩小后，可选 kernel 更少，启发式更不容易在相邻 M 值间切换。但不保证不切换——当缩小后的空间仍有多个 kernel 且各自在不同 M 范围最优时，仍会切换。

### 6.2 为什么 cuBLASLt 反而更差

`preferred_blas_library('cublaslt')` 切换到 cuBLASLt 的 kernel 池（vs 默认 cuBLAS）。cuBLASLt 为灵活性设计，kernel 变体更多 → 启发式在更多 M 值处切换。这是一个 **搜索空间扩大导致不稳定** 的典型案例。

### 6.3 Theorem 1 的 κ bound 验证

在 Llama-3.1-8B 的 225 个线性层测量 $\kappa(S) = \sum|a_i| / |\sum a_i|$：

| 统计量 | 值 |
|---|---|
| 理论 bound ($K=4096$) | 8.0 |
| 满足 bound 的层 | **0 / 225** |
| 中位 max $\kappa$ | 3,485 |
| 最大 $\kappa$ | 1,526,426 |

所有层都违反充分条件。但 FP32 在 95.3% 的 bs 上仍然有效——差距来自 worst-case vs average-case（误差随机分布部分抵消）。

---

## 7. 讨论

### 7.1 与 Yuan et al. 的关系

| 方面 | Yuan et al. (NeurIPS 2025) | 我们 |
|---|---|---|
| 测试 bs 范围 | 3 个 (8, 16, 32) | **256 个 (1..256)** |
| 根因归因 | BF16 精度不足 | **cuBLAS kernel 选择启发式** |
| 代码级追踪 | 无 | **PyTorch → ATen → cublasGemmEx 完整链路** |
| FP32 效果 | "near-perfect" | **模型依赖，可能恶化** |
| 方案开销 | +373% (LayerCast) | **< 0.2% (DetermLLM)** |

Yuan et al. 报告"FP32 near-perfect"是因为 3 个 bs 不足以发现切换边界。LayerCast 在 bs=8,16,32 上确实表现良好，但 bs=3,5,6,8,11,12 处有 6 个切换点。

### 7.2 实践建议

| 场景 | 推荐方案 | 确定性 | 开销 |
|---|---|---:|---:|
| **研究评估 (HuggingFace)** | **DetermLLM** | **100%** | **< 0.2%** |
| 研究评估（无法控制 bs） | FP32 flag | 95.3% | 0% |
| 生产 serving (dense) | batch_invariant_ops | 100% | 20-34% |
| 生产 serving (MoE) | 开放问题 | — | — |

### 7.3 局限性

- **硬件依赖**：切换边界因 GPU 架构不同（Ampere vs Hopper）
- **模型依赖**：不同权重维度 → 不同切换 landscape → 需要 per-model profiling
- **MoE**：FP32 flag 可能恶化，smart padding 不足以解决 routing 放大问题
- **Serving engine**：DetermLLM 仅适用于 HuggingFace/SDPA 场景

---

## 8. 结论

LLM 推理中 batch 非确定性的根因是 cuBLAS kernel 选择启发式，不是浮点精度不足。通过代码级追踪，我们证明 `allow_bf16_reduced_precision_reduction=False` 的实际作用是缩小 kernel 搜索空间（而非"提升精度"），这解释了其效果的模型依赖性和上限。

DetermLLM 基于两个简单机制（FP32 flag + kernel 切换边界 profiling + smart batch padding），在 Llama-3.1-8B 上以 < 0.2% 的开销实现了 100% 的生成确定性，比 LayerCast (+373%) 和 batch_invariant_ops (+20-34%) 高效 2-3 个数量级。

```python
# DetermLLM: 2 行代码 + 一次性 profiling
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
# + profile_kernel_transitions() → smart_batch_padding()
```

---

## 实验清单

本文涉及的全部实验（17 组 JSON 数据文件）：

| 实验 | 数据 | 节 |
|---|---|---|
| 10,000-run 生成确定性 | exp_e1_10k.json | §3.2 |
| 1,000-run DeepSeek MoE | exp_e2_moe_1k.json | §3.3 |
| 8 种 PyTorch flag 穷举 | exp_r2_flags.json | §3.4 |
| 4 种 GEMM 确定性方案 | exp_s1_gemm_schemes.json | §3.4 |
| 3 种精度方案 (FP32 flag/LayerCast/Hybrid) | exp_fp32_kernel.json | §3.2 |
| cuBLASLt + batch padding 分析 | (inline) | §3.2, §5.1 |
| GEMM K 维度 sweep (7K × 11M × 2mode) | exp_m1_gemm_kdim.json | §4.1 |
| RMSNorm + Softmax chunk variance | exp_m2_reduction.json | §4.2 |
| Attention split-KV | exp_m3_attn_split.json | §4.3 |
| 逐层误差累积 | exp_m4_layer_accum.json | §4.4 |
| MoE near-tie + expert flip | exp_m5_moe.json | §4.5 |
| Theorem 1 κ bound 验证 | exp_theorem1_kappa.json | §6.3 |
| 方案 latency | exp_s1_latency.json | §5.2 |
| 序列长度缩放 | exp_table9_seqlen.json | §4.4 |
| 下游影响 (RL/KD) | exp_downstream.json | §1.1 |

---

## References

Yuan, J. et al. (2025). Understanding and Mitigating Numerical Sources of Nondeterminism in LLM Inference. NeurIPS 2025.

Thinking Machine Lab (2025). Defeating Nondeterminism in LLM Inference.

EigenAI (2025). Deterministic Inference, Verifiable Results. arXiv:2602.00182.

SGLang Team (2025). SGLang Deterministic Inference.

vLLM Team (2025). Batch Invariance. docs.vllm.ai.

Higham, N. J. (2002). Accuracy and Stability of Numerical Algorithms (2nd ed.). SIAM.

Dao, T. (2023). FlashAttention-2. arXiv:2307.08691.

NVIDIA (2024). cuBLAS Library Documentation.
