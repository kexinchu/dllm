# Research Plan v3: Deterministic LLM Inference

## 核心发现（Phase 0 结论）

FP32 accumulation **不能消除** cuBLAS kernel 切换，只是改变了切换模式：
- BF16: 6 个切换点，bs=1 就与 bs=2 不同
- FP32: 4 个切换点，bs=1-56 稳定，但 bs=57+ 仍有切换

**根因不是 accumulation 精度，而是 cuBLAS 的 kernel 选择启发式本身。**
不同 batch size → 不同 kernel algorithm (不同 tiling/split-K) → 不同规约顺序 → 不同结果。
FP32 accum 缩小了"不同规约顺序"导致的误差范围，但当 kernel algorithm 完全不同时，
仍然可能穿透 BF16 rounding quantum。

---

## 论文新定位

**Title**: Deterministic LLM Inference: Anatomy, Analysis, and a Practical System

**核心叙事**:
LLM 推理中的 batch non-determinism 来自 5 个独立机制。
我们逐一分解、分析、提出针对性方案，并整合为一个实用系统，
在不同场景下提供 deterministic inference，同时最小化性能开销。

**与旧版本的区别**：
- 旧版：FP32 flag 零成本解决 → 实际不成立
- 新版：系统化分解 + 逐个击破 + 整合为实用方案

---

## 论文结构

### §1 Introduction
- Batch non-determinism 问题定义与实际影响
- 现有方案（vLLM 20-35%, SGLang 34%）的开销过高
- 我们的目标：理解根因，最小化开销实现确定性

### §2 Background
- BF16/FP32 精度、非结合律、cuBLAS kernel 选择
- Transformer 规约操作分类
- Continuous batching 与 serving engine 架构

### §3 Anatomy: 五种非确定性来源

**§3.1 Source 1: GEMM Kernel Selection**
- Observation 1: cuBLAS 在不同 M 下选择不同 kernel algorithm
- 实验: 精细扫描 bs=1..256，映射 kernel 切换边界
- Observation 2: FP32 accum 减少切换次数（6→4）但不消除
- Observation 3: K 维度越大，FP32 残余误差越大（K=2048 完美，K=4096 有残余）
- 根因分析: 不同 kernel = 不同 tiling/split-K = 结构性不同的计算图

**§3.2 Source 2: Reduction Ops (RMSNorm, Softmax)**
- Observation 4: 分块规约在不同 batch 下可能走不同分块策略
- 实验: RMSNorm chunk variance, Softmax chunk variance
- FP32 效果: 完美消除（κ=1 for RMSNorm, 近似完美 for Softmax）

**§3.3 Source 3: Attention Split-KV**
- Observation 5: 不同 split 数 = 不同计算图（乘法缩放链）
- 实验: 固定 split vs 变化 split，BF16 vs FP32
- FP32 效果: 完全无效（结构性问题，非精度问题）

**§3.4 Source 4: Layer Accumulation**
- Observation 6: 即使每个 op 只差 0.5 ULP，32 层累积后 logit 差异可达 6-9 ULP
- 实验: 逐层 tracking，每层的 logit diff 如何增长
- 关键区分: logit-level variance vs generation-level variance
- Observation 7: generation (argmax) 在大多数情况下仍然一致，因为 top-1 margin >> accumulated error

**§3.5 Source 5: MoE Routing Amplification**
- Observation 8: 100% token 位于 near-tie 区间（DeepSeek-V2-Lite）
- Observation 9: 0.5 ULP GEMM 残余 → softmax 放大 → top-k 翻转 expert
- FP32 效果: 必要但不充分

### §4 Theoretical Framework
- Theorem 1: 加法规约的 FP32 充分条件
- Theorem 1 的保守性分析（0/225 层满足，但多数情况有效）
- Probabilistic refinement: √N average-case bound
- Theorem 2: Split-KV 的结构不可修复性
- Theorem 3: 固定 split + FP32 恢复确定性

### §5 DetermLLM: A Practical Deterministic Inference System
（这是论文的核心 contribution — 一个组合方案）

**§5.1 Layer 1: GEMM Determinism**
- 方案 A: cuBLAS FP32 flag（零成本，消除大部分 kernel 切换）
- 方案 B: 固定 cuBLAS algorithm selection（禁用启发式）
  - CUBLAS_WORKSPACE_CONFIG=:4096:8 + torch.use_deterministic_algorithms(True)
  - 或: cublasSetMathMode 强制特定 algorithm
- 方案 C: 方案 A + B 组合
- 实验: 对比三种方案的确定性 + 开销

**§5.2 Layer 2: Reduction Op Determinism**
- RMSNorm: HuggingFace 已在 FP32 内部计算（零额外成本）
- Softmax: FP32 accumulation（零成本）
- 对 serving engine: 可能需要自定义 kernel

**§5.3 Layer 3: Attention Determinism**
- HuggingFace (SDPA): 已 batch-invariant（不用 split-KV）→ 零成本
- Serving engine: 固定 split-KV 边界（Theorem 3 方案）
- 实验: 固定 split=256 vs 动态 split 的确定性 + overhead

**§5.4 Layer 4: MoE Routing Determinism**
- 方案: FP32 gate accum + deterministic top-k（tie-breaking by index）
- 实验: 在 DeepSeek-V2-Lite 上验证
- 局限性: near-tie 下 0.5 ULP 仍可能翻转（open problem）

**§5.5 Integration: DetermLLM System**
- 组合 Layer 1-4 的最佳方案
- 配置: 一组 flags/settings 的组合
- 适用范围: HuggingFace / vLLM / SGLang

### §6 Evaluation

**§6.1 Generation Determinism (大规模测试)**
- Llama-3.1-8B: 10000 runs, bs=1..256 随机
- 对比: BF16 baseline / FP32 only / DetermLLM full
- 指标: unique outputs, kernel 切换次数

**§6.2 MoE Determinism**
- DeepSeek-V2-Lite: 1000 runs
- Qwen3.5-35B-A3B: 500 runs (if available)

**§6.3 Performance Overhead**
- 逐 layer 开销分解
- 与 vLLM/SGLang 确定性模式对比

**§6.4 Downstream Impact**
- RL reward variance
- Distillation KL divergence
- MoE expert flip rate

### §7 Related Work
### §8 Conclusion

---

## 实验计划

### Phase 1: Motivation Observations (§3 的数据)

| ID | 实验 | §对应 | GPU需求 | 预计时间 |
|----|------|-------|---------|----------|
| M1 | GEMM kernel 切换边界精确映射: bs=1..256, BF16/FP32, 输出每个 bs 的 generation hash | §3.1 | 1 GPU | 已完成(V1c) |
| M2 | GEMM K-dim 影响: K=[1024,2048,4096,8192,14336], M=[1,32,64,128,256], BF16/FP32 | §3.1 | 1 GPU | 15 min |
| M3 | RMSNorm/Softmax chunk variance: dim=[2048,4096], chunks=[1..64], BF16/FP32 | §3.2 | 1 GPU | 5 min |
| M4 | Attention split-KV: splits=[1,2,4,8,16,32], seq=[128,512,2048], BF16/FP32 | §3.3 | 1 GPU | 10 min |
| M5 | 逐层 error accumulation: hook 每层输出，记录 logit diff 随层增长 | §3.4 | 1 GPU | 15 min |
| M6 | MoE near-tie + expert flip: DeepSeek-V2-Lite, 500 tokens, 计算 near-tie prevalence | §3.5 | 2 GPU | 20 min |

### Phase 2: DetermLLM System 实验 (§5 的数据)

| ID | 实验 | §对应 | GPU需求 | 预计时间 |
|----|------|-------|---------|----------|
| S1 | GEMM 三种方案对比: (A) FP32 flag, (B) deterministic algo, (C) A+B | §5.1 | 1 GPU | 30 min |
| S2 | torch.use_deterministic_algorithms(True) 能否消除 kernel 切换? | §5.1 | 1 GPU | 20 min |
| S3 | CUBLAS_WORKSPACE_CONFIG 对 GEMM 确定性的影响 | §5.1 | 1 GPU | 15 min |

### Phase 3: End-to-End Evaluation (§6 的数据)

| ID | 实验 | §对应 | GPU需求 | 预计时间 |
|----|------|-------|---------|----------|
| E1 | 10000-run Llama: BF16 / FP32 / DetermLLM, bs=1..256 随机 | §6.1 | 1 GPU | 3-4 hours |
| E2 | 1000-run DeepSeek-V2-Lite MoE: 同上 | §6.2 | 2 GPU | 1.5 hours |
| E3 | Latency benchmark: 各方案的 overhead | §6.3 | 1 GPU | 30 min |
| E4 | Downstream impact: RL/KD/MoE flip, BF16 vs FP32 vs DetermLLM | §6.4 | 1 GPU | 30 min |

### Phase 4: 论文撰写

基于 Phase 1-3 的数据，重写论文。

---

## 关键技术方案详细设计

### GEMM Determinism (§5.1)

```python
# 方案 A: FP32 accumulation (零成本)
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

# 方案 B: 固定 cuBLAS algorithm (可能有少量开销)
torch.use_deterministic_algorithms(True)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# 方案 C: A + B 组合 (预期最佳确定性)
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
torch.use_deterministic_algorithms(True)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
```

核心问题: 方案 B 是否真的固定了 kernel 选择？还是只影响 backward？
需要 S2 实验验证。

### Attention Determinism (§5.3)

对 HuggingFace SDPA: 无需额外处理（已 batch-invariant）
对 serving engine: 需要固定 split-KV 边界

### MoE Determinism (§5.4)

```python
# Deterministic top-k: 在 score 相等时按 index 排序
def deterministic_topk(scores, k):
    # 添加微小的 index-based perturbation 打破 tie
    n_experts = scores.shape[-1]
    tiebreaker = torch.arange(n_experts, device=scores.device) * 1e-10
    return torch.topk(scores + tiebreaker, k, dim=-1)
```

---

## 执行顺序

```
Phase 1 (并行):  M2, M3, M4 同时跑 (op-level, 不需要模型)
                  M5 跑 (需要 Llama)
                  M6 跑 (需要 DeepSeek)

Phase 2 (串行):  S1 → S2 → S3 (需要根据 S1 结果调整 S2)

Phase 3 (串行):  E1 (最耗时, 3-4 hours) → E2 → E3 → E4

Phase 4:         论文撰写
```

总计 GPU 时间: ~8-10 hours
可并行部分: Phase 1 的 op-level 实验

---

## 风险评估

| 风险 | 影响 | 缓解 |
|------|------|------|
| 方案 B (deterministic algo) 不影响 forward | 高: 无法固定 GEMM kernel | 需要研究 cuBLAS API 的 algo selection |
| 10000-run 发现更多 FP32 切换点 | 中: 论文叙述调整 | 已预期，论文已按"不完美"设计 |
| Qwen3.5 下载失败 | 低: 只影响 MoE 泛化性 | DeepSeek-V2-Lite 作为主 MoE 模型 |
| Deterministic top-k 无法解决 MoE | 低: 已定位为 open problem | 论文诚实讨论局限 |

---

## 预期论文 Contribution

1. **首次系统化分解** LLM batch non-determinism 的 5 种来源
2. **精确映射** cuBLAS kernel 切换边界（BF16: 6 transitions, FP32: 4 transitions in bs=1..256）
3. **理论框架** (Theorems 1-3) + 理论-实践鸿沟分析
4. **DetermLLM 系统**: 组合方案，逐层消除非确定性
5. **大规模验证**: 10000-run 测试，多模型（dense + MoE）
6. **下游影响量化**: RL reward variance, distillation KL, MoE routing
