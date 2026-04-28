# SRP Performance Optimization Log

5-题 DeepSeek-R1-Distill-Qwen-7B / MATH500 验证 (greedy 256 tokens, bs ∈ {1, 2, 4, 8, 16})。

## 实验记录

### Baseline (BF16, no patch)
- Avg_Std@top1_prob: **3.97e-3**
- Total wall: 99.4 s
- Per-bs steady: 6.5 - 7.0 s

### Step 0: PyTorch FP32 attention fallback (initial Triton det_gemm Linear path)
- Avg_Std: **0** (strict bit-exact)
- Total: 150.2 s (51% overhead)
- Per-bs: 9.9 - 10.2 s

### Step A (failed): BF16 cuBLAS GEMV decode-attn path
- 假设 M=1 → cuBLAS 不 split-K → bs-invariant
- **错了**：cuBLAS 对 batched matmul 按 batch_count = B×H 选不同 algo
- Avg_Std: 4.04e-3 (BROKE bit-exactness)
- Reverted

### Step B: Triton decode-specialized kernel (M_q=1)
- 单 program per (B, H)，固定 BLOCK_N=64 顺序遍历 KV
- Avg_Std: **0** ✓ (维持 bit-exact)
- Total: 147.4 s (49% overhead)
- 改善 ~3 s vs Step 0；attention kernel 跟 PyTorch FP32 fallback 同速

### Step C trial 1: cuBLASLt fixed-algo Linear backend (default)
- cuBLASLt 两阶段：NONE → COMPUTE_TYPE → FP32 fallback
- 性能极好：Total 104.8 s (5% overhead)
- 但 Avg_Std: 2.02e-3（**仅减 50% vs BF16**，远离 FP32 floor 3.59e-7）
- 原因：cuBLASLt heuristic 对不同 M 选不同 tile config，即使 NONE scheme 也不严格 bit-exact

### Step C trial 2: hybrid M-based dispatch
- M ≤ 16: cuBLASLt（decode）
- M > 16: Triton det_gemm（prefill）
- Total 105.7 s, Avg_Std 2.02e-3 — 跟纯 cuBLASLt 几乎一样
- 因为 decode kernel 也不严格 bit-exact

### Step C trial 3: torch.compile (mode='reduce-overhead')
- 期望：dynamo 把 Triton kernels 收进 cudagraph
- 实际：dynamo 把 Triton kernel 当 graph break，无效
- Total: 1.17 s (vs no-compile 1.17 s — 0% improvement)

## 50% overhead 拆解

每 decode step 单层 11 个 kernel launch × 28 layers = 308 launches/step：
- BF16 (cuBLAS): ~10-20 us/launch → 308 × 15us = 4.6 ms/step dispatch
- SRP (Triton):  ~30-50 us/launch → 308 × 40us = 12.3 ms/step dispatch
- 差: ~7.7 ms/step × 256 steps = **2.0 s extra dispatch per generation**

加 attention kernel 本身 ~0.1ms/call × 28 layers × 256 steps = 0.7 s extra

总 ≈ 2.7 s 额外开销（实测 3.3 s）。

## 决定：Step D - CUDA Graphs

把整个 decode step capture 成 CUDA graph，单次 launch 整个 graph，跳过 per-kernel dispatch。预计：
- 节省 dispatch overhead ~2 s
- 50% → ~25-30% overhead（如能配合 attention kernel 优化，可压到 15-20%）

## Trade-off 总结表

5-题 DeepSeek-7B / MATH500 / bs ∈ {1,2,4,8,16}：

| Backend | Avg_Std | Total | Overhead | 评价 |
|---|---|---|---|---|
| HF BF16 (DynamicCache) | 3.97e-3 | 194s | 0% (HF ref) | 我们的 baseline |
| HF + cuBLASLt fixed-algo | 2.02e-3 | 105s | -2% (more vars) | std 不够强 |
| HF + cuBLASLt + Triton attn | 2.51e-3 | 109s | -1% | 同上 |
| **HF + SRP-FP32 Triton (我方)** | **0** | **279s** | **+44%** | strict bit-exact ✓ |
| HF + SRP overlay on vLLM-BI | 2e-8 | 278s | +43% | FP32 floor 同量级 |
| HF + StaticCache eager | varies | 217-312s | +12-61% | StaticCache 单用是 regression |
| **vLLM BF16 native (cudagraph)** | 2.60e-3 | 179s | -8% (vLLM ref) | vLLM baseline 比 HF 快 |
| vLLM BF16 (apples-to-apples FLASH_ATTN config) | 2.87e-3 | 177s | -9% (vLLM ref) | 与 BI 同 attention_config 的 fair re-run（差异在 noise 内）|
| **vLLM + BATCH_INVARIANT (full)** | **2.55e-5** | **402s** | **+108% vs HF, +125% vs vLLM-BF16** | vLLM 官方 deterministic |
| LayerCast (估计) | ~5e-5 | ~600+ | ~300% | 不可行 |
| FP32-all | 3.59e-7 | ~470 | +140% | reference floor |

## 🎯 论文核心数据点（2026-04-26）

**我方 HF SRP-FP32 (Triton fixed-plan) 比 vLLM 官方 batch_invariant 都更优**：

| 维度 | 我方 HF+SRP | vLLM full + BI |
|---|---|---|
| Avg_Std | **0** (strict, by construction) | 2.55e-5 (precision-amplification 概率性) |
| 总 wall-clock | **279s** | 402s |
| vs vLLM BF16 baseline | **+56%** | +125% |
| vs HF BF16 baseline | +44% | +108% |

→ **写论文时这是核心 contribution datapoint。**
我方 fixed-plan Triton kernels (no split-K, no split-KV, fixed BLOCK_K=64)
比 vLLM 报告的 batch_invariant_ops 提供更强的 deterministic guarantee
AND 更低的 wall-clock，破除"必须用 vLLM stack 才能拿低开销"的假设。
