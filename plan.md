# Execution Plan — aligned to NIPS-2027.md (post 2026-04-26 reframe)

仅记录"具体执行什么 / 怎么执行"。叙事 / RQ / 风险 在 [`NIPS-2027.md`](NIPS-2027.md)。图表细节在 [`docs/figures_plan.md`](docs/figures_plan.md)。

---

## ✅ 2026-04-26 Step E 完成：vLLM 集成对比 — 论文核心数据点

经 5-题 DeepSeek-7B / MATH500 全栈验证（[`docs/srp_perf_log.md`](docs/srp_perf_log.md)）：

| Backend | Avg_Std | Total | vs HF-BF16 | vs vLLM-BF16 | 评价 |
|---|---|---|---|---|---|
| HF BF16 (DynamicCache) | 3.97e-3 | 194s | 0% | — | HF 参照 |
| vLLM BF16 (cudagraph) | 2.60e-3 | 179s | -8% | 0% | vLLM 参照（更快） |
| **HF + SRP-FP32 (我方 Triton)** | **0** strict | **279s** | **+44%** | **+56%** | **bit-exact** ✓ |
| **vLLM + BATCH_INVARIANT (官方)** | 2.55e-5 | 402s | +108% | +125% | vLLM 官方 deterministic |
| FP32-all | 3.59e-7 | ~470s | +140% | +163% | 上界参照 |

### 🎯 论文核心结论

**我方 HF+SRP-FP32 在两个维度上同时优于 vLLM 官方 batch_invariant_ops**：

1. **Determinism**：我方 strict bit-exact (Avg_Std=0) 优于 vLLM-BI 的 2.55e-5 概率性保证
2. **Wall-clock**：279s vs 402s（**快 30%**）

→ **直接破除"必须用 vLLM stack 才能拿低开销"的假设**。我方 fixed-plan Triton kernels (no split-K, no split-KV, fixed BLOCK_K=64) 是更优解。

### Step D（CUDA Graphs）状态：放弃

CUDA graphs 在 HF eager 路径上不可行：
- 手工 capture：HF Llama/Qwen2 attention 含 Python 控制流（`.item()` 在 capture 时被烤进），token 在 idx=1 即分叉
- `torch.compile(mode='reduce-overhead')`：dynamo 把 Triton kernel 当 graph break，无效

**Step E 已经替代 Step D**：与其在 HF 里手搓 cudagraph，不如对比真正用 cudagraph 的 vLLM 官方实现 → 结果显示我方仍更优（既严格又快）。

### 论文写作建议（contribution datapoint）

> Our SRP-FP32 with fixed-plan Triton kernels in HF eager achieves strict bit-exact (Avg_Std=0) at 279s/5-prob/MATH500 on DeepSeek-7B, while vLLM 0.19's official `VLLM_BATCH_INVARIANT=1` mode (with cudagraph + flash-attn) yields 2.55e-5 at 402s — **strictly worse on both axes**. This refutes the assumption that achieving batch invariance requires the vLLM serving stack with cuda graphs.

---

## 0. 当前共识（源码核对后）

源码事实（HF transformers / PyTorch ATen / Dao-AILab FA-2，详见 NIPS-2027.md §1.3）：

- HF `LlamaRMSNorm.forward` **已 FP32 累加器**（`hidden_states.to(torch.float32).pow(2).mean(...)`）
- `torch.nn.functional.scaled_dot_product_attention` 在所有 4 个 CUDA 后端（FA-2 / EFFICIENT / CUDNN / MATH）**累加器都是 FP32**（FA-2 `kernel_traits.h:17` 硬编码 `ElementAccum=float`）
- cuBLAS BF16 GEMM **始终** 用 FP32 inner-product 累加器（`CUBLAS_COMPUTE_32F`）
- `torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False` 控制的是 **split-K 跨段聚合**精度（`CUBLASLT_REDUCTION_SCHEME_*`），不是内积累加器

→ **FP32 累加器在现代栈里已是默认**。**真正的 batch-non-determinism 来源是 cuBLAS split-K + FA split-KV 按 bs 改 plan**。SRP 真正提供的是 **fixed-plan kernels at the two sites that actually have bs-dependent plans: MatMul-K + Attention**。

---

## 1. Method 定义

| 方法 | 含义 | 实现 |
|---|---|---|
| `BF16` | HF 默认行为 | trivial（`method_BF16`，no-op）|
| `FP32-all` | weights + compute 整体 FP32 | model load `dtype=torch.float32` |
| `LayerCast` | Yuan et al.：BF16 weights + per-Linear FP32 cast | `research/layercast.py` |
| `SRP-FP32` | 在 `linear, rmsnorm, attention, softmax` 全部 4 个 site 用我们的 fixed-plan kernel（FP32 累加器+无 split-K/无 split-KV） | `methods.py:method_SRP_FP32(model, ALL_SITES)` |
| `SRP-FP32-Critical` | 仅 `linear + attention`（NIPS-2027 §5.2 表格证明只这两个真有 bs-dependent plan） | `methods.py:method_SRP_FP32(model, CRITICAL_SITES)` |
| `SRP-FP32-MatMul-K` | site ablation: 仅 GEMM | site=`('linear',)` |
| `SRP-FP32-Norm-Stat` | site ablation: 仅 RMSNorm（**预期与 BF16 几乎一致**）| site=`('rmsnorm',)` |
| `SRP-FP32-Attn` | site ablation: 仅 attention（QK + Softmax + V）| site=`('attention',)` |
| `SRP-FP32-Softmax-Red (non-attn)` | site ablation: 仅 LM head / MoE gate softmax（**预期与 BF16 几乎一致**）| site=`('softmax',)` |

**已删除**：`SRP-FP64`（A6000 没 BF16→FP64 tensor core 路径，详见 NIPS-2027.md §5.1 注）。

---

## 2. 模型 / 数据集 / GPU 配置

| 维度 | A6000 ×2 上能做 | defer |
|---|---|---|
| 模型 | M1 DeepSeek-R1-Distill-Qwen-7B + M4 Llama-3.1-8B-Instruct | M2 DeepSeek-R1-Distill-Llama-8B（待下载）, M3 Qwen2.5-7B-Instruct |
| 数据集 | D1 MATH500（含 level）+ D2 AIME25（替 AIME24）| D3 LiveCodeBench |
| Batch size | {1, 8, 16, 32}（mem 允许时 64）| — |
| GPU count / TP size | ❌ 单机 | future work（借机器） |
| GPU type | ❌ 仅 A6000 | future work（借机器） |

---

## 3. NIPS-2027 §7 实验 → 我方实施

| NIPS plan | A6000 实施 | 脚本 | 状态 |
|---|---|---|---|
| **E1** Pareto (tok/s × ExactMatch × mem) | (M1+M4) × (D1+D2) × {BF16, LayerCast, SRP-FP32, SRP-FP32-Critical} × bs={1,8,16,32} | `exp_core_eval.py` | Phase 2 跑 bs={1,8,16}；缺 32 |
| **E2** BS sweep | 同 E1，bs 维度展开 | `exp_core_eval.py` | 同上 |
| ~~E3 GPU count~~ | ❌ defer | — | limitation |
| ~~E4 GPU type~~ | ❌ defer | — | limitation |
| **E5** ⭐ Site ablation | M1+M4 × D1+D2 × `make_methods(include_site_ablation=True)` × bs=8 | `exp_E5_site_ablation.py`（待写）| 等 Phase 2 完 |
| ~~E6 FP32 vs FP64~~ | ❌ 已删除（hardware）| — | — |
| **E7** First div CDF + logit margin | M1+M4 × D1 × {BF16, SRP-FP32} × bs={1, 16}，manual decode + logit dump | `exp_E7_logit_margin.py`（待写）| 等 Phase 2 完 |
| **E8** Efficiency vs LayerCast | (M1+M4) × {BF16, LayerCast, SRP-FP32, SRP-FP32-Critical, FP32-all} × bs={1,8,32}：dedicated TTFT/TPOT bench（多次取中位数）| `exp_E8_efficiency.py`（待写）| 等 Phase 2 完 |

---

## 4. Phase 进度

| Phase | 内容 | 状态 |
|---|---|---|
| 1. 基础设施 | `methods.py` + `srp_fp64_*.py` (后已删) + `exp_core_eval.py` smoke 通过 | ✅ |
| 2. 主线对比 | E1/E2/E8 部分数据：M1+M4 × D1+D2 × {BF16, LayerCast, SRP-FP32} × bs={1,8,16} × N=30 | 🔄 GPU 0+1 同时跑 |
| 3. 机制实验 | E5 site ablation；同样 (M1+M4)×(D1+D2) | ⏸ 等 Phase 2 |
| 4. 误差分析 | E7 logit margin (manual decode + logit dump) | ⏸ 等 Phase 2 |
| 5. 效率对比 | E8 dedicated benchmark（含 FP32-all reload）| ⏸ 等 Phase 2 |
| 6. 绘图 + 写论文 | `research/figs/` 重写 8 张图 + Tables | ⏸ 等数据 |
| **7. vLLM 对比（Step E）** | HF+SRP vs vLLM-BF16 vs vLLM+BI 三方对比，5-题 quick eval | ✅ **完成** — 我方 279s/Std=0 vs vLLM+BI 402s/Std=2.55e-5 |

---

## 5. Phase 2 当前运行（修订前启动，仍有效用）

| GPU | model | dataset | methods | bs | N | gen_len | 启动时刻 |
|---|---|---|---|---|---|---|---|
| 0 | Llama-8B | MATH500 → AIME25 | BF16, LayerCast, SRP-FP32 | {1,8,16} | 30 | 512 | 2026-04-25 12:11 |
| 1 | DeepSeek-7B | MATH500 → AIME25 | BF16, LayerCast, SRP-FP32 | {1,8,16} | 30 | 1024 | 2026-04-25 12:11 |

`SRP-FP32` 在这里是 all-sites 版本（4 个 site 全打开）。Phase 3 的 site ablation 会**用同样数据点**反过来验证 NIPS-2027.md §5.2 的 site 分类。

监控：

```bash
tail -f research/P0_summary/queue_phase2_gpu{0,1}.log
tail -f research/exp_main/{llama8b,deepseek7b}_{math500,aime25}.log
```

---

## 6. 限制（写进论文 Limitations）

1. **单机 2× A6000**：无法做 TP / 多 GPU 实验；E3/E4 缺位
2. **AIME24 用 AIME25 替代**（30 题，同质）；DeepSeek-R1-Distill-Llama-8B 待下载
3. **FP64 不在范围**：Ampere 消费卡硬件限制
4. **LayerCast** 实现复用 [`research/layercast.py`](research/layercast.py)，在 wall-time 测量上可能与 Yuan et al. 的 vLLM 原版略有差异
5. **non-determinism 来源仅覆盖单卡范围内**：cross-GPU AllReduce 的非确定（NCCL 算法选择）需要多卡环境，留作 future work
