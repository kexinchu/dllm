# Cleanup Proposal — 对齐 NIPS-2027 Plan

读完 `NIPS-2027.md` 全文后，跟现状对比：**新 plan 的方法论是 SRP（选择性高精度 reduction），跟我之前主推的 DetermLLM (Triton fixed-plan kernel) 不在同一条线上。** 主线变了，需要重组。

---

## 1. 方法名映射

| NIPS-2027 plan 方法 | 现状 | 处理 |
|---|---|---|
| `BF16` | ✅ trivial | 保留 |
| `FP32-all` | ❌ 没定义为独立 method | **新增**：load weights `dtype=float32` |
| `LayerCast` | ✅ `motivation/test_layercast_latency.py:apply_layercast` | 保留，把 method 名固定为 `LayerCast` |
| `SRP-FP32` | ✅ 实质等于 `FP32/model_patcher.py` 全开（GEMM + RMSNorm + Attn + Softmax） | **重命名**为 `SRP-FP32` |
| ~~`SRP-FP64`~~ | ❌ 已删除（2026-04-25）—— A6000 tensor core 不支持 BF16 → FP64 累加器路径 | — |
| `SRP-site only`（单一 site） | ✅ `model_patcher.py` 提供 `patch_linear / patch_rmsnorm / patch_attn / patch_softmax` 单独开关 | 现成 |

**消失的方法（在新 plan 里不存在）**：
- `FP32flag`（cuBLAS `allow_bf16_reduced_precision_reduction=False`）→ 实质是 MatMul-K only 的 SRP 子集，可改名 `SRP-MatMulK-flag` 作 site-ablation 对照
- `DetermLLM`（Triton fixed-plan GEMM）→ 这是 bit-exact 路线，不属 SRP；建议**降级为附录 baseline** 或彻底移除主线
- `DetermLLM+attn`（DetermLLM + FP32 attn matmul upcast）→ 同上

---

## 2. NIPS-2027 reduction site 分类 → 现有实现

| Site | 已实现 | 文件 |
|---|---|---|
| `MatMul-K`（GEMM K 维 accum） | ✅ FP32 accum | [FP32/gemm_fp32_accum.py](FP32/gemm_fp32_accum.py) |
| `Softmax-Red`（attn softmax max/sum） | ✅ FP32 accum | [FP32/attention_fp32_accum.py](FP32/attention_fp32_accum.py) |
| `Softmax-Red`（其他 softmax，如 MoE gate） | ✅ FP32 accum | [FP32/softmax_fp32_accum.py](FP32/softmax_fp32_accum.py) |
| `Norm-Stat`（RMSNorm/LayerNorm） | ✅ FP32 accum | [FP32/rmsnorm_fp32_accum.py](FP32/rmsnorm_fp32_accum.py) |
| `Attn-Value-Red`（score @ V） | ✅ FP32 accum | [FP32/attention_fp32_accum.py](FP32/attention_fp32_accum.py) |
| `CrossGPU-AllReduce` | ❌ 单卡环境无 TP | **物理不可行**，标注 limitation |

**好消息**：FP32 accum kernels 全部已写好，`model_patcher.py` 已经能做 site 级开关。SRP-FP32 实际 99% 现成。

---

## 3. 实验 mapping

| NIPS-2027 实验 | 我做的 P0 | 是否可用 |
|---|---|---|
| E1 Pareto（tok/s × ExactMatch） | 部分（E7 overhead + E4 bit-exact） | ⚠️ method 重命名后可拼 |
| E2 BatchSize → Div_Percent | E4 部分（bs={1,8,16}） | ⚠️ NIPS 要 bs={1,8,32,64}，需补 32/64 |
| E3 GPU count → Div_Percent | 无 | ❌ 单卡做不到 |
| E4 GPU type → Div_Percent | 无 | ❌ 物理做不到 |
| **E5 Reduction site ablation** | 完全没做 | ❌ **新 plan 最关键的机制实验** |
| ~~**E6 FP32 vs FP64**~~ | 已删除（2026-04-25）| 见 NIPS-2027.md §7.E6 |
| E7 First div CDF + logit margin | E2 prob gap + E4 div_idx 部分有 | ⚠️ 重组可用 |
| E8 vs LayerCast efficiency | E7 overhead | ⚠️ 重命名 method 后直接可用 |

**即将卡住的**：E3、E4 单卡做不到。Limitation 写明，或者借 H100/A100 节点跑（要确认资源）。

---

## 4. 数据保留 / 删除清单（待你确认）

### 4.1 ✅ 必保留

| 路径 | 内容 | 重组后用途 |
|---|---|---|
| `FP32/*.py`, `FP32/*.so` | FP32 accum kernels | SRP-FP32 全部站点 |
| `FP32/model_patcher.py` | site-level patch infra | E5 site ablation |
| `motivation/test_layercast_latency.py` (apply/remove_layercast) | LayerCast 实现 | LayerCast 对照 |
| `research/math500_cached.json` (含 level 字段) | MATH500 + level | E1/E2/E5/E6 主数据集 |
| `research/math500_cached.json.bak` | 原始 cache 备份 | 安全网 |
| `/home/kec23008/docker-sys/DynaQuant/calibration_datasets/requests/aime25_available_30.jsonl` | AIME25 | E1/E2/E5/E6 主数据集（即使 plan 写 AIME24，AIME25 同质） |
| `research/figs/_style.py` + `figXX_*.py` | 绘图基础设施 | 改 method 名后继续用 |
| `research/exp_E7/E7_Llama-3.1-8B.json` | Llama overhead 数据 | 重命名 method（`DetermLLM`→`SRP-FP32`，`DetermLLM+attn`→ 删，`FP32flag`→`SRP-MatMulK-only`）后能拼 E1/E8 |
| `docs/figures_plan.md`, `docs/batch_invariance.md` | 现有文档 | 新 plan 配套补充 |

### 4.2 🟡 重命名后可用（不删，改 schema）

| 路径 | 现状问题 | 处理 |
|---|---|---|
| `research/exp_E4/E4_llama8b_math500.json` (eager) | method 列表 = `[BF16, FP32flag, LayerCast, DetermLLM, DetermLLM+attn]` | 写映射脚本：`FP32flag`→`SRP-MatMulK-only`，`DetermLLM`/`+attn` 标 `(legacy bit-exact)` 列入附录 |
| `research/exp_E3/E3_llama8b.json` | 同上 method 命名 | 同上重命名 |
| `research/exp_E7/E7_Llama-3.1-8B.json` | 同上 method 命名 | 同上重命名 |

### 4.3 ❌ 建议删除（无效或不可信）

| 路径 | 原因 |
|---|---|
| `motivation/batch_invariance_calibration.json/log/py` | D4 single-prompt N=1000 micro-bench；NIPS 新 plan 完全没这场景 |
| `research/exp_E4/E4_llama8b_aime25.json` | 老 sdpa 版 + 只跑了 BF16/FP32flag 就 OOM 崩溃，数据不可信 |
| `research/exp_E4/E4_deepseek7b_math500.json` | 老 sdpa 版（attention patch bypass），DetermLLM+attn 数据失真 |
| `research/exp_math500_overnight/*.json` | 老一批 dllm_cublaslt/dllm_triton/dllm_hybrid 数据，全用旧 method 名，不属于新 plan |
| `motivation/determinism_*_results.json`, `motivation/fp32_*_results.json`, `motivation/layercast_*_results.json`, `motivation/reduction_results.json` | motivation/ 下旧 micro-bench 输出，论文不会引用 |
| `motivation/fp32_reduction_*_run.log`, `motivation/batch_invariance_calibration.log` | 对应日志 |
| `research/exp_*.json`（散在 research/ 根目录的 30+ 个 JSON：`exp_d1_moe_rollback.json`, `exp_decompose_*.json`, `exp_kvcache_*.json`, `exp_logprob_nondet.json`, `exp_m{1,2,3,4}_*.json`, `exp_e1_10k.json`, `exp_e2e.json`, `exp_e2_moe_1k.json`, `exp_det_*.json`, `exp_d_rollback_llama.json`, `exp_downstream.json`, `exp_kmax_sweep.json`, `exp_fp32_kernel.json` …） | 旧 DetermLLM-时代杂项实验，新 plan 无对应 |
| `research/_archive/paper_*.md`, `research/_archive/exp_*.md` | 旧论文草稿 / 旧实验 markdown，跟新 plan 完全不搭 |
| `research/exp_math500_overnight/` 整目录 | 4.7 MB 旧数据全为 deprecated 方法 |

### 4.4 🟡 代码现状/去留待定

| 路径 | 问题 | 备选 |
|---|---|---|
| `research/determ_llm.py` | 实现 Triton fixed-plan kernel，不属 SRP 主线 | (a) 移到 `research/legacy/` 作附录 baseline；(b) 完全删除 |
| `research/exp_E4_div_index.py`（带 `DetermLLM/+attn` schemes） | method enum 跟新 plan 不一致 | 重写：method enum 改成 `BF16/FP32-all/LayerCast/SRP-FP32/SRP-FP64`（+ 各 site only） |
| `research/exp_E2_prob_gap.py`（用 DetermLLM+attn 作 ours） | method 名错 | 同上 |
| `research/exp_E3_prob_std.py` | method 名错 | 同上 |
| `research/exp_E7_overhead.py` | method 名错 | 同上 |
| `research/run_P0_*.sh` (4 个 queue 脚本) | 都跑老 method | 删；写新版 |
| `research/run_math500_eval.py` | 旧 method 命名 | 删（被 exp_E4 取代） |
| `motivation/run_topk_tie_test.{py,sh}`, `motivation/test_motivation_*.py`, `motivation/test_topk_tie_breaking.py`, `motivation/test_three_scenarios.py`, `motivation/test_layercast_logic.py`, `motivation/test_continuous_batching_determinism.py`, `motivation/test_determinism_*.py`, `motivation/test_fp32_*.py`, `motivation/test_reduction_*.py`, `motivation/moe_gating_determinism_test.py` | 老一代 motivation/diagnose 脚本 | 大概率新 plan 用不上，建议**移到** `motivation/_legacy/` 而非删除 |

---

## 5. 当前正在跑的 GPU 队列（要立刻决定）

| GPU | PID | 跑的内容 | 建议 |
|---|---|---|---|
| 0 | 3399007 (python) | E4 DeepSeek MATH500 last scheme `DetermLLM+attn`（~1h 剩）→ AIME25 → E2 | **杀掉**，method 名错 |
| 1 | 3770235 (python) | E4 Llama AIME25 eager rerun（method 同上） | **杀掉** |

杀掉后 GPU 全空，准备新 SRP 版实验。

---

## 6. 新增工作量（按 NIPS plan 必做）

| 任务 | 估时 | 必要性 |
|---|---|---|
| 写 6 个 method 标签（BF16 / FP32-all / LayerCast / SRP-FP32 / SRP-FP64 / SRP-site-X）的统一 enable/disable 接口 | 1h | P0 |
| 实现 FP32 → FP64 accumulator kernels（gemm/rmsnorm/attn/softmax 各一份） | 4-8h（CUDA + Triton 改 dtype） | P0 |
| 重写 `exp_E4_div_index.py` 用新 method enum | 30 min | P0 |
| **新增 E5 site ablation 主实验脚本** | 1h | P0（新 plan 最关键） |
| 重命名/迁移现有 E3/E4/E7 输出 JSON 的 method 字段 | 30 min | P0 |
| 删除候选清单 4.3 一次性执行（移 trash 而非真 rm） | 5 min | 待你点头 |

---

## 7. 我的具体建议（请逐条确认 ✅/❌）

1. **杀掉当前 GPU 0 / GPU 1 上跑的两个队列**（method 名错，跑完也用不上）
2. **执行 4.3 的删除清单**（先 mv 到 `_trash_$(date)/` 而非 `rm`）
3. **`research/determ_llm.py` 移到 `research/legacy/`**（保留代码作附录可选 baseline，不再出现在主线）
4. **保留 4.1 / 4.2 的代码和数据**，重命名 method
5. **新增 SRP-FP64 kernels**（FP64 accumulator，同样的 op patch 框架）
6. **更新 `plan.md` 与 `docs/figures_plan.md`** 对齐 NIPS-2027 plan 的 8 个实验编号 + 5 个 method 命名
7. **是否要做 E3/E4（GPU count / TP / GPU type）**：单卡做不到，要不要：(a) 写 limitation；(b) 申请 H100/A100 节点；(c) 用 NCCL/torch.distributed 模拟 multi-GPU（仍单卡，假 TP，不真实）

请回 ✅/❌ 以及第 7 条 (a/b/c)。
