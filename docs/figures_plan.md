# Figures Plan — NIPS-2027 alignment (Figures 1–7)

每张图 **先固定 X / Y / Why**，再列**收集数据 schema**，最后跟脚本绑定。

---

## ⚠️ 2026-04-25 概念修正

源码核对（HF transformers / PyTorch ATen / FA-2）后确认：

- **FP32 累加器在现代栈里已是默认**（HF RMSNorm、SDPA-FA-2、cuBLAS BF16 GEMM 都是）
- **真正驱动 batch-non-determinism 的是 cuBLAS split-K 与 FA split-KV 按 bs 改 plan**，不是累加器精度
- **真正需要 patch 的位点只有两个**：MatMul-K（GEMM）和 Attention。RMSNorm / non-attn Softmax 在 dense LLM 路径上 HF 默认就是 batch-invariant

因此 SRP 的核心叙事应当更新为 "**Selective Fixed-Plan Reductions**"——FP32 累加器是必要前提，但**固定 plan 才是消除 bs-variance 的真正手段**。这不影响主图编号，但影响 E5（site ablation）的 X 轴含义和预期结论。详见 NIPS-2027.md §1.3 / §5.1 / §5.2。

参考 [NIPS-2027.md §8](../NIPS-2027.md) 推荐图清单：

| Paper Figure | NIPS plan 编号 | 实验 |
|---|---|---|
| Fig 1 | 方法示意图 | 概念图（SRP vs LayerCast） |
| Fig 2 | E1 | Reproducibility-Efficiency Pareto |
| Fig 3 | E2 | Batch Size → Div_Percent |
| Fig 4 | E3 | GPU Count / TP Size → Div_Percent ❌（defer）|
| Fig 5 | E5 | Reduction Site Ablation ⭐ |
| ~~Fig 6~~ | ~~E6~~ | ~~FP32 vs FP64~~ — **已删除（2026-04-25）** |
| Fig 7 | E7 | First Divergence CDF + Logit Margin |

附加（NIPS plan §8 也提到）：
| Fig 8 | E8 | LayerCast 效率对比（3 panel） |

---

## Fig 1 — 方法示意图（concept）

| 项 | 内容 |
|---|---|
| 图类型 | 流程图 / box-and-arrow |
| 内容 | 一个 transformer block 内部的 4 个 reduction 站点（MatMul-K / Norm-Stat / Softmax-Red / Attn-Value-Red），分别用不同颜色标 LayerCast vs SRP 的处理 |
| Why | 让 reviewer 一眼看到"我们改的是 reduction accumulator，不是整个 compute path" |
| 数据需求 | 无（纯示意） |
| 脚本 | 手画或 TikZ；**这张图不用程序生成** |

---

## Fig 2 — Pareto（E1）

| 项 | 内容 |
|---|---|
| 图类型 | 气泡散点图 |
| X | `Throughput (tok/s)` |
| Y | `Exact Match Consistency (%)` |
| 气泡大小 | `Peak Memory (GB)` |
| 颜色 | 方法 ∈ {BF16, LayerCast, SRP-FP32, FP32-all} |
| Subpanels | 2×2 = (M1, M4) × (D1, D2) |
| Why | 图想讲的故事：BF16 在右下，LayerCast/FP32-all 在左上，SRP 应在右上（更好的 Pareto front） |
| 数据需求 | 见下表 |
| 脚本 | `research/figs/fig2_pareto.py` |

**数据 schema (per row)**：

| model | dataset | method | bs | tok_s | exact_match_consistency | peak_mem_GB | div_percent |
|---|---|---|---|---|---|---|---|

每个气泡 = 一个 (model, dataset, method) triple；bs 维度可以聚合（取 mean 或单独画多个气泡）。

**数据来源**：`research/exp_E1_pareto.py` 输出 → `research/exp_E1/E1_*.json`

---

## Fig 3 — Batch Size → Div_Percent (E2)

| 项 | 内容 |
|---|---|
| 图类型 | 折线图（主 + 附） |
| 主图 X | `Batch Size` ∈ {1, 8, 32, 64} |
| 主图 Y | `Div_Percent (%)` |
| 附图 Y | `Exact Match Consistency (%)` |
| 线条 | 方法 |
| Subpanels | 2×2 = (M1, M4) × (D1, D2) |
| Why | BF16 折线随 bs 升高（diverge 增加），SRP 折线扁平 |
| 数据需求 | per (model, dataset, method, bs)：div_percent, exact_match_consistency, std_len, std_top1_prob |
| 脚本 | `research/figs/fig3_bs_sweep.py` |

**数据 schema**：

| model | dataset | method | bs | div_percent | exact_match_consistency | std_len | std_top1_prob |
|---|---|---|---|---|---|---|---|

**数据来源**：`research/exp_E2_bs_sweep.py` → `research/exp_E2/E2_*.json`

---

## Fig 4 — GPU Count → Div_Percent (E3) ❌

**defer to future hardware**。论文中表为 limitation。

---

## Fig 5 — Reduction Site Ablation (E5) ⭐ 最关键（2026-04-25 修订）

| 项 | 内容 |
|---|---|
| 图类型 | 分组柱状图（2 panel） |
| Panel A — X | `Promoted Reduction Site` |
| Panel A — Y | `Div_Percent (%)` |
| Panel B — Y | `Throughput (tok/s)` |
| X 取值 | {`BF16` / `+ MatMul-K fixed` / `+ Norm-Stat fixed` / `+ Attn fixed` / `+ Softmax-Red(non-attn) fixed` / `+ MatMul-K & Attn fixed` / `LayerCast` / `FP32-all`} |
| 颜色分组 | (model, dataset) 组合 |
| Why | **关键预期**：`+ MatMul-K & Attn fixed` 应当显著降低 div%，**接近** `LayerCast`；`+ Norm-Stat` 和 `+ Softmax-Red(non-attn)` 应当**与 BF16 几乎一致**（HF 默认已 batch-invariant）。这是论文核心机制 claim 的 datapoint。|
| 数据需求 | per (model, dataset, site)：div_percent, exact_match, tok_s, peak_mem, first_div_idx |
| 脚本 | `research/exp_E5_site_ablation.py`（用 `methods.py:make_methods(include_site_ablation=True)`）|

**数据 schema**：

| model | dataset | promoted_site | div_percent | exact_match_consistency | first_div_idx_median | tok_s | peak_mem_GB |
|---|---|---|---|---|---|---|---|

`promoted_site` ∈ {`none` (= BF16), `linear`, `rmsnorm`, `attention`, `softmax`, `linear+attention`, `all`, `layercast`(=参考), `fp32-all`(=参考)}

**论文叙事关键点**：

1. `linear` 单独打开能 **大幅降低 div%**（GEMM 是主要 batch-variance 源 of two）
2. `attention` 单独打开也能 **大幅降低 div%**（split-KV 是另一主源 of two）
3. `linear + attention` 接近 `LayerCast`（证明这两个就够，无需 FP32-all）
4. `rmsnorm` / `softmax(non-attn)` 与 baseline 几乎一致（证明 HF 默认已 batch-invariant，不需修补）

如果 (1)/(2)/(3) 实测成立 → "Selective Fixed-Plan Reductions" 论点立得住。
如果 (4) 实测成立 → 反过来证明现代框架已经在 norm/softmax 做对了；本文不重复 reinvent。

**数据来源**：`research/exp_E5_site_ablation.py` → `research/exp_E5/E5_*.json`

---

## Fig 6 ❌ — 已删除（2026-04-25）

原计划：`Precision Policy` ∈ {BF16, SRP-FP32, SRP-FP64, LayerCast, FP32-all} 的 Div_Percent / Throughput 对比。

**删除原因**：A6000 / Ampere 消费卡硬件不支持 BF16 → FP64 累加器路径（tensor core 仅支持 FP32 累加器）。任何 FP64 实现需把 input 整体 upcast 到 FP64 走标量 ALU，开销由 "FP64 ALU 慢 ~60×" 主导而非 reduction precision 本身——破坏了 RQ4 想要的"干净对照"。论文中放入 Limitations / Future Work（在 H100 + 特殊 dtype 配置上重做）。

**E7 升级为 E6（First Divergence + Logit Margin）**，原编号顺延。

---

## Fig 7 — First Divergence + Logit Margin (E7)

| 项 | 内容 |
|---|---|
| 图类型 | 2 子图 |
| 子图 A | CDF：X = `First Divergence Token Index`，Y = CDF |
| 子图 B | 散点 / 分桶折线：X = `Top1−Top2 Margin Bin`，Y = `Divergence Probability` |
| 线条 | 方法（A）/ 方法（B） |
| Why | 解释机制：分叉发生在 top-1/top-2 margin 小的位置；SRP 应当显著降低这些脆弱点的分叉概率 |
| 数据需求 | per sample：first_div_idx, top1_top2_margin_at_div, final_correct, output_len |
| 脚本 | `research/figs/fig7_first_div_margin.py` |

**数据 schema**（per sample）：

| sample_id | model | dataset | method | first_div_idx | top1_top2_margin_at_div | final_correct | output_len |
|---|---|---|---|---|---|---|---|

**数据来源**：`research/exp_E7_logit_margin.py` → `research/exp_E7/E7_*.json`

---

## Fig 8 — LayerCast Efficiency (E8)

| 项 | 内容 |
|---|---|
| 图类型 | 分组柱状图（3 panel） |
| Panel A | Y = `Throughput (tok/s)` |
| Panel B | Y = `Peak Memory (GB)` |
| Panel C | Y = `TTFT_ms` 和 `TPOT_ms` |
| X | 方法 |
| Why | 直接对比 SRP 的效率优势 |
| 数据需求 | per (model, method, bs)：tok_s, peak_mem, TTFT, TPOT |
| 脚本 | `research/figs/fig8_efficiency.py` |

**数据 schema**：

| model | dataset | method | bs | tok_s | TTFT_ms | TPOT_ms | peak_mem_GB |
|---|---|---|---|---|---|---|---|

**数据来源**：`research/exp_E8_efficiency.py` → `research/exp_E8/E8_*.json`（与 E1 数据有重叠，可复用）

---

## Tables（NIPS plan §8 + §9）

| Table | 内容 | 数据源 |
|---|---|---|
| T1 主结果表 | 不同 model × dataset × method 的 (acc, ExactMatch, div%, std_acc, std_len, std_top1_prob, tok/s, TTFT, TPOT, mem) | E1 / E2 |
| T2 efficiency vs LayerCast | tok/s × TTFT × TPOT × mem | E8 |
| T3 site ablation | 每个 site 的 (div%, ExactMatch, first_div_idx, tok/s, mem) | E5 |

---

## 旧图清理

`research/figures/` 下的 figure1_gemm_variance / figure2_hash_distribution 等是 DetermLLM-时代图，已迁移到 `research/legacy/figures/`，不再使用。

`research/figs/` 是新版（NIPS-2027 对齐）：
| 文件 | 状态 |
|---|---|
| `_style.py` | ✅ keep（颜色/字体/loader） |
| `fig1_motivation_accuracy_vs_bs.py` | ❌ 旧（DetermLLM-era），下一步重写为 fig3_bs_sweep.py |
| `fig2_prob_gap.py` | ❌ 旧 E2 prob gap，下一步重写为 fig7_first_div_margin.py |
| `fig3_avg_std.py` | ❌ 旧 E3 avg_std，下一步移除（NIPS plan 用 E5 site ablation 替代） |
| `fig4_div_index_cdf.py` | ❌ 旧 E4，下一步重写为 fig7（CDF 部分） |
| `fig5_bit_exact.py` | ❌ 旧 E6 bit-exact，下一步移除（NIPS plan 用 ExactMatch 取代） |
| `fig6_difficulty.py` | ❌ 旧 E5 difficulty，下一步移除或合并到 E5 site ablation |
| `fig7_overhead.py` | ⚠️ 重命名 → fig8_efficiency.py |

**Action**: 等 E1–E8 数据出来后，按本文档重写 figs/，旧版同步迁移到 `research/legacy/figs/`。
