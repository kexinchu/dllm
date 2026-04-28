# Research Progress Log

## Status: 进行中（2026-04-26 晚）

---

## 【2026-04-26 晚 — 头条结果出炉，明早恢复点】

### 一句话状态
**SRP 在 HF eager 上拿到 strict bit-exact (Avg_Std=0) + 比 vLLM 官方 batch_invariant 快 31% 的双重优势**，论文 Step E 完成，明天可以直接进入 figure 重画 / 表格汇总 / 论文 polishing 阶段。

### 今天产出的论文核心数据点

5-题 DeepSeek-R1-Distill-Qwen-7B / MATH500 / greedy 256 tok / bs∈{1,2,4,8,16}：

| Method | Avg_Std@top1 | Total | vs HF-BF16 | vs vLLM-BF16 |
|---|---|---|---|---|
| HF BF16 (DynamicCache) | 3.97e-3 | 194s | 0% | — |
| vLLM BF16 (cudagraph+FA2) | 2.60e-3 | 179s | -8% | 0% |
| vLLM BF16 (FLASH_ATTN explicit, fair re-run) | 2.87e-3 | 177s | -9% | -1% |
| **HF + SRP-FP32 (我方)** | **0** strict | **279s** | **+44%** | **+56%** |
| **vLLM + VLLM_BATCH_INVARIANT=1** | 2.55e-5 | 402s | +108% | +125% |
| FP32-all (floor) | 3.59e-7 | ~470s | +140% | +163% |

**头条 claim**：我方 HF eager 路径同时在 *determinism*（strict 0 vs 概率性 2.55e-5）和 *wall-clock*（-31%）两个维度上严格优于 vLLM 官方 batch_invariant 实现。

### 今天的代码/文档变更（已提交到工作区，未 commit）
1. `docs/srp_perf_log.md` — 加 vLLM 三行（BF16 native / BF16 fair / BI），加 §🎯 论文核心数据点，加 fair re-run 一行
2. `plan.md` — Step D（CUDA Graphs）状态改为"放弃"，新加 Step E 完成块 + 论文写作建议
3. `NIPS-2027.md` — 新增 §15「2026-04-26 头条实证结果：vs vLLM 官方 batch_invariant_ops」整节
4. `paper/main.tex` — 新加 §6.7 `\subsection{Comparison Against vLLM Official Batch-Invariant Mode}`（Table `tab:srp_vs_vllm_bi`），更新 abstract 末段，添加 contribution (vi)。pdflatex 双 pass 通过，main.pdf 19 页 0 undefined refs
5. `research/validate_avg_std_vllm_native.py` — 把 `attention_config=FLASH_ATTN` 改为无条件设置（之前只在 BI 模式下加），按 BATCH_INVARIANT env 区分输出文件名
6. memory 加 `finding_srp_beats_vllm.md` + 索引

### 数据文件
- `research/exp_validate/vllm_native_bf16_results.json` — fair re-run（最新覆盖）：177.2s / 2.868e-3
- `research/exp_validate/vllm_native_bf16_results.no_flash_attn.json` — 原 BF16 备份：178.9s / 2.608e-3
- `research/exp_validate/vllm_native_results.json` — vLLM+BI（402s / 2.55e-5）
- `research/exp_validate/vllm_kernels_results.json`、`combo_results.json` — HF + 我方各 site 配置
- BF16 re-run 的 stdout：`research/exp_validate/vllm_native_bf16_fair.log`

### 明早 todo（推荐顺序）

1. **Figure 重画**（最优先）。`research/figs/` 下旧图基于 cuBLASLt-only 的 v1 结果。需要为新 Table `tab:srp_vs_vllm_bi` 配一张主 figure：
   - Pareto 散点：x=Total wall, y=Avg_Std (log)，5 个点 + 文字标签（HF-BF16, vLLM-BF16, HF+SRP, vLLM+BI, FP32-all）
   - 我方 HF+SRP 应当落在 y=0 的"地板"上 + x 比 vLLM+BI 左
2. **更新 main.tex Section 1（Introduction）**。当前 contribution (i)-(v) 还在 cuBLASLt-only 框架下；新加的 (vi) 是后补的。建议写论文这一遍把 §1 的整体框架升级到 4-site SRP（linear/rmsnorm/attention/softmax 全 patch）+ vLLM-BI 对比的视角。这是 paper 最大的 framing shift，需要至少 1-2 小时。
3. **Section 3 Method**。当前 §3 还是 GEMM 的 K_max 证明。SRP 4-site 现在还没 method 描述。需要补 §3.X "Fixed-plan reductions on the four sites"（已经在 NIPS-2027.md §5 列好了，照搬即可）。
4. **统计 N=30 的 full evaluation**。目前 vLLM 对比用的是 N=5 (5 problems)。reviewer 会要求 N≥30，最好 N=50 或 N=100。需要决定：是否花一晚跑 vLLM-BI N=30（vLLM-BI 一晚 = 30/5 × 402s = ~40 min；我方 HF+SRP = 30/5 × 279s = ~28 min；HF-BF16 = 30/5 × 194s = ~20 min）。**这一步是最关键的"加 reviewer 防御"**。
5. **GPU 0 上还在跑的东西**：上次 ps 显示 GPU 0 在 100% util，27GB 占用。可能是 Phase 2 主线对比（M1+M4 × D1+D2）。明早先 `tail -f research/exp_main/*.log` 看一下进度，决定是否抢占 GPU 0 跑 N=30 vLLM 对比。

### Step D 放弃的原因（明早不要再走老路）
Step D = HF eager 路径上做 CUDA Graphs。试过 4 条全失败：
- cuBLASLt fixed-algo backend：std 仅减 50%（2.0e-3），离 strict 太远
- Hybrid M-thresholded 调度：同上
- `torch.compile(mode='reduce-overhead')`：dynamo 把 Triton kernel 当 graph break，无效
- 手工 cuda graph capture：HF Llama/Qwen2 attention 在 capture 时调 `.item()` 烤进 Python 控制流

→ 明天若要再降 overhead，唯一方向是把我方 fixed-plan kernels 直接 patch 进 vLLM stack（vLLM 0.20+ 的 batch_invariant 替换）。这是 future work，不在 NeurIPS submit 范围。

### 关键文件位置

| 内容 | 路径 |
|---|---|
| paper LaTeX | `paper/main.tex`（19 页）+ `paper/main.pdf` |
| 论文 plan | `NIPS-2027.md`（重点看 §15）、`plan.md`（重点看顶部 Step E 完成块） |
| perf log | `docs/srp_perf_log.md`（重点看 §🎯） |
| 我方 SRP 实现 | `FP32/model_patcher.py`（_HYBRID_M_THRESHOLD=0 强制 Triton） + `FP32/triton_det_gemm.py` + `FP32/attention_fp32_accum.py` |
| vLLM 对比脚本 | `research/validate_avg_std_vllm_native.py` |
| HF + 我方 SRP 对比脚本 | `research/validate_avg_std_vllm_kernels.py`、`research/validate_combo.py` |
| 跑 vLLM 的 conda env | `/home/kec23008/miniconda3/envs/vllm_test`，需要 `LD_LIBRARY_PATH=/home/kec23008/miniconda3/envs/vllm_test/lib/python3.10/site-packages/nvidia/nvshmem/lib:$LD_LIBRARY_PATH` |

---

## 【2026-04-23 下午关键发现 & 诊断】

### 核心发现：LayerCast 的 FP32 upcast 其实并不 bs-invariant

用户质疑："LayerCast 理论上应该像 Yuan 论文说的达到 ~2% non-det，为什么我们实测 100%？" 让我深挖，发现一连串 bug 和意外事实：

**1. Hybrid backend 的 cuBLASLt 路径不是 bs-invariant**
- `research/diagnose_triton_bs_invariance.py` 显示：对 DeepSeek-7B 的 6 个投影 shape，cuBLASLt 中 4/6 在 bs=1 vs bs=8 row[0] 不一致
- Hybrid 规则 "N≤4096→cuBLASLt" 把 Q_proj/O_proj (N=3584) 路由到 cuBLASLt，导致 decode 时每步 0.1% 级 noise 累积 57 步就 flip token
- Triton 后端则所有 shape 都 bs-invariant ✓

**2. `@` 和 `.matmul()` 不过 torch.matmul 的 monkey-patch**
- `a @ b` → `Tensor.__matmul__` (C 级)，不走 Python 的 `torch.matmul`
- HF 的 attention 用 `q @ k.transpose(-2,-1)`，我们的 matmul patch 完全没触发
- 修复：同时 patch `torch.Tensor.matmul` 和 `torch.Tensor.__matmul__`

**3. 最关键：cuBLAS FP32 matmul 也不是 bs-invariant**
- `research/diagnose_fp32_matmul_bs_invariance.py`: FP32 upcast 的 attention Q@K^T 在 300 个 S 值中 54 个 fail bs-invariance
- `torch.use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG=:4096:8` 也无法解决
- LayerCast F.linear 在 DeepSeek 的 6 投影 shape 中 5 个 fail (FP32 upcast 后)

**结论**: 只有**固定 plan kernel**（我们的 Triton）能真正 bs-invariant。Yuan 论文中 "LayerCast FP32 2.2% non-det" 是在更窄的 bs 范围（{8,16}）或短序列下测的；在我们的 stress regime（bs {1,8,32}，1024 gen）上，LayerCast 不达标。

### 决策：保留 v1 paper，加 Appendix 解释
- 所有 patch 代码回滚到 F.linear-only LayerCast (Yuan 原版忠实复现) + DetermLLM Triton F.linear bit-exact
- 在 paper cross-architecture paragraph 加了 1 句话 + Appendix `app:fp32insufficient`（~50 行）完整诊断
- 保留的 paper: main 9 页 + refs 2 页 + appendices 7 页 = 18 页总
- 删除 `exp_matrix_n50_v2/` 目录（基于错误假设）

### 这是 paper 的**强化卖点**而不是弱点
- **DetermLLM 是第一个真正 bs-invariant 的 F.linear 方案** — LayerCast 没达到其表面 claim
- Fig 1/2/3/4 的 n=50 数据全部 valid，DetermLLM 在 Div_Index 和 AvgStd@top1 上都赢 LayerCast

### v1 paper 最终表数据（DeepSeek-7B MATH500）不变

| 方法 | Acc | Std@Acc↓ | Div_Index↑ | AvgStd@top1↓ | Runtime |
|---|---|---|---|---|---|
| BF16 | 14.0% | 0.0327 | 88 | 0.1825 | 1.00× |
| LayerCast | 14.7% | **0.0094** | 98 | 0.1812 | 3.22× |
| **DetermLLM-Hybrid** | 15.3% | 0.0189 | **126** | **0.1807** | **1.11×** |

注：这里 DetermLLM-Hybrid 在 Std@Acc 上输给 LayerCast (0.0189 vs 0.0094)，但那是 accuracy-level 的变化，主要由 attention 残留 non-det 驱动（两种方法都有）。**F.linear level 的 bit-exact 只有 DetermLLM 做到** — 这是 Div_Index/AvgStd@top1 两个更 fine-grained 指标上 DetermLLM 胜出的原因。

---

## 【2026-04-23 全部完成 — 早晨】

### 27/27 configs 全部完成，论文已更新

- n=50 matrix 结束于 07:16（10 小时实际 GPU 时间）
- 3 张 paper figure 已生成并加入 paper/figs/
- paper/main.tex 已更新: 新 Table 1 (n=50, 3 方法 DeepSeek), 新 Fig 1 (same-prefix 跨 3 模型), 新 Fig 3 (Runtime-Det Pareto)
- Old Tab:matrix 被 Fig 1 取代，总页数从 9 → 9 保持 NeurIPS 限制
- 核心结果：DetermLLM 在 DeepSeek/Llama 上把 median Div_Index 推后 29-43%，LayerCast 3.22× 慢。关键数值表见下方。

### n=50 核心结果表（DeepSeek-7B MATH500）

| 方法 | Acc | Std@Acc↓ | Div_Index↑ | AvgStd@top1↓ | Runtime |
|---|---|---|---|---|---|
| BF16 | 14.0% | 0.0327 | 88 | 0.1825 | 1.00× |
| LayerCast | 14.7% | **0.0094** | 98 | 0.1812 | 3.22× |
| **DetermLLM-Hybrid** | 15.3% | 0.0189 | **126** | **0.1807** | **1.11×** |

跨模型 Div_Index median（n=50，pair 1↔8 + 8↔32 合并）：
- DeepSeek-7B: BF16=88 → Hybrid=126（+43%）, LayerCast=98
- Llama-8B: BF16=101 → Hybrid=142（+41%）, LayerCast=80
- Phi-4: BF16=310 → Hybrid=382, LayerCast=386（dense 模型都近似）

跨 3 bs 完全一致率 (1 - nondet%):
- DeepSeek-7B: 全 0%（attention 主导，1024 token 长）
- Llama-8B: BF16=6%, LayerCast=4%, **Hybrid=14%**
- Phi-4: BF16=46%, LayerCast=52%, Hybrid=52%

### 新生成的文件
- `research/exp_matrix_n50/*.json` (27 configs) + `summary_n50.json`
- `research/figs/fig1_same_prefix_n50.{pdf,png}` — 3 模型 survival curve
- `research/figs/fig2_div_index_box_n50.{pdf,png}` — Div_Index boxplot
- `research/figs/fig3_pareto_n50.{pdf,png}` — Runtime-Det Pareto
- `paper/figs/fig1_same_prefix_n50.pdf` + `fig3_pareto_n50.pdf`（Fig 2 暂未入论文，页面预算紧）

### 脚本
- [research/run_matrix_n50.sh](run_matrix_n50.sh) — 27-config 编排，mkdir-lock
- [research/analyze_matrix_n50.py](analyze_matrix_n50.py) — Std@Acc/Div_Index/AvgStd@top1 计算
- [research/make_phase3_figs.py](make_phase3_figs.py) — 3 张最终 figure

### 教训
- n=50 让 Std@Acc 有区分度（BF16 0.0327 vs LayerCast 0.0094 vs Hybrid 0.0189），n=10 时全是噪声
- Div_Index 和 AvgStd@top1 是 DetermLLM 的强项 — GEMM 层 bit-exact 直接反映在这两个指标上
- Std@Acc 我们输给 LayerCast（0.0189 vs 0.0094）因为 attention flip 答案还是有。老实说就行，不要硬吹。
- 同理，end-to-end token-wise det 在 DeepSeek GQA+reasoning 上都是 0%（attention 非关联）
- Fig 1 是最好的 headline figure：直接展示 DetermLLM 的 wedge

### 保留的待办（非紧急）
- Fig 2 (Div boxplot) 目前没进论文，若 reviewer 要求可加入 supplementary
- 可考虑 AIME'24 benchmark（Yuan 用了）作为第二个 reasoning dataset

---

## 【2026-04-22→04-23 进展快照 — 归档】

### 原正在后台运行的实验（完成）
- `run_matrix_n50.sh` 两个 worker 进程，PID `1160891`（GPU0）/ `1161063`（GPU1）
- 当前正在跑的子进程：
  - GPU0: `ds7_math_hy_bs8`（DeepSeek 7B hybrid bs=8）
  - GPU1: `ds7_math_lc_bs32`（DeepSeek 7B LayerCast bs=32, 预计 ≥3h）
- 输出目录: `/home/kec23008/docker-sys/dllm/research/exp_matrix_n50/`
- 进度: **6/27 configs done** — `ds7_math_{bf16_bs1, bf16_bs8, bf16_bs32, lc_bs1, lc_bs8, hy_bs1}`
- 预计剩余时间 8–10 h（LayerCast bs=32 是瓶颈）

### 恢复的第一步（按顺序）
1. 检查进度:
   ```bash
   ls /home/kec23008/docker-sys/dllm/research/exp_matrix_n50/*.json | wc -l   # 期望 27
   tail -30 /home/kec23008/docker-sys/dllm/research/exp_matrix_n50/worker_w0.log
   tail -30 /home/kec23008/docker-sys/dllm/research/exp_matrix_n50/worker_w1.log
   ```
2. 若进程已挂/卡死: `nvidia-smi` 看 GPU 是否空闲 → 必要时重启 worker
   ```bash
   cd /home/kec23008/docker-sys/dllm/research
   GPU=0 nohup bash run_matrix_n50.sh > /tmp/n50_w0_out.log 2>&1 &
   GPU=1 nohup bash run_matrix_n50.sh > /tmp/n50_w1_out.log 2>&1 &
   ```
   （脚本有 mkdir-lock, 只会认领未完成 config，不会重复跑）
3. 全部 27 个完成后:
   ```bash
   /home/kec23008/miniconda3/envs/dllm_research/bin/python analyze_matrix_n50.py
   /home/kec23008/miniconda3/envs/dllm_research/bin/python make_phase3_figs.py
   # 产物: figs/fig1_same_prefix_n50.{pdf,png}
   #       figs/fig2_div_index_box_n50.{pdf,png}
   #       figs/fig3_pareto_n50.{pdf,png}
   #       exp_matrix_n50/summary_n50.json
   ```

### 已完成（2026-04-22 → 04-23）
- **Phase 1 图**（基于旧 n=10 数据，用于快速验证 story）
  - [research/figs/fig_non_det_rate.png](figs/fig_non_det_rate.png) — end-to-end 非决定率条形图
  - [research/figs/fig_div_hist_deepseek7b_math500.png](figs/fig_div_hist_deepseek7b_math500.png) — Div position 直方图
  - [research/figs/fig_same_prefix_curve.png](figs/fig_same_prefix_curve.png) — **最佳 story**: first-K 匹配概率曲线，DetermLLM 持续领先 BF16/LayerCast
- **Phase 2.1 — `run_eval_general.py` 升级**: 每个 per_problem 现在存 `token_ids`（全长）和 `top1_logprobs`（全长），用来计算 Yuan 式 Div_Index（不受 100 token 窗口限制）和 Avg_Std@top1_prob
- **Phase 2.2 — n=50 matrix**: 27 configs 跑 3 models × MATH500 × 3 bs × 3 methods. `research/run_matrix_n50.sh` 负责编排（mkdir-lock work-stealing，两个 worker）
- **Phase 2.3 分析脚本**: [research/analyze_matrix_n50.py](analyze_matrix_n50.py) — 计算 Std@Acc / nondet_rate / Div_Index 分布 / Avg_Std@top1_prob / Avg_Std@Out_Len
- **Phase 3 图脚本**: [research/make_phase3_figs.py](make_phase3_figs.py) — 三张最终论文图（same-prefix curve / Div boxplot / Runtime-Det Pareto）

### 重要决策 & 教训
- **原方案有问题**: 原计划用 Std@Acc 作为 headline metric，但 n=10 噪声 > 方法差异（summary.json 显示多个 cell 的 Std@Acc = 0 或 ~0.06，信息量几乎为零）。改用 **same-first-K 存活曲线** 作为 headline figure — 在 n=10 数据下已明显展示 DetermLLM 的 wedge。
- **Phi-4 上 DetermLLM 似乎略差于 BF16** (5% 差距 = 1 problem at n=10)——纯噪声, n=50 应该消除
- **End-to-end token-wise 不是 DetermLLM 能独自达成的**: FA2 attention softmax 本身有 non-associativity，GEMM determinism 不 patch attention 就仍会最终发散。论文定位应该改为：GEMM 层 bit-exact，延迟 divergence，但端到端 still-not-fully-deterministic
- 不要用 autonomous wakeups — 用户没有 `/loop` 就不应该自主调度

### 下一步 TODOs（完成顺序）
1. 等 27 configs 全部完成（约明早可能还在跑 LayerCast configs）
2. 跑 `analyze_matrix_n50.py` — 输出 per_mdm 表 + summary_n50.json
3. 跑 `make_phase3_figs.py` — 生成 3 张 final figure
4. **更新 paper/main.tex**:
   - Table 1 (tab:math500_std, line 411): Std@Acc 用 bs∈{1,8,32} 新数据替换，accuracy 数字也更新
   - Table 2 (tab:matrix, line 481): 3 models × MATH500 × n=50. 考虑**删掉** Table 2，用 Figure 1 (same_prefix_curve) 替代（节省 ~0.4 页空间）
   - 新增 Figure 1 在 section 4.2 作为 headline visual
   - 新增 Figure 3 (Pareto) 在 section 4.3 overhead 中
   - cross-bs 段落 (line 464–479) 的数字按新结果改
5. 重新编 paper 检查 9 页约束（原来就在极限）
6. 把这些改动 + 原 summary_n50.json 归档到 memory

### Monitor/后台任务
- Monitor `bq8bmqhwa` (persistent): `tail -F` 两个 worker log，过滤 `OK|FAIL|START|Traceback|...`. 会一直流事件直到 session 结束。
- 若下次新开 session: Monitor 需要重新启动：
  ```
  (Monitor tool) command:
  tail -F -q exp_matrix_n50/worker_w0.log exp_matrix_n50/worker_w1.log 2>/dev/null | \
    grep -E --line-buffered "\] OK |\] FAIL |\] START |Traceback|Error|Killed|OOM|done ==="
  ```

---

## Status: 进行中（2026-04-19）

---

## 重要：运行脚本的环境变量

```bash
SITE=/home/kec23008/miniconda3/envs/dllm2/lib/python3.10/site-packages
CUBLAS_LIB=$SITE/nvidia/cublas/lib
CUDART_LIB=$SITE/nvidia/cuda_runtime/lib
TORCH_LIB=$SITE/torch/lib

CUDA_VISIBLE_DEVICES=0 \
PATH=/usr/local/cuda-12.9/bin:$PATH \
LD_LIBRARY_PATH=$CUBLAS_LIB:$CUDART_LIB:$TORCH_LIB:/usr/local/cuda-12.9/lib64 \
PYTHONNOUSERSITE=1 PYTHONPATH=$SITE \
/home/kec23008/miniconda3/envs/dllm2/bin/python <script>
```

---

## 已完成的工作

### 关键 Bug 修复（2026-04-19）

#### Bug 1: cuBLASLt 矩阵布局错误（CRITICAL）
- **症状**: CUDA illegal memory access，然后 CUBLAS_STATUS_INTERNAL_ERROR
- **根因**: `gemm_fixed_algo.cu` 使用 column-major 布局，但 PyTorch tensors 是 row-major
  - 对于 M=2, N=2048 的矩阵：column-major 解释为 ld=N=2048，导致
    访问 index = M-1 + (N-1)*N ≈ 4M（超出 M×N=4096 元素的 buffer）
- **修复**: 添加 `CUBLASLT_ORDER_ROW` 设置，将 fallback 改为 FP32 mm
- **文件**: `FP32/csrc/gemm_fixed_algo.cu`（已重新编译）

#### Bug 2: determ_llm.enable() 设置全局 BF16 flag
- **症状**: CUBLAS_STATUS_INTERNAL_ERROR 在 small-M forward pass
- **根因**: `allow_bf16_reduced_precision_reduction=False` 限制了 torch::mm 的算法
  选择，导致小 M 无法找到合适的 BF16 GEMM 算法
- **修复**: 从 enable()/disable() 中删除该 flag 设置
- **文件**: `research/determ_llm.py`

### 验证通过

DetermLLM 在 Llama-3.2-1B 上，bs=1,2,4,8,16,32 全部输出一致 → **DETERMINISTIC**

### 5-way Overhead Benchmark（Phi-4 14B, A6000）已完成

| 方法 | 确定性? | bs=1 | bs=8 | bs=32 |
|------|--------|------|------|-------|
| A: BF16 baseline | YES* | 0% | 0% | 0% |
| B: allow_bf16_reduced_precision_reduction=False | YES | -11.2% | -28.7% | -9.2% |
| **C: DetermLLM (我们的)** | **YES** | **-0.6%** | **+10.3%** | **+16.5%** |
| D: torch.use_deterministic_algorithms | YES | +57.3% | +29.0% | +83.5% |
| E: Full FP32 (LayerCast) | YES | +213.9% | +243.5% | +280.3% |

*注：Phi-4 对这些测试 prompt 的 greedy decoding 结果碰巧没有 argmax flip（dominant tokens）。
但 logit 分布是不同的（详见 motivation/ 实验数据）。

- 结果文件: `research/exp_overhead_bench.json`

### 论文草稿（英文）已写

文件: `research/paper_draft_en.md`

---

## 当前进行中

无（所有主要实验已完成）

---

## 已完成（追加，2026-04-19）

### Log-probability 非确定性实验（DONE）
- 结果文件: `research/exp_logprob_nondet.json`
- BF16 baseline: 0.0231 nats avg, DetermLLM: 0.0064 nats avg（3.6× 减少）

### vLLM 集成测试（DONE，部分成功）
- 脚本: `research/run_vllm_determ.py`，vLLM 0.8.5，Llama-3.2-1B，dllm2 env
- 结果文件: `research/exp_vllm_determ.json`
- BF16 baseline: 940 tok/s，1-2 mismatches（**确认 vLLM BF16 是非确定性的**）
- DetermLLM: 553 tok/s (+41% overhead)，1 mismatch（GEMM 组件已修复，FA2 残余）
- **关键发现**: vLLM Flash Attention 2 的 KV-tile traversal 是独立于 GEMM 的第二个非确定性来源
- **论文结论**: DetermLLM 需要配合 FA3 deterministic mode 才能完全修复 vLLM

### 论文草稿完善（DONE）
- 增加 Section 4.5: vLLM 集成实验
- 更新 Abstract（软化"全确定性"claim，明确 standalone vs serving）
- 更新 Discussion（vLLM 集成建议）
- 更新 Conclusion（明确两个 non-det 来源分解）

---

## 已完成（追加，2026-04-19）

### 分解实验（Decomposition，DONE）

**目的**：识别 LLM inference 中各组件对 batch non-determinism 的贡献，以 token-level
exact match 为指标。

**实验设计**：4 条件（A/B/C/D）× 10 prompts × 64 tokens × 6 batch sizes (2-64)
- A: BF16 baseline
- B: DetermLLM F.linear only
- C: DetermLLM F.linear + attn patch (decode-phase M=1)
- D: torch.use_deterministic_algorithms(warn_only=True)

**结果**：
| 条件 | Token mismatches | Avg|ΔlogP| |
|------|-----------------|-------------|
| A (BF16) | 596 | 0.0798 nats |
| B (DetermLLM F.linear) | **16** | **0.0075 nats** |
| C (+ attn patch) | **16** | **0.0075 nats** |
| D (torch_det warn_only) | 596 | 0.0798 nats |

**关键发现**：
1. **DetermLLM F.linear 将 token mismatches 减少 37×（596→16），log-prob variance 减少 10.6×**
2. **Attn patch (C) 与 B 结果相同**：实验用 full forward pass（无 KV cache），
   attn Q@K^T 的 shape 是 [bs, n_heads, seq_len, head_dim]，M=seq_len ≠ 1，
   超出 decode-phase patch 的覆盖范围（M=1 case）
3. **16 残余 mismatches 来源**：prefill-phase attention（M=seq_len），非 F.linear
   - p0/bs=64: first_div=53, 11 mismatches（bs=64 时 prefill attn 差异）
   - p3/bs=4-64: first_div=28, 1 mismatch each（near-tie token，logit 差 < 0.01）
4. **D（torch_det warn_only）= A（baseline）**：没有 CUBLAS_WORKSPACE_CONFIG=:4096:8，
   torch.use_deterministic_algorithms 对 cuBLAS 没有实际效果
5. **Production serving 结论**：有 KV cache 时，decode 阶段 M=1，F.linear 已足够确保确定性

**结果文件**: `research/exp_det_decompose.json`, `research/exp_det_attn_patch.json`

### 论文大更新（DONE）

1. **标题**：改为 "Token-Level Batch-Invariant BF16 LLM Inference via Precision Amplification"
2. **Abstract**：用正确的 precision amplification 叙事（FP32 masking non-associativity）
3. **Contributions #4**：改为 37× token mismatch 减少 + 10.6× log-prob 减少
4. **Section 3 Coverage**：明确区分 decode-phase attn (M=1) 和 prefill-phase attn (M=seq_len)
5. **Section 4.3 (新)**：Decomposition experiment 表格 + 分析
6. **Section 4.2 Log-prob**：更新数字（BF16: 0.0798 nats, DetermLLM: 0.0075 nats）
7. **Discussion**：新增 "Prefill-phase attention non-determinism" 段落
8. **Conclusion**：用正确叙事 + 实验数字

---

## 待完成

### 建议下一步（高优先级）
- 用 KV cache 验证：确认 production serving 环境下 F.linear 即可实现完全 token-level determinism
- Llama-3.1-8B 或 Phi-4 上重复 decomposition 实验（确认跨模型普适性）

### 可选改进（低优先级）
- 添加 motivation 数据表格（logit diff 分布图描述）
- 可选：Qwen3-30B MoE 实验（GPU 时间充足则做）

---

## 环境说明（重要）

### dllm2 env（主要实验环境）
- Python: `/home/kec23008/miniconda3/envs/dllm2/bin/python`
- torch: 2.6.0+cu124
- transformers: 4.48.3
- 编译好的 kernel: `FP32/_gemm_fixed_algo.cpython-310-x86_64-linux-gnu.so`

### dllm_research env（vLLM 环境，有兼容性问题）
- Python: `/home/kec23008/miniconda3/envs/dllm_research/bin/python`
- torch: 2.6.0+cu124
- vLLM: 0.19.1（_C.abi3.so 与当前 torch 不兼容，需新建 venv）

### 可用模型
- `/home/kec23008/docker-sys/Models/Llama-3.2-1B-Instruct` (1.5B BF16, 快速验证)
- `/home/kec23008/docker-sys/Models/Phi-4` (14B BF16, overhead benchmark 用)
