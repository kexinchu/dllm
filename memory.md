# Project Resume Memory — DetermLLM / SRP NeurIPS 2027

**Last updated:** 2026-04-28 ~15:00 EDT, on host with /home/kec23008/docker-sys/dllm
**Reason for snapshot:** moving to a different machine because GPU 0+1 here are saturated by another user (`ieb24002` running Qwen3-32B eval since ~10:30 AM, ~44 GB used per GPU).

---

## ⚠️ Before you switch machines — sync the working tree

The repo has **many uncommitted/untracked changes that must travel with you**, including the entire `paper/` directory and recent `research/` updates. `git pull` alone on the new machine will not bring them.

```bash
git status -s   # large list — most of paper/, research/, FP32/, NIPS-2027.md, plan.md, docs/ are untracked
```

**Options before leaving:**
1. Commit + push everything important to a working branch (cleanest), or
2. `rsync -avz /home/kec23008/docker-sys/dllm/ <new-host>:/path/to/dllm/` (preserves uncommitted state)

The Claude `memory/` index lives at `~/.claude/projects/-home-kec23008-docker-sys-dllm/memory/` — it does **not** travel via git. On the new machine, a fresh Claude session will rely on this `memory.md` (in the repo) instead of those entries. Three memory files worth re-creating manually if you want continuity:
- `finding_srp_beats_vllm.md` — same content as §"Headline Result" below
- `project_role.md` — Claude leads the project; user assists
- `finding_padding_rope.md` — HF batched inference non-determinism is RoPE position shift from left-padding (legacy finding, still relevant background)

---

## TL;DR — Where we are

We're building a NeurIPS 2027 paper claiming **Selective Reduction Precision (SRP)**: fixed-plan Triton kernels at the 4 reduction sites (`linear / rmsnorm / attention / softmax`) achieve strict bit-exact batch invariance (Avg_Std@top1 = 0) at lower wall-clock than vLLM's official `VLLM_BATCH_INVARIANT=1` mode. Today's main contribution datapoint is locked in. We're now hardening it for reviewer defense (N=30) and rewriting the paper §1/§3 framing.

**Today's status:**
- ✅ Paper headline data captured (N=5 + partial N=30, see below)
- ✅ paper/main.tex updated with new §6.7 + abstract + contribution (vi); compiles cleanly to 19 pages
- ✅ N=30 Phase 1 (HF stack — BF16, SRP-FP32, vLLM-BI aten overrides) finished overnight
- ⏸ N=30 Phase 2 + Phase 3 (vLLM native BF16, vLLM+BATCH_INVARIANT) **blocked**: GPU contention from another user
- 🔲 §1/§3 framing rewrite still pending — current §1 talks about cuBLAS-only split-K, doesn't mention 4-site SRP yet

---

## Headline Result (the paper claim)

5-prob DeepSeek-R1-Distill-Qwen-7B / MATH500 / greedy 256 tok / bs∈{1,2,4,8,16}:

| Method | Avg_Std@top1 | Total | vs HF-BF16 | vs vLLM-BF16 |
|---|---|---|---|---|
| HF BF16 (DynamicCache) | 3.97e-3 | 194s | 0% | — |
| vLLM BF16 (cudagraph + FA2) | 2.60e-3 | 179s | -8% | 0% |
| vLLM BF16 (FA2 explicit, fair re-run) | 2.87e-3 | 177s | -9% | -1% |
| **HF + SRP-FP32 (ours)** | **0** strict | **279s** | **+44%** | **+56%** |
| vLLM + `VLLM_BATCH_INVARIANT=1` | 2.55e-5 | 402s | +108% | +125% |
| FP32-all (reference floor) | 3.59e-7 | ~470s | +140% | +163% |

→ Our method is **simultaneously stricter** (Std=0 by construction) **and faster** (-31%) than vLLM's official deterministic mode.

### Same finding holds at N=30 (HF stack, finished 2026-04-28 02:21 AM)

| Method | Avg_Std@top1 | Total | Overhead |
|---|---|---|---|
| HF BF16 | 3.54e-3 | 998.8s | +0% |
| **HF + SRP-FP32 (ours)** | **0** strict | 1470.5s | +47.2% |
| HF + vLLM-BI aten overrides | 7.26e-4 | 1639.5s | +64.1% |

(N=30 result from `research/exp_validate/vllm_kernels_results_n30.json`. Linear scaling 5.3× from N=5 datapoint, confirming N=5 wasn't a noise artifact.)

**Same-stack ablation observation (important for paper):** when you put both methods inside the *same* HF eager pipeline (controlled experiment, no vLLM cudagraph confound), our SRP still beats vLLM's `batch_invariant_ops` aten overrides on **both** axes — strictly better Avg_Std (0 vs 7.3e-4) AND lower wall (+47% vs +64%). This is a stronger claim than the cross-stack comparison because it isolates the algorithm.

---

## What to do on the new machine

### Step 0: sync repo + envs
- Bring full `dllm/` tree (rsync, branch push, or otherwise — see top of file)
- Recreate two conda envs:
  - `dllm_research` (HF stack): `transformers==4.48.x`, `torch==2.6.0+cu124`, `triton`, plus our `research/srp_kernels/batch_invariant_vllm.py` (vendored from vLLM with stubs)
  - `vllm_test` (vLLM 0.19.x): needs `flash-attn 2.8.3` and `nvidia-nvshmem-cu12` site-package. Run with `LD_LIBRARY_PATH=/path/to/site-packages/nvidia/nvshmem/lib:$LD_LIBRARY_PATH`
- Bring DeepSeek-R1-Distill-Qwen-7B model weights to `<new>/Models/DeepSeek-R1-Distill-Qwen-7B` — script paths are hardcoded as `/home/kec23008/docker-sys/Models/...`, so either symlink or sed-replace
- Bring `research/math500_cached.json` (500-prob cached MATH500)

### Step 1: run Phase 2 + 3 (vLLM stack at N=30)

Two scripts already prepared and ready:
- `research/run_n30_vllm_only.sh` — runs Phase 2 (vLLM BF16) + Phase 3 (vLLM+BI) sequentially, ~58 min total

```bash
cd /path/to/dllm
nohup bash research/run_n30_vllm_only.sh > research/exp_validate/n30_vllm_nohup.out 2>&1 &
```

Both scripts honor `N_PROB` env override; default is 5. Output JSONs go to:
- `research/exp_validate/vllm_native_bf16_results_n30.json`
- `research/exp_validate/vllm_native_bi_results_n30.json`

Check progress: `tail -f research/exp_validate/n30_vllm_master.log`

### Step 2: update paper with N=30 numbers

Once Phase 2+3 finish, replace the table in `paper/main.tex` §6.7 (Table `tab:srp_vs_vllm_bi`) with N=30 data. Also update `docs/srp_perf_log.md` and `NIPS-2027.md §15`.

### Step 3 (highest leverage, paper writing): rewrite §1 + §3

Current main.tex §1 (Introduction) and §3 (Method) are still in the old "DetermLLM = cuBLAS GEMM split-K fix" framing. The paper now has a wider claim covering 4 reduction sites and direct vLLM-BI superiority. Source material already prepared:
- `NIPS-2027.md §15` — vLLM comparison narrative (drop into §1 punchline)
- `NIPS-2027.md §5.2 + §5.3` — 4-site classification + implementation details (drop into §3)
- `NIPS-2027.md §1.3` — source-code-verified facts about FP32 accumulator defaults (drop into §2 Background)

Estimated 2-3h of LaTeX work.

### Step 4: figures
The Pareto scatter (x = Total wall, y = log Avg_Std) for the 5 methods in the headline table is the missing main figure. matplotlib 30 min job once N=30 numbers are final.

---

## Key files map

| File | Purpose |
|---|---|
| [`memory.md`](memory.md) | This file — resume context |
| [`NIPS-2027.md`](NIPS-2027.md) | Paper plan; §15 has the 2026-04-26 headline narrative |
| [`plan.md`](plan.md) | Execution plan; top section has Step E completion + Step D abandonment notes |
| [`docs/srp_perf_log.md`](docs/srp_perf_log.md) | Performance log with all method comparisons |
| [`paper/main.tex`](paper/main.tex) | LaTeX — abstract + contribution (vi) + §6.7 already updated |
| [`paper/main.pdf`](paper/main.pdf) | 19-page compiled PDF |
| [`research/legacy/agent_experience.md`](research/legacy/agent_experience.md) | Day-by-day project log; 2026-04-26 evening entry has more detail |

### SRP implementation
| File | What it does |
|---|---|
| `FP32/model_patcher.py` | Monkey-patches `nn.Linear`, `LlamaRMSNorm`, attention, softmax. Sets `_HYBRID_M_THRESHOLD = 0` → forces all GEMMs to Triton det_gemm |
| `FP32/triton_det_gemm.py` | Fixed-plan no-split-K GEMM |
| `FP32/attention_fp32_accum.py` | FP32-accumulator attention (decode-specialized + prefill paths) |
| `research/methods.py` | Method context managers: `method_BF16`, `method_LayerCast`, `method_SRP_FP32`, `method_FP32_all` |
| `research/srp_kernels/batch_invariant_vllm.py` | Vendored vLLM batch_invariant.py with stubs for `vllm.envs/logger/platforms` (works in HF env without vllm package) |

### Validation scripts
| File | What it runs |
|---|---|
| `research/validate_avg_std_vllm_kernels.py` | HF stack: BF16, SRP-FP32 (ours), vLLM-BI aten overrides. Honors `N_PROB` env |
| `research/validate_avg_std_vllm_native.py` | vLLM stack: BF16 (no flag) or +BI (with `VLLM_BATCH_INVARIANT=1`). Honors `N_PROB` and `VLLM_BATCH_INVARIANT` envs |
| `research/validate_combo.py` | Combo experiments (vLLM-BI + ours overlay, ablations) |
| `research/run_n30_overnight.sh` | All 3 phases (HF + vLLM BF16 + vLLM BI). Phase 1 already done; only re-run if you want fresh Phase 1 |
| `research/run_n30_vllm_only.sh` | **Use this on new machine** — only Phase 2+3 (vLLM) at N=30 |

### Data files (already produced)
| File | Result |
|---|---|
| `research/exp_validate/vllm_kernels_results.json` | N=5 HF stack 3-way |
| `research/exp_validate/vllm_kernels_results_n30.json` | **N=30 HF stack 3-way (today's overnight Phase 1)** |
| `research/exp_validate/vllm_native_bf16_results.json` | N=5 vLLM BF16 (apples-to-apples FLASH_ATTN) |
| `research/exp_validate/vllm_native_bf16_results.no_flash_attn.json` | N=5 vLLM BF16 (original, default attn backend) |
| `research/exp_validate/vllm_native_results.json` | N=5 vLLM+BI (will be regenerated as `vllm_native_bi_results_n5.json` if re-run; actual filename-suffix logic was added today) |
| `research/exp_validate/combo_results.json` | N=5 ablation: BF16, vLLM-BI alone, vLLM-BI + Triton attn, vLLM-BI + full SRP overlay |

---

## Failed paths — DO NOT retry on new machine (lessons §6.7)

The following were tried in HF eager and exhausted; none can hit `Avg_Std=0` strict:
1. **cuBLASLt fixed-algo backend** — Std drops to 2e-3 only (50%); not strict
2. **Hybrid M-thresholded dispatch** (M ≤ 16 cuBLASLt, M > 16 Triton) — same 2e-3 floor
3. **`torch.compile(mode='reduce-overhead')`** — dynamo treats Triton kernels as graph breaks, no cuda graph happens
4. **Manual cuda graph capture of decode step** — HF Llama/Qwen2 attention has Python `.item()` sync points baked in at capture time; tokens diverge at idx=1
5. **HF StaticCache eager alone** — +12 to +61% regression with no determinism benefit

→ The path forward for *lower-than-44% overhead* is integrating our fixed-plan Triton kernels directly into vLLM's stack (replacing their `batch_invariant_ops`). This is **future work**, not in NeurIPS scope.

---

## Risks reviewers may raise (and our defenses)

| Risk | Defense |
|---|---|
| "N=5 isn't enough" | N=30 Phase 1 (HF stack) confirms; finishing N=30 Phase 2+3 on new machine closes this |
| "vLLM+BI's +125% overhead is way above their reported 34% — did you misconfigure?" | Same-stack ablation (HF+SRP +47% beats HF+vLLM-BI-aten +64%) shows the algorithm gap is real, independent of vLLM stack overhead. Add a paragraph noting their 34% is on *vLLM* hardware/scheduler, not a portable algorithm property |
| "Only 1 model 1 dataset" | Llama-8B + AIME25 are next priorities (use same `validate_avg_std_vllm_*.py` with model-path env override or sed-replace) |
| "Bit-exact = 0 sounds suspicious" | Our claim is "by construction": no split-K, no split-KV, fixed BLOCK_K — the reduction order does not depend on bs. This is structural, not statistical |

---

## TODO when resumed

1. ⏳ Run Phase 2 + 3 at N=30 on the new GPU (`bash research/run_n30_vllm_only.sh`)
2. 🔲 Update `paper/main.tex` Table `tab:srp_vs_vllm_bi` with N=30 numbers
3. 🔲 Rewrite `paper/main.tex` §1 Introduction (4-site SRP framing, vLLM-BI punchline) — biggest leverage
4. 🔲 Rewrite `paper/main.tex` §3 Method (4-site fixed-plan kernel description)
5. 🔲 Generate Pareto main figure (matplotlib, ~30 min)
6. 🔲 If time: Llama-8B + AIME25 quick eval (N=5 enough for "model-generality" defense)
