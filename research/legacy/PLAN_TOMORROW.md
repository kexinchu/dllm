# Plan for 2026-04-22

## Overnight status (2026-04-22 ~00:30)

Pipeline v2 running on both A6000 GPUs with work-stealing workers.
- **16/18 configs done**
- Last 2 still running: `dllm_cublaslt_bs16_seed1` (GPU 0), `dllm_triton_bs16_seed1` (GPU 1)
- Expected completion: ~01:30-02:00

Workers:
- `bash run_worker.sh` × 2 (PIDs 3223299, 3223376), shell-disowned
- Logs: `exp_math500_overnight/worker_w0.log`, `worker_w1.log`
- Monitor: `bin09hece` edge-triggered, notifies on each completion + the final "ALL DONE"

## Today's key progress

### 1. Triton +14.5% overhead fix → Hybrid backend

Root cause: Triton kernel has a **~67μs launch-overhead floor**. On DeepSeek's
GQA K/V projections (K=3584, N=512), this floor dwarfs the actual compute
(baseline 12-16μs per call). Large-N shapes (FFN, lm_head) amortize the floor
and Triton wins there.

Fix: `_gemm()` in `research/determ_llm.py` gained a `hybrid` backend that
routes by `N`:
- `N <= 4096` → cuBLASLt path (low dispatch floor)
- `N > 4096` → Triton path (amortized, sometimes faster than baseline)

Per-op benchmark on DeepSeek shapes (summed):
| Backend | Total μs | vs BF16 |
|---------|----------|---------|
| BF16 baseline | 5469 | 0% |
| Triton pure | 5745 | +5.0% |
| **Hybrid** | **5464** | **−0.1%** |
| cuBLASLt pure | 5487 | +0.3% |
| LayerCast | 46197 | +407% |

### 2. MATH500 end-to-end results (on DeepSeek-R1-Distill-Qwen-7B)

Per-config runtime (s) on 50 problems × 2048 gen_len:
- bf16_baseline: ~3400 (bs=8) / ~4500 (bs=16)
- dllm_cublaslt: ~3400 / ~4560
- **dllm_hybrid: ~3555 / ~4725** (+4.9-5.6%, the new recommendation)
- dllm_triton: ~4025 / ~4990 (+19.5-10.8%, pure Triton)
- layercast: ~11000 / ~12200 (+227% / +172%, the competing method!)

Cross-seed bit-exactness verified for all deterministic methods:
- hybrid bs=8 seed=0/1: both 36.0%, rt 3556s vs 3554s (0.06%)
- hybrid bs=16 seed=0/1: both 38.0%, rt 4723s vs 4729s (0.13%)
- cublaslt bs=8 seed=0/1: both 42.0%
- triton bs=8 seed=0/1: both 40.0%

### 3. Paper updates so far

- `paper/main.tex` Related Work: added Yuan et al. 2025 as closest prior with
  explicit LayerCast comparison (weights stay BF16 in our backends vs.
  LayerCast doubling weight bandwidth)
- `paper/references.bib`: added `yuan2025numerical`, `deepseekr1`,
  `hendrycks2021math`
- `paper/experiments_v2_draft.tex`: draft of new Experiments section centered
  on MATH500 + Std@Acc (headline) with `{TBD}` placeholders for real numbers.
  Not yet spliced into `main.tex`.
- Paper compiles, still 9 pages main content, bibliography on page 10-11.

## Tomorrow work queue

1. **Wait for last 2 configs** (should be done by 02:00). If worker died and
   configs missing, restart with `GPU=0 bash run_worker.sh &` etc.
2. **Aggregate**: `python aggregate_math500.py` → `exp_math500_overnight/aggregate.{json,md}`
3. **Fill in draft**: replace `{TBD}` in `experiments_v2_draft.tex` with
   actual numbers from aggregate.json.
4. **Integrate** draft into `paper/main.tex`. Open question: does the new
   section *replace* the old Llama-1B token-mismatch-centric experiments, or
   sit alongside? My recommendation: lead with MATH500 (new), demote the
   token mismatch decomposition to Appendix or a later "Decomposition"
   subsection.
5. **Recompile + 9-page check**. Likely will need trimming.
6. **Investigate low hybrid accuracy**: hybrid 36-38% vs BF16 44-42% on
   MATH500. 6-8% absolute is a lot. Candidates:
   - Triton kernel correctness bug (unlikely given op-level bench at 1e-2
     tolerance)
   - Prompt template inconsistency across methods
   - Something in `set_method()` cleanup
   - Real numerical effect: FP32 compute changes which problems are correct
     (Yuan et al. describe this as "precision changes accuracy")
   Quick check: diff first 5 generated tokens for the same problem across
   methods; if they diverge early, it's a real precision effect; if late
   divergence, probably a bug.

## Files written today

- `research/determ_llm.py` — added `hybrid` backend, `_gemm()` routing
- `research/run_worker.sh` — parallel work-stealing worker
- `research/profile_triton_deepseek.py` — per-shape overhead profiler
- `research/profile_triton_vs_layercast.py` — direct LayerCast comparison
- `research/bench_cublaslt_small_shapes.py` — justifies hybrid threshold
- `research/measure_triton_overhead.py` — confirms Triton launch floor
- `research/investigate_triton_configs.py` — dumps autotune cache
- `paper/experiments_v2_draft.tex` — new Experiments skeleton
- `paper/references.bib` — Yuan 2025, DeepSeek-R1, MATH datasets

## Files updated

- `paper/main.tex` — Related Work now cites Yuan et al. and compares against
  LayerCast directly
- `paper/references.bib` — 3 new entries
- This plan doc
