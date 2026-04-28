#!/bin/bash
# Phase-2-v2: rerun with corrected attention patching.
#   * LayerCast now patches torch.matmul and torch.bmm (was F.linear-only).
#   * DetermLLM uses triton backend (bit-exact F.linear) with attn=True
#     (FP32-upcast attention matmul). Hybrid backend's cuBLASLt path is
#     not bs-invariant; diagnose_triton_bs_invariance.py shows this.
# Focused on DeepSeek-7B MATH500 first; extend to Llama/Phi-4 after
# confirming the numbers.
#
# Usage: GPU=0 bash run_matrix_n50_v2.sh
#        GPU=1 bash run_matrix_n50_v2.sh
set -u

DLLM=/home/kec23008/docker-sys/dllm
RES=$DLLM/research
PY=/home/kec23008/miniconda3/envs/dllm_research/bin/python
OUT=$RES/exp_matrix_n50_v2
mkdir -p "$OUT"

GPU_ID="${GPU:-0}"
WID="w${GPU_ID}"
LOG=$OUT/worker_${WID}.log

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [${WID}] $*" | tee -a "$LOG"; }
log "=== v2 worker ${WID} on GPU ${GPU_ID} starting ==="

MODEL_DEEPSEEK=/home/kec23008/docker-sys/Models/DeepSeek-R1-Distill-Qwen-7B

# 6 configs: DeepSeek × {LayerCast-new, DetermLLM-Triton+attn} × {bs=1,8,32}
# BF16 not rerun (no code changes for it; reuse exp_matrix_n50/*bf16*).
CONFIGS=(
  "ds7_math_lc_bs1|$MODEL_DEEPSEEK|math500|layercast|1|1024|50"
  "ds7_math_lc_bs8|$MODEL_DEEPSEEK|math500|layercast|8|1024|50"
  "ds7_math_lc_bs32|$MODEL_DEEPSEEK|math500|layercast|32|1024|50"
  "ds7_math_tri_bs1|$MODEL_DEEPSEEK|math500|dllm_triton|1|1024|50"
  "ds7_math_tri_bs8|$MODEL_DEEPSEEK|math500|dllm_triton|8|1024|50"
  "ds7_math_tri_bs32|$MODEL_DEEPSEEK|math500|dllm_triton|32|1024|50"
)

log "v2 matrix: ${#CONFIGS[@]} configs"

RUN_IDX=0
for entry in "${CONFIGS[@]}"; do
  IFS='|' read -r tag model_path dataset method bs gen_len n_prob <<< "$entry"
  RUN_IDX=$((RUN_IDX + 1))
  OUT_FILE="$OUT/${tag}.json"
  LOCK="${OUT_FILE}.lock"

  if [ -f "$OUT_FILE" ]; then
    log "[$RUN_IDX] SKIP $tag (done)"
    continue
  fi
  if ! mkdir "$LOCK" 2>/dev/null; then
    log "[$RUN_IDX] SKIP $tag (claimed by other)"
    continue
  fi
  echo "$WID $$ $(date -u +%s)" > "$LOCK/info"
  trap 'rm -rf "$LOCK"' EXIT INT TERM

  log "[$RUN_IDX] START $tag  ds=$dataset method=$method bs=$bs"
  t_start=$(date +%s)
  cd "$RES" && CUDA_VISIBLE_DEVICES=$GPU_ID "$PY" run_eval_general.py \
    --model-path "$model_path" --dataset "$dataset" --method "$method" \
    --batch-size "$bs" --seed 0 --n-problems "$n_prob" \
    --gen-len "$gen_len" --out "$OUT_FILE" >> "$LOG" 2>&1
  rc=$?
  t_end=$(date +%s); duration=$((t_end - t_start))

  rm -rf "$LOCK"; trap - EXIT INT TERM

  if [ $rc -eq 0 ] && [ -f "$OUT_FILE" ]; then
    log "[$RUN_IDX] OK $tag duration=${duration}s"
  else
    log "[$RUN_IDX] FAIL $tag rc=$rc duration=${duration}s"
    [ -f "$OUT_FILE" ] && mv "$OUT_FILE" "${OUT_FILE}.failed" || true
  fi
done
log "=== v2 worker ${WID} done ==="
