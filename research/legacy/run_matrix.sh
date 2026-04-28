#!/bin/bash
# Comprehensive determinism matrix across models × datasets × batch sizes × methods.
#
# Usage: GPU=0 bash run_matrix.sh
#        GPU=1 bash run_matrix.sh
# Workers race for configs via mkdir-lock. Each config writes to a single JSON.
set -u

DLLM=/home/kec23008/docker-sys/dllm
RES=$DLLM/research
PY=/home/kec23008/miniconda3/envs/dllm_research/bin/python
OUT=$RES/exp_matrix
mkdir -p "$OUT"

GPU_ID="${GPU:-0}"
WID="w${GPU_ID}"
LOG=$OUT/worker_${WID}.log

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [${WID}] $*" | tee -a "$LOG"; }
log "=== Matrix worker ${WID} on GPU ${GPU_ID} starting ==="

# Models:
#   DeepSeek-R1-Distill-Qwen-7B  reasoning, GQA
#   Llama-3.1-8B-Instruct         non-reasoning, GQA
#   Phi-4                         non-reasoning, dense (no GQA)
MODEL_DEEPSEEK=/home/kec23008/docker-sys/Models/DeepSeek-R1-Distill-Qwen-7B
MODEL_LLAMA=/home/kec23008/docker-sys/Models/Llama-3.1-8B-Instruct
MODEL_PHI4=/home/kec23008/docker-sys/Models/Phi-4

# Config format: "short_name|model_path|dataset|method|bs|gen_len|n_problems"
# 3 models * 2 datasets * 3 bs * 3 methods = 54 configs, n=10 per config
CONFIGS=(
  # --- DeepSeek-7B (reasoning) ---
  # MATH500
  "ds7_math_bf16_bs1|$MODEL_DEEPSEEK|math500|bf16_baseline|1|1024|10"
  "ds7_math_bf16_bs8|$MODEL_DEEPSEEK|math500|bf16_baseline|8|1024|10"
  "ds7_math_bf16_bs32|$MODEL_DEEPSEEK|math500|bf16_baseline|32|1024|10"
  "ds7_math_lc_bs1|$MODEL_DEEPSEEK|math500|layercast|1|1024|10"
  "ds7_math_lc_bs8|$MODEL_DEEPSEEK|math500|layercast|8|1024|10"
  "ds7_math_lc_bs32|$MODEL_DEEPSEEK|math500|layercast|32|1024|10"
  "ds7_math_hy_bs1|$MODEL_DEEPSEEK|math500|dllm_hybrid|1|1024|10"
  "ds7_math_hy_bs8|$MODEL_DEEPSEEK|math500|dllm_hybrid|8|1024|10"
  "ds7_math_hy_bs32|$MODEL_DEEPSEEK|math500|dllm_hybrid|32|1024|10"
  # GSM8K
  "ds7_gsm_bf16_bs1|$MODEL_DEEPSEEK|gsm8k|bf16_baseline|1|1024|10"
  "ds7_gsm_bf16_bs8|$MODEL_DEEPSEEK|gsm8k|bf16_baseline|8|1024|10"
  "ds7_gsm_bf16_bs32|$MODEL_DEEPSEEK|gsm8k|bf16_baseline|32|1024|10"
  "ds7_gsm_lc_bs1|$MODEL_DEEPSEEK|gsm8k|layercast|1|1024|10"
  "ds7_gsm_lc_bs8|$MODEL_DEEPSEEK|gsm8k|layercast|8|1024|10"
  "ds7_gsm_lc_bs32|$MODEL_DEEPSEEK|gsm8k|layercast|32|1024|10"
  "ds7_gsm_hy_bs1|$MODEL_DEEPSEEK|gsm8k|dllm_hybrid|1|1024|10"
  "ds7_gsm_hy_bs8|$MODEL_DEEPSEEK|gsm8k|dllm_hybrid|8|1024|10"
  "ds7_gsm_hy_bs32|$MODEL_DEEPSEEK|gsm8k|dllm_hybrid|32|1024|10"

  # --- Llama-3.1-8B-Instruct (non-reasoning) --- shorter gen, faster
  "llm8_math_bf16_bs1|$MODEL_LLAMA|math500|bf16_baseline|1|512|10"
  "llm8_math_bf16_bs8|$MODEL_LLAMA|math500|bf16_baseline|8|512|10"
  "llm8_math_bf16_bs32|$MODEL_LLAMA|math500|bf16_baseline|32|512|10"
  "llm8_math_lc_bs1|$MODEL_LLAMA|math500|layercast|1|512|10"
  "llm8_math_lc_bs8|$MODEL_LLAMA|math500|layercast|8|512|10"
  "llm8_math_lc_bs32|$MODEL_LLAMA|math500|layercast|32|512|10"
  "llm8_math_hy_bs1|$MODEL_LLAMA|math500|dllm_hybrid|1|512|10"
  "llm8_math_hy_bs8|$MODEL_LLAMA|math500|dllm_hybrid|8|512|10"
  "llm8_math_hy_bs32|$MODEL_LLAMA|math500|dllm_hybrid|32|512|10"
  "llm8_gsm_bf16_bs1|$MODEL_LLAMA|gsm8k|bf16_baseline|1|512|10"
  "llm8_gsm_bf16_bs8|$MODEL_LLAMA|gsm8k|bf16_baseline|8|512|10"
  "llm8_gsm_bf16_bs32|$MODEL_LLAMA|gsm8k|bf16_baseline|32|512|10"
  "llm8_gsm_lc_bs1|$MODEL_LLAMA|gsm8k|layercast|1|512|10"
  "llm8_gsm_lc_bs8|$MODEL_LLAMA|gsm8k|layercast|8|512|10"
  "llm8_gsm_lc_bs32|$MODEL_LLAMA|gsm8k|layercast|32|512|10"
  "llm8_gsm_hy_bs1|$MODEL_LLAMA|gsm8k|dllm_hybrid|1|512|10"
  "llm8_gsm_hy_bs8|$MODEL_LLAMA|gsm8k|dllm_hybrid|8|512|10"
  "llm8_gsm_hy_bs32|$MODEL_LLAMA|gsm8k|dllm_hybrid|32|512|10"

  # --- Phi-4 14B (non-reasoning, no GQA) ---
  "phi4_math_bf16_bs1|$MODEL_PHI4|math500|bf16_baseline|1|512|10"
  "phi4_math_bf16_bs8|$MODEL_PHI4|math500|bf16_baseline|8|512|10"
  "phi4_math_bf16_bs32|$MODEL_PHI4|math500|bf16_baseline|32|512|10"
  "phi4_math_lc_bs1|$MODEL_PHI4|math500|layercast|1|512|10"
  "phi4_math_lc_bs8|$MODEL_PHI4|math500|layercast|8|512|10"
  "phi4_math_lc_bs32|$MODEL_PHI4|math500|layercast|32|512|10"
  "phi4_math_hy_bs1|$MODEL_PHI4|math500|dllm_hybrid|1|512|10"
  "phi4_math_hy_bs8|$MODEL_PHI4|math500|dllm_hybrid|8|512|10"
  "phi4_math_hy_bs32|$MODEL_PHI4|math500|dllm_hybrid|32|512|10"
  "phi4_gsm_bf16_bs1|$MODEL_PHI4|gsm8k|bf16_baseline|1|512|10"
  "phi4_gsm_bf16_bs8|$MODEL_PHI4|gsm8k|bf16_baseline|8|512|10"
  "phi4_gsm_bf16_bs32|$MODEL_PHI4|gsm8k|bf16_baseline|32|512|10"
  "phi4_gsm_lc_bs1|$MODEL_PHI4|gsm8k|layercast|1|512|10"
  "phi4_gsm_lc_bs8|$MODEL_PHI4|gsm8k|layercast|8|512|10"
  "phi4_gsm_lc_bs32|$MODEL_PHI4|gsm8k|layercast|32|512|10"
  "phi4_gsm_hy_bs1|$MODEL_PHI4|gsm8k|dllm_hybrid|1|512|10"
  "phi4_gsm_hy_bs8|$MODEL_PHI4|gsm8k|dllm_hybrid|8|512|10"
  "phi4_gsm_hy_bs32|$MODEL_PHI4|gsm8k|dllm_hybrid|32|512|10"
)

log "Matrix: ${#CONFIGS[@]} configs"

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

  log "[$RUN_IDX] START $tag  model=$(basename $model_path) ds=$dataset method=$method bs=$bs"
  t_start=$(date +%s)
  cd "$RES" && CUDA_VISIBLE_DEVICES=$GPU_ID "$PY" run_eval_general.py \
    --model-path "$model_path" \
    --dataset "$dataset" \
    --method "$method" \
    --batch-size "$bs" \
    --seed 0 \
    --n-problems "$n_prob" \
    --gen-len "$gen_len" \
    --out "$OUT_FILE" \
    >> "$LOG" 2>&1
  rc=$?
  t_end=$(date +%s)
  duration=$((t_end - t_start))

  rm -rf "$LOCK"
  trap - EXIT INT TERM

  if [ $rc -eq 0 ] && [ -f "$OUT_FILE" ]; then
    log "[$RUN_IDX] OK $tag duration=${duration}s"
  else
    log "[$RUN_IDX] FAIL $tag rc=$rc duration=${duration}s"
    [ -f "$OUT_FILE" ] && mv "$OUT_FILE" "${OUT_FILE}.failed" || true
  fi
done
log "=== Matrix worker ${WID} done ==="
