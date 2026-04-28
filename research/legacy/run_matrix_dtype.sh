#!/bin/bash
# Batch-size sweep: 3 methods × 3 models × 7 bs × MATH500, n=10 per config.
#
# Methods:
#   1. bf16    : model dtype=bfloat16, no patch                → baseline
#   2. fp32    : model dtype=float32, no patch                 → full-FP32
#   3. dllm    : model dtype=bfloat16, DetermLLM hybrid patch  → ours
#
# Phi-4 FP32 is skipped at high bs — 14B × 4bytes = 56 GB does not fit the
# RTX A6000 (48 GB) together with a bs=128 KV cache.
#
# Usage: GPU=0 bash run_matrix_dtype.sh
#        GPU=1 bash run_matrix_dtype.sh
set -u

DLLM=/home/kec23008/docker-sys/dllm
RES=$DLLM/research
PY=/home/kec23008/miniconda3/envs/dllm_research/bin/python
OUT=$RES/exp_matrix_dtype
mkdir -p "$OUT"

GPU_ID="${GPU:-0}"
WID="w${GPU_ID}"
LOG=$OUT/worker_${WID}.log

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [${WID}] $*" | tee -a "$LOG"; }
log "=== dtype-sweep worker ${WID} on GPU ${GPU_ID} starting ==="

MODEL_DS=/home/kec23008/docker-sys/Models/DeepSeek-R1-Distill-Qwen-7B
MODEL_LM=/home/kec23008/docker-sys/Models/Llama-3.1-8B-Instruct
MODEL_P4=/home/kec23008/docker-sys/Models/Phi-4

BS_LIST=(1 4 8 16 32 64 128)
GEN_LEN=256
N_PROB=10

# Format: "short_name|model_path|method|dtype|bs"
declare -a CONFIGS=()
for BS in "${BS_LIST[@]}"; do
  # DeepSeek-7B × 3 methods
  CONFIGS+=("ds7_bf16_bs${BS}|$MODEL_DS|bf16_baseline|bfloat16|${BS}")
  CONFIGS+=("ds7_fp32_bs${BS}|$MODEL_DS|bf16_baseline|float32|${BS}")
  CONFIGS+=("ds7_dllm_bs${BS}|$MODEL_DS|dllm_hybrid|bfloat16|${BS}")
  # Llama-8B × 3 methods
  CONFIGS+=("lm8_bf16_bs${BS}|$MODEL_LM|bf16_baseline|bfloat16|${BS}")
  CONFIGS+=("lm8_fp32_bs${BS}|$MODEL_LM|bf16_baseline|float32|${BS}")
  CONFIGS+=("lm8_dllm_bs${BS}|$MODEL_LM|dllm_hybrid|bfloat16|${BS}")
  # Phi-4 × 2 methods (skip FP32 — model won't fit in 48 GB)
  CONFIGS+=("p4_bf16_bs${BS}|$MODEL_P4|bf16_baseline|bfloat16|${BS}")
  CONFIGS+=("p4_dllm_bs${BS}|$MODEL_P4|dllm_hybrid|bfloat16|${BS}")
done

log "dtype-sweep: ${#CONFIGS[@]} configs, n_prob=${N_PROB}, gen_len=${GEN_LEN}"

RUN_IDX=0
for entry in "${CONFIGS[@]}"; do
  IFS='|' read -r tag model_path method dtype bs <<< "$entry"
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

  log "[$RUN_IDX] START $tag  model=$(basename $model_path) method=$method dtype=$dtype bs=$bs"
  t_start=$(date +%s)
  cd "$RES" && CUDA_VISIBLE_DEVICES=$GPU_ID "$PY" run_eval_general.py \
    --model-path "$model_path" --dataset math500 --method "$method" --dtype "$dtype" \
    --batch-size "$bs" --seed 0 --n-problems "$N_PROB" --gen-len "$GEN_LEN" \
    --out "$OUT_FILE" >> "$LOG" 2>&1
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
log "=== dtype-sweep worker ${WID} done ==="
