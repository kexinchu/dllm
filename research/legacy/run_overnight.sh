#!/bin/bash
# Overnight MATH500 evaluation orchestrator (v2).
#
# v2 changes:
#   - Skip bs=32 (too slow, user decided)
#   - Add dllm_hybrid method (cuBLASLt for small-N, Triton for large-N)
#   - layercast only runs seed=0 (already have enough data showing 3x slowdown)
#
# Output: exp_math500_overnight/<method>_bs<N>_seed<S>.json
# Idempotent: already-completed configs are skipped.

set -u

DLLM=/home/kec23008/docker-sys/dllm
RES=$DLLM/research
PY=/home/kec23008/miniconda3/envs/dllm_research/bin/python
OUT=$RES/exp_math500_overnight
mkdir -p "$OUT"
LOG=$OUT/orchestrator.log

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
}

log "=== Overnight MATH500 eval v2 starting ==="
log "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
log "Free memory: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader | head -1)"

# Skip download if already present.
MODEL_DIR=/home/kec23008/docker-sys/Models/DeepSeek-R1-Distill-Qwen-7B
if [ ! -f "$MODEL_DIR/config.json" ]; then
  log "ERROR: model not downloaded. Run download_deepseek.py first."
  exit 1
fi
log "Model present at $MODEL_DIR"

N_PROBLEMS=50
GEN_LEN=2048

# Matrix
#   bf16_baseline, dllm_cublaslt, dllm_triton, dllm_hybrid: full grid (2bs × 2seed = 4)
#   layercast: seed=0 only (already shown 3x slower; more configs just cost time)
# Total unique configs: 4 * 4 + 1 * 2 = 18 (of which 8 already done: we just skip).

BATCH_SIZES=(8 16)
SEEDS=(0 1)

log "Matrix: methods = bf16/cublaslt/triton/hybrid across ${#BATCH_SIZES[@]}bs x ${#SEEDS[@]}seed; layercast seed=0 only"
log "n_problems=$N_PROBLEMS, gen_len=$GEN_LEN"

RUN_IDX=0

run_one() {
  local method=$1
  local bs=$2
  local seed=$3
  RUN_IDX=$((RUN_IDX + 1))
  local OUT_FILE="$OUT/${method}_bs${bs}_seed${seed}.json"
  if [ -f "$OUT_FILE" ]; then
    log "[$RUN_IDX] SKIP ${method} bs=${bs} seed=${seed}"
    return
  fi
  log "[$RUN_IDX] START ${method} bs=${bs} seed=${seed}"
  local t_start; t_start=$(date +%s)
  cd "$RES" && CUDA_VISIBLE_DEVICES=0 "$PY" run_math500_eval.py \
    --method "$method" \
    --batch-size "$bs" \
    --seed "$seed" \
    --n-problems "$N_PROBLEMS" \
    --gen-len "$GEN_LEN" \
    --out "$OUT_FILE" \
    >> "$LOG" 2>&1
  local rc=$?
  local t_end; t_end=$(date +%s)
  local duration=$((t_end - t_start))

  if [ $rc -eq 0 ] && [ -f "$OUT_FILE" ]; then
    log "[$RUN_IDX] OK ${method} bs=${bs} seed=${seed} duration=${duration}s"
  else
    log "[$RUN_IDX] FAIL ${method} bs=${bs} seed=${seed} rc=$rc"
    [ -f "$OUT_FILE" ] && mv "$OUT_FILE" "${OUT_FILE}.failed" || true
  fi
}

# Priority order: finish the 'new' methods first so we get clear picture sooner.
# Interleave by method to get a per-method row across bs × seed early.
for seed in "${SEEDS[@]}"; do
  for bs in "${BATCH_SIZES[@]}"; do
    # New method first (missing from previous run)
    run_one "dllm_hybrid"    "$bs" "$seed"
    # Bf16 baseline for comparison (seed=0 already done)
    run_one "bf16_baseline"  "$bs" "$seed"
    run_one "dllm_cublaslt"  "$bs" "$seed"
    run_one "dllm_triton"    "$bs" "$seed"
    # layercast only seed=0
    if [ "$seed" -eq 0 ]; then
      run_one "layercast"    "$bs" "$seed"
    fi
  done
done

log "=== Overnight eval v2 done. Attempted $RUN_IDX runs. ==="
log "Next: run aggregate_math500.py"
