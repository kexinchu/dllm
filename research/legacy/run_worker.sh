#!/bin/bash
# Worker for parallel MATH500 evaluation.
#
# Usage: GPU=0 bash run_worker.sh
#        GPU=1 bash run_worker.sh
#
# Two workers on separate GPUs can run concurrently. Each worker loops through
# the config matrix and atomically claims configs via `mkdir <lockfile>`.
# A config is done when its output json exists (no lock, json present).

set -u

DLLM=/home/kec23008/docker-sys/dllm
RES=$DLLM/research
PY=/home/kec23008/miniconda3/envs/dllm_research/bin/python
OUT=$RES/exp_math500_overnight
mkdir -p "$OUT"

GPU_ID="${GPU:-0}"
WORKER_ID="w${GPU_ID}"
LOG=$OUT/worker_${WORKER_ID}.log

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] [${WORKER_ID}] $*" | tee -a "$LOG"
}

log "=== Worker ${WORKER_ID} on GPU ${GPU_ID} starting ==="

MODEL_DIR=/home/kec23008/docker-sys/Models/DeepSeek-R1-Distill-Qwen-7B
if [ ! -f "$MODEL_DIR/config.json" ]; then
  log "ERROR: model not downloaded."
  exit 1
fi

N_PROBLEMS=50
GEN_LEN=2048

# Full config matrix. Workers race for each entry via mkdir lock.
# Format: "method|bs|seed"
CONFIGS=(
  # seed=0 new (hybrid)
  "dllm_hybrid|8|0"
  "dllm_hybrid|16|0"
  # seed=1 full grid
  "dllm_hybrid|8|1"
  "dllm_hybrid|16|1"
  "bf16_baseline|8|1"
  "bf16_baseline|16|1"
  "dllm_cublaslt|8|1"
  "dllm_cublaslt|16|1"
  "dllm_triton|8|1"
  "dllm_triton|16|1"
)

for entry in "${CONFIGS[@]}"; do
  IFS='|' read -r method bs seed <<< "$entry"

  OUT_FILE="$OUT/${method}_bs${bs}_seed${seed}.json"
  LOCK="${OUT_FILE}.lock"

  # Skip if output already exists (idempotent)
  if [ -f "$OUT_FILE" ]; then
    log "SKIP ${method} bs=${bs} seed=${seed} (already done)"
    continue
  fi

  # Try to claim: mkdir is atomic on local filesystems.
  if ! mkdir "$LOCK" 2>/dev/null; then
    log "SKIP ${method} bs=${bs} seed=${seed} (claimed by other worker)"
    continue
  fi

  # Write PID into the lock for debugging
  echo "$WORKER_ID $$ $(date -u +%s)" > "$LOCK/info"

  # Cleanup on unexpected exit
  trap 'rm -rf "$LOCK"' EXIT INT TERM

  log "START ${method} bs=${bs} seed=${seed}"
  t_start=$(date +%s)
  cd "$RES" && CUDA_VISIBLE_DEVICES="$GPU_ID" "$PY" run_math500_eval.py \
    --method "$method" \
    --batch-size "$bs" \
    --seed "$seed" \
    --n-problems "$N_PROBLEMS" \
    --gen-len "$GEN_LEN" \
    --out "$OUT_FILE" \
    >> "$LOG" 2>&1
  rc=$?
  t_end=$(date +%s)
  duration=$((t_end - t_start))

  rm -rf "$LOCK"
  trap - EXIT INT TERM

  if [ $rc -eq 0 ] && [ -f "$OUT_FILE" ]; then
    log "OK ${method} bs=${bs} seed=${seed} duration=${duration}s"
  else
    log "FAIL ${method} bs=${bs} seed=${seed} rc=$rc"
    [ -f "$OUT_FILE" ] && mv "$OUT_FILE" "${OUT_FILE}.failed" || true
  fi
done

log "=== Worker ${WORKER_ID} done. ==="
