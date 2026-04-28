#!/bin/bash
# Parallel runner for dllm_hybrid_attn 4 configs on 2 GPUs.
# Also run layercast bs=16 seed=1 to complete the matrix for cross-bs analysis.

DLLM=/home/kec23008/docker-sys/dllm
RES=$DLLM/research
PY=/home/kec23008/miniconda3/envs/dllm_research/bin/python
OUT=$RES/exp_math500_overnight

GPU=$1  # 0 or 1
shift
JOBS=("$@")

LOG=$OUT/worker_attn_gpu${GPU}.log
echo "[$(date)] [gpu${GPU}] starting jobs: ${JOBS[@]}" >> "$LOG"

for job in "${JOBS[@]}"; do
  IFS='|' read -r method bs seed <<< "$job"
  OUT_FILE=$OUT/${method}_bs${bs}_seed${seed}.json
  if [ -f "$OUT_FILE" ]; then
    echo "[$(date)] [gpu${GPU}] skip $job" >> "$LOG"
    continue
  fi
  echo "[$(date)] [gpu${GPU}] start $job" >> "$LOG"
  t0=$(date +%s)
  cd "$RES" && CUDA_VISIBLE_DEVICES=$GPU "$PY" run_math500_eval.py \
    --method "$method" --batch-size "$bs" --seed "$seed" \
    --n-problems 50 --gen-len 2048 --out "$OUT_FILE" \
    >> "$LOG" 2>&1
  rc=$?
  t1=$(date +%s)
  echo "[$(date)] [gpu${GPU}] done $job rc=$rc dur=$((t1-t0))s" >> "$LOG"
done
echo "[$(date)] [gpu${GPU}] ALL DONE" >> "$LOG"
