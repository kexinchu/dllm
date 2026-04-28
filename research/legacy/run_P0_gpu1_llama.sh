#!/bin/bash
# Wait for current E4 Llama MATH500 (PID $1) to finish, then run remaining
# Llama experiments on GPU 1. Finally aggregate.
set -e
WAIT_PID="${1:-}"
LOGDIR=/home/kec23008/docker-sys/dllm/research/P0_summary
mkdir -p "$LOGDIR"
Q=$LOGDIR/queue_gpu1.log
PY=/home/kec23008/miniconda3/envs/dllm_research/bin/python
cd /home/kec23008/docker-sys/dllm

log() { echo "[$(date '+%F %T')] $*" | tee -a "$Q"; }

if [[ -n "$WAIT_PID" ]]; then
  log "waiting for PID $WAIT_PID (current E4 Llama MATH500) ..."
  while ps -p "$WAIT_PID" >/dev/null 2>&1; do sleep 60; done
  log "PID $WAIT_PID done."
fi

# 1. E4 Llama AIME25 — ~1.5h
log "START: E4 llama8b aime25"
CUDA_VISIBLE_DEVICES=1 $PY research/exp_E4_div_index.py \
  --model llama8b --dataset aime25 --n-problems 30 \
  >> research/exp_E4/E4_llama8b_aime25.log 2>&1
log "DONE: E4 llama8b aime25"

# 2. (Optional) rerun E3 on DeepSeek for completeness if GPU 0 done
# Not adding here — E3 DeepSeek could be run if time permits.

# Wait a bit for GPU 0 queue to be near done, then aggregate
log "waiting briefly for GPU 0 stragglers ..."
# Aggregate what we have
log "START: analyze_P0_results"
$PY research/analyze_P0_results.py > "$LOGDIR/final_summary.log" 2>&1 || true
log "DONE: initial aggregation (re-run when DeepSeek is fully done)"
