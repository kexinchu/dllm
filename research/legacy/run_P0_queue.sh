#!/bin/bash
# P0 experiment queue. Waits for the currently running E4 (pid passed via $1)
# to finish, then runs each subsequent experiment on GPU 1 sequentially.
# Logs to /home/kec23008/docker-sys/dllm/research/P0_summary/queue.log

set -e
WAIT_PID="${1:-}"
LOGDIR=/home/kec23008/docker-sys/dllm/research/P0_summary
mkdir -p "$LOGDIR"
Q=$LOGDIR/queue.log

PY=/home/kec23008/miniconda3/envs/dllm_research/bin/python
cd /home/kec23008/docker-sys/dllm

log() { echo "[$(date '+%F %T')] $*" | tee -a "$Q"; }

if [[ -n "$WAIT_PID" ]]; then
  log "waiting for PID $WAIT_PID ..."
  while ps -p "$WAIT_PID" >/dev/null 2>&1; do sleep 60; done
  log "PID $WAIT_PID done."
fi

# 1. E4 Llama AIME25
log "START: E4 llama8b aime25"
CUDA_VISIBLE_DEVICES=1 $PY research/exp_E4_div_index.py \
  --model llama8b --dataset aime25 --n-problems 30 \
  >> research/exp_E4/E4_llama8b_aime25.log 2>&1
log "DONE: E4 llama8b aime25"

# 2. E4 DeepSeek MATH500
log "START: E4 deepseek7b math500"
CUDA_VISIBLE_DEVICES=1 $PY research/exp_E4_div_index.py \
  --model deepseek7b --dataset math500 --n-problems 30 \
  >> research/exp_E4/E4_deepseek7b_math500.log 2>&1
log "DONE: E4 deepseek7b math500"

# 3. E4 DeepSeek AIME25
log "START: E4 deepseek7b aime25"
CUDA_VISIBLE_DEVICES=1 $PY research/exp_E4_div_index.py \
  --model deepseek7b --dataset aime25 --n-problems 30 \
  >> research/exp_E4/E4_deepseek7b_aime25.log 2>&1
log "DONE: E4 deepseek7b aime25"

# 4. E7 DeepSeek rerun (single model only)
log "START: E7 DeepSeek (rerun)"
mkdir -p research/exp_E7
CUDA_VISIBLE_DEVICES=1 $PY research/exp_E7_overhead.py \
  --model DeepSeek-R1-Qwen-7B --no-fp32full \
  >> research/exp_E7/E7_deepseek_rerun.log 2>&1
log "DONE: E7 DeepSeek"

# 5. E2 prob gap on DeepSeek
log "START: E2 prob-gap (DeepSeek)"
mkdir -p research/exp_E2
CUDA_VISIBLE_DEVICES=1 $PY research/exp_E2_prob_gap.py \
  --model deepseek7b --n-problems 30 \
  >> research/exp_E2/E2_deepseek.log 2>&1
log "DONE: E2 prob-gap"

# 6. Final aggregation
log "START: analyze_P0_results"
$PY research/analyze_P0_results.py > "$LOGDIR/final_summary.log" 2>&1
log "DONE: all P0 experiments"
