#!/bin/bash
# GPU 0: heavy DeepSeek experiments (eager attention).
# Each step logs to its own file; queue progress to queue_gpu0_eager.log.
LOGDIR=/home/kec23008/docker-sys/dllm/research/P0_summary
mkdir -p "$LOGDIR"
Q=$LOGDIR/queue_gpu0_eager.log
PY=/home/kec23008/miniconda3/envs/dllm_research/bin/python
cd /home/kec23008/docker-sys/dllm

log() { echo "[$(date '+%F %T')] $*" | tee -a "$Q"; }

# Don't use `set -e` — if one step OOMs we want others to continue.

# 1. E4 DeepSeek MATH500 eager (~6h)
log "START: E4 deepseek7b math500 (eager, 30 problems)"
mkdir -p research/exp_E4
CUDA_VISIBLE_DEVICES=0 $PY research/exp_E4_div_index.py \
  --model deepseek7b --dataset math500 --n-problems 30 \
  >> research/exp_E4/E4_deepseek7b_math500_eager.log 2>&1
log "DONE: E4 deepseek7b math500"

# 2. E4 DeepSeek AIME25 eager (~3h)
log "START: E4 deepseek7b aime25 (eager, 30 problems)"
CUDA_VISIBLE_DEVICES=0 $PY research/exp_E4_div_index.py \
  --model deepseek7b --dataset aime25 --n-problems 30 \
  >> research/exp_E4/E4_deepseek7b_aime25_eager.log 2>&1
log "DONE: E4 deepseek7b aime25"

# 3. E2 prob-gap DeepSeek (eager) — mechanism figure (~1h)
log "START: E2 prob-gap (deepseek, eager)"
mkdir -p research/exp_E2
CUDA_VISIBLE_DEVICES=0 $PY research/exp_E2_prob_gap.py \
  --model deepseek7b --n-problems 30 \
  >> research/exp_E2/E2_deepseek_eager.log 2>&1
log "DONE: E2 prob-gap deepseek"

log "GPU 0 eager queue FINISHED"
