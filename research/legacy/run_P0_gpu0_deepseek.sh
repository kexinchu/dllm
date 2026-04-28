#!/bin/bash
# DeepSeek experiments on GPU 0 (parallel to Llama on GPU 1).
set -e
LOGDIR=/home/kec23008/docker-sys/dllm/research/P0_summary
mkdir -p "$LOGDIR"
Q=$LOGDIR/queue_gpu0.log
PY=/home/kec23008/miniconda3/envs/dllm_research/bin/python
cd /home/kec23008/docker-sys/dllm

log() { echo "[$(date '+%F %T')] $*" | tee -a "$Q"; }

# 1. E7 DeepSeek (overhead) — 20 min
log "START: E7 DeepSeek"
mkdir -p research/exp_E7
CUDA_VISIBLE_DEVICES=0 $PY research/exp_E7_overhead.py \
  --model DeepSeek-R1-Qwen-7B --no-fp32full \
  >> research/exp_E7/E7_deepseek.log 2>&1
log "DONE: E7 DeepSeek"

# 2. E4 DeepSeek MATH500 — ~3h
log "START: E4 deepseek7b math500"
mkdir -p research/exp_E4
CUDA_VISIBLE_DEVICES=0 $PY research/exp_E4_div_index.py \
  --model deepseek7b --dataset math500 --n-problems 30 \
  >> research/exp_E4/E4_deepseek7b_math500.log 2>&1
log "DONE: E4 deepseek7b math500"

# 3. E4 DeepSeek AIME25 — ~2h
log "START: E4 deepseek7b aime25"
CUDA_VISIBLE_DEVICES=0 $PY research/exp_E4_div_index.py \
  --model deepseek7b --dataset aime25 --n-problems 30 \
  >> research/exp_E4/E4_deepseek7b_aime25.log 2>&1
log "DONE: E4 deepseek7b aime25"

# 4. E2 DeepSeek prob-gap — ~1h
log "START: E2 prob-gap DeepSeek"
mkdir -p research/exp_E2
CUDA_VISIBLE_DEVICES=0 $PY research/exp_E2_prob_gap.py \
  --model deepseek7b --n-problems 30 \
  >> research/exp_E2/E2_deepseek.log 2>&1
log "DONE: E2 prob-gap DeepSeek"

log "GPU 0 queue FINISHED"
