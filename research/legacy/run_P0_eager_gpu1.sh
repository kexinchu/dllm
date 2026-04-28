#!/bin/bash
# GPU 1: lighter Llama experiments (eager attention) + DeepSeek prefill.
LOGDIR=/home/kec23008/docker-sys/dllm/research/P0_summary
mkdir -p "$LOGDIR"
Q=$LOGDIR/queue_gpu1_eager.log
PY=/home/kec23008/miniconda3/envs/dllm_research/bin/python
cd /home/kec23008/docker-sys/dllm

log() { echo "[$(date '+%F %T')] $*" | tee -a "$Q"; }

# 1. E4 Llama MATH500 eager (re-do correctly with eager, 30 problems)
log "START: E4 llama8b math500 (eager, 30 problems)"
mkdir -p research/exp_E4
CUDA_VISIBLE_DEVICES=1 $PY research/exp_E4_div_index.py \
  --model llama8b --dataset math500 --n-problems 30 \
  >> research/exp_E4/E4_llama8b_math500_eager.log 2>&1
log "DONE: E4 llama8b math500 eager"

# 2. E4 Llama AIME25 eager (~1.5h)
log "START: E4 llama8b aime25 (eager, 30 problems)"
CUDA_VISIBLE_DEVICES=1 $PY research/exp_E4_div_index.py \
  --model llama8b --dataset aime25 --n-problems 30 \
  >> research/exp_E4/E4_llama8b_aime25_eager.log 2>&1
log "DONE: E4 llama8b aime25 eager"

# 3. E3 DeepSeek prefill (10 min)  — adds the second curve to Fig 3
log "START: E3 deepseek7b prefill"
mkdir -p research/exp_E3
CUDA_VISIBLE_DEVICES=1 $PY research/exp_E3_prob_std.py \
  --out research/exp_E3/E3_deepseek7b.json \
  >> research/exp_E3/E3_deepseek7b.log 2>&1
log "DONE: E3 deepseek7b"

log "GPU 1 eager queue FINISHED"
