#!/bin/bash
# Phase 2 — GPU 1: DeepSeek-R1-Distill-Qwen-7B on MATH500 + AIME25.
LOGDIR=/home/kec23008/docker-sys/dllm/research/P0_summary
mkdir -p "$LOGDIR" /home/kec23008/docker-sys/dllm/research/exp_main
Q=$LOGDIR/queue_phase2_gpu1.log
PY=/home/kec23008/miniconda3/envs/dllm_research/bin/python
cd /home/kec23008/docker-sys/dllm
log() { echo "[$(date '+%F %T')] $*" | tee -a "$Q"; }

log "START Phase 2 GPU 1 (DeepSeek-7B)"

# DeepSeek-7B / MATH500
log "[run] deepseek7b math500 BF16+LayerCast+SRP-FP32"
CUDA_VISIBLE_DEVICES=1 $PY research/exp_core_eval.py \
  --model deepseek7b --dataset math500 \
  --methods BF16 LayerCast SRP-FP32 \
  --bs 1 8 16 --n-problems 30 \
  --out research/exp_main/deepseek7b_math500.json \
  >> research/exp_main/deepseek7b_math500.log 2>&1
log "DONE: deepseek7b math500"

# DeepSeek-7B / AIME25
log "[run] deepseek7b aime25 BF16+LayerCast+SRP-FP32"
CUDA_VISIBLE_DEVICES=1 $PY research/exp_core_eval.py \
  --model deepseek7b --dataset aime25 \
  --methods BF16 LayerCast SRP-FP32 \
  --bs 1 8 16 --n-problems 30 \
  --out research/exp_main/deepseek7b_aime25.json \
  >> research/exp_main/deepseek7b_aime25.log 2>&1
log "DONE: deepseek7b aime25"

log "Phase 2 GPU 1 FINISHED"
