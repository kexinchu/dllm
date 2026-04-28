#!/bin/bash
# Phase 2 — GPU 0: Llama-8B on MATH500 + AIME25 with main 3 methods.
LOGDIR=/home/kec23008/docker-sys/dllm/research/P0_summary
mkdir -p "$LOGDIR" /home/kec23008/docker-sys/dllm/research/exp_main
Q=$LOGDIR/queue_phase2_gpu0.log
PY=/home/kec23008/miniconda3/envs/dllm_research/bin/python
cd /home/kec23008/docker-sys/dllm
log() { echo "[$(date '+%F %T')] $*" | tee -a "$Q"; }

log "START Phase 2 GPU 0 (Llama-8B)"

# Llama-8B / MATH500
log "[run] llama8b math500 BF16+LayerCast+SRP-FP32"
CUDA_VISIBLE_DEVICES=0 $PY research/exp_core_eval.py \
  --model llama8b --dataset math500 \
  --methods BF16 LayerCast SRP-FP32 \
  --bs 1 8 16 --n-problems 30 \
  --out research/exp_main/llama8b_math500.json \
  >> research/exp_main/llama8b_math500.log 2>&1
log "DONE: llama8b math500"

# Llama-8B / AIME25
log "[run] llama8b aime25 BF16+LayerCast+SRP-FP32"
CUDA_VISIBLE_DEVICES=0 $PY research/exp_core_eval.py \
  --model llama8b --dataset aime25 \
  --methods BF16 LayerCast SRP-FP32 \
  --bs 1 8 16 --n-problems 30 \
  --out research/exp_main/llama8b_aime25.json \
  >> research/exp_main/llama8b_aime25.log 2>&1
log "DONE: llama8b aime25"

log "Phase 2 GPU 0 FINISHED"
