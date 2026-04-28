#!/bin/bash
# E5 site ablation on Llama-8B / MATH500.
# Goal:
#   1. Validate the 3 predictions (linear+attn ≈ all sites; rmsnorm/softmax ≈ BF16)
#   2. URGENT: identify which site causes the SRP-FP32 acc=0% regression seen in Phase 2
LOGDIR=/home/kec23008/docker-sys/dllm/research/P0_summary
mkdir -p "$LOGDIR" /home/kec23008/docker-sys/dllm/research/exp_E5
Q=$LOGDIR/queue_E5_validate.log
PY=/home/kec23008/miniconda3/envs/dllm_research/bin/python
cd /home/kec23008/docker-sys/dllm
log() { echo "[$(date '+%F %T')] $*" | tee -a "$Q"; }

log "START: E5 Llama-8B / MATH500 / 4 site-ablation methods"

CUDA_VISIBLE_DEVICES=0 $PY research/exp_core_eval.py \
  --model llama8b --dataset math500 \
  --methods SRP-FP32-linear SRP-FP32-rmsnorm SRP-FP32-attention SRP-FP32-Critical \
  --bs 1 8 16 --n-problems 30 \
  --out research/exp_E5/E5_llama8b_math500.json \
  > research/exp_E5/E5_llama8b_math500.log 2>&1
log "DONE: E5 Llama-8B / MATH500"
