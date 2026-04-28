#!/bin/bash
LOGDIR=/home/kec23008/docker-sys/dllm/research/P0_summary
mkdir -p "$LOGDIR"
Q=$LOGDIR/queue_gpu1_fixes.log
PY=/home/kec23008/miniconda3/envs/dllm_research/bin/python
cd /home/kec23008/docker-sys/dllm
log() { echo "[$(date '+%F %T')] $*" | tee -a "$Q"; }

# 1. E3 DeepSeek prefill (~10 min)
log "START: E3 deepseek7b prefill (real)"
CUDA_VISIBLE_DEVICES=1 $PY research/exp_E3_prob_std.py --model deepseek7b \
  >> research/exp_E3/E3_deepseek7b_real.log 2>&1
log "DONE: E3 deepseek7b"

# 2. E4 Llama AIME25 eager (~1h, 30 problems)
log "START: E4 llama8b aime25 eager (rerun)"
CUDA_VISIBLE_DEVICES=1 $PY research/exp_E4_div_index.py \
  --model llama8b --dataset aime25 --n-problems 30 \
  >> research/exp_E4/E4_llama8b_aime25_eager.log 2>&1
log "DONE: E4 llama8b aime25 eager"

# 3. Re-render figures with everything we have
log "START: render figures"
$PY research/figs/render_all.py >> "$LOGDIR/figs_render.log" 2>&1
log "DONE: figures"

log "GPU 1 fixes queue FINISHED"
