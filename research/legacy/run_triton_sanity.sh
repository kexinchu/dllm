#!/bin/bash
# Quick test: DeepSeek-7B with pure Triton backend (not hybrid) across bs
# to confirm F.linear bit-exact covers full residual but attention still limits.
set -u

DLLM=/home/kec23008/docker-sys/dllm
RES=$DLLM/research
PY=/home/kec23008/miniconda3/envs/dllm_research/bin/python
OUT=$RES/exp_matrix_dtype
mkdir -p "$OUT"

MODEL=/home/kec23008/docker-sys/Models/DeepSeek-R1-Distill-Qwen-7B

for BS in 1 4 8 16 32; do
  OUT_FILE="$OUT/ds7_triton_bs${BS}.json"
  if [ -f "$OUT_FILE" ]; then
    echo "SKIP ds7_triton_bs${BS} (done)"
    continue
  fi
  echo "START ds7_triton_bs${BS}"
  t0=$(date +%s)
  CUDA_VISIBLE_DEVICES=1 "$PY" run_eval_general.py \
    --model-path "$MODEL" --dataset math500 --method dllm_triton \
    --dtype bfloat16 --batch-size "$BS" --seed 0 --n-problems 10 \
    --gen-len 256 --out "$OUT_FILE" 2>&1 | tail -4
  t1=$(date +%s)
  echo "OK ds7_triton_bs${BS} duration=$((t1-t0))s"
done
