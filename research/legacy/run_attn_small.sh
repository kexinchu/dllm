#!/bin/bash
# Fast attn-patch experiment: n=10 problems, gen_len=1024
PY=/home/kec23008/miniconda3/envs/dllm_research/bin/python
OUT=/home/kec23008/docker-sys/dllm/research/exp_math500_overnight

for bs in 8 16; do
  OUT_FILE=$OUT/dllm_hybrid_attn_bs${bs}_seed0_small.json
  if [ -f "$OUT_FILE" ]; then continue; fi
  CUDA_VISIBLE_DEVICES=0 $PY run_math500_eval.py \
    --method dllm_hybrid_attn --batch-size $bs --seed 0 \
    --n-problems 10 --gen-len 1024 \
    --out "$OUT_FILE" >> $OUT/attn_small.log 2>&1
done

# Also run a matching layercast with n=10 for fair comparison
for bs in 8 16; do
  OUT_FILE=$OUT/layercast_bs${bs}_seed0_small.json
  if [ -f "$OUT_FILE" ]; then continue; fi
  CUDA_VISIBLE_DEVICES=0 $PY run_math500_eval.py \
    --method layercast --batch-size $bs --seed 0 \
    --n-problems 10 --gen-len 1024 \
    --out "$OUT_FILE" >> $OUT/attn_small.log 2>&1
done

# And matching hybrid (no attn) for 3-way comparison
for bs in 8 16; do
  OUT_FILE=$OUT/dllm_hybrid_bs${bs}_seed0_small.json
  if [ -f "$OUT_FILE" ]; then continue; fi
  CUDA_VISIBLE_DEVICES=0 $PY run_math500_eval.py \
    --method dllm_hybrid --batch-size $bs --seed 0 \
    --n-problems 10 --gen-len 1024 \
    --out "$OUT_FILE" >> $OUT/attn_small.log 2>&1
done

echo "=== ALL SMALL RUNS DONE ===" >> $OUT/attn_small.log
