#!/bin/bash
# N=30 reviewer-defense run: HF + vLLM 三方对比 (5 methods total)
# Sequential on GPU 1; total wall ~2h 13min.

# NOTE: deliberately no `set -u` — we substitute $LD_LIBRARY_PATH which may be unset.
# We also do NOT `set -e` because we want phase 3 to attempt even if phase 2 fails.

# Ensure LD_LIBRARY_PATH is at least defined (empty) so ${LD_LIBRARY_PATH:-} substitution is safe.
: "${LD_LIBRARY_PATH:=}"

LOGDIR=/home/kec23008/docker-sys/dllm/research/exp_validate
MASTER=$LOGDIR/n30_master.log
N=30

cd /home/kec23008/docker-sys/dllm

date > "$MASTER"
echo "=== N=30 overnight runner started ===" >> "$MASTER"

# Phase 1: HF stack (BF16 + SRP-FP32 + vLLM-BI aten overrides via vendored module)
echo -e "\n[Phase 1/3] HF kernels.py (3 methods, ~75 min)" | tee -a "$MASTER"
date | tee -a "$MASTER"
N_PROB=$N CUDA_VISIBLE_DEVICES=1 \
  /home/kec23008/miniconda3/envs/dllm_research/bin/python \
  research/validate_avg_std_vllm_kernels.py \
  > "$LOGDIR/n30_hf_kernels.log" 2>&1
echo "[Phase 1 exit=$?]" | tee -a "$MASTER"
date | tee -a "$MASTER"

# Phase 2: vLLM stack BF16 (cudagraph + flash-attn)
echo -e "\n[Phase 2/3] vLLM native BF16 (~18 min)" | tee -a "$MASTER"
date | tee -a "$MASTER"
N_PROB=$N CUDA_VISIBLE_DEVICES=1 \
  LD_LIBRARY_PATH=/home/kec23008/miniconda3/envs/vllm_test/lib/python3.10/site-packages/nvidia/nvshmem/lib:$LD_LIBRARY_PATH \
  VLLM_USE_V1=1 \
  /home/kec23008/miniconda3/envs/vllm_test/bin/python \
  research/validate_avg_std_vllm_native.py \
  > "$LOGDIR/n30_vllm_bf16.log" 2>&1
echo "[Phase 2 exit=$?]" | tee -a "$MASTER"
date | tee -a "$MASTER"

# Phase 3: vLLM stack BATCH_INVARIANT
echo -e "\n[Phase 3/3] vLLM native + VLLM_BATCH_INVARIANT=1 (~40 min)" | tee -a "$MASTER"
date | tee -a "$MASTER"
N_PROB=$N CUDA_VISIBLE_DEVICES=1 \
  VLLM_BATCH_INVARIANT=1 \
  LD_LIBRARY_PATH=/home/kec23008/miniconda3/envs/vllm_test/lib/python3.10/site-packages/nvidia/nvshmem/lib:$LD_LIBRARY_PATH \
  VLLM_USE_V1=1 \
  /home/kec23008/miniconda3/envs/vllm_test/bin/python \
  research/validate_avg_std_vllm_native.py \
  > "$LOGDIR/n30_vllm_bi.log" 2>&1
echo "[Phase 3 exit=$?]" | tee -a "$MASTER"
date | tee -a "$MASTER"

# Final summary
echo -e "\n=== N=30 overnight runner DONE ===" | tee -a "$MASTER"
date | tee -a "$MASTER"

echo -e "\n--- Phase 1 tail ---" | tee -a "$MASTER"
tail -20 "$LOGDIR/n30_hf_kernels.log" >> "$MASTER" 2>&1

echo -e "\n--- Phase 2 tail ---" | tee -a "$MASTER"
grep -E "TOTAL|Avg_Std|problem [0-9]+:" "$LOGDIR/n30_vllm_bf16.log" | tail -10 >> "$MASTER" 2>&1

echo -e "\n--- Phase 3 tail ---" | tee -a "$MASTER"
grep -E "TOTAL|Avg_Std|problem [0-9]+:" "$LOGDIR/n30_vllm_bi.log" | tail -10 >> "$MASTER" 2>&1
