# FP32 Accumulation Only: all reduction ops use FP32 accumulators, I/O stays BF16.
#
# Modules:
#   gemm_fp32_accum      - GEMM/Linear with FP32 accumulation (cuBLASLt/Triton)
#   rmsnorm_fp32_accum   - RMSNorm with FP32 variance accumulator
#   softmax_fp32_accum   - Softmax with FP32 max/sum accumulators
#   attention_fp32_accum - Flash-style attention with FP32 accumulators (no split-KV)
#   model_patcher        - Unified monkey-patch for HuggingFace models
#   reduction_ops        - Low-level reduction simulations (legacy)
