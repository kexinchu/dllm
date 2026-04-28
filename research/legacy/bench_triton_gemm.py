"""
Benchmark Triton deterministic GEMM against:
  - Baseline BF16 (torch.nn.functional.linear, cuBLAS)
  - Our cuBLASLt wrapper (gemm_fixed_algo)

Reports per-call latency in microseconds and the overhead relative to baseline.
"""
import os, sys, time
import torch
import torch.nn.functional as F

DLLM_DIR = os.path.dirname(os.path.abspath(__file__))
FP32_DIR = os.path.join(DLLM_DIR, '..', 'FP32')
sys.path.insert(0, FP32_DIR)
sys.path.insert(0, DLLM_DIR)

from triton_det_gemm import det_gemm
import determ_llm  # loads cuBLASLt kernel on demand

# Explicitly load the cuBLASLt kernel for direct benchmarking.
_CUBLASLT = determ_llm._load_kernel()


device = 'cuda:0'

SHAPES = [
    ('1B attn',        2048, 2048),
    ('1B FFN up',      8192, 2048),
    ('1B FFN down',    2048, 8192),
    ('8B attn',        4096, 4096),
    ('8B FFN up',     14336, 4096),
    ('8B FFN down',    4096, 14336),
    ('Phi-4 attn',     5120, 5120),
    ('Phi-4 FFN up',  17920, 5120),
]

BATCH_SIZES = [1, 8, 32]

N_WARMUP = 20
N_ITER = 200


def bench_one(func, A, B):
    # warmup (also triggers autotune for Triton)
    for _ in range(N_WARMUP):
        _ = func(A, B)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(N_ITER):
        _ = func(A, B)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / N_ITER
    return ms * 1000  # microseconds


def bench_baseline(A, B):
    return F.linear(A, B)


def bench_cublaslt(A, B):
    return _CUBLASLT.gemm_fixed_algo(A, B)


def bench_triton(A, B):
    return det_gemm(A, B)


print(f"Benchmark: per-call latency in microseconds")
print(f"warmup={N_WARMUP}, iterations={N_ITER}\n")

header = f"{'shape':<14} {'M':>3} | {'baseline':>10} {'cuBLASLt':>10} {'Triton':>10} | {'cuBLASLt ovhd':>14} {'Triton ovhd':>13}"
print(header)
print('-' * len(header))

results = {}
for label, N, K in SHAPES:
    for M in BATCH_SIZES:
        torch.manual_seed(42)
        A = torch.randn(M, K, device=device, dtype=torch.bfloat16)
        B = (torch.randn(N, K, device=device, dtype=torch.float32) / (K ** 0.5)).to(torch.bfloat16)

        t_base = bench_one(bench_baseline, A, B)
        t_cublaslt = bench_one(bench_cublaslt, A, B)
        t_triton = bench_one(bench_triton, A, B)

        cublaslt_ovhd = (t_cublaslt / t_base - 1) * 100
        triton_ovhd = (t_triton / t_base - 1) * 100

        print(f"{label:<14} {M:>3} | {t_base:>10.1f} {t_cublaslt:>10.1f} {t_triton:>10.1f}"
              f" | {cublaslt_ovhd:>+13.1f}% {triton_ovhd:>+12.1f}%")

        results[(label, M)] = {
            'baseline_us': t_base,
            'cublaslt_us': t_cublaslt,
            'triton_us': t_triton,
            'cublaslt_ovhd_pct': cublaslt_ovhd,
            'triton_ovhd_pct': triton_ovhd,
        }

# Summary
print()
print('=' * 80)
print('SUMMARY (avg over all shapes)')
print('=' * 80)
import statistics
for bs in BATCH_SIZES:
    cu = [v['cublaslt_ovhd_pct'] for (lab, m), v in results.items() if m == bs]
    tr = [v['triton_ovhd_pct'] for (lab, m), v in results.items() if m == bs]
    print(f"M={bs:>2}:  cuBLASLt avg ovhd = {statistics.mean(cu):+.1f}%   "
          f"Triton avg ovhd = {statistics.mean(tr):+.1f}%")

# Save JSON
import json
out_file = os.path.join(DLLM_DIR, 'exp_bench_triton.json')
serialized = {f'{k[0]}|M={k[1]}': v for k, v in results.items()}
with open(out_file, 'w') as f:
    json.dump({'per_call_us': serialized}, f, indent=2)
print(f"\nSaved -> {out_file}")
