"""Benchmark cuBLASLt kernel on the small-N DeepSeek shapes where Triton struggles.

If cuBLASLt handles small N faster than Triton, a hybrid dispatcher can win.
"""
import os, sys
import torch
import torch.nn.functional as F

DLLM = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DLLM)
sys.path.insert(0, os.path.join(DLLM, '..', 'FP32'))

from triton_det_gemm import det_gemm
import determ_llm
cublaslt_kernel = determ_llm._load_kernel()


SHAPES = [
    ('attn_q',  3584,  3584),
    ('attn_k',  512,   3584),
    ('attn_v',  512,   3584),
    ('attn_o',  3584,  3584),
    ('ffn_gate', 18944, 3584),
    ('ffn_down', 3584, 18944),
]

BATCH_SIZES = [1, 8, 16, 32]
device = 'cuda:0'
torch.manual_seed(42)


def bench(func, A, B, n_iter=100):
    for _ in range(5):
        _ = func(A, B)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        _ = func(A, B)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_iter * 1000


print(f"{'shape':<10} {'M':>3} {'K':>6} {'N':>7} | {'BF16':>8} {'cuBLASLt':>10} {'Triton':>9} | {'winner':>12}")
print('-' * 85)

for label, N, K in SHAPES:
    for M in BATCH_SIZES:
        A = torch.randn(M, K, device=device, dtype=torch.bfloat16).contiguous()
        B = ((torch.randn(N, K, device=device, dtype=torch.float32) / (K**0.5)).to(torch.bfloat16)).contiguous()

        t_base = bench(F.linear, A, B)
        t_lt = bench(cublaslt_kernel.gemm_fixed_algo, A, B)
        t_tri = bench(det_gemm, A, B)

        winner = 'cuBLASLt' if t_lt < t_tri else 'Triton'
        margin = abs(t_lt - t_tri)
        print(f"{label:<10} {M:>3} {K:>6} {N:>7} | {t_base:>7.1f}us {t_lt:>9.1f}us {t_tri:>8.1f}us | {winner:>8} by {margin:.1f}us")
