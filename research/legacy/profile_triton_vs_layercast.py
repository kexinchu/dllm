"""Apples-to-apples: Triton vs LayerCast on DeepSeek shapes (both deterministic FP32 compute).

Also includes BF16 baseline as reference (non-deterministic).
"""
import os, sys
import torch
import torch.nn.functional as F

DLLM = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DLLM)
sys.path.insert(0, os.path.join(DLLM, '..', 'FP32'))

from triton_det_gemm import det_gemm
import layercast


SHAPES = [
    ('attn_q',   3584,   3584),
    ('attn_k',   512,    3584),
    ('attn_v',   512,    3584),
    ('attn_o',   3584,   3584),
    ('ffn_gate', 18944,  3584),
    ('ffn_up',   18944,  3584),
    ('ffn_down', 3584,   18944),
    ('lm_head',  152064, 3584),
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
    return start.elapsed_time(end) / n_iter * 1000  # us


def layercast_gemm(A, B):
    """What LayerCast does per matmul: upcast BF16 -> FP32, compute FP32, cast back."""
    x32 = A.to(torch.float32)
    w32 = B.to(torch.float32)
    out = F.linear(x32, w32)
    return out.to(torch.bfloat16)


print(f"{'shape':<12} {'M':>3} {'K':>6} {'N':>7} | {'BF16':>8} {'Triton':>8} {'LayerCast':>10} | {'Tri vs LC':>10}")
print('-' * 95)

totals = {'bf16': 0, 'triton': 0, 'layercast': 0}

for label, N, K in SHAPES:
    for M in BATCH_SIZES:
        A = torch.randn(M, K, device=device, dtype=torch.bfloat16)
        B = (torch.randn(N, K, device=device, dtype=torch.float32) / (K ** 0.5)).to(torch.bfloat16)

        t_bf16 = bench(F.linear, A, B)
        t_tri = bench(det_gemm, A, B)
        t_lc = bench(layercast_gemm, A, B)

        diff = (t_tri / t_lc - 1) * 100
        flag = '  ✓' if diff < 0 else '  ✗'
        print(f"{label:<12} {M:>3} {K:>6} {N:>7} | {t_bf16:>7.1f}us {t_tri:>7.1f}us {t_lc:>9.1f}us | {diff:>+9.1f}%{flag}")

        totals['bf16'] += t_bf16
        totals['triton'] += t_tri
        totals['layercast'] += t_lc

print()
print(f"Aggregated (sum across shapes × batch sizes):")
print(f"  BF16 baseline:     {totals['bf16']:>8.1f} us")
print(f"  Triton:            {totals['triton']:>8.1f} us  (+{(totals['triton']/totals['bf16']-1)*100:.1f}% vs BF16)")
print(f"  LayerCast:         {totals['layercast']:>8.1f} us  (+{(totals['layercast']/totals['bf16']-1)*100:.1f}% vs BF16)")
print()
print(f"Triton vs LayerCast: {(totals['triton']/totals['layercast']-1)*100:+.1f}%")
