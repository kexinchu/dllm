"""Profile Triton vs baseline F.linear on DeepSeek-R1-Distill-Qwen-7B shapes.

Finds which specific shapes are slow, so we know where to focus the fix.
"""
import os, sys, time, statistics
import torch
import torch.nn.functional as F

DLLM = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DLLM)
sys.path.insert(0, os.path.join(DLLM, '..', 'FP32'))

from triton_det_gemm import det_gemm

# DeepSeek-R1-Distill-Qwen-7B shapes (one GEMM per type per layer):
# hidden=3584, intermediate=18944, num_heads=28, num_kv_heads=4, head_dim=128
SHAPES = [
    # label,          N,      K          count_per_layer  role
    ('attn_q',        3584,   3584),   # 1 per layer (out = hidden)
    ('attn_k',        512,    3584),   # 1 per layer (out = num_kv_heads * head_dim = 512)
    ('attn_v',        512,    3584),   # 1 per layer
    ('attn_o',        3584,   3584),   # 1 per layer
    ('ffn_gate',      18944,  3584),   # 1 per layer
    ('ffn_up',        18944,  3584),   # 1 per layer
    ('ffn_down',      3584,   18944),  # 1 per layer
    ('lm_head',       152064, 3584),   # once per decode step
]

BATCH_SIZES = [1, 8, 16, 32]
device = 'cuda:0'
torch.manual_seed(42)

N_WARMUP = 5
N_ITER = 100

def bench(func, A, B):
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
    return start.elapsed_time(end) / N_ITER * 1000  # us

print(f"Benchmark DeepSeek shapes, warmup={N_WARMUP} iter={N_ITER}\n")
print(f"{'shape':<12} {'M':>3}  {'K':>6} {'N':>7} | {'baseline':>10} {'Triton':>10}  {'Triton ovhd':>13}")
print('-' * 80)

problems = {}
for label, N, K in SHAPES:
    for M in BATCH_SIZES:
        A = torch.randn(M, K, device=device, dtype=torch.bfloat16)
        B = (torch.randn(N, K, device=device, dtype=torch.float32) / (K ** 0.5)).to(torch.bfloat16)

        t_base = bench(lambda a, b: F.linear(a, b), A, B)
        t_tri = bench(det_gemm, A, B)
        ovhd = (t_tri / t_base - 1) * 100

        flag = ' !!' if ovhd > 20 else ''
        print(f"{label:<12} {M:>3}  {K:>6} {N:>7} | {t_base:>9.1f}us {t_tri:>9.1f}us  {ovhd:>+11.1f}%{flag}")

        if ovhd > 20:
            problems.setdefault(label, []).append((M, N, K, ovhd))

print()
print("=== Problem shapes (>20% overhead) ===")
for label, cases in problems.items():
    print(f"{label}:")
    for M, N, K, ovhd in cases:
        print(f"  M={M} K={K} N={N} -> +{ovhd:.1f}%")
