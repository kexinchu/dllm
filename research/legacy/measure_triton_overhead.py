"""Measure Triton launch overhead decomposition:
  1. Minimal Triton kernel (no autotune, no masks)
  2. Our det_gemm with autotune
  3. cuBLASLt (via torch.mm) for reference
"""
import os, sys, time
import torch
import triton
import triton.language as tl

DLLM = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DLLM)
sys.path.insert(0, os.path.join(DLLM, '..', 'FP32'))

# Load our kernel to compare
from triton_det_gemm import det_gemm

# Minimal Triton kernel: just launch and do trivial work
@triton.jit
def _minimal_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(y_ptr + offs, x, mask=mask)


def bench(func, *args, n_iter=100):
    for _ in range(5):
        _ = func(*args)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        _ = func(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_iter * 1000  # us


device = 'cuda:0'
torch.manual_seed(0)

# Test setup: small shape where overhead dominates
M, K, N = 8, 3584, 3584
A = torch.randn(M, K, device=device, dtype=torch.bfloat16)
B_bf16 = (torch.randn(N, K, device=device, dtype=torch.float32) / (K ** 0.5)).to(torch.bfloat16)
B_fp32 = B_bf16.to(torch.float32)
A_fp32 = A.to(torch.float32)

# Output buffers
y_out = torch.empty(M, N, device=device, dtype=torch.bfloat16)

print(f"Shape: M={M}, K={K}, N={N}\n")

# 1. Minimal kernel (just shows Triton's per-call floor)
def minimal_call():
    n = 1024
    x = torch.zeros(n, device=device, dtype=torch.float32)
    y = torch.empty_like(x)
    grid = (triton.cdiv(n, 128),)
    _minimal_kernel[grid](x, y, n, BLOCK=128)
    return y

t_min = bench(minimal_call)
print(f"Minimal Triton kernel (copy 1024 floats):  {t_min:>6.1f} us")

# 2. Our det_gemm
t_det = bench(det_gemm, A, B_bf16)
print(f"Our det_gemm (K=N=3584, M=8):              {t_det:>6.1f} us")

# 3. Baseline torch F.linear
import torch.nn.functional as F
t_f = bench(F.linear, A, B_bf16)
print(f"torch F.linear (K=N=3584, M=8):            {t_f:>6.1f} us")

# 4. torch.mm in FP32 (reference for 'FP32 compute' cost without Triton)
def fp32_mm():
    return torch.mm(A_fp32, B_fp32.t()).to(torch.bfloat16)
t_fp32 = bench(fp32_mm)
print(f"torch.mm FP32 (K=N=3584, M=8):             {t_fp32:>6.1f} us")

# 5. Raw BF16 mm
def bf16_mm():
    return torch.mm(A, B_bf16.t())
t_bf16 = bench(bf16_mm)
print(f"torch.mm BF16 (K=N=3584, M=8):             {t_bf16:>6.1f} us")

print()
print(f"Our Triton overhead over baseline:         {t_det - t_f:>6.1f} us")
print(f"Minimal kernel floor:                      {t_min:>6.1f} us")
print(f"=> dispatch + launch cost ~ {t_min:.1f}us, so ~{t_det - t_f - t_min:.1f}us is kernel-specific")
