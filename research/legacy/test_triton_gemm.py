"""
Correctness + determinism test for the Triton deterministic GEMM.

Checks:
  1. Output matches torch.nn.functional.linear within BF16 tolerance.
  2. Bit-exact batch invariance: row 0 at M=2 equals row 0 at M=32 across
     all tested LLM shapes.
"""
import os, sys, time
import torch
import torch.nn.functional as F

DLLM_DIR = os.path.dirname(os.path.abspath(__file__))
FP32_DIR = os.path.join(DLLM_DIR, '..', 'FP32')
sys.path.insert(0, FP32_DIR)

from triton_det_gemm import det_gemm

device = 'cuda:0'
torch.manual_seed(0)

# Cover the full LLM shape space: attention projs, FFN up/gate/down.
# (M, N, K) tuples typical of 1B, 8B, 14B.
SHAPES = [
    # (label, N, K)
    ('1B attn proj',   2048, 2048),
    ('1B FFN gate/up', 8192, 2048),
    ('1B FFN down',    2048, 8192),
    ('8B attn proj',   4096, 4096),
    ('8B FFN gate/up', 14336, 4096),
    ('8B FFN down',    4096, 14336),
    ('Phi-4 attn',     5120, 5120),
    ('Phi-4 FFN up',   17920, 5120),
]

BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]

print('=' * 80)
print('CORRECTNESS TEST: Triton vs torch.nn.functional.linear')
print('=' * 80)
print(f"{'shape':<22} {'M':>4} {'max|diff|':>12} {'match?':>8}")
print('-' * 60)

correctness_pass = True
for label, N, K in SHAPES:
    for M in [1, 8, 32]:
        torch.manual_seed(42)
        A = torch.randn(M, K, device=device, dtype=torch.bfloat16)
        B = (torch.randn(N, K, device=device, dtype=torch.float32) / (K ** 0.5)).to(torch.bfloat16)

        y_triton = det_gemm(A, B)
        y_ref = F.linear(A, B)

        diff = (y_triton.float() - y_ref.float()).abs().max().item()
        # BF16 quantization step ~ 7.8e-3. Allow one ULP tolerance.
        ok = diff <= 1.0e-2
        if not ok:
            correctness_pass = False
        print(f"{label:<22} {M:>4} {diff:>12.4e} {'OK' if ok else 'FAIL':>8}")

print()
print('=' * 80)
print('DETERMINISM TEST: row 0 at M=2 vs M=32 should be bit-exact')
print('=' * 80)
print(f"{'shape':<22} {'linf diff':>12} {'zero rate':>12} {'bit-exact?':>12}")
print('-' * 60)

det_pass = True
for label, N, K in SHAPES:
    torch.manual_seed(42)
    A_full = torch.randn(32, K, device=device, dtype=torch.bfloat16)
    B = (torch.randn(N, K, device=device, dtype=torch.float32) / (K ** 0.5)).to(torch.bfloat16)

    y_m2 = det_gemm(A_full[:2].contiguous(), B)
    y_m32 = det_gemm(A_full[:32].contiguous(), B)

    d = y_m2[0].float() - y_m32[0].float()
    linf = d.abs().max().item()
    zero_rate = (d == 0).float().mean().item()
    ok = (linf == 0.0)
    if not ok:
        det_pass = False
    print(f"{label:<22} {linf:>12.4e} {zero_rate*100:>11.2f}% {'YES' if ok else 'NO':>12}")

print()
print('=' * 80)
print('DETERMINISM TEST ACROSS BATCH SIZES: row 0 at every bs identical?')
print('=' * 80)

all_bs_pass = True
for label, N, K in SHAPES[:3]:  # sample 3 shapes to keep runtime tight
    torch.manual_seed(42)
    A_big = torch.randn(max(BATCH_SIZES), K, device=device, dtype=torch.bfloat16)
    B = (torch.randn(N, K, device=device, dtype=torch.float32) / (K ** 0.5)).to(torch.bfloat16)

    refs = []
    for bs in BATCH_SIZES:
        y = det_gemm(A_big[:bs].contiguous(), B)
        refs.append(y[0].clone())

    all_equal = all((refs[i] == refs[0]).all().item() for i in range(1, len(refs)))
    if not all_equal:
        all_bs_pass = False
        # find where they differ
        for i in range(1, len(refs)):
            if not (refs[i] == refs[0]).all():
                linf = (refs[i].float() - refs[0].float()).abs().max().item()
                print(f"  {label}: bs={BATCH_SIZES[i]} differs from bs={BATCH_SIZES[0]}, linf={linf:.4e}")

    print(f"{label:<22} all {len(BATCH_SIZES)} batch sizes identical: {'YES' if all_equal else 'NO'}")

print()
print('=' * 80)
print(f"CORRECTNESS:  {'PASS' if correctness_pass else 'FAIL'}")
print(f"DETERMINISM:  {'PASS' if det_pass else 'FAIL'}")
print(f"ALL-BS:       {'PASS' if all_bs_pass else 'FAIL'}")
print('=' * 80)
