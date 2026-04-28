"""
Verify GEMV/GEMM padding fix.
Checks that M=1 padded to M=2 gives same row-0 result as M=N.
Runs on a representative LLM linear layer shape.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'FP32'))
sys.path.insert(0, os.path.dirname(__file__))

import torch
import determ_llm

# ── Setup kernel ──────────────────────────────────────────────────────────────
kern = determ_llm._load_kernel()
device = torch.device('cuda:0')

def gemm_m(A_row, W, M):
    """Compute GEMM with M copies of A_row, return row 0 output."""
    x = A_row.unsqueeze(0).expand(M, -1).contiguous()
    return kern.gemm_fixed_algo(x, W)[0]

def gemm_m_padded(A_row, W):
    """Compute GEMM with M=1 padded to M=2, return row 0 output."""
    x = A_row.unsqueeze(0).expand(2, -1).contiguous()
    return kern.gemm_fixed_algo(x, W)[0]

def gemm_m1_raw(A_row, W):
    """Compute GEMM with M=1 (no padding), return row 0 output."""
    x = A_row.unsqueeze(0).contiguous()
    return kern.gemm_fixed_algo(x, W)[0]

# ── Test shapes: all actual Llama-3.2-1B and Llama-3.1-8B linear layers ───────
# Llama-3.2-1B: hidden=2048, intermediate=8192, num_heads=32, num_kv_heads=8,
#               head_dim=64, num_layers=16
# Llama-3.1-8B: hidden=4096, intermediate=14336, num_heads=32, num_kv_heads=8,
#               head_dim=128
shapes = [
    # (label, K=in_features, N=out_features)
    # --- 1B attention ---
    ('1B q_proj  K=2048 N=2048', 2048, 2048),
    ('1B k_proj  K=2048 N= 512', 2048,  512),
    ('1B v_proj  K=2048 N= 512', 2048,  512),
    ('1B o_proj  K=2048 N=2048', 2048, 2048),
    # --- 1B FFN ---
    ('1B gate    K=2048 N=8192', 2048, 8192),
    ('1B up      K=2048 N=8192', 2048, 8192),
    ('1B down    K=8192 N=2048', 8192, 2048),
    # --- 8B attention ---
    ('8B q_proj  K=4096 N=4096', 4096, 4096),
    ('8B k_proj  K=4096 N=1024', 4096, 1024),
    ('8B o_proj  K=4096 N=4096', 4096, 4096),
    # --- 8B FFN ---
    ('8B gate    K=4096 N=14336', 4096, 14336),
    ('8B down    K=14336 N=4096', 14336, 4096),
]

batch_sizes = [2, 4, 8, 32, 64]

print(f"{'Shape':<32} {'M1_raw→M2_pad':<16} {'M2_pad→M=64_max':<18} Status")
print('-' * 80)

all_pass = True
for label, K, N in shapes:
    torch.manual_seed(42)
    A_row = torch.randn(K, dtype=torch.bfloat16, device=device)
    W = torch.randn(N, K, dtype=torch.bfloat16, device=device) * (K ** -0.5)

    out_m1_raw = gemm_m1_raw(A_row, W)
    out_m2_pad = gemm_m_padded(A_row, W)

    diff_raw_vs_pad = (out_m1_raw.float() - out_m2_pad.float()).abs().max().item()

    max_diff_pad_vs_N = 0.0
    for bs in batch_sizes:
        out_mN = gemm_m(A_row, W, bs)
        d = (out_m2_pad.float() - out_mN.float()).abs().max().item()
        max_diff_pad_vs_N = max(max_diff_pad_vs_N, d)

    fixed = (max_diff_pad_vs_N == 0.0)
    if not fixed:
        all_pass = False
    status = '✓' if fixed else '✗'

    print(f"  {label:<30} {diff_raw_vs_pad:.3e}{'':>5} {max_diff_pad_vs_N:.3e}{'':>5} {status}")

print()
print(f"Overall: {'ALL PASS ✓' if all_pass else 'SOME SHAPES FAIL ✗'}")
print()
print("M1_raw→M2_pad : gap closed by padding (GEMV/GEMM boundary).")
print("M2_pad→M=64   : residual cross-M variation after padding fix.")
