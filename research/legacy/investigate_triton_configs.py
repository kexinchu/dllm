"""Investigate: which autotune configs were selected for DeepSeek shapes?"""
import os, sys, time
import torch
import torch.nn.functional as F

DLLM = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DLLM)
sys.path.insert(0, os.path.join(DLLM, '..', 'FP32'))

from triton_det_gemm import det_gemm, _det_gemm_kernel, _m_class

SHAPES = [
    ('attn_q',        3584,   3584),
    ('attn_k',        512,    3584),
    ('ffn_gate',      18944,  3584),
    ('ffn_down',      3584,   18944),
    ('lm_head',       152064, 3584),
]

BATCH_SIZES = [1, 8, 16, 32]
device = 'cuda:0'
torch.manual_seed(42)

# Trigger autotune for each shape
print("Triggering autotune for each shape...")
for label, N, K in SHAPES:
    for M in BATCH_SIZES:
        A = torch.randn(M, K, device=device, dtype=torch.bfloat16)
        B = (torch.randn(N, K, device=device, dtype=torch.float32) / (K ** 0.5)).to(torch.bfloat16)
        _ = det_gemm(A, B)

# Now inspect best_config dict
print("\nAutotune cache contents:\n")
print(f"{'M_class':>8} {'N':>7} {'K':>6} | {'BLOCK_M':>8} {'BLOCK_N':>8} {'BLOCK_K':>8} {'warps':>6} {'stages':>6}")
print('-' * 80)

# The best_config is on the wrapper object
cache = _det_gemm_kernel.cache
print(f"cache type: {type(cache)}, len: {len(cache)}")
if cache:
    first_key = next(iter(cache.keys()))
    print(f"first key type: {type(first_key)}, val: {first_key}")
for key, config in list(cache.items())[:20]:
    print(f"key={key}")
    print(f"  config kwargs: {getattr(config, 'kwargs', config)}")
    print(f"  num_warps: {getattr(config, 'num_warps', '?')}, num_stages: {getattr(config, 'num_stages', '?')}")

print(f"\nTotal cached configs: {len(cache)}")
