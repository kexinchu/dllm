"""Diagnose whether DetermLLM's _det_linear (Triton backend) is truly
batch-invariant per row on the shapes that occur in DeepSeek-7B inference.

Test: synthesize x of shape (1, K) and (8, K); run F.linear with our patched
kernel; verify row[0] of bs=8 equals row[0] of bs=1 at bit level.
"""
import os
import sys

import torch
import torch.nn.functional as F

DLLM = os.path.dirname(os.path.abspath(__file__))
FP32 = os.path.join(DLLM, "..", "FP32")
sys.path.insert(0, DLLM); sys.path.insert(0, FP32)
import determ_llm

# Representative DeepSeek-7B linear shapes. (K, N):
#  - Q proj: 3584 x 3584
#  - K/V proj (GQA, kv_heads=4): 3584 x 512
#  - O proj:  3584 x 3584
#  - Gate/Up: 3584 x 18944  (MLP)
#  - Down:    18944 x 3584
SHAPES = [
    ("Q_proj",  3584, 3584),
    ("KV_proj", 3584,  512),   # GQA small-N, routes to cuBLASLt under hybrid
    ("O_proj",  3584, 3584),
    ("Gate/Up", 3584, 18944),  # large-N, routes to Triton under hybrid
    ("Down",   18944, 3584),
    ("lm_head", 3584, 151936),
]


def test(label, K, N, backend="hybrid"):
    torch.manual_seed(42)
    x = torch.randn(1, K, dtype=torch.bfloat16, device="cuda") * 0.5
    W = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") * (1.0 / K ** 0.5)
    determ_llm.disable()
    determ_llm.enable(backend=backend, attn=False)

    x1 = x
    x8 = x.repeat(8, 1).contiguous()

    y1 = F.linear(x1, W)           # (1, N)
    y8 = F.linear(x8, W)           # (8, N)

    determ_llm.disable()

    # Compare: y1[0] vs y8[0], y8[1], ..., y8[7]
    y1_row = y1[0]
    eq = [torch.equal(y1_row, y8[i]) for i in range(8)]
    max_abs_diff = max(float((y1_row - y8[i]).abs().max()) for i in range(8))
    all_eq = all(eq)
    print(f"  [{label:10s}] (K={K}, N={N})  all_eq={all_eq}  max|diff|={max_abs_diff:.2e}  eq_mask={eq}")


def main():
    print("=== hybrid backend ===")
    for name, K, N in SHAPES:
        test(name, K, N, backend="hybrid")
    print("\n=== triton backend ===")
    for name, K, N in SHAPES:
        test(name, K, N, backend="triton")
    print("\n=== cublaslt backend ===")
    for name, K, N in SHAPES:
        test(name, K, N, backend="cublaslt")


if __name__ == "__main__":
    main()
