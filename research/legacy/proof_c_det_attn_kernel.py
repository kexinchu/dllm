"""PROOF C — does our new Triton attention kernel produce per-row
bit-exact output across batch sizes?

Tests both attention matmuls with decode shapes:
  * Q @ K^T:  [B,H,1,D] @ [B,H,D,S]  -> [B,H,1,S]   (K^T is transpose view)
  * attn @ V: [B,H,1,S] @ [B,H,S,D]  -> [B,H,1,D]

And prefill shape variants.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'FP32'))

import torch
from triton_det_attn import det_attn_matmul


def test_shape(label, H, M, K_dim, N, transpose_B=False, bs_list=(1, 4, 8, 16, 32)):
    """Generate a random A [1,H,M,K_dim] and B [1,H,K_dim,N] (or transposed
    view), then repeat across bs and verify per-row bit-exact output."""
    torch.manual_seed(0)
    A_1 = torch.randn(1, H, M, K_dim, dtype=torch.bfloat16, device='cuda') * 0.5
    if transpose_B:
        # B is generated as [1,H,N,K_dim] contiguous then transposed; this
        # mirrors how HF does K.transpose(-2, -1) on a contiguous K.
        B_src = torch.randn(1, H, N, K_dim, dtype=torch.bfloat16, device='cuda') * 0.5
        B_1 = B_src.transpose(-2, -1)   # [1, H, K_dim, N], non-contiguous
    else:
        B_1 = torch.randn(1, H, K_dim, N, dtype=torch.bfloat16, device='cuda') * 0.5

    ref = det_attn_matmul(A_1, B_1)          # [1, H, M, N]
    ref_row0 = ref[0]                         # [H, M, N]

    statuses = []
    for bs in bs_list:
        A_bs = A_1.expand(bs, -1, -1, -1).contiguous()
        if transpose_B:
            # Re-materialise transpose on a repeated contig tensor so
            # strides match "real" usage.
            B_src_bs = B_src.expand(bs, -1, -1, -1).contiguous()
            B_bs = B_src_bs.transpose(-2, -1)
        else:
            B_bs = B_1.expand(bs, -1, -1, -1).contiguous()

        out = det_attn_matmul(A_bs, B_bs)    # [bs, H, M, N]
        row0 = out[0]                         # compare same-input row
        same = torch.equal(row0, ref_row0)
        maxd = float((row0 - ref_row0).abs().max())
        statuses.append((bs, same, maxd))

    print(f"  [{label:30s}]  M={M} K={K_dim} N={N} H={H}")
    for bs, same, maxd in statuses:
        mark = "OK" if same else "DIFF"
        print(f"    bs={bs:>3}  {mark}  max|diff|={maxd:.3e}")


def main():
    print("Hardware:", torch.cuda.get_device_name())

    # Decode: Q@K^T (K_dim=D, S varies).
    print("\n--- decode Q@K^T: [B,H,1,D] @ [B,H,D,S] (transpose view of K) ---")
    for S in [32, 64, 128, 256, 512]:
        test_shape(f"Q@K^T S={S}", H=28, M=1, K_dim=128, N=S, transpose_B=True)

    # Decode: attn@V (K_dim=S, N=D=128).
    print("\n--- decode attn@V: [B,H,1,S] @ [B,H,S,D] ---")
    for S in [32, 64, 128, 256, 512]:
        test_shape(f"attn@V S={S}", H=28, M=1, K_dim=S, N=128, transpose_B=False)

    # Prefill Q@K^T (M and N both > 1).
    print("\n--- prefill Q@K^T: [B,H,M,D] @ [B,H,D,M] ---")
    for M in [16, 64, 256]:
        test_shape(f"prefill Q@K^T M={M}", H=28, M=M, K_dim=128, N=M, transpose_B=True)


if __name__ == "__main__":
    main()
