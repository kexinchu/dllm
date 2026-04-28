"""PROOF A — does FP32-reduction alone make BF16 cuBLAS matmul bs-invariant?

The user's specification is: matmul stays BF16 in/out, reduction uses FP32.
PyTorch exposes ``torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction``
which controls whether BF16 matmul is allowed to use BF16 reduction (True) or
is forced to FP32 reduction (False). We test bs-invariance at both settings.

Hypothesis: setting the flag to False is necessary but NOT sufficient, because
cuBLAS still picks batch-dependent split-K strategies even under FP32 reduction.
"""
import torch

DEVICE = 'cuda'

ATTN_SHAPES = [
    # ("label", H, S, D, bs_ref, bs_probe)
    ("decode S=32",   28,  32, 128, 1, 8),
    ("decode S=64",   28,  64, 128, 1, 8),
    ("decode S=128",  28, 128, 128, 1, 8),
    ("decode S=256",  28, 256, 128, 1, 8),
    ("decode S=512",  28, 512, 128, 1, 8),
    ("decode S=1024", 28,1024, 128, 1, 8),
    ("decode S=128 bs=32", 28, 128, 128, 1, 32),
    ("decode S=256 bs=32", 28, 256, 128, 1, 32),
]

LINEAR_SHAPES = [
    # (label, K, N)
    ("Q_proj",  3584, 3584),
    ("O_proj",  3584, 3584),
    ("Gate/Up", 3584, 18944),
    ("Down",   18944, 3584),
    ("lm_head", 3584, 151936),
]


def test_flag(flag_value):
    print(f"\n{'='*76}")
    print(f"allow_bf16_reduced_precision_reduction = {flag_value}")
    print('='*76)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = flag_value

    print("\n--- attention Q@K^T: BF16 in -> BF16 out ---")
    for label, H, S, D, bs_a, bs_b in ATTN_SHAPES:
        torch.manual_seed(0)
        Q = torch.randn(1, H, 1, D, dtype=torch.bfloat16, device=DEVICE) * 0.5
        K = torch.randn(1, H, S, D, dtype=torch.bfloat16, device=DEVICE) * 0.5

        Qa = Q.repeat(bs_a, 1, 1, 1).contiguous()
        Qb = Q.repeat(bs_b, 1, 1, 1).contiguous()
        Ka = K.repeat(bs_a, 1, 1, 1).contiguous()
        Kb = K.repeat(bs_b, 1, 1, 1).contiguous()

        out_a = Qa @ Ka.transpose(-2, -1)
        out_b = Qb @ Kb.transpose(-2, -1)

        eq = torch.equal(out_a[0], out_b[0])
        maxd = float((out_a[0] - out_b[0]).abs().max())
        print(f"  {label:24s}  bs={bs_a}v{bs_b}  bit-exact? {eq}  max|diff|={maxd:.3e}")

    print("\n--- F.linear: [M,K]·[K,N] = [M,N] BF16 in -> BF16 out ---")
    import torch.nn.functional as F
    for label, K, N in LINEAR_SHAPES:
        torch.manual_seed(1)
        W = torch.randn(N, K, dtype=torch.bfloat16, device=DEVICE) * (1.0/K**0.5)
        x = torch.randn(1, K, dtype=torch.bfloat16, device=DEVICE) * 0.5
        x_bs1 = x
        x_bs8 = x.repeat(8, 1).contiguous()
        y1 = F.linear(x_bs1, W)
        y8 = F.linear(x_bs8, W)
        eq = torch.equal(y1[0], y8[0])
        maxd = float((y1[0] - y8[0]).abs().max())
        print(f"  {label:10s} K={K:<5} N={N:<6}  bit-exact? {eq}  max|diff|={maxd:.3e}")


def main():
    print("Hardware:", torch.cuda.get_device_name())
    print("Test: per-row bit-equality across batch sizes (same input repeated)")
    test_flag(True)   # current setting — explicitly allows BF16 reduction
    test_flag(False)  # user-requested — FP32 reduction accumulator


if __name__ == "__main__":
    main()
