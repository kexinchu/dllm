"""Does FP32 cuBLAS torch.matmul produce bit-identical output per batch row
across batch sizes? Test the attention Q@K^T shape at various sequence
lengths (prompt + generated tokens).

If this test shows divergence: FP32 upcast alone does not fix attention
non-determinism — cuBLAS FP32 also picks batch-dependent split-K.
"""
import torch

torch.manual_seed(0)


def test(label, B, H, S, D, bs_list=(1, 8, 32)):
    """Test Q @ K^T where Q,K shape [B*bs_per, H, S, D], but only the first
    row should be compared."""
    print(f"\n== {label}  (H={H}, S={S}, D={D})  ==")
    # Reference
    Q1 = torch.randn(1, H, S, D, dtype=torch.bfloat16, device="cuda") * 0.5
    K1 = torch.randn(1, H, S, D, dtype=torch.bfloat16, device="cuda") * 0.5
    Q1_fp = Q1.to(torch.float32)
    K1_fp = K1.to(torch.float32)

    for bs in bs_list:
        # bs=1 is the reference
        Qbs = Q1.repeat(bs, 1, 1, 1).contiguous()
        Kbs = K1.repeat(bs, 1, 1, 1).contiguous()
        Qbs_fp = Qbs.to(torch.float32)
        Kbs_fp = Kbs.to(torch.float32)

        # BF16 direct matmul
        out_bf = torch.matmul(Qbs, Kbs.transpose(-2, -1))
        # FP32 upcast matmul (what our patched LayerCast / DetermLLM does)
        out_fp = torch.matmul(Qbs_fp, Kbs_fp.transpose(-2, -1)).to(torch.bfloat16)

        ref_bf = torch.matmul(Q1,    K1.transpose(-2, -1))
        ref_fp = torch.matmul(Q1_fp, K1_fp.transpose(-2, -1)).to(torch.bfloat16)

        same_bf = torch.equal(out_bf[0], ref_bf[0])
        same_fp = torch.equal(out_fp[0], ref_fp[0])
        max_bf = float((out_bf[0] - ref_bf[0]).abs().max())
        max_fp = float((out_fp[0] - ref_fp[0]).abs().max())
        print(f"  bs={bs:>2}  BF16: same={same_bf}  max|diff|={max_bf:.3e}   "
              f"FP32-upcast: same={same_fp}  max|diff|={max_fp:.3e}")


def main():
    # DeepSeek-7B attention parameters: 28 heads, head_dim=128, GQA with 4 kv heads
    # Q@K^T decode: Q=[B, 28, 1, 128], K=[B, 28, S, 128] (after GQA expansion)
    print("=== Decode attention: Q@K^T with Q=[B,28,1,128] K=[B,28,S,128] ===")
    # Can't easily test decode shape where Q has S=1 but K has S=variable;
    # simplest: repeat Q to match
    for S in (32, 128, 512, 1024):
        test(f"decode-step at context S={S}", B=1, H=28, S=S, D=128)

    # Prefill: Q,K are both [B, 28, S, 128]
    print("\n=== Prefill attention: Q@K^T with Q,K=[B,28,S,128] ===")
    for S in (32, 128, 512):
        test(f"prefill S={S}", B=1, H=28, S=S, D=128)


if __name__ == "__main__":
    main()
