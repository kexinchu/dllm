"""
RMSNorm with FP32 accumulation: input/output/weight stay BF16,
only the reduction accumulator (mean of x^2) uses FP32.

Pattern: BF16 load -> FP32 accumulator -> cast to BF16 -> store
"""
from __future__ import annotations

import torch

TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    pass


if TRITON_AVAILABLE:
    @triton.jit
    def _rmsnorm_fp32_accum_kernel(
        x_ptr, w_ptr, out_ptr,
        stride_x_row, stride_out_row,
        N, eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        """One program per row -- data-parallel over batch, no cross-row reduction."""
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < N

        # Load BF16
        x = tl.load(x_ptr + row * stride_x_row + offs, mask=mask, other=0.0)
        w = tl.load(w_ptr + offs, mask=mask, other=0.0)

        # FP32 accumulator for variance reduction
        x_fp32 = x.to(tl.float32)
        var = tl.sum(x_fp32 * x_fp32, axis=0) / N  # FP32 accumulator
        rstd = 1.0 / tl.sqrt(var + eps)             # FP32

        # Normalize in FP32, cast back to BF16, then multiply weight (BF16)
        normed = (x_fp32 * rstd).to(tl.bfloat16)
        out = normed * w  # BF16 * BF16 = BF16
        tl.store(out_ptr + row * stride_out_row + offs, out, mask=mask)


def rmsnorm_fp32_accum_triton(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Triton RMSNorm: BF16 in/out, FP32 accumulator for reduction."""
    assert x.is_cuda and weight.is_cuda
    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1])
    M, N = x_2d.shape
    out = torch.empty_like(x_2d)

    BLOCK_SIZE = triton.next_power_of_2(N)

    _rmsnorm_fp32_accum_kernel[(M,)](
        x_2d, weight, out,
        x_2d.stride(0), out.stride(0),
        N, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out.reshape(orig_shape)


def rmsnorm_fp32_accum_pytorch(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """PyTorch fallback: BF16 in/out, FP32 accumulator for variance reduction."""
    x_fp32 = x.float()
    variance = x_fp32.pow(2).mean(-1, keepdim=True)  # FP32 accumulator
    rstd = torch.rsqrt(variance + eps)                # FP32
    # x stays BF16, rstd cast to BF16, weight stays BF16
    return (x * rstd.to(x.dtype)) * weight


def rmsnorm_fp32_accum(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm with FP32 accumulation. Prefer Triton on CUDA, else PyTorch fallback."""
    if TRITON_AVAILABLE and x.is_cuda:
        return rmsnorm_fp32_accum_triton(x, weight, eps)
    return rmsnorm_fp32_accum_pytorch(x, weight, eps)
