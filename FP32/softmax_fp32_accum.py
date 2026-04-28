"""
Softmax with FP32 accumulation: input/output stay BF16,
only the reduction accumulators (max and sum of exp) use FP32.

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
    def _softmax_fp32_accum_kernel(
        input_ptr, output_ptr,
        stride_input_row, stride_output_row,
        n_cols,
        BLOCK_SIZE: tl.constexpr,
    ):
        """One program per row -- data-parallel over batch, FP32 accumulator for reductions."""
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < n_cols

        # Load BF16, use -inf for masked positions (correct for max/softmax)
        x = tl.load(input_ptr + row * stride_input_row + offs, mask=mask, other=float('-inf'))
        x_fp32 = x.to(tl.float32)

        # FP32 reductions
        row_max = tl.max(x_fp32, axis=0)              # FP32
        numerator = tl.exp(x_fp32 - row_max)           # FP32
        denominator = tl.sum(numerator, axis=0)        # FP32 accumulator

        out = (numerator / denominator).to(tl.bfloat16)  # cast to BF16
        tl.store(output_ptr + row * stride_output_row + offs, out, mask=mask)


def softmax_fp32_accum_triton(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Triton Softmax: BF16 in/out, FP32 accumulator for max and sum reductions."""
    assert x.is_cuda
    if dim < 0:
        dim = x.dim() + dim
    assert dim == x.dim() - 1, "Triton kernel only supports softmax over last dim"

    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1])
    M, N = x_2d.shape
    out = torch.empty_like(x_2d)

    BLOCK_SIZE = triton.next_power_of_2(N)

    _softmax_fp32_accum_kernel[(M,)](
        x_2d, out,
        x_2d.stride(0), out.stride(0),
        N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out.reshape(orig_shape)


def softmax_fp32_accum_pytorch(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """PyTorch fallback: BF16 in/out, FP32 accumulator for softmax reductions."""
    return torch.softmax(x.float(), dim=dim).to(x.dtype)


def softmax_fp32_accum(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Softmax with FP32 accumulation. Prefer Triton on CUDA, else PyTorch fallback."""
    if TRITON_AVAILABLE and x.is_cuda and (dim == -1 or dim == x.dim() - 1):
        return softmax_fp32_accum_triton(x, dim)
    return softmax_fp32_accum_pytorch(x, dim)


def log_softmax_fp32_accum(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Log-softmax with FP32 accumulation. BF16 in/out."""
    return torch.log_softmax(x.float(), dim=dim).to(x.dtype)
