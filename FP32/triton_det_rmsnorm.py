"""Deterministic RMSNorm via Triton — fixed reduction plan per row.

Hugging Face's Qwen2RMSNorm computes
    var = hidden_states.pow(2).mean(-1, keepdim=True)
    out = self.weight * (hidden_states * rsqrt(var + eps)).to(dtype)
Mathematically per-row, but PyTorch's ``mean(-1)`` switches between data-
parallel (one row per core) and split-reduction (one row across multiple
cores) based on total tensor size — with batch size 1 it has fewer rows
than SMs, so it splits a single row across cores, changing the reduction
tree relative to bs=4 / 8 / etc. and producing a per-row output that is
not bit-exact across bs.

This kernel forces a **single program per row** with a sequential K-tile
loop of fixed BLOCK_N. Per-row reduction tree is identical regardless of
how many rows we process, so the output is bit-invariant across bs.

References:
  - Thinking Machines, "Defeating Nondeterminism in LLM Inference":
    > "decreasing our batch size will eventually lead to having more cores
    >  than batch elements ... [split-reduction] loses batch invariance."
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(
    X_ptr, W_ptr, OUT_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_om, stride_on,
    eps,
    BLOCK_N: tl.constexpr,    # fixed across all bs/M
):
    """One program per row. Sequential K-tile loop with fixed BLOCK_N.

    HF formula reproduced (FP32 inside, BF16 cast at the end):
        var = mean(x**2)
        out = weight * (x * rsqrt(var + eps))
    """
    pid = tl.program_id(0)
    if pid >= M:
        return

    # Pass 1: sum of squares (FP32 accumulator).
    sum_sq = tl.zeros((), dtype=tl.float32)
    for n_start in range(0, tl.cdiv(N, BLOCK_N)):
        offs = n_start * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(X_ptr + pid * stride_xm + offs * stride_xn,
                    mask=mask, other=0.0).to(tl.float32)
        sum_sq += tl.sum(x * x)

    var = sum_sq / N
    rstd = 1.0 / tl.sqrt(var + eps)

    # Pass 2: apply norm and weight, store BF16.
    for n_start in range(0, tl.cdiv(N, BLOCK_N)):
        offs = n_start * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(X_ptr + pid * stride_xm + offs * stride_xn,
                    mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = (x * rstd * w).to(tl.bfloat16)
        tl.store(OUT_ptr + pid * stride_om + offs * stride_on, y, mask=mask)


def det_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Deterministic RMSNorm. ``x`` shape ``[..., N]`` BF16 CUDA. Returns
    same shape BF16."""
    assert x.dtype == torch.bfloat16, f"x.dtype={x.dtype}"
    assert weight.dtype == torch.bfloat16
    assert x.is_cuda and weight.is_cuda

    orig_shape = x.shape
    N = orig_shape[-1]
    assert weight.shape == (N,), f"weight shape {weight.shape} != ({N},)"

    x2 = x.reshape(-1, N)
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    M = x2.shape[0]

    out = torch.empty_like(x2)

    # Pick BLOCK_N once for the whole shape. The autotune key is just N
    # (per-row width), so the same BLOCK_N is used for any M / batch size.
    BLOCK_N = 1024 if N >= 1024 else 256

    _rmsnorm_kernel[(M,)](
        x2, weight, out,
        M, N,
        x2.stride(0), x2.stride(1),
        out.stride(0), out.stride(1),
        eps,
        BLOCK_N=BLOCK_N,
    )

    return out.reshape(*orig_shape)
