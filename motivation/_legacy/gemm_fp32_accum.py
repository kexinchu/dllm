"""
Option A: GEMM kernel with FP32 accumulation along K, output cast to BF16.
Triton kernel when available; PyTorch fallback otherwise.
Move to FP32/ or use from FP32 via import.
"""
from __future__ import annotations
from typing import Optional
import torch

TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    pass


def _matmul_fp32_accum_triton(a: torch.Tensor, b: torch.Tensor, BLOCK_SIZE_M: int = 64, BLOCK_SIZE_N: int = 64, BLOCK_SIZE_K: int = 32, GROUP_SIZE_M: int = 8) -> torch.Tensor:
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)

    @triton.jit
    def _kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
        pid_n = (pid % num_pid_in_group) // group_size_m
        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            mask_a = offs_k[None, :] < K - k * BLOCK_SIZE_K
            mask_b = offs_k[:, None] < K - k * BLOCK_SIZE_K
            a_block = tl.load(a_ptrs, mask=mask_a, other=0.0)
            b_block = tl.load(b_ptrs, mask=mask_b, other=0.0)
            accumulator = tl.dot(a_block, b_block, acc=accumulator, out_dtype=tl.float32)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk
        c_block = accumulator.to(tl.bfloat16)
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c_block, mask=c_mask)

    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)
    _kernel[grid](a, b, c, M=M, N=N, K=K, stride_am=a.stride(0), stride_ak=a.stride(1), stride_bk=b.stride(0), stride_bn=b.stride(1), stride_cm=c.stride(0), stride_cn=c.stride(1), BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K, GROUP_SIZE_M=GROUP_SIZE_M)
    return c


def matmul_fp32_accum(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """C = A @ B: BF16 in, FP32 accum along K, BF16 out."""
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        a, b = a.to(torch.bfloat16), b.to(torch.bfloat16)
    if not a.is_cuda or not TRITON_AVAILABLE:
        return (a.float() @ b.float()).to(torch.bfloat16)
    return _matmul_fp32_accum_triton(a, b)


def linear_fp32_accum(input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Linear with FP32 accumulation; drop-in for F.linear / nn.Linear."""
    out = matmul_fp32_accum(input, weight.T)
    if bias is not None:
        out = out + bias.to(out.dtype)
    return out


class LinearFP32Accum(torch.nn.Module):
    """Drop-in replacement for nn.Linear: same in/out, matmul uses FP32 accumulation then BF16."""

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int

    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return linear_fp32_accum(input, self.weight, self.bias)
