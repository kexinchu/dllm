"""
Option A: GEMM kernel with FP32 accumulation along K, output cast to BF16.
Prefer fused engine kernel (cuBLASLt) when built; else Triton; else PyTorch fallback.
"""
from __future__ import annotations
import os
from typing import Optional
import torch

# Prefer inference-engine fused kernel (cuBLASLt) when extension is built
_CUBLASLT_EXTENSION = None
def _try_load_cublaslt_extension():
    global _CUBLASLT_EXTENSION
    # Ensure torch lib is on LD_LIBRARY_PATH so extension can load libc10.so etc.
    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    if os.path.exists(torch_lib):
        ld = os.environ.get("LD_LIBRARY_PATH", "")
        if torch_lib not in ld.split(os.pathsep):
            os.environ["LD_LIBRARY_PATH"] = torch_lib + (os.pathsep + ld if ld else "")
    try:
        from FP32._gemm_fp32_accum_cuda import gemm_fp32_accum_cuda as _cublaslt_gemm
        _CUBLASLT_EXTENSION = _cublaslt_gemm
    except ImportError:
        try:
            from _gemm_fp32_accum_cuda import gemm_fp32_accum_cuda as _cublaslt_gemm
            _CUBLASLT_EXTENSION = _cublaslt_gemm
        except ImportError:
            pass
_try_load_cublaslt_extension()

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
    """C = A @ B: BF16 in, FP32 accum along K, BF16 out.
    Prefer cuBLASLt extension when built and on CUDA; else Triton; else PyTorch float."""
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        a, b = a.to(torch.bfloat16), b.to(torch.bfloat16)
    if not a.is_cuda:
        return (a.float() @ b.float()).to(torch.bfloat16)
    if _CUBLASLT_EXTENSION is not None:
        try:
            return _CUBLASLT_EXTENSION(a, b)
        except RuntimeError:
            # cuBLASLt heuristic can fail (e.g. CUBLAS_STATUS_INVALID_VALUE) on some configs; fallback
            pass
    if TRITON_AVAILABLE:
        return _matmul_fp32_accum_triton(a, b)
    return (a.float() @ b.float()).to(torch.bfloat16)


def linear_fp32_accum(input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Linear with FP32 accumulation; drop-in for F.linear / nn.Linear. Supports 2D (N, C_in) and 3D (N, L, C_in)."""
    orig_shape = input.shape
    if input.dim() == 3:
        input = input.reshape(-1, input.size(-1))
    out = matmul_fp32_accum(input, weight.T)
    if bias is not None:
        out = out + bias.to(out.dtype)
    if len(orig_shape) == 3:
        out = out.reshape(orig_shape[0], orig_shape[1], -1)
    return out


class LinearFP32Accum(torch.nn.Module):
    """
    Drop-in replacement for nn.Linear: same in/out, but matmul uses FP32 accumulation then BF16.
    Use: model.fc = LinearFP32Accum(in_features, out_features); model.fc.load_state_dict(orig_fc.state_dict())
    """

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
