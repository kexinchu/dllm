"""Deterministic BF16 4D attention matmul via Triton.

Fixed-plan kernel for the two attention GEMMs in HuggingFace eager attention:
  * Q @ K^T: [B, H, M, D] @ [B, H, D, S] -> [B, H, M, S]
  * attn @ V: [B, H, M, S] @ [B, H, S, D] -> [B, H, M, D]

Design principle (same as ``triton_det_gemm.det_gemm``):
  * BF16 inputs, BF16 output (matmul precision unchanged).
  * FP32 accumulator inside the kernel (HMMA.16816.F32 on Ampere).
  * **No split-K across programs.** Each output tile is computed by exactly
    one program, so the K-reduction is a single sequential loop inside that
    program. Grid size scales with batch/heads but the per-element reduction
    path is byte-identical across batch sizes — output is bit-exact per row.
  * Arbitrary strides on inputs (handles ``K.transpose(-2,-1)`` without a
    .contiguous() copy).
  * Autotune key is (M_class, N, K) so re-entry at a different B*H reuses the
    same cached tile plan.
"""
import torch
import triton
import triton.language as tl


def _autotune_configs():
    """Autotune configs — **BLOCK_K is pinned to 128** (single value) across
    all (M_class, N, K) keys to guarantee a single, identical K-tile
    sequence for every bs. Varying BLOCK_K across M_class would split the
    FP32 accumulator into different partial-sum trees and re-introduce
    bs-dependent rounding even within the fixed-plan design.

    BLOCK_M and BLOCK_N remain tunable — they do not affect reduction order.
    """
    configs = []
    BLOCK_K = 128
    for bm in [16, 32, 64, 128]:
        for bn in [16, 32, 64, 128]:
            for nw in [4, 8]:
                for ns in [2, 3, 4]:
                    configs.append(triton.Config(
                        {'BLOCK_M': bm, 'BLOCK_N': bn, 'BLOCK_K': BLOCK_K},
                        num_warps=nw, num_stages=ns,
                    ))
    return configs


@triton.autotune(
    configs=_autotune_configs(),
    key=['M_class', 'N', 'K'],
)
@triton.jit
def _det_attn_kernel(
    A_ptr, B_ptr, C_ptr,
    BH, M, N, K, M_class,
    stride_a_bh, stride_a_m, stride_a_k,
    stride_b_bh, stride_b_k, stride_b_n,
    stride_c_bh, stride_c_m, stride_c_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """C[bh, m, n] = sum_k A[bh, m, k] * B[bh, k, n]  in FP32 accumulator.

    Grid: (BH, ceil(M/BLOCK_M), ceil(N/BLOCK_N)).
    Each program owns a single (bh, m_tile, n_tile) output tile and does the
    entire K reduction sequentially — no split-K.
    """
    pid_bh = tl.program_id(0)
    pid_m  = tl.program_id(1)
    pid_n  = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    A_batch = A_ptr + pid_bh * stride_a_bh
    B_batch = B_ptr + pid_bh * stride_b_bh
    C_batch = C_ptr + pid_bh * stride_c_bh

    a_ptrs = A_batch + offs_m[:, None] * stride_a_m + offs_k[None, :] * stride_a_k
    b_ptrs = B_batch + offs_k[:, None] * stride_b_k + offs_n[None, :] * stride_b_n

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Sequential K reduction in a single program (bit-exact by construction).
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offset = k * BLOCK_K
        k_base = offs_k + k_offset
        k_mask = k_base < K

        a = tl.load(a_ptrs,
                    mask=(offs_m[:, None] < M) & k_mask[None, :],
                    other=0.0)
        b = tl.load(b_ptrs,
                    mask=k_mask[:, None] & (offs_n[None, :] < N),
                    other=0.0)

        acc += tl.dot(a, b)

        a_ptrs += BLOCK_K * stride_a_k
        b_ptrs += BLOCK_K * stride_b_k

    c_ptrs = C_batch + offs_m[:, None] * stride_c_m + offs_n[None, :] * stride_c_n
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=mask)


def _m_class(M: int) -> int:
    if M == 1:   return 1
    if M <= 4:   return 4
    if M <= 16:  return 16
    if M <= 64:  return 64
    if M <= 256: return 256
    return 1024


def det_attn_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Batched 4D matmul C = A @ B with fixed-plan K reduction.

    Accepts 4D ``[..., M, K]`` and ``[..., K, N]`` with arbitrary strides on
    the last two dims. Leading batch dimensions must match and must have the
    outer stride equal to H * stride_1 (true for contiguous 4D and for the
    ``.transpose(-2, -1)`` view used by HuggingFace attention).

    Returns a 4D BF16 tensor of shape ``[..., M, N]``.
    """
    assert A.dtype == torch.bfloat16 and B.dtype == torch.bfloat16
    assert A.is_cuda and B.is_cuda
    assert A.dim() == 4 and B.dim() == 4, f"expected 4D, got {A.shape}/{B.shape}"
    assert A.shape[:-2] == B.shape[:-2], "batch dims must match"

    B1, H, M, K = A.shape
    _, _, Kb, N = B.shape
    assert K == Kb, f"contraction mismatch: {K} vs {Kb}"

    # Outer stride trick: for a 4D tensor whose outer axis 0 is contiguous
    # over axis 1 (stride_0 == stride_1 * size_1), the flat bh index
    # (b * H + h) can be strided by stride(1).
    assert A.stride(0) == A.stride(1) * H, f"A outer stride mismatch {A.stride()}"
    assert B.stride(0) == B.stride(1) * H, f"B outer stride mismatch {B.stride()}"

    BH = B1 * H
    C = torch.empty((B1, H, M, N), dtype=torch.bfloat16, device=A.device)

    grid = lambda META: (
        BH,
        triton.cdiv(M, META['BLOCK_M']),
        triton.cdiv(N, META['BLOCK_N']),
    )
    _det_attn_kernel[grid](
        A, B, C,
        BH, M, N, K, _m_class(M),
        A.stride(1), A.stride(2), A.stride(3),
        B.stride(1), B.stride(2), B.stride(3),
        C.stride(1), C.stride(2), C.stride(3),
    )
    return C
