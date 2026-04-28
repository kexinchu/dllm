"""FP64 SRP ops — Selective Reduction Precision with float64 accumulators.

Mirrors the FP32 variants in this directory but performs the reduction in
double precision. The CUDA extensions used by the FP32 path (cuBLAS-with-FP32-
accumulator GEMM) don't exist for FP64 of BF16 inputs; we therefore route
everything through PyTorch ops with explicit ``.to(torch.float64)`` casts.

These ops are paid-for in throughput (FP64 GEMM on consumer GPUs is 1/32 to
1/64 of FP32). That is the point: NIPS-2027 §4 / RQ4 asks whether FP64 is
actually necessary or if FP32 already suffices. This module is the upper-bound
"throw FP64 at the reduction" reference.

Drop-in for ``model_patcher`` site-level patches. All inputs/outputs stay in
the original (BF16/FP16) dtype; only intra-kernel accumulation is FP64.
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F


# ── Linear (GEMM K-dim accumulation in FP64) ──────────────────────────────────
def linear_fp64_accum(input: torch.Tensor, weight: torch.Tensor,
                      bias: torch.Tensor | None = None) -> torch.Tensor:
    out = F.linear(input.to(torch.float64), weight.to(torch.float64), None)
    if bias is not None:
        out = out + bias.to(torch.float64)
    return out.to(input.dtype)


# ── RMSNorm (variance accum FP64) ─────────────────────────────────────────────
def rmsnorm_fp64_accum(x: torch.Tensor, weight: torch.Tensor,
                      eps: float) -> torch.Tensor:
    x64 = x.to(torch.float64)
    var = (x64 * x64).mean(dim=-1, keepdim=True)
    out = x64 * torch.rsqrt(var + eps) * weight.to(torch.float64)
    return out.to(x.dtype)


# ── Softmax (sum accum FP64) — for non-attention paths (e.g. MoE gate) ────────
def softmax_fp64_accum(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.softmax(input.to(torch.float64), dim=dim).to(input.dtype)


# ── Attention (QK^T, softmax, score@V all in FP64) ────────────────────────────
def _expand_kv_for_gqa(K: torch.Tensor, V: torch.Tensor, num_q_heads: int):
    num_kv_heads = K.shape[1]
    if num_kv_heads == num_q_heads:
        return K, V
    assert num_q_heads % num_kv_heads == 0, \
        f"Q heads {num_q_heads} not divisible by KV heads {num_kv_heads}"
    repeat = num_q_heads // num_kv_heads
    return K.repeat_interleave(repeat, dim=1), V.repeat_interleave(repeat, dim=1)


def attention_fp64_accum(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
    is_causal: bool = False, attn_mask: torch.Tensor | None = None,
    scale: float | None = None,
) -> torch.Tensor:
    """Reference attention with FP64 reductions for QK^T, softmax, and score@V.

    Shapes: [B, H_q, L_q, D], [B, H_kv, L_k, D], [B, H_kv, L_k, D].
    Handles GQA (H_kv ≤ H_q) by repeat_interleave on K/V.
    Returns: [B, H_q, L_q, D] in the input dtype.
    """
    if query.dim() == 4 and key.shape[1] != query.shape[1]:
        key, value = _expand_kv_for_gqa(key, value, query.shape[1])

    q64 = query.to(torch.float64)
    k64 = key.to(torch.float64)
    v64 = value.to(torch.float64)

    if scale is None:
        scale = 1.0 / math.sqrt(query.shape[-1])

    scores = torch.matmul(q64, k64.transpose(-2, -1)) * scale
    if is_causal:
        L_q = query.shape[-2]; L_k = key.shape[-2]
        mask = torch.ones(L_q, L_k, device=query.device, dtype=torch.bool).tril(L_k - L_q)
        scores = scores.masked_fill(~mask, float("-inf"))
    if attn_mask is not None:
        scores = scores + attn_mask.to(torch.float64)

    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, v64)
    return out.to(query.dtype)
