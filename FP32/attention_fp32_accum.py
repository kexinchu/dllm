"""
Flash-style Attention with FP32 accumulation: Q/K/V/Output stay BF16,
only reduction accumulators (QK^T dot, online softmax, attn@V dot) use FP32.

Key design:
- Tiled online softmax (FlashAttention algorithm) -> O(1) extra memory
- No split-KV: each (batch, head, q_block) program iterates over full KV sequence
  -> reduction order depends only on seq_len and BLOCK_N, not on batch size
  -> naturally batch-invariant
- All accumulators (m_i, l_i, acc) are FP32
- tl.dot(..., out_dtype=tl.float32) for QK^T and attn@V reductions

Pattern: BF16 load -> FP32 accumulator -> cast to BF16 -> store
"""
from __future__ import annotations

import math
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
    def _attention_fp32_accum_fwd_kernel(
        Q, K, V, Out,
        sm_scale,
        stride_qz, stride_qh, stride_qm, stride_qk,
        stride_kz, stride_kh, stride_kn, stride_kk,
        stride_vz, stride_vh, stride_vn, stride_vk,
        stride_oz, stride_oh, stride_om, stride_ok,
        Z, H, N_CTX,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_DMODEL: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
    ):
        # Program: one per (q_block, batch*head)
        start_m = tl.program_id(0)
        off_hz = tl.program_id(1)
        off_z = off_hz // H
        off_h = off_hz % H

        # Base pointers for this batch and head
        q_offset = off_z * stride_qz + off_h * stride_qh
        k_offset = off_z * stride_kz + off_h * stride_kh
        v_offset = off_z * stride_vz + off_h * stride_vh
        o_offset = off_z * stride_oz + off_h * stride_oh

        # Offsets for Q block rows
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_DMODEL)

        # Load Q block [BLOCK_M, BLOCK_DMODEL] -- BF16
        q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
        q_mask = offs_m[:, None] < N_CTX
        q = tl.load(q_ptrs, mask=q_mask, other=0.0)

        # FP32 accumulators for online softmax
        m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)  # running max
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)                 # running sum(exp)
        acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)   # output accumulator

        # KV iteration range
        if IS_CAUSAL:
            kv_end = tl.minimum(N_CTX, (start_m + 1) * BLOCK_M)
        else:
            kv_end = N_CTX

        # Iterate over KV blocks -- fixed order, no split-KV
        offs_n = tl.arange(0, BLOCK_N)
        for start_n in range(0, kv_end, BLOCK_N):
            cur_offs_n = start_n + offs_n
            kv_mask = cur_offs_n < N_CTX

            # Load K block [BLOCK_N, BLOCK_DMODEL] -- BF16
            k_ptrs = K + k_offset + cur_offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
            k = tl.load(k_ptrs, mask=kv_mask[:, None], other=0.0)

            # QK^T: BF16 inputs, FP32 accumulator
            # [BLOCK_M, BLOCK_DMODEL] x [BLOCK_DMODEL, BLOCK_N] -> [BLOCK_M, BLOCK_N]
            qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32)
            qk *= sm_scale

            # Causal mask
            if IS_CAUSAL:
                causal_mask = offs_m[:, None] >= cur_offs_n[None, :]
                qk = tl.where(causal_mask, qk, float('-inf'))

            # Mask out-of-bounds KV positions
            qk = tl.where(kv_mask[None, :], qk, float('-inf'))

            # Online softmax update -- all FP32
            m_ij = tl.max(qk, axis=1)                      # [BLOCK_M] FP32
            m_new = tl.maximum(m_i, m_ij)                   # [BLOCK_M] FP32
            alpha = tl.exp(m_i - m_new)                     # [BLOCK_M] FP32
            p = tl.exp(qk - m_new[:, None])                 # [BLOCK_M, BLOCK_N] FP32

            # Update running sum and accumulator
            l_i = l_i * alpha + tl.sum(p, axis=1)           # FP32 accumulator

            # Load V block [BLOCK_N, BLOCK_DMODEL] -- BF16
            v_ptrs = V + v_offset + cur_offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
            v = tl.load(v_ptrs, mask=kv_mask[:, None], other=0.0)

            # attn@V: cast p to BF16 for dot input, accumulator stays FP32
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v, out_dtype=tl.float32)
            m_i = m_new

        # Final normalization (FP32) -> cast to BF16
        acc = acc / l_i[:, None]
        out = acc.to(tl.bfloat16)

        # Store output
        o_ptrs = Out + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
        o_mask = offs_m[:, None] < N_CTX
        tl.store(o_ptrs, out, mask=o_mask)


def attention_fp32_accum_triton(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    is_causal: bool = False,
) -> torch.Tensor:
    """
    Triton Flash-style attention: BF16 in/out, FP32 accumulators.
    Q, K, V: [batch, heads, seq_len, head_dim] BF16
    Returns: [batch, heads, seq_len, head_dim] BF16
    """
    assert Q.is_cuda
    Z, H, N_CTX, BLOCK_DMODEL = Q.shape
    assert K.shape == (Z, H, N_CTX, BLOCK_DMODEL) or K.shape[0] == Z
    assert V.shape[2] == K.shape[2]  # KV seq_len must match

    sm_scale = 1.0 / math.sqrt(BLOCK_DMODEL)
    KV_LEN = K.shape[2]

    Out = torch.empty_like(Q)

    # Tile sizes
    BLOCK_M = 64
    BLOCK_N = 64
    # Ensure BLOCK_DMODEL is power of 2 for triton
    assert BLOCK_DMODEL in (16, 32, 64, 128, 256), f"head_dim={BLOCK_DMODEL} not supported"

    grid = (triton.cdiv(N_CTX, BLOCK_M), Z * H)

    _attention_fp32_accum_fwd_kernel[grid](
        Q, K, V, Out,
        sm_scale,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        Out.stride(0), Out.stride(1), Out.stride(2), Out.stride(3),
        Z, H, N_CTX,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL,
        IS_CAUSAL=is_causal,
    )
    return Out


def attention_fp32_accum_pytorch(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    is_causal: bool = False,
    attn_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    PyTorch fallback: BF16 in/out, FP32 accumulators for all reductions.
    Warning: O(n^2) memory for attention scores. Use only for short sequences / validation.
    Q, K, V: [..., seq_len, head_dim] BF16
    """
    scale = Q.shape[-1] ** -0.5

    # QK^T: FP32 accumulator
    scores = torch.matmul(Q.float(), K.float().transpose(-2, -1)) * scale

    if is_causal:
        L, S = scores.shape[-2], scores.shape[-1]
        mask = torch.triu(torch.ones(L, S, device=scores.device, dtype=torch.bool), diagonal=1)
        scores.masked_fill_(mask, float('-inf'))

    if attn_mask is not None:
        scores = scores + attn_mask.float()

    # Softmax: FP32 accumulator
    weights = torch.softmax(scores, dim=-1)

    # attn@V: FP32 accumulator
    out = torch.matmul(weights, V.float())

    return out.to(Q.dtype)


def _expand_kv_for_gqa(K: torch.Tensor, V: torch.Tensor, num_q_heads: int) -> tuple:
    """Expand K/V heads to match Q heads for GQA (Grouped Query Attention)."""
    num_kv_heads = K.shape[1]
    if num_kv_heads == num_q_heads:
        return K, V
    assert num_q_heads % num_kv_heads == 0, f"Q heads {num_q_heads} not divisible by KV heads {num_kv_heads}"
    repeat = num_q_heads // num_kv_heads
    K = K.repeat_interleave(repeat, dim=1)
    V = V.repeat_interleave(repeat, dim=1)
    return K, V


if TRITON_AVAILABLE:
    @triton.jit
    def _attention_decode_fp32_accum_kernel(
        Q_ptr, K_ptr, V_ptr, Out_ptr,
        sm_scale,
        # Q: [B*H, D]      contiguous along last dim
        stride_qbh, stride_qd,
        # K, V: [B*H, L_k, D]
        stride_kbh, stride_kn, stride_kd,
        stride_vbh, stride_vn, stride_vd,
        # Out: [B*H, D]
        stride_obh, stride_od,
        BH, KV_LEN,
        BLOCK_N: tl.constexpr,
        BLOCK_DMODEL: tl.constexpr,
    ):
        """Decode attention kernel for M_q=1.

        One program per (batch, head). Each program iterates the full KV in
        fixed BLOCK_N tiles -- no split-KV across programs and no split-K
        within a tile, so the per-output reduction order is byte-identical
        regardless of batch size.

        BF16 loads, FP32 accumulators (m_i, l_i, acc), BF16 store.
        """
        bh = tl.program_id(0)
        offs_d = tl.arange(0, BLOCK_DMODEL)
        offs_n = tl.arange(0, BLOCK_N)

        # Load Q row [D] -> FP32
        q_ptr = Q_ptr + bh * stride_qbh + offs_d * stride_qd
        q = tl.load(q_ptr).to(tl.float32)

        m_i = float('-inf')
        l_i = 0.0
        acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)

        # KV iteration in fixed BLOCK_N tiles (no split-KV).
        for start_n in range(0, KV_LEN, BLOCK_N):
            cur_offs_n = start_n + offs_n
            kv_mask = cur_offs_n < KV_LEN

            k_ptrs = (K_ptr + bh * stride_kbh
                      + cur_offs_n[:, None] * stride_kn
                      + offs_d[None, :] * stride_kd)
            k = tl.load(k_ptrs, mask=kv_mask[:, None], other=0.0).to(tl.float32)

            # qk: [BLOCK_N] = sum_d (q[d] * k[n, d])  -- FP32 reduction
            qk = tl.sum(q[None, :] * k, axis=1) * sm_scale
            qk = tl.where(kv_mask, qk, float('-inf'))

            # Online softmax update (all FP32)
            m_ij = tl.max(qk, axis=0)
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(qk - m_new)
            l_i = l_i * alpha + tl.sum(p, axis=0)

            v_ptrs = (V_ptr + bh * stride_vbh
                      + cur_offs_n[:, None] * stride_vn
                      + offs_d[None, :] * stride_vd)
            v = tl.load(v_ptrs, mask=kv_mask[:, None], other=0.0).to(tl.float32)

            # acc[d] = acc[d] * alpha + sum_n p[n] * v[n, d]   -- FP32 reduction
            acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
            m_i = m_new

        acc = acc / l_i
        o_ptr = Out_ptr + bh * stride_obh + offs_d * stride_od
        tl.store(o_ptr, acc.to(tl.bfloat16))


def attention_decode_fp32_accum_triton(Q: torch.Tensor, K: torch.Tensor,
                                       V: torch.Tensor,
                                       valid_kv_len: int | None = None) -> torch.Tensor:
    """Triton decode kernel: bit-exact across bs by construction.
    Q: [B, H, 1, D] BF16. K, V: [B, H, L_k, D] BF16.
    valid_kv_len: only attend to K[:, :, :valid_kv_len]. Used with StaticCache
                  where the K buffer is allocated at max_seq_len but only the
                  first valid_kv_len entries are written. Defaults to L_k
                  (full).
    Returns [B, H, 1, D] BF16.
    """
    B, H, M_q, D = Q.shape
    assert M_q == 1, "decode kernel requires M_q=1"
    L_k = K.shape[2]
    if valid_kv_len is None:
        valid_kv_len = L_k
    assert valid_kv_len <= L_k
    assert K.shape == (B, H, L_k, D) and V.shape == (B, H, L_k, D)
    assert D in (16, 32, 64, 128, 256), f"head_dim={D} not supported"

    Q2 = Q.reshape(B * H, D).contiguous() if not Q.is_contiguous() else Q.view(B * H, D)
    K2 = K.reshape(B * H, L_k, D).contiguous() if not K.is_contiguous() else K.view(B * H, L_k, D)
    V2 = V.reshape(B * H, L_k, D).contiguous() if not V.is_contiguous() else V.view(B * H, L_k, D)
    Out = torch.empty((B * H, D), dtype=torch.bfloat16, device=Q.device)

    sm_scale = 1.0 / math.sqrt(D)
    BLOCK_N = 64
    grid = (B * H,)

    _attention_decode_fp32_accum_kernel[grid](
        Q2, K2, V2, Out,
        sm_scale,
        Q2.stride(0), Q2.stride(1),
        K2.stride(0), K2.stride(1), K2.stride(2),
        V2.stride(0), V2.stride(1), V2.stride(2),
        Out.stride(0), Out.stride(1),
        B * H, valid_kv_len,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=D,
    )
    return Out.view(B, H, 1, D)


# Module-level "current valid KV length" override. When the caller (e.g. a
# CUDA-graph-replay decode loop) sets this, the dispatcher uses it instead of
# inspecting attn_mask. This avoids `.item()` inside graph-captured regions.
# Set/clear via `set_decode_valid_kv_len(int_or_tensor or None)`.
_CURRENT_VALID_KV_LEN = None


def set_decode_valid_kv_len(val):
    """Override valid_kv_len for the next attention_decode call. Pass None to
    revert to attn_mask-driven inference. ``val`` may be a Python int or a
    0-d tensor (the latter useful for graph capture, where reading via
    .item() in the dispatcher itself works at capture time and the graph
    bakes the value).
    """
    global _CURRENT_VALID_KV_LEN
    _CURRENT_VALID_KV_LEN = val


def _infer_valid_kv_len_from_mask(attn_mask: torch.Tensor) -> int | None:
    """Extract `valid_kv_len` from a HF additive 4-D causal mask of shape
    (bs, 1, q_len, kv_len), where 0 = valid and -inf (or large negative) = masked.
    Returns None if mask cannot be interpreted as such (e.g. 2-D padding mask).
    Uses `.item()` -> implies a CPU sync; not safe inside a captured graph,
    but fine in eager mode.
    """
    if attn_mask is None:
        return None
    # Expect (B, 1, q_len, kv_len). Read row 0 of last query position.
    if attn_mask.dim() == 4 and attn_mask.shape[-2] >= 1:
        last_row = attn_mask[0, 0, -1]   # [kv_len]
        # valid if entry is finite (== 0); masked if -inf
        # Count valid positions: those with finite values (sum of (mask >= 0))
        valid = (last_row >= 0).sum().item()
        return int(valid)
    return None


def attention_decode_fp32_accum(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Decode attention (M_q=1): use Triton fixed-plan kernel for bit-exact
    output across batch sizes. Supports `attn_mask` (e.g. from StaticCache)
    by inferring `valid_kv_len`. Falls back to FP32 PyTorch when Triton path
    cannot run (head_dim not supported, non-CUDA).
    """
    if (TRITON_AVAILABLE and Q.is_cuda
            and Q.dim() == 4 and Q.shape[-2] == 1
            and Q.shape[-1] in (16, 32, 64, 128, 256)):
        # Determine valid_kv_len: precedence is module-level override > mask > full.
        if _CURRENT_VALID_KV_LEN is not None:
            v = _CURRENT_VALID_KV_LEN
            valid_kv_len = int(v.item()) if torch.is_tensor(v) else int(v)
        else:
            valid_kv_len = _infer_valid_kv_len_from_mask(attn_mask)
        return attention_decode_fp32_accum_triton(Q, K, V, valid_kv_len=valid_kv_len)
    # Fall back to FP32 PyTorch (precision amplification ⇒ near-bit-exact in
    # practice; slower since K/V get fully cast).
    return attention_fp32_accum_pytorch(Q, K, V, is_causal=False, attn_mask=attn_mask)


def attention_fp32_accum(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    is_causal: bool = False,
    attn_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Attention with FP32 accumulation. Supports GQA (K/V may have fewer heads than Q).
    Dispatch:
      M_q=1 (decode)   -> BF16 GEMV path (fast, bs-invariant via M=1)
      M_q>1 (prefill)  -> Triton FA-style fixed-plan kernel (bs-invariant by no split-KV)
                          falls back to FP32 PyTorch when Triton path can't run
                          (non-contig, head_dim mismatch, attn_mask present, etc.)
    Q: [batch, q_heads, seq_len, head_dim] BF16
    K, V: [batch, kv_heads, seq_len, head_dim] BF16 (kv_heads <= q_heads)
    """
    # Handle GQA: expand K/V to match Q's head count
    if Q.dim() == 4 and K.shape[1] != Q.shape[1]:
        K, V = _expand_kv_for_gqa(K, V, Q.shape[1])

    M_q = Q.shape[-2]
    # Decode fast path (M_q=1): cuBLAS BF16 GEMV is naturally batch-invariant.
    if M_q == 1 and Q.is_cuda:
        return attention_decode_fp32_accum(Q, K, V, attn_mask)

    if (
        TRITON_AVAILABLE
        and Q.is_cuda
        and attn_mask is None
        and Q.dim() == 4
        and Q.shape[-1] in (16, 32, 64, 128, 256)
        and Q.is_contiguous()
        and K.is_contiguous()
        and V.is_contiguous()
        and K.shape[2] == Q.shape[2]   # Triton kernel currently only handles M_q == KV_LEN
    ):
        return attention_fp32_accum_triton(Q, K, V, is_causal)
    return attention_fp32_accum_pytorch(Q, K, V, is_causal, attn_mask)
