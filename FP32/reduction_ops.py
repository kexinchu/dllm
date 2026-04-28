"""Reduction ops: BF16 vs FP32 then BF16 vs deterministic. Ref: Paper-List.md; batch_invariance.md.

注意：本模块是 PyTorch 层面的数值/行为模拟，用于比较 latency 与 deterministic。
若要在真实推理中做到「仅规约用 FP32、再压回 BF16」，需要修改 kernel：
  - 方案 A：修改 GEMM kernel，在规约维度上用 FP32 累加，输出前 cast 回 BF16；
  - 方案 B：GEMM 输出 partial sums（BF16），再用单独的 FP32 reduction kernel 求和并 cast 回 BF16。
当前 PyTorch 的 x.float().sum().to(bf16) 会走通用 reduction，并非定制 kernel。
"""

from __future__ import annotations

import random
from typing import Optional

import torch

def reduce_bf16_atomic_style(x, dim=-1, num_splits=None, seed=None):
    if x.dtype != torch.bfloat16:
        x = x.to(torch.bfloat16)
    K = x.shape[dim]
    if num_splits is None:
        num_splits = min(32, max(1, K // 4))
    num_splits = min(num_splits, K)
    chunk_size = K // num_splits
    if chunk_size == 0:
        num_splits, chunk_size = K, 1
    chunks = []
    for i in range(num_splits):
        start = i * chunk_size
        end = K if i == num_splits - 1 else (i + 1) * chunk_size
        sl = [slice(None)] * x.dim()
        sl[dim] = slice(start, end)
        chunks.append(x[tuple(sl)].sum(dim=dim, keepdim=True))
    stacked = torch.cat(chunks, dim=dim)
    order = list(range(num_splits))
    (random.Random(seed) if seed is not None else random).shuffle(order)
    out = stacked[..., order[0]].clone()
    for idx in order[1:]:
        out = (out + stacked[..., idx]).to(torch.bfloat16)
    return out

def reduce_fp32_then_bf16(x, dim=-1):
    return x.to(torch.float32).sum(dim=dim).to(torch.bfloat16)

def reduce_deterministic_sequential(x, dim=-1):
    return x.to(torch.float32).sum(dim=dim).to(torch.bfloat16)

def reduce_bf16_naive(x, dim=-1):
    if x.dtype != torch.bfloat16:
        x = x.to(torch.bfloat16)
    return x.sum(dim=dim)
