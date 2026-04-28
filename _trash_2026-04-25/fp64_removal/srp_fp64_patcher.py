"""FP64 site-level patcher, parallel to ``FP32/model_patcher.py``.

Same site coverage (Linear, RMSNorm, Attention, non-attn Softmax) but every
patched op runs reductions in FP64. Inputs/outputs stay in the original BF16/
FP16 dtype.
"""
from __future__ import annotations

import types
from contextlib import contextmanager
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from FP32.srp_fp64_ops import (
    linear_fp64_accum,
    rmsnorm_fp64_accum,
    softmax_fp64_accum,
    attention_fp64_accum,
)
from FP32.model_patcher import (
    _is_rmsnorm,
    _is_attention,
)


def _patch_all_linear_fp64(model: nn.Module) -> Dict[str, object]:
    originals = {}
    def _fwd(self, input):
        return linear_fp64_accum(input, self.weight, self.bias)
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            originals[name] = module.forward
            module.forward = types.MethodType(_fwd, module)
    return originals


def _patch_all_rmsnorm_fp64(model: nn.Module) -> Dict[str, object]:
    originals = {}
    def _fwd(self, hidden_states):
        eps = getattr(self, "variance_epsilon", getattr(self, "eps", 1e-6))
        return rmsnorm_fp64_accum(hidden_states, self.weight, eps)
    for name, module in model.named_modules():
        if _is_rmsnorm(module):
            originals[name] = module.forward
            module.forward = types.MethodType(_fwd, module)
    return originals


def _make_attention_forward_fp64(original_forward):
    def _patched(self, *args, **kwargs):
        original_sdpa = F.scaled_dot_product_attention
        def _fp64_sdpa(query, key, value, attn_mask=None, dropout_p=0.0,
                       is_causal=False, scale=None, **extra):
            return attention_fp64_accum(query, key, value,
                                        is_causal=is_causal,
                                        attn_mask=attn_mask, scale=scale)
        F.scaled_dot_product_attention = _fp64_sdpa
        try:
            return original_forward(*args, **kwargs)
        finally:
            F.scaled_dot_product_attention = original_sdpa
    return _patched


def _patch_all_attention_fp64(model: nn.Module) -> Dict[str, object]:
    originals = {}
    for name, module in model.named_modules():
        if _is_attention(module):
            originals[name] = module.forward
            module.forward = types.MethodType(
                _make_attention_forward_fp64(module.forward), module
            )
    return originals


def _patch_non_attn_softmax_fp64() -> object:
    original_softmax = F.softmax
    def _fwd(input, dim=None, _stacklevel=3, dtype=None):
        if input.dtype in (torch.bfloat16, torch.float16) and dtype is None:
            return original_softmax(input.to(torch.float64), dim=dim).to(input.dtype)
        return original_softmax(input, dim=dim, dtype=dtype)
    F.softmax = _fwd
    return original_softmax


def _restore(model, originals):
    for category, patches in originals.items():
        if category == "softmax":
            F.softmax = patches; continue
        for name, original_forward in patches.items():
            mod = model
            for p in name.split("."):
                mod = getattr(mod, p)
            mod.forward = original_forward


def apply_fp64_accum_all(
    model: nn.Module,
    patch_linear: bool = True,
    patch_rmsnorm: bool = True,
    patch_attention: bool = True,
    patch_softmax: bool = True,
) -> Dict[str, object]:
    """Apply FP64-accum patches. Mirrors ``FP32.model_patcher.apply_fp32_accum_all``."""
    originals: Dict[str, object] = {}
    if patch_linear:    originals["linear"]    = _patch_all_linear_fp64(model)
    if patch_rmsnorm:   originals["rmsnorm"]   = _patch_all_rmsnorm_fp64(model)
    if patch_attention: originals["attention"] = _patch_all_attention_fp64(model)
    if patch_softmax:   originals["softmax"]   = _patch_non_attn_softmax_fp64()
    return originals


def restore_fp64_accum_all(model: nn.Module, originals: Dict[str, object]):
    _restore(model, originals)


@contextmanager
def fp64_accum_mode(model, **kw):
    orig = apply_fp64_accum_all(model, **kw)
    try: yield orig
    finally: _restore(model, orig)
