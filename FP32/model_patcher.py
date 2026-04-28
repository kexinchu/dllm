"""
Unified monkey-patch infrastructure for FP32 accumulation.

Patches all reduction-bearing ops in a HuggingFace model:
- Linear (GEMM): FP32 accumulator along K dimension
- RMSNorm/LayerNorm: FP32 accumulator for variance reduction
- Attention: FP32 accumulators for QK^T, softmax, attn@V
- Softmax (non-attention path, e.g. MoE gating): FP32 accumulator

All input/output/weights stay BF16. Only reduction accumulators use FP32.

Supported models: Llama, Qwen3-MoE (and any model using standard HF attention/norm classes).
"""
from __future__ import annotations

import types
from contextlib import contextmanager
from typing import Dict, Optional

import torch
import torch.nn as nn

from FP32.gemm_fp32_accum import linear_fp32_accum
from FP32.rmsnorm_fp32_accum import rmsnorm_fp32_accum
from FP32.attention_fp32_accum import attention_fp32_accum


# ---------------------------------------------------------------------------
# Linear patch: replace cuBLAS BF16 GEMM with a fixed-plan Triton kernel.
#
# Why not just the cuBLAS flag?
#   `allow_bf16_reduced_precision_reduction=False` only affects cross-split
#   precision when split-K is used; it does NOT prevent cuBLAS heuristics
#   from picking different split-K plans for different M.  Different bs ->
#   different splits -> different FP32-accumulator paths -> different BF16
#   outputs at the ~5e-4 prob-std level (verified 2026-04-25 on
#   DeepSeek-7B / MATH500 with 5-problem validation).
#
# Fix: use the autotuned no-split-K Triton GEMM in FP32/triton_det_gemm.py.
#   - Each output tile computed by exactly one program (no cross-program
#     reduction).
#   - BLOCK_K pinned to 64 across all M classes -> reduction order is fixed
#     w.r.t. K regardless of M.
#   - Autotune key includes M-bucket so the plan stabilises within a bucket.
#
# We still set the cuBLAS flag as belt-and-suspenders for any F.linear call
# that bypasses our per-module patch (e.g. external modules).
# ---------------------------------------------------------------------------

# Lazy-import the deterministic GEMM kernel.
#
# Backend selection by environment variable SRP_LINEAR_BACKEND:
#   'cublaslt' (default) — cuBLASLt fixed-algo extension; near-native cuBLAS
#       speed; bit-exact via "precision amplification" (FP32 accumulator masks
#       split-K rounding to within BF16 ulp). Probabilistic argument but holds
#       in practice for typical LLM shapes.
#   'triton'  — Triton det_gemm (no split-K, fixed BLOCK_K=64). Bit-exact BY
#       CONSTRUCTION; ~1.5x baseline cuBLAS per call + ~30s first-time autotune.
#
# To switch:  SRP_LINEAR_BACKEND=triton python ...
_DET_GEMM = None
_DET_GEMM_BACKEND = None
def _load_det_gemm():
    global _DET_GEMM, _DET_GEMM_BACKEND
    if _DET_GEMM is not None:
        return _DET_GEMM

    import os
    pref = os.environ.get('SRP_LINEAR_BACKEND', 'cublaslt').lower()

    def _try_cublaslt():
        global _DET_GEMM, _DET_GEMM_BACKEND
        import sys
        ext_dir = os.path.dirname(os.path.abspath(__file__))
        if ext_dir not in sys.path:
            sys.path.insert(0, ext_dir)
        from _gemm_fixed_algo import gemm_fixed_algo as _kernel
        _DET_GEMM = _kernel
        _DET_GEMM_BACKEND = 'cublaslt'

    def _try_triton():
        global _DET_GEMM, _DET_GEMM_BACKEND
        from FP32.triton_det_gemm import det_gemm as _kernel
        _DET_GEMM = _kernel
        _DET_GEMM_BACKEND = 'triton'

    if pref == 'triton':
        try: _try_triton()
        except ImportError: _try_cublaslt()
    else:  # cublaslt (default) or anything else
        try: _try_cublaslt()
        except ImportError: _try_triton()
    return _DET_GEMM


# Hybrid dispatch threshold for the experimental cuBLASLt/Triton split:
#   M <= _HYBRID_M_THRESHOLD  -> cuBLASLt fixed-algo (faster, but cuBLAS
#                                heuristic picks per-M tile config so output
#                                is NOT strictly bit-exact across batch sizes)
#   M >  _HYBRID_M_THRESHOLD  -> Triton det_gemm (fixed BLOCK_K=64, strictly
#                                bit-exact by construction)
#
# Default 0 means "M ≤ 0 is always false, so all calls go to Triton det_gemm"
# -> strict Avg_Std=0 (verified 2026-04-25 on DeepSeek-7B / MATH500).
# Set higher (e.g. 16) to trade strict bit-exactness for speed in workloads
# where ~5% std reduction at near-zero overhead is acceptable.
_HYBRID_M_THRESHOLD = 0

# Lazy load both backends so dispatcher can choose per-call.
_CUBLASLT_GEMM = None
_TRITON_GEMM   = None
def _load_both_gemm():
    global _CUBLASLT_GEMM, _TRITON_GEMM
    if _CUBLASLT_GEMM is None:
        import os, sys
        ext_dir = os.path.dirname(os.path.abspath(__file__))
        if ext_dir not in sys.path:
            sys.path.insert(0, ext_dir)
        from _gemm_fixed_algo import gemm_fixed_algo
        _CUBLASLT_GEMM = gemm_fixed_algo
    if _TRITON_GEMM is None:
        from FP32.triton_det_gemm import det_gemm
        _TRITON_GEMM = det_gemm
    return _CUBLASLT_GEMM, _TRITON_GEMM


def _det_linear_forward(self, input: torch.Tensor) -> torch.Tensor:
    """nn.Linear.forward replacement, hybrid backend by M:
       small M (decode path)     -> cuBLASLt fixed-algo (NONE scheme, bit-exact)
       large M (prefill path)    -> Triton det_gemm (fixed BLOCK_K, bit-exact)
    """
    weight = self.weight
    bias = self.bias
    if (input.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16
            and input.is_cuda and input.dim() >= 2):
        orig_shape = input.shape
        x2d = input.reshape(-1, orig_shape[-1])
        if not x2d.is_contiguous():
            x2d = x2d.contiguous()
        w = weight if weight.is_contiguous() else weight.contiguous()
        cublaslt_gemm, triton_gemm = _load_both_gemm()
        M = x2d.shape[0]
        if M <= _HYBRID_M_THRESHOLD:
            out = cublaslt_gemm(x2d, w)
        else:
            out = triton_gemm(x2d, w)
        out = out.reshape(*orig_shape[:-1], w.shape[0])
        if bias is not None:
            out = out + bias.to(out.dtype)
        return out
    import torch.nn.functional as F
    return F.linear(input, weight, bias)


def _patch_all_linear(model: nn.Module) -> Dict[str, object]:
    """
    Replace every nn.Linear.forward with a fixed-plan Triton GEMM (FP32 accum,
    no split-K). Also set cuBLAS flag for safety on any unpatched F.linear.
    """
    originals = {}

    # Belt-and-suspenders: also set cuBLAS flag for any F.linear that bypasses
    # our per-module patch (e.g. third-party modules).
    if hasattr(torch.backends.cuda.matmul, 'allow_bf16_reduced_precision_reduction'):
        originals['__bf16_reduced_flag'] = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            originals[name] = module.forward
            module.forward = types.MethodType(_det_linear_forward, module)
    return originals


# ---------------------------------------------------------------------------
# RMSNorm patch
# ---------------------------------------------------------------------------

# Known RMSNorm class names in HuggingFace transformers
_RMSNORM_CLASS_NAMES = {
    "LlamaRMSNorm",
    "Qwen2RMSNorm",
    "Qwen3MoeRMSNorm",
    "MistralRMSNorm",
    "GemmaRMSNorm",
    "InternLM2RMSNorm",
    "DeepseekV2RMSNorm",
}


def _is_rmsnorm(module: nn.Module) -> bool:
    """Check if module is an RMSNorm variant (by class name)."""
    return type(module).__name__ in _RMSNORM_CLASS_NAMES


def _patch_all_rmsnorm(model: nn.Module) -> Dict[str, object]:
    """Replace RMSNorm forward with FP32-accumulation version."""
    originals = {}

    def _fp32_accum_rmsnorm_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        eps = getattr(self, "variance_epsilon", getattr(self, "eps", 1e-6))
        return rmsnorm_fp32_accum(hidden_states, self.weight, eps)

    for name, module in model.named_modules():
        if _is_rmsnorm(module):
            originals[name] = module.forward
            module.forward = types.MethodType(_fp32_accum_rmsnorm_forward, module)
    return originals


# ---------------------------------------------------------------------------
# Attention patch
# ---------------------------------------------------------------------------

# Known attention class names in HuggingFace transformers
_ATTENTION_CLASS_NAMES = {
    "LlamaSdpaAttention",
    "LlamaFlashAttention2",
    "LlamaAttention",
    "Qwen2SdpaAttention",
    "Qwen2FlashAttention2",
    "Qwen2Attention",
    "Qwen3MoeSdpaAttention",
    "Qwen3MoeFlashAttention2",
    "Qwen3MoeAttention",
}


def _is_attention(module: nn.Module) -> bool:
    return type(module).__name__ in _ATTENTION_CLASS_NAMES


def _make_attention_forward(original_forward):
    """
    Create a patched attention forward that:
    1. Runs Q/K/V projections (using already-patched Linear layers)
    2. Applies RoPE (unchanged, per-element op)
    3. Replaces F.scaled_dot_product_attention with our FP32 accum version
    """
    def _patched_forward(self, *args, **kwargs):
        # Temporarily replace torch.nn.functional.scaled_dot_product_attention
        import torch.nn.functional as F
        original_sdpa = F.scaled_dot_product_attention

        def _fp32_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **extra_kwargs):
            # Ignore scale, enable_gqa, etc. -- our kernel handles these internally
            return attention_fp32_accum(query, key, value, is_causal=is_causal, attn_mask=attn_mask)

        F.scaled_dot_product_attention = _fp32_sdpa
        try:
            return original_forward(*args, **kwargs)
        finally:
            F.scaled_dot_product_attention = original_sdpa

    return _patched_forward


def _patch_all_attention(model: nn.Module) -> Dict[str, object]:
    """Replace attention modules' SDPA call with FP32 accum attention."""
    originals = {}

    for name, module in model.named_modules():
        if _is_attention(module):
            originals[name] = module.forward
            module.forward = types.MethodType(
                _make_attention_forward(module.forward), module
            )
    return originals


# ---------------------------------------------------------------------------
# Softmax patch (non-attention paths only, e.g. MoE gating)
# ---------------------------------------------------------------------------

def _patch_non_attn_softmax() -> Optional[object]:
    """
    Override torch.nn.functional.softmax globally for non-attention paths.
    Attention softmax is handled inside the fused attention kernel.
    Returns the original function for restoration.
    """
    import torch.nn.functional as F
    original_softmax = F.softmax

    def _fp32_softmax(input, dim=None, _stacklevel=3, dtype=None):
        if input.dtype in (torch.bfloat16, torch.float16) and dtype is None:
            # FP32 accumulation for reduction, cast back
            return original_softmax(input.float(), dim=dim).to(input.dtype)
        return original_softmax(input, dim=dim, dtype=dtype)

    F.softmax = _fp32_softmax
    return original_softmax


def _restore_softmax(original: object):
    import torch.nn.functional as F
    F.softmax = original


# ---------------------------------------------------------------------------
# Unified context manager
# ---------------------------------------------------------------------------

def _restore_all(model: nn.Module, originals: Dict[str, Dict[str, object]]):
    """Restore all patched modules."""
    for category, patches in originals.items():
        if category == 'softmax':
            _restore_softmax(patches)
            continue
        if category == 'linear' and '__bf16_reduced_flag' in patches:
            # Restore the cuBLAS flag
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = patches['__bf16_reduced_flag']
            continue
        for name, original_forward in patches.items():
            # Walk model to find the module by name
            parts = name.split('.')
            mod = model
            for p in parts:
                mod = getattr(mod, p)
            mod.forward = original_forward


@contextmanager
def fp32_accum_mode(
    model: nn.Module,
    patch_linear: bool = True,
    patch_rmsnorm: bool = False,
    patch_attention: bool = False,
    patch_softmax: bool = False,
):
    """
    Context manager: enable FP32 accumulation for reduction ops.

    Default: only patch Linear (cuBLAS flag, zero overhead).
    RMSNorm/Attention are NOT patched by default because:
    - HuggingFace LlamaRMSNorm already uses FP32 internally
    - SDPA/FlashAttention already uses FP32 accumulators internally
    - Custom kernels produce micro-different FP32 results that compound over layers
    RMSNorm/Attention patches are for serving engines (vLLM/SGLang) that use
    different kernel backends with dynamic split strategies.

    Args:
        model: HuggingFace model to patch
        patch_linear: Force FP32 accumulation in GEMM via cuBLAS flag (zero overhead)
        patch_rmsnorm: Replace RMSNorm with custom Triton kernel (for serving engines only)
        patch_attention: Replace attention with fixed-split Triton kernel (for serving engines only)
        patch_softmax: Patch F.softmax globally (for non-attention paths like MoE gating)
    """
    originals: Dict[str, object] = {}

    if patch_linear:
        originals['linear'] = _patch_all_linear(model)
    if patch_rmsnorm:
        originals['rmsnorm'] = _patch_all_rmsnorm(model)
    if patch_attention:
        originals['attention'] = _patch_all_attention(model)
    if patch_softmax:
        originals['softmax'] = _patch_non_attn_softmax()

    try:
        yield originals
    finally:
        _restore_all(model, originals)


def apply_fp32_accum_all(
    model: nn.Module,
    patch_linear: bool = True,
    patch_rmsnorm: bool = False,
    patch_attention: bool = False,
    patch_softmax: bool = False,
) -> Dict[str, object]:
    """
    Non-context-manager version: apply patches and return originals dict.
    Call restore_fp32_accum_all(model, originals) to undo.
    Default: only patch Linear (cuBLAS flag). See fp32_accum_mode docstring.
    """
    originals: Dict[str, object] = {}
    if patch_linear:
        originals['linear'] = _patch_all_linear(model)
    if patch_rmsnorm:
        originals['rmsnorm'] = _patch_all_rmsnorm(model)
    if patch_attention:
        originals['attention'] = _patch_all_attention(model)
    if patch_softmax:
        originals['softmax'] = _patch_non_attn_softmax()
    return originals


def restore_fp32_accum_all(model: nn.Module, originals: Dict[str, object]):
    """Restore all patches applied by apply_fp32_accum_all."""
    _restore_all(model, originals)
