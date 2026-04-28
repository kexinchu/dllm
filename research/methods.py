"""NIPS-2027 method enum + enter/exit context managers.

Aligned with /home/kec23008/docker-sys/dllm/NIPS-2027.md §5.1 and §5.2.

CORRECTED FRAMING (2026-04-25, after source-level verification):
  - HF LlamaRMSNorm already uses an FP32 accumulator
    (modeling_llama.py:62-67, hidden_states.to(torch.float32).pow(2).mean(...)).
  - PyTorch SDPA backends (FA-2 / EFFICIENT / CUDNN / MATH) all use FP32
    accumulators (FA-2 kernel_traits.h:17 hardcodes ElementAccum=float).
  - cuBLAS BF16 GEMM ALWAYS uses FP32 inner-product accumulator
    (CUBLAS_COMPUTE_32F unconditionally; CUDABlas.cpp:625-635).
  - The flag `allow_bf16_reduced_precision_reduction=False` does NOT change
    the inner-product accumulator. It only forbids reduced-precision
    cross-split reduction in split-K kernels (CUBLASLT_REDUCTION_SCHEME_*).

  =>  FP32 accumulator is the default in modern stacks. The remaining batch-
  non-determinism comes from BS-DEPENDENT KERNEL PLAN SELECTION (cuBLAS
  split-K heuristic + FlashAttention split-KV), NOT from accumulator
  precision. SRP's real lever is fixed-plan reduction kernels.

  =>  Only MatMul-K and Attention sites genuinely need patching for batch-
  invariance. RMSNorm and non-attention Softmax patches are no-ops in the
  HF/PyTorch dense path (kept available for serving-stack experiments).

Methods:

  BF16          baseline (no patch). Default HF behaviour. FP32 accumulators
                are already used everywhere; only the kernel plan varies with bs.

  FP32-all      all relevant compute uses FP32 (impl: load weights as fp32).

  LayerCast     Yuan et al.'s baseline. BF16 storage + per-Linear FP32 cast.

  SRP-FP32      Selective Fixed-Plan Reduction with FP32 accumulator on the
                requested `sites`. The "FP32" in the name preserves NIPS plan
                terminology -- semantics is "force FP32 accumulator AND fixed
                reduction plan"; in practice the accumulator was already FP32,
                the meaningful change is the fixed plan.

FP64 is intentionally absent: BF16 -> FP64 has no tensor core path on Ampere
(A6000), and BF16 inputs cannot be accumulated to FP64 by hardware. Removed
2026-04-25.

Each method is a context manager:  ``with method_X(model): run(...)``.
The model dtype/weights for FP32-all is decided at load time, not patch
time, so its CM is a no-op (must be paired with the appropriate model load).
"""
from __future__ import annotations

import os, sys
from contextlib import contextmanager
from typing import Iterable

import torch
import torch.nn as nn

# Make sibling packages importable
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from FP32.model_patcher import apply_fp32_accum_all, restore_fp32_accum_all

# layercast modules at research/ root
from research.layercast import enable as _layercast_enable, disable as _layercast_disable


# ── Method enum (canonical strings used in JSON / plot labels) ────────────────
# All sites SRP supports. The two that genuinely matter for batch-invariance
# in dense LLMs are `linear` (cuBLAS split-K) and `attention` (FA-2 split-KV).
# `rmsnorm` and `softmax` are kept for completeness / serving-stack experiments
# but in HF dense inference they're already batch-invariant by default.
ALL_SITES = ("linear", "rmsnorm", "attention", "softmax")
CRITICAL_SITES = ("linear", "attention")  # the real "must-patch" set

# Site → human label (used in figures)
SITE_LABEL = {
    "linear":     "MatMul-K",
    "rmsnorm":    "Norm-Stat",
    "attention":  "Attn (QK + Softmax + V)",
    "softmax":    "Softmax-Red (non-attn)",
}


# ── Method context managers ──────────────────────────────────────────────────
@contextmanager
def method_BF16(model):
    """No-op. Model must be loaded in BF16."""
    # Make sure no leftover patch from a previous call is active.
    yield


@contextmanager
def method_FP32_all(model):
    """No-op patch. Caller must load model with ``dtype=torch.float32``.

    We assert the dtype here so misuse fails loudly.
    """
    p = next(model.parameters())
    assert p.dtype == torch.float32, (
        "FP32-all requires the model to be loaded with dtype=torch.float32; "
        f"got {p.dtype}. Reload the model and re-enter this context."
    )
    yield


@contextmanager
def method_LayerCast(model):
    """Yuan et al.'s baseline: BF16 weights, per-Linear FP32 compute, BF16 output."""
    _layercast_enable()
    try: yield
    finally: _layercast_disable()


@contextmanager
def method_SRP_FP32(model, sites: Iterable[str] = ALL_SITES):
    """Selective Reduction Precision with FP32 accumulators."""
    sites = tuple(sites)
    orig = apply_fp32_accum_all(
        model,
        patch_linear="linear" in sites,
        patch_rmsnorm="rmsnorm" in sites,
        patch_attention="attention" in sites,
        patch_softmax="softmax" in sites,
    )
    try: yield
    finally: restore_fp32_accum_all(model, orig)


# ── Method registry: name → (context manager, kwargs) ─────────────────────────
# Used by experiments to iterate over all methods uniformly.

def make_methods(include_site_ablation: bool = False, critical_only: bool = False):
    """Return list of (name, ctx_factory) where ctx_factory(model) → CM.

    Args:
      include_site_ablation: also include single-site SRP-FP32 variants.
      critical_only: if True, "SRP-FP32" patches only critical sites
                     (linear+attention); RMSNorm/non-attn-Softmax skipped
                     since HF defaults are already batch-invariant.
    """
    main_sites = CRITICAL_SITES if critical_only else ALL_SITES
    main_label = "SRP-FP32" if not critical_only else "SRP-FP32-Critical"
    methods = [
        ("BF16",       lambda m: method_BF16(m)),
        ("LayerCast",  lambda m: method_LayerCast(m)),
        (main_label,   lambda m: method_SRP_FP32(m, main_sites)),
    ]
    if include_site_ablation:
        for site in ALL_SITES:
            label = f"SRP-FP32-{SITE_LABEL[site]}"
            methods.append((label, (lambda s: lambda m: method_SRP_FP32(m, (s,)))(site)))
    # FP32-all isn't included here because it requires a different model load.
    return methods
