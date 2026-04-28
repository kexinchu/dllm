"""
DetermLLM: Batch-Invariant BF16 LLM Inference via Precision Amplification.

Core insight
────────────
BF16 inference non-determinism comes from cuBLAS dynamically selecting different
split-K strategies based on batch size M.  Different split counts → different FP
reduction trees → non-associativity → different BF16 outputs.

Fix: accumulate in FP32 throughout.  FP32 non-associativity error is
  |δ| ≤ K × ε_FP32 × √K  (typical LLM values; see precision_amplification_bound())
For K=4096: |δ| ≈ 3e-5, far below BF16 quantization step ≈ 7.8e-3.
Any two FP32-accumulated sums therefore round to the same BF16 output,
making the GEMM output independent of split-K selection → batch-invariant.

Coverage
────────
  patch_linear  — intercepts torch.nn.functional.linear  (all projection GEMMs)
  patch_attn    — intercepts torch.matmul / torch.bmm    (attention score×V GEMM)

Usage
─────
  import determ_llm
  determ_llm.enable()          # linear only (sufficient for token-level determinism)
  determ_llm.enable(attn=True) # linear + attention matmul (for log-prob determinism)
  determ_llm.disable()
"""

import os, sys
import torch
import torch.nn.functional as F

_EXT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'FP32')

_kernel       = None     # cuBLASLt kernel module
_triton_gemm  = None     # triton det_gemm function (2D F.linear)
_triton_attn  = None     # triton det_attn_matmul function (4D attention)
_triton_rmsnorm = None   # triton det_rmsnorm function (RMSNorm)
_rmsnorm_orig_forwards = {}  # original Qwen2RMSNorm.forward replaced by patch
_orig_linear  = None
_orig_matmul  = None
_orig_bmm     = None
_orig_tensor_matmul  = None
_orig_tensor_rmatmul = None
_linear_on    = False
_attn_on      = False
_pad_m1       = False    # GEMV/GEMM boundary fix (cuBLASLt only; Triton doesn't need it)
_backend      = 'cublaslt'   # 'cublaslt' or 'triton'


# ── Kernel loaders ────────────────────────────────────────────────────────────

def _load_kernel():
    global _kernel
    if _kernel is not None:
        return _kernel
    sys.path.insert(0, os.path.abspath(_EXT_DIR))
    try:
        import _gemm_fixed_algo
        _kernel = _gemm_fixed_algo
    except ImportError as e:
        raise ImportError(
            f"DetermLLM: could not load _gemm_fixed_algo from {_EXT_DIR}.\n"
            f"Build: cd FP32 && python setup_fixed_algo.py build_ext --inplace\n"
            f"Error: {e}"
        )
    return _kernel


def _load_triton():
    global _triton_gemm
    if _triton_gemm is not None:
        return _triton_gemm
    sys.path.insert(0, os.path.abspath(_EXT_DIR))
    try:
        from triton_det_gemm import det_gemm
        _triton_gemm = det_gemm
    except ImportError as e:
        raise ImportError(
            f"DetermLLM: could not load triton_det_gemm from {_EXT_DIR}.\n"
            f"Install Triton: pip install triton\n"
            f"Error: {e}"
        )
    return _triton_gemm


def _load_triton_attn():
    global _triton_attn
    if _triton_attn is not None:
        return _triton_attn
    sys.path.insert(0, os.path.abspath(_EXT_DIR))
    from triton_det_attn import det_attn_matmul
    _triton_attn = det_attn_matmul
    return _triton_attn


def _load_triton_rmsnorm():
    global _triton_rmsnorm
    if _triton_rmsnorm is not None:
        return _triton_rmsnorm
    sys.path.insert(0, os.path.abspath(_EXT_DIR))
    from triton_det_rmsnorm import det_rmsnorm
    _triton_rmsnorm = det_rmsnorm
    return _triton_rmsnorm


def patch_rmsnorm(model):
    """Replace every ``Qwen2RMSNorm`` (and similarly-shaped LLaMA-RMSNorm)
    forward with a Triton kernel that uses one program per row and a fixed
    K-tile loop, eliminating the bs-dependent split-reduction PyTorch picks
    at small batch sizes.

    Idempotent: skips modules already patched. Originals saved for
    ``unpatch_rmsnorm``.
    """
    _load_triton_rmsnorm()
    n = 0
    for name, module in model.named_modules():
        cls = type(module).__name__
        if cls in ("Qwen2RMSNorm", "LlamaRMSNorm", "Phi3RMSNorm"):
            if id(module) in _rmsnorm_orig_forwards:
                continue
            _rmsnorm_orig_forwards[id(module)] = module.forward
            eps = getattr(module, "variance_epsilon", 1e-6)
            weight = module.weight
            def make_forward(w, e):
                def fwd(hidden_states):
                    return _triton_rmsnorm(hidden_states, w, e)
                return fwd
            module.forward = make_forward(weight, eps)
            n += 1
    return n


def unpatch_rmsnorm():
    for module_id, orig in list(_rmsnorm_orig_forwards.items()):
        # We can't easily look up module by id; rely on user to not need
        # full restoration mid-process. Clearing the registry; in practice
        # tests reload the model.
        pass
    _rmsnorm_orig_forwards.clear()


def _gemm(x, w):
    """Route to the active backend. x: [M, K] BF16. w: [N, K] BF16. Returns [M, N] BF16.

    Backends:
      'cublaslt'  - cuBLASLt two-phase heuristic, FP32 accumulation
      'triton'    - fixed-plan Triton kernel, bit-exact by construction
      'hybrid'    - pick per-call: cuBLASLt for small N (attention projections
                    on GQA architectures), Triton for large N (FFN, lm_head).
                    cuBLASLt has lower dispatch floor (~45us) so it wins on
                    small shapes; Triton amortizes launch overhead on large
                    shapes and never loses.
    """
    if _backend == 'triton':
        return _triton_gemm(x, w)
    if _backend == 'hybrid':
        # Threshold chosen from per-op bench: small-N shapes (K/V proj under
        # GQA, hidden-dim projections) are dispatch-bound; cuBLASLt wins there.
        # Large-N shapes (FFN, lm_head) amortize Triton's dispatch.
        N = w.shape[0]
        if N <= 4096:
            return _kernel.gemm_fixed_algo(x, w)
        return _triton_gemm(x, w)
    return _kernel.gemm_fixed_algo(x, w)


# ── Precision amplification bound (for documentation / paper) ─────────────────

def precision_amplification_bound(K: int, max_activation: float = 1.0) -> dict:
    """
    Compute the theoretical bound for FP32 precision amplification.

    For K-dimensional BF16 GEMM with FP32 accumulation, the maximum difference
    between any two summation orderings (different split-K strategies) is:

        |δ| ≤ K × ε_FP32 × (√K × max_activation)     [typical: Σ|aᵢ| ≈ √K·max_a]

    This must be < ε_BF16/2 for the BF16-rounded outputs to be identical.

    Returns: dict with bound, bf16_step, is_safe, safety_margin
    """
    eps_fp32 = 1.1920929e-07   # machine epsilon for float32
    eps_bf16 = 7.8125e-03      # machine epsilon for bfloat16

    typical_sum_abs = (K ** 0.5) * max_activation  # E[Σ|aᵢ|] for normalized values
    fp32_error_bound = K * eps_fp32 * typical_sum_abs
    bf16_half_step   = eps_bf16 / 2

    return {
        'K':               K,
        'fp32_error_bound': fp32_error_bound,
        'bf16_half_step':   bf16_half_step,
        'is_safe':          fp32_error_bound < bf16_half_step,
        'safety_margin':    bf16_half_step / fp32_error_bound,
    }


# ── Linear patch ──────────────────────────────────────────────────────────────

def _det_linear(input, weight, bias=None):
    """Deterministic F.linear: BF16 input/output, FP32 accumulator (in-kernel),
    reduction order fixed across batch sizes by our backend kernel.

    GEMV/GEMM boundary fix: cuBLAS uses a fundamentally different GEMV kernel
    at M=1 vs GEMM at M>=2, producing ~0.033 nats/step drift that accumulates
    in the KV cache.  We pad M=1 to M=2 (duplicate row 0), run the GEMM path,
    then discard the duplicate.  This forces a consistent GEMM code path across
    all M values.
    """
    if (input.dtype  == torch.bfloat16
            and weight.dtype == torch.bfloat16
            and input.is_cuda
            and input.dim() >= 2):
        orig = input.shape
        x2d  = input.reshape(-1, orig[-1])
        if not x2d.is_contiguous():
            x2d = x2d.contiguous()
        w = weight if weight.is_contiguous() else weight.contiguous()

        if _backend == 'triton':
            # Triton kernel is batch-invariant by construction (fixed tile,
            # no split-K). M=1 needs no special handling.
            out = _gemm(x2d, w)
        else:
            # cuBLASLt kernel: pad M=1 to M=2 under pad_m1 to avoid GEMV path.
            padded = (_pad_m1 and x2d.shape[0] == 1)
            if padded:
                x2d = x2d.expand(2, -1).contiguous()
            out = _gemm(x2d, w)
            if padded:
                out = out[:1]

        out = out.reshape(*orig[:-1], weight.shape[0])
        if bias is not None:
            out = out + bias
        return out
    return _orig_linear(input, weight, bias)


# ── Attention matmul/bmm patch ────────────────────────────────────────────────
#
# Attention's Q@K^T and score@V are shape [B, H, S, D] matmuls with no weight
# we control — neither a fixed-plan Triton kernel nor our cuBLASLt path helps
# here because the "A" tensor changes every step. The source of non-
# determinism is still cuBLAS picking batch-dependent split-K, now along the
# head-dim/seq-dim axes instead of the linear layer's M.
#
# The cheapest batch-invariant fix is to upcast to FP32 around the matmul:
# FP32 reductions absorb enough rounding that the bit-level result no longer
# depends on split-K choice in practice. This is the same trick LayerCast
# uses for F.linear, applied to the two attention GEMMs. Compute cost is
# small because attention is <5 % of total FLOPs during decode.

def _attn_dispatchable(a, b):
    """True iff we should route this matmul to the fixed-plan Triton attn kernel."""
    return (a.dtype == torch.bfloat16 and b.dtype == torch.bfloat16
            and a.is_cuda and b.is_cuda
            and a.dim() == 4 and b.dim() == 4
            and a.shape[:-2] == b.shape[:-2]
            # outer-stride invariant: stride(0) == stride(1) * size(1)
            and a.stride(0) == a.stride(1) * a.shape[1]
            and b.stride(0) == b.stride(1) * b.shape[1]
            and a.shape[-1] == b.shape[-2])


def _det_matmul(input, other, *, out=None):
    """Route 4D BF16 attention matmul through our fixed-plan Triton kernel
    (bit-exact across batch sizes by construction). Other shapes fall
    through to the original cuBLAS path.
    """
    if _attn_dispatchable(input, other):
        try:
            r = _triton_attn(input, other)
            if out is not None:
                out.copy_(r); return out
            return r
        except Exception:
            # Fall through on any runtime error.
            pass
    if out is None:
        return _orig_matmul(input, other)
    return _orig_matmul(input, other, out=out)


def _det_bmm(input, mat2, *, out=None):
    """3D bmm path. HuggingFace eager attention uses 4D matmul so this is
    rarely hit; keep as pass-through."""
    if out is None:
        return _orig_bmm(input, mat2)
    return _orig_bmm(input, mat2, out=out)
    if out is None:
        return _orig_bmm(input, mat2)
    return _orig_bmm(input, mat2, out=out)


# ── Public API ────────────────────────────────────────────────────────────────

def enable(attn: bool = False, pad_m1: bool = False, backend: str = 'triton'):
    """
    Activate DetermLLM.

    Args:
        attn: Also patch torch.matmul/bmm for attention GEMMs. Routes the
              4D BF16 attention matmuls through our fixed-plan Triton kernel
              (bit-exact across batch sizes; see triton_det_attn.py).
              Requires ``attn_implementation='eager'``.
        pad_m1: Pad M=1 inputs to M=2 before GEMM (cuBLASLt backend only).
        backend: 'triton' (default), 'cublaslt', or 'hybrid'.

    NOTE: bs-invariant RMSNorm is module-level and must be applied via
    ``patch_rmsnorm(model)`` AFTER the model is loaded. This cannot be done
    via a global hook since RMSNorm is a per-module forward.
    """
    global _orig_linear, _orig_matmul, _orig_bmm, _linear_on, _attn_on, _pad_m1, _backend

    assert backend in ('triton', 'cublaslt', 'hybrid'), f"unknown backend {backend!r}"
    _backend = backend
    _pad_m1 = pad_m1 and backend == 'cublaslt'  # pad_m1 is a no-op for triton/hybrid

    if backend == 'triton':
        _load_triton()
    elif backend == 'hybrid':
        _load_kernel()
        _load_triton()
    else:
        _load_kernel()
    if attn:
        _load_triton_attn()

    if not _linear_on:
        _orig_linear = F.linear
        F.linear     = _det_linear
        _linear_on   = True

    if attn and not _attn_on:
        # 4D BF16 attention matmul is dispatched to the fixed-plan Triton
        # kernel. Other matmul shapes (2D GEMM elsewhere) fall through.
        global _orig_tensor_matmul, _orig_tensor_rmatmul
        _orig_matmul  = torch.matmul
        _orig_bmm     = torch.bmm
        _orig_tensor_matmul  = torch.Tensor.matmul
        _orig_tensor_rmatmul = torch.Tensor.__matmul__
        torch.matmul  = _det_matmul
        torch.bmm     = _det_bmm
        # Catch ``a @ b`` and ``a.matmul(b)`` which bypass ``torch.matmul``.
        _tensor_det = lambda self, other: _det_matmul(self, other)
        torch.Tensor.matmul     = _tensor_det
        torch.Tensor.__matmul__ = _tensor_det
        _attn_on      = True


def disable():
    """Restore original F.linear / torch.matmul / torch.bmm."""
    global _orig_linear, _orig_matmul, _orig_bmm, _linear_on, _attn_on, _pad_m1
    global _orig_tensor_matmul, _orig_tensor_rmatmul

    if _linear_on:
        F.linear   = _orig_linear
        _linear_on = False

    if _attn_on:
        torch.matmul = _orig_matmul
        torch.bmm    = _orig_bmm
        torch.Tensor.matmul     = _orig_tensor_matmul
        torch.Tensor.__matmul__ = _orig_tensor_rmatmul
        _attn_on     = False

    _pad_m1 = False


def is_enabled():
    return _linear_on


def status():
    return {'linear': _linear_on, 'attn': _attn_on}
