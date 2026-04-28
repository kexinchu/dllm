"""
LayerCast baseline (Yuan et al. NeurIPS 2025).

Weights are stored in BF16 for memory efficiency, but every linear layer
upcasts weights and inputs to FP32 at forward time and runs matmul in FP32.
This gives FP32-like numerical stability at BF16 memory cost.

Their original implementation is a vLLM patch. We replicate the same math
as an F.linear patch so it composes with the rest of our evaluation harness.

Contract match (as described in Yuan et al. Section 3):
  - Weights stay BF16 in memory (as model was loaded).
  - Input activation is cast BF16 -> FP32 before matmul.
  - Weight is cast BF16 -> FP32 before matmul.
  - torch.nn.functional.linear runs in FP32 (compute and output).
  - Output is cast back to BF16 before returning.

Note (documented in Appendix): patching F.linear (or F.linear + torch.matmul
with an FP32 upcast) does NOT yield bit-exact output across batch sizes in
stress tests (bs=1 vs 32, 1024-token generation): cuBLAS's FP32 GEMM still
picks batch-dependent split-K at many matrix shapes. This is a limitation of
the LayerCast design, not an implementation bug — see
research/diagnose_triton_bs_invariance.py and
research/diagnose_fp32_matmul_bs_invariance.py for the empirical evidence.
"""
import torch
import torch.nn.functional as F

_orig_linear = None
_on = False


def _layercast_linear(input, weight, bias=None):
    """Upcast weight and input to FP32, compute in FP32, cast result back."""
    if (input.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16
            and input.is_cuda and input.dim() >= 2):
        x_fp32 = input.to(torch.float32)
        w_fp32 = weight.to(torch.float32)
        b_fp32 = bias.to(torch.float32) if bias is not None else None
        out_fp32 = _orig_linear(x_fp32, w_fp32, b_fp32)
        return out_fp32.to(torch.bfloat16)
    return _orig_linear(input, weight, bias)


def enable():
    global _orig_linear, _on
    if _on:
        return
    _orig_linear = F.linear
    F.linear = _layercast_linear
    _on = True


def disable():
    global _orig_linear, _on
    if not _on:
        return
    F.linear = _orig_linear
    _on = False


def is_enabled():
    return _on
