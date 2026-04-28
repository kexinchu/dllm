"""True LayerCast: BF16 weights + FP32 activation pipeline (no cast-back).

Yuan et al.'s LayerCast paper intends: keep BF16 weights for memory,
upcast per matmul to FP32 for precision. But the standard implementation
casts the output back to BF16, which re-exposes rounding at every layer
and re-introduces the batch-dependent drift LayerCast was supposed to kill.

This module implements the "true" design:
  * F.linear:    BF16 weight -> FP32 cast, FP32 matmul, **FP32 output kept**
  * torch.matmul / bmm / __matmul__: FP32 upcast, **FP32 output kept**

With FP32 activations flowing through every layer, RMSNorm / softmax /
residuals all stay in FP32 and the rounding-driven batch-composition
drift is absorbed by FP32 precision. Weights remain BF16 in memory (~0.5x
memory vs full-FP32 model).
"""
import torch
import torch.nn.functional as F

_orig_linear = None
_orig_matmul = None
_orig_bmm = None
_orig_tensor_matmul = None
_orig_tensor_rmatmul = None
_on = False


def _lc_linear(input, weight, bias=None):
    """FP32 matmul, FP32 output. Works for BF16 or FP32 inputs/weights."""
    if input.is_cuda and input.dim() >= 2:
        x32 = input if input.dtype == torch.float32 else input.to(torch.float32)
        w32 = weight if weight.dtype == torch.float32 else weight.to(torch.float32)
        b32 = None if bias is None else (bias if bias.dtype == torch.float32 else bias.to(torch.float32))
        return _orig_linear(x32, w32, b32)
    return _orig_linear(input, weight, bias)


def _lc_matmul(input, other, *, out=None):
    if input.is_cuda and other.is_cuda and input.dim() >= 2 and other.dim() >= 2:
        x32 = input if input.dtype == torch.float32 else input.to(torch.float32)
        y32 = other if other.dtype == torch.float32 else other.to(torch.float32)
        r = _orig_matmul(x32, y32)
        if out is not None:
            out.copy_(r); return out
        return r
    if out is None:
        return _orig_matmul(input, other)
    return _orig_matmul(input, other, out=out)


def _lc_bmm(input, mat2, *, out=None):
    if input.is_cuda and mat2.is_cuda and input.dim() == 3 and mat2.dim() == 3:
        x32 = input if input.dtype == torch.float32 else input.to(torch.float32)
        y32 = mat2  if mat2.dtype  == torch.float32 else mat2.to(torch.float32)
        r = _orig_bmm(x32, y32)
        if out is not None:
            out.copy_(r); return out
        return r
    if out is None:
        return _orig_bmm(input, mat2)
    return _orig_bmm(input, mat2, out=out)


def _lc_tensor_matmul(self, other):
    return _lc_matmul(self, other)


def enable():
    global _orig_linear, _orig_matmul, _orig_bmm, _on
    global _orig_tensor_matmul, _orig_tensor_rmatmul
    if _on:
        return
    _orig_linear = F.linear
    _orig_matmul = torch.matmul
    _orig_bmm    = torch.bmm
    _orig_tensor_matmul  = torch.Tensor.matmul
    _orig_tensor_rmatmul = torch.Tensor.__matmul__
    F.linear     = _lc_linear
    torch.matmul = _lc_matmul
    torch.bmm    = _lc_bmm
    torch.Tensor.matmul     = _lc_tensor_matmul
    torch.Tensor.__matmul__ = _lc_tensor_matmul
    _on = True


def disable():
    global _orig_linear, _orig_matmul, _orig_bmm, _on
    global _orig_tensor_matmul, _orig_tensor_rmatmul
    if not _on:
        return
    F.linear     = _orig_linear
    torch.matmul = _orig_matmul
    torch.bmm    = _orig_bmm
    torch.Tensor.matmul     = _orig_tensor_matmul
    torch.Tensor.__matmul__ = _orig_tensor_rmatmul
    _on = False


def is_enabled():
    return _on
