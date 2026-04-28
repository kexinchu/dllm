"""Run reduction tests: python -m FP32.run_tests from repo root.

本测试是数值/行为模拟（PyTorch 层面）。方案 A 的 GEMM kernel 见 gemm_fp32_accum.py（Triton + PyTorch fallback）。
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from FP32.reduction_ops import reduce_bf16_naive, reduce_bf16_atomic_style, reduce_fp32_then_bf16, reduce_deterministic_sequential
import torch
import time
import json

try:
    from FP32.gemm_fp32_accum import matmul_fp32_accum, linear_fp32_accum
    GEMM_FP32_AVAILABLE = True
except Exception:
    GEMM_FP32_AVAILABLE = False

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    M, K = 32, 4096
    torch.manual_seed(42)
    x = torch.randn(M, K, device=device, dtype=torch.float32).to(torch.bfloat16)
    n = 500
    results = {}
    for name, fn in [("bf16_naive", lambda: reduce_bf16_naive(x.clone(), -1)), ("bf16_atomic", lambda: reduce_bf16_atomic_style(x.clone(), -1, seed=0)), ("fp32_then_bf16", lambda: reduce_fp32_then_bf16(x.clone(), -1)), ("deterministic", lambda: reduce_deterministic_sequential(x.clone(), -1))]:
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        results[name + "_ms"] = (time.perf_counter() - t0) / n * 1000
    print("Latency (ms):", results)
    runs = 20
    bf16 = torch.stack([reduce_bf16_atomic_style(x.clone(), -1, num_splits=32, seed=r) for r in range(runs)], 0)
    fp32 = torch.stack([reduce_fp32_then_bf16(x.clone(), -1) for _ in range(runs)], 0)
    det = torch.stack([reduce_deterministic_sequential(x.clone(), -1) for _ in range(runs)], 0)
    print("Determinism std_mean: bf16", bf16.float().std(0).mean().item(), "fp32", fp32.float().std(0).mean().item(), "det", det.float().std(0).mean().item())
    row = torch.randn(K, device=device, dtype=torch.float32).to(torch.bfloat16)
    ref_fp32 = reduce_fp32_then_bf16(row.unsqueeze(0), -1).squeeze(0)
    ref_bf16 = reduce_bf16_atomic_style(row.unsqueeze(0), -1, seed=0).squeeze(0)
    for M in (1, 8, 32):
        xx = torch.cat([row.unsqueeze(0)] + [torch.randn(1, K, device=device, dtype=torch.bfloat16) for _ in range(M - 1)], 0)
        d_fp32 = torch.abs(reduce_fp32_then_bf16(xx, -1)[0] - ref_fp32).max().item()
        d_bf16 = torch.abs(reduce_bf16_atomic_style(xx, -1, seed=0)[0] - ref_bf16).max().item()
        print("Batch M=%d: fp32_then_bf16 max_diff=%.2e, bf16_atomic max_diff=%.2e" % (M, d_fp32, d_bf16))
    out = {
        "latency_ms": results,
        "determinism_bf16_std": float(bf16.float().std(0).mean().item()),
        "determinism_fp32_std": float(fp32.float().std(0).mean().item()),
        "determinism_det_std": float(det.float().std(0).mean().item()),
    }

    # 方案 A：GEMM kernel（FP32 规约 -> BF16）正确性 + latency（仅 CUDA 下运行）
    if not GEMM_FP32_AVAILABLE:
        print("(GEMM kernel skipped: import failed)")
    elif device.type != "cuda":
        print("(GEMM kernel skipped: no CUDA)")
    if GEMM_FP32_AVAILABLE and device.type == "cuda":
        print()
        print("=== GEMM kernel (Option A: FP32 accum -> BF16) ===")
        try:
            from FP32.gemm_fp32_accum import _CUBLASLT_EXTENSION
            backend = "cuBLASLt (fused)" if _CUBLASLT_EXTENSION is not None else "Triton/PyTorch"
            print("  Backend:", backend)
        except Exception:
            print("  Backend: Triton/PyTorch")
        torch.manual_seed(123)
        Mm, Km, Nm = 128, 1024, 512
        A = torch.randn(Mm, Km, device=device, dtype=torch.bfloat16)
        B = torch.randn(Km, Nm, device=device, dtype=torch.bfloat16)
        ours = matmul_fp32_accum(A, B)
        ref = (A.float() @ B.float()).to(torch.bfloat16)
        max_diff = (ours.float() - ref.float()).abs().max().item()
        print("  Correctness vs (A.float()@B.float()).to(bf16): max_diff = %.2e" % max_diff)
        n = 200
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            matmul_fp32_accum(A, B)
        if device.type == "cuda":
            torch.cuda.synchronize()
        gemm_ms = (time.perf_counter() - t0) / n * 1000
        t0 = time.perf_counter()
        for _ in range(n):
            torch.nn.functional.linear(A, B.T)
        if device.type == "cuda":
            torch.cuda.synchronize()
        bf16_linear_ms = (time.perf_counter() - t0) / n * 1000
        print("  Latency: gemm_fp32_accum %.4f ms, torch linear (bf16) %.4f ms" % (gemm_ms, bf16_linear_ms))
        out["gemm_fp32_accum_ms"] = gemm_ms
        out["gemm_bf16_linear_ms"] = bf16_linear_ms
        out["gemm_correctness_max_diff"] = max_diff
        # GEMM determinism: same A,B -> same output every run?
        n_runs = 20
        gemm_outputs = []
        for _ in range(n_runs):
            gemm_outputs.append(matmul_fp32_accum(A, B))
        gemm_stack = torch.stack([o.float() for o in gemm_outputs], 0)
        gemm_std = gemm_stack.std(0).mean().item()
        out["gemm_deterministic_std_mean"] = gemm_std
        print("  Determinism: %d runs, std_mean = %.2e (0 => deterministic)" % (n_runs, gemm_std))

    with open(os.path.join(os.path.dirname(__file__), "reduction_fp32_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print()
    print("Results saved to FP32/reduction_fp32_results.json")

if __name__ == "__main__":
    main()
