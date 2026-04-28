"""R3b: Triton fixed-tile matmul — batch-invariant GEMM."""
import torch
import triton
import triton.language as tl
import time

@triton.jit
def matmul_fixed_tile(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Fixed-tile matmul kernel. No dynamic tile selection."""
    pid = tl.program_id(0)
    grid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // grid_n
    pid_n = pid % grid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        mask_a = (offs_m[:, None] < M) & ((k_start + offs_k[None, :]) < K)
        mask_b = ((k_start + offs_k[:, None]) < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(tl.bfloat16)
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=mask_c)


def triton_linear(input, weight, BLOCK_M=64, BLOCK_N=64, BLOCK_K=64):
    """input: [M, K], weight: [N, K] -> [M, N]"""
    M, K = input.shape
    N = weight.shape[0]
    B_t = weight.t().contiguous()  # [K, N]
    C = torch.empty((M, N), device=input.device, dtype=input.dtype)

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    matmul_fixed_tile[grid](
        input, B_t, C,
        M, N, K,
        input.stride(0), input.stride(1),
        B_t.stride(0), B_t.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return C


if __name__ == '__main__':
    print('=== Triton Fixed-Tile Matmul: Batch Invariance ===')
    torch.manual_seed(42)
    W = torch.randn(4096, 4096, device='cuda', dtype=torch.bfloat16)
    x = torch.randn(1, 4096, device='cuda', dtype=torch.bfloat16)

    ref = triton_linear(x, W)

    all_match = True
    for M in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
        x_b = x.repeat(M, 1)
        out = triton_linear(x_b, W)
        d = (ref[0].float() - out[0].float()).abs().max().item()
        ok = 'OK' if d == 0 else f'DIFF={d:.4e}'
        if d != 0: all_match = False
        print(f'  M={M:>3}: {ok}')

    print(f'\nBatch-invariant: {all_match}')

    # Latency
    print('\n=== Latency vs cuBLAS ===')
    for M in [1, 8, 32, 128]:
        A = torch.randn(M, 4096, device='cuda', dtype=torch.bfloat16)
        for _ in range(10):
            triton_linear(A, W)
            torch.mm(A, W.t())
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(200):
            triton_linear(A, W)
        torch.cuda.synchronize()
        t_ms = (time.perf_counter() - t0) / 200 * 1000
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(200):
            torch.mm(A, W.t())
        torch.cuda.synchronize()
        c_ms = (time.perf_counter() - t0) / 200 * 1000
        print(f'  M={M:>3}: triton={t_ms:.3f}ms  cuBLAS={c_ms:.3f}ms  overhead={((t_ms/c_ms)-1)*100:+.1f}%')

    # K-dimension sweep
    print('\n=== K-dimension sweep ===')
    for K in [1024, 2048, 4096, 8192, 14336]:
        W_k = torch.randn(4096, K, device='cuda', dtype=torch.bfloat16)
        x_k = torch.randn(1, K, device='cuda', dtype=torch.bfloat16)
        ref_k = triton_linear(x_k, W_k)
        bi = True
        for M in [1, 32, 64, 128, 256]:
            out_k = triton_linear(x_k.repeat(M, 1), W_k)
            if (ref_k[0].float() - out_k[0].float()).abs().max().item() != 0:
                bi = False
                break
        print(f'  K={K:>5}: batch_invariant={bi}')
