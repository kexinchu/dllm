"""M1: GEMM batch variance across K dimensions, wide sweep."""
import torch, json, itertools
torch.manual_seed(42)

device = 'cuda'
OUT = '/home/kec23008/docker-sys/dllm/research/exp_m1_gemm_kdim.json'

# Comprehensive K and M sweep
K_dims = [512, 1024, 2048, 4096, 8192, 11008, 14336]
M_vals = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
N = 4096  # output dim fixed

results = []
for K in K_dims:
    W = torch.randn(N, K, device=device, dtype=torch.bfloat16)
    torch.manual_seed(42)
    x_single = torch.randn(1, K, device=device, dtype=torch.bfloat16)

    for mode, flag in [('BF16', True), ('FP32', False)]:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = flag

        ref = x_single @ W.T  # [1, N] — bs=1 reference

        for M in M_vals:
            x_batch = x_single.repeat(M, 1)  # all rows identical
            out = x_batch @ W.T  # [M, N]
            diff = (ref[0].float() - out[0].float()).abs()
            max_d = diff.max().item()
            mean_d = diff.mean().item()
            nonzero = (diff > 0).sum().item()
            results.append({
                'K': K, 'N': N, 'M': M, 'mode': mode,
                'max_diff': max_d, 'mean_diff': mean_d,
                'nonzero_frac': nonzero / N
            })

        # Also test: is bs=1 run-to-run deterministic?
        ref2 = x_single @ W.T
        r2r = (ref.float() - ref2.float()).abs().max().item()
        results.append({
            'K': K, 'N': N, 'M': 0, 'mode': mode,
            'max_diff': r2r, 'mean_diff': 0, 'nonzero_frac': 0,
            'note': 'run-to-run bs=1'
        })

# Print summary
print(f"{'K':>6} {'mode':<5} | ", end='')
for M in M_vals:
    print(f"M={M:<4}", end=' ')
print()
print('-' * (14 + 7 * len(M_vals)))

for K in K_dims:
    for mode in ['BF16', 'FP32']:
        print(f"{K:>6} {mode:<5} | ", end='')
        for M in M_vals:
            r = [x for x in results if x['K']==K and x['M']==M and x['mode']==mode]
            if r:
                d = r[0]['max_diff']
                print(f"{d:<6.3f}", end=' ')
        print()
    print()

with open(OUT, 'w') as f:
    json.dump(results, f, indent=2)
print(f"Saved {len(results)} records to {OUT}")
