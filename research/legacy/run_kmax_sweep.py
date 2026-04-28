"""
K_max sweep: empirical verification of the Corollary 1 bound.

For each K, build a synthetic BF16 GEMM with Llama-like scaling (N=K):
  A_M : [M, K]  ~ N(0, 1)              (post-LayerNorm activations)
  B   : [K, K]  ~ N(0, 1/K)             (Xavier-initialized weights)

Run F.linear at M=2 and M=32 (both within the GEMM code path so GEMV boundary
does not confound).  Measure:
  (a) max |C_M=2[0] - C_M=32[0]|   (l_inf row 0 deviation)
  (b) fraction of the 32 output cells where C_M=2[0,j] == C_M=32[0,j]

(b) is the quantity the Corollary controls: it predicts (b)->1 as K drops
below K_max.

Output: research/exp_kmax_sweep.json
"""
import os, sys, json
import torch

DLLM_DIR = os.path.dirname(os.path.abspath(__file__))
FP32_DIR = os.path.join(DLLM_DIR, '..', 'FP32')
sys.path.insert(0, DLLM_DIR)
sys.path.insert(0, FP32_DIR)

import determ_llm

OUT_FILE = os.path.join(DLLM_DIR, 'exp_kmax_sweep.json')

K_VALUES = [2048, 3072, 4096, 5120, 6144, 7168, 7680, 8192, 10240, 12288, 14336, 16384]
M_REF = 2     # both M values take the GEMM code path
M_LARGE = 32
N_TRIALS = 10

torch.manual_seed(0)
device = 'cuda:0'

results = {
    'meta': {
        'K_max_theoretical': 7700,
        'M_ref': M_REF, 'M_large': M_LARGE, 'n_trials': N_TRIALS,
        'dtype': 'bfloat16', 'gpu': torch.cuda.get_device_name(0),
        'note': 'N = K (square GEMM); zero_rate is fraction of output cells '
                'where row 0 matches bit-for-bit between M=2 and M=32.',
    },
    'sweep': [],
}

print(f"K_max sweep: K in {K_VALUES}, trials={N_TRIALS}")
print(f"{'K':>6} {'BF16 linf':>12} {'dllm linf':>12} {'BF16 zero%':>12} {'dllm zero%':>12}")
print('-' * 60)

for K in K_VALUES:
    N = K
    bf16_linfs = []
    dllm_linfs = []
    bf16_zero_rates = []
    dllm_zero_rates = []

    for trial in range(N_TRIALS):
        torch.manual_seed(1000 + trial)
        A = torch.randn(M_LARGE, K, device=device, dtype=torch.float32).to(torch.bfloat16)
        B = (torch.randn(N, K, device=device, dtype=torch.float32) / (K ** 0.5)).to(torch.bfloat16)

        # BF16 default
        determ_llm.disable()
        y_ref_bf16 = torch.nn.functional.linear(A[:M_REF], B)
        y_large_bf16 = torch.nn.functional.linear(A[:M_LARGE], B)
        d_bf16 = (y_ref_bf16[0] - y_large_bf16[0])
        bf16_linfs.append(d_bf16.abs().max().item())
        bf16_zero_rates.append((d_bf16 == 0).float().mean().item())

        # dllm (FP32 accum)
        determ_llm.enable()
        y_ref_dllm = torch.nn.functional.linear(A[:M_REF], B)
        y_large_dllm = torch.nn.functional.linear(A[:M_LARGE], B)
        d_dllm = (y_ref_dllm[0] - y_large_dllm[0])
        dllm_linfs.append(d_dllm.abs().max().item())
        dllm_zero_rates.append((d_dllm == 0).float().mean().item())
        determ_llm.disable()

    import statistics
    bf16_linf_mean = statistics.mean(bf16_linfs)
    dllm_linf_mean = statistics.mean(dllm_linfs)
    bf16_zero_mean = statistics.mean(bf16_zero_rates)
    dllm_zero_mean = statistics.mean(dllm_zero_rates)

    entry = {
        'K': K, 'N': N,
        'bf16_linf_mean': bf16_linf_mean,
        'dllm_linf_mean': dllm_linf_mean,
        'bf16_zero_rate_mean': bf16_zero_mean,
        'dllm_zero_rate_mean': dllm_zero_mean,
        'bf16_linf_trials': bf16_linfs,
        'dllm_linf_trials': dllm_linfs,
    }
    results['sweep'].append(entry)

    print(f"{K:>6} {bf16_linf_mean:>12.4e} {dllm_linf_mean:>12.4e}"
          f" {bf16_zero_mean*100:>11.1f}% {dllm_zero_mean*100:>11.1f}%")

with open(OUT_FILE, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved -> {OUT_FILE}")
