"""Verify Theorem 1 condition number bound on real Llama-3.1-8B layers."""
import sys, json, torch, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/kec23008/docker-sys/dllm')
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = '/home/kec23008/docker-sys/Models/Llama-3.1-8B-Instruct'
OUT = '/home/kec23008/docker-sys/dllm/research/exp_theorem1_kappa.json'

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map='auto')
model.eval()
device = next(model.parameters()).device

PROMPT = "What is deterministic inference in large language models and why does it matter for production systems?"
inputs = tokenizer(PROMPT, return_tensors='pt').to(device)

# Hook all linear layers to capture inputs
layer_data = {}
hooks = []

def make_hook(name):
    def hook_fn(module, inp, out):
        x = inp[0].detach()  # [bs, seq, K]
        w = module.weight.detach()  # [N, K]
        layer_data[name] = (x, w, module.weight.shape)
    return hook_fn

for name, mod in model.named_modules():
    if isinstance(mod, torch.nn.Linear):
        hooks.append(mod.register_forward_hook(make_hook(name)))

print("Running forward pass...")
with torch.no_grad():
    model(**inputs)

for h in hooks:
    h.remove()

# Compute kappa for sampled output elements
results = []
N_SAMPLES = 100  # sample output elements per layer

print(f"\nComputing kappa(S) for {len(layer_data)} linear layers, {N_SAMPLES} samples each...\n")
print(f"{'Layer':<55} {'K':>5} {'N':>6} {'mean_k':>10} {'med_k':>10} {'p99_k':>10} {'max_k':>10} {'bound':>8} {'safe':>5}")
print("-" * 130)

for name, (x, w, shape) in sorted(layer_data.items()):
    N_out, K = shape
    # x: [1, seq, K], w: [N_out, K]
    # For each sampled output element (i,j): y[i,j] = sum_k W[j,k] * x[i,k]
    x_2d = x.reshape(-1, K)  # [tokens, K]
    n_tokens = x_2d.shape[0]

    # Sample random (token, output_dim) pairs
    torch.manual_seed(42)
    sample_tokens = torch.randint(0, n_tokens, (N_SAMPLES,))
    sample_dims = torch.randint(0, N_out, (N_SAMPLES,))

    kappas = []
    for si in range(N_SAMPLES):
        ti, di = sample_tokens[si].item(), sample_dims[si].item()
        # Element-wise products a_k = W[di, k] * x[ti, k] — compute in FP64
        a = (w[di].double() * x_2d[ti].double())  # [K] in FP64
        sum_abs = a.abs().sum().item()
        abs_sum = a.sum().abs().item()
        if abs_sum > 1e-30:
            kappa = sum_abs / abs_sum
        else:
            kappa = float('inf')  # degenerate case
        kappas.append(kappa)

    kappas = np.array(kappas)
    finite_kappas = kappas[np.isfinite(kappas)]
    if len(finite_kappas) == 0:
        continue

    # Theoretical bound: kappa < 2^15 / (K-1)
    bound = 2**15 / (K - 1)
    mean_k = np.mean(finite_kappas)
    med_k = np.median(finite_kappas)
    p99_k = np.percentile(finite_kappas, 99)
    max_k = np.max(finite_kappas)
    safe = "YES" if max_k < bound else "NO"

    short_name = name if len(name) < 55 else "..." + name[-52:]
    print(f"{short_name:<55} {K:>5} {N_out:>6} {mean_k:>10.2f} {med_k:>10.2f} {p99_k:>10.2f} {max_k:>10.2f} {bound:>8.2f} {safe:>5}")

    results.append({
        "layer": name, "K": K, "N": N_out,
        "mean_kappa": float(mean_k), "median_kappa": float(med_k),
        "p99_kappa": float(p99_k), "max_kappa": float(max_k),
        "n_inf": int(np.sum(~np.isfinite(kappas))),
        "theoretical_bound": float(bound),
        "satisfies_bound": bool(max_k < bound),
        "margin": float(bound / max_k) if max_k > 0 else float('inf')
    })

# Summary
n_total = len(results)
n_safe = sum(1 for r in results if r['satisfies_bound'])
print(f"\n{'='*80}")
print(f"SUMMARY: {n_safe}/{n_total} layers satisfy Theorem 1 bound")
print(f"Layers that VIOLATE bound:")
for r in results:
    if not r['satisfies_bound']:
        print(f"  {r['layer']}: max_kappa={r['max_kappa']:.2f} > bound={r['theoretical_bound']:.2f}")

with open(OUT, 'w') as f:
    json.dump({"layers": results, "n_total": n_total, "n_safe": n_safe,
               "summary": f"{n_safe}/{n_total} layers satisfy bound"}, f, indent=2)
print(f"\nSaved to {OUT}")
