"""M5: MoE near-tie + expert flip analysis on DeepSeek-V2-Lite."""
import torch, json, warnings, types
warnings.filterwarnings('ignore')
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = '/home/kec23008/docker-sys/Models/Llama-3.1-8B-Instruct'
# We simulate MoE routing using synthetic gate weights to avoid needing MoE model
# Also test with real DeepSeek-V2-Lite if available
OUT = '/home/kec23008/docker-sys/dllm/research/exp_m5_moe.json'

device = 'cuda'
results = {}

# ===== Part A: Synthetic MoE routing analysis =====
print("=== Part A: Synthetic MoE Routing ===")
# Simulate gate projection with realistic hidden dimensions
for n_experts, top_k, hidden_dim in [(64, 6, 2048), (128, 8, 4096), (8, 2, 4096)]:
    print(f"\n  Config: {n_experts} experts, top-{top_k}, hidden={hidden_dim}")

    gate_weight = torch.randn(n_experts, hidden_dim, device=device, dtype=torch.bfloat16) * 0.02
    n_tokens = 500

    for mode, flag in [('BF16', True), ('FP32', False)]:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = flag

        torch.manual_seed(42)
        hidden = torch.randn(n_tokens, hidden_dim, device=device, dtype=torch.bfloat16)

        # Reference: bs=1 (each token independently)
        ref_logits = hidden @ gate_weight.T  # [n_tokens, n_experts]
        ref_probs = torch.softmax(ref_logits.float(), dim=-1)
        ref_topk_vals, ref_topk_idx = torch.topk(ref_probs, top_k, dim=-1)

        # Near-tie analysis on reference
        sorted_probs, _ = torch.sort(ref_probs, dim=-1, descending=True)
        gap_k = sorted_probs[:, top_k-1] - sorted_probs[:, top_k]  # gap between k-th and (k+1)-th

        for tau in [0.1, 0.01, 0.001, 0.0001]:
            near_tie_frac = (gap_k < tau).float().mean().item()
            results.setdefault('near_tie', []).append({
                'config': f'{n_experts}e_top{top_k}_h{hidden_dim}',
                'mode': mode, 'tau': tau,
                'near_tie_frac': near_tie_frac
            })

        print(f"    {mode} near-tie (tau<0.001): {(gap_k < 0.001).float().mean().item():.2%}")

        # Batch variance: compare routing at different batch sizes
        for bs in [4, 8, 16, 32, 64]:
            hidden_batch = hidden[:bs].repeat(1, 1)  # just first bs tokens

            # bs=1 reference for these tokens
            ref_b = hidden_batch @ gate_weight.T
            ref_p = torch.softmax(ref_b.float(), dim=-1)
            _, ref_idx = torch.topk(ref_p, top_k, dim=-1)

            # Simulate batch effect: add tiny perturbation to gate logits
            # (simulating the GEMM batch variance of 0.5 ULP)
            perturb = torch.zeros_like(ref_b)
            # Add 0.5 ULP perturbation to random positions
            torch.manual_seed(bs)  # different perturbation per bs
            ulp = 2**-8  # BF16 ULP at scale ~1
            mask = torch.rand_like(ref_b) < 0.1  # 10% of elements
            perturb[mask] = ulp * 0.5 * (2 * torch.rand(mask.sum().item(), device=device) - 1).to(torch.bfloat16)

            pert_b = ref_b + perturb
            pert_p = torch.softmax(pert_b.float(), dim=-1)
            _, pert_idx = torch.topk(pert_p, top_k, dim=-1)

            # Count expert flips
            flips = (ref_idx != pert_idx).any(dim=-1).sum().item()
            flip_rate = flips / bs

            results.setdefault('expert_flips', []).append({
                'config': f'{n_experts}e_top{top_k}_h{hidden_dim}',
                'mode': mode, 'bs': bs,
                'flips': flips, 'flip_rate': flip_rate
            })

        print(f"    {mode} expert flips at bs=8: {[r for r in results['expert_flips'] if r['bs']==8 and r['mode']==mode and r['config']==f'{n_experts}e_top{top_k}_h{hidden_dim}'][0]['flip_rate']:.2%}")

# ===== Part B: Real model GEMM batch variance at gate projection =====
print("\n=== Part B: Real Gate GEMM Batch Variance (Llama as proxy) ===")
# Use Llama's linear layers as proxy for MoE gate
# Test: does batch size affect the gate logits?
tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map='auto')
model.eval()
dev = next(model.parameters()).device

# Extract a representative weight matrix (down_proj, K=14336 — largest K)
W = None
for name, mod in model.named_modules():
    if 'mlp.down_proj' in name and isinstance(mod, torch.nn.Linear):
        W = mod.weight.data.clone()  # [4096, 14336]
        print(f"  Using {name}: shape={W.shape}")
        break

if W is not None:
    K = W.shape[1]
    N = W.shape[0]
    torch.manual_seed(42)
    x = torch.randn(1, K, device=dev, dtype=torch.bfloat16)

    print(f"\n  Gate-proxy GEMM (K={K}, N={N}):")
    for mode, flag in [('BF16', True), ('FP32', False)]:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = flag
        ref = x @ W.T

        for M in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
            x_b = x.repeat(M, 1)
            out = x_b @ W.T
            diff = (ref[0].float() - out[0].float()).abs()
            max_d = diff.max().item()
            mean_d = diff.mean().item()
            results.setdefault('gate_gemm', []).append({
                'K': K, 'N': N, 'M': M, 'mode': mode,
                'max_diff': max_d, 'mean_diff': mean_d
            })

        diffs_str = [f"M={r['M']}:{r['max_diff']:.3f}" for r in results['gate_gemm']
                     if r['mode']==mode and r['K']==K]
        print(f"    {mode}: {', '.join(diffs_str[:6])}")

del model
torch.cuda.empty_cache()

# ===== Part C: Real DeepSeek-V2-Lite MoE routing =====
print("\n=== Part C: DeepSeek-V2-Lite Real MoE Routing ===")
try:
    from transformers import AutoModelForCausalLM as AMCLM
    moe_model = AMCLM.from_pretrained(
        'deepseek-ai/DeepSeek-V2-Lite', dtype=torch.bfloat16,
        device_map='auto', trust_remote_code=True
    )
    moe_model.eval()
    moe_tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-V2-Lite', trust_remote_code=True)

    prompt = "Explain deterministic inference."
    inp = moe_tokenizer(prompt, return_tensors='pt').to(dev)
    sl = inp['input_ids'].shape[1]

    # Hook first MoE gate to capture routing
    gate_outputs = {}
    def make_gate_hook(name, storage):
        def hook(module, inp, out):
            if isinstance(out, tuple):
                storage[name] = [x.detach().clone() if isinstance(x, torch.Tensor) else x for x in out]
            else:
                storage[name] = out.detach().clone()
        return hook

    # Find gate modules
    hooked = False
    for name, mod in moe_model.named_modules():
        if 'gate' in name.lower() and hasattr(mod, 'weight'):
            mod.register_forward_hook(make_gate_hook(name, gate_outputs))
            hooked = True
            break

    if hooked:
        for mode, flag in [('BF16', True), ('FP32', False)]:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = flag
            gate_outputs.clear()

            with torch.no_grad():
                moe_model(**inp)

            if gate_outputs:
                first_key = list(gate_outputs.keys())[0]
                gate_out = gate_outputs[first_key]
                if isinstance(gate_out, list):
                    gate_out = gate_out[0]
                print(f"  {mode}: gate output shape={gate_out.shape}")

    del moe_model
    torch.cuda.empty_cache()
    print("  DeepSeek-V2-Lite analysis done")
except Exception as e:
    print(f"  DeepSeek-V2-Lite not available: {e}")
    results['deepseek_error'] = str(e)

with open(OUT, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {OUT}")
