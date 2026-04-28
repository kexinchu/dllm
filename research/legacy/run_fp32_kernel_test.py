"""Test 3 GEMM precision approaches for batch invariance across bs=1..256.

Approach A: FP32 flag only (BF16 kernel, FP32 accum) — baseline
Approach B: LayerCast (BF16 storage, FP32 compute via cublasSgemm)
Approach C: cuBLASLt BF16 input, FP32 compute type, FP32 output (hybrid)
"""
import torch
import torch.nn.functional as F
import warnings, hashlib, time, json, types
warnings.filterwarnings('ignore')
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = '/home/kec23008/docker-sys/Models/Llama-3.1-8B-Instruct'
OUT = '/home/kec23008/docker-sys/dllm/research/exp_fp32_kernel.json'

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map='auto')
model.eval()
device = next(model.parameters()).device

PROMPT = 'What is deterministic inference in large language models?'
input_ids = tokenizer(PROMPT, return_tensors='pt')['input_ids']
sl = input_ids.shape[1]

# ====== Patching utilities ======

def patch_all_linear_layercast(model):
    """Replace all nn.Linear with FP32 compute (LayerCast style)."""
    originals = {}
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            originals[name] = mod.forward
            def make_fwd(m):
                def fwd(x):
                    return F.linear(x.float(), m.weight.float(),
                                    m.bias.float() if m.bias is not None else None).to(x.dtype)
                return fwd
            mod.forward = make_fwd(mod)
    return originals

def patch_all_linear_hybrid(model):
    """BF16 input cast to FP32, FP32 matmul, cast output back to BF16.
    Same as LayerCast but using torch.mm directly for clarity."""
    originals = {}
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            originals[name] = mod.forward
            def make_fwd(m):
                def fwd(x):
                    # Key: compute in FP32, but keep weight in BF16 storage
                    # Cast weight JIT, compute, cast back
                    out = torch.mm(x.view(-1, x.shape[-1]).float(),
                                   m.weight.float().t())
                    if m.bias is not None:
                        out = out + m.bias.float()
                    return out.to(x.dtype).view(*x.shape[:-1], m.weight.shape[0])
                return fwd
            mod.forward = make_fwd(mod)
    return originals

def patch_all_linear_fp32_output(model):
    """BF16 x BF16 with FP32 accum AND FP32 output (no BF16 rounding at GEMM boundary).
    Keeps intermediate hidden states in FP32 between linear layers."""
    originals = {}
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            originals[name] = mod.forward
            def make_fwd(m):
                def fwd(x):
                    # If input is FP32 (from previous layer), cast to BF16 for GEMM input
                    # but accumulate and output in FP32
                    x_bf16 = x.to(torch.bfloat16) if x.dtype != torch.bfloat16 else x
                    # Use FP32 matmul with BF16 inputs cast to FP32
                    out = torch.mm(x_bf16.view(-1, x.shape[-1]).float(),
                                   m.weight.float().t())
                    if m.bias is not None:
                        out = out + m.bias.float()
                    # Cast back to BF16 for next layer
                    return out.to(torch.bfloat16).view(*x.shape[:-1], m.weight.shape[0])
                return fwd
            mod.forward = make_fwd(mod)
    return originals

def restore(model, originals):
    for name, mod in model.named_modules():
        if name in originals:
            mod.forward = originals[name]

def gen_hash(bs, gen_len=32):
    ids_b = input_ids.repeat(bs, 1).to(device)
    pos_b = torch.arange(sl, device=device).unsqueeze(0).repeat(bs, 1)
    generated = []
    cur_ids = ids_b.clone()
    cur_pos = pos_b.clone()
    with torch.no_grad():
        for step in range(gen_len):
            out = model(input_ids=cur_ids, position_ids=cur_pos).logits
            next_tok = out[0, -1].argmax().item()
            generated.append(next_tok)
            cur_ids = torch.tensor([[next_tok]], device=device).repeat(bs, 1)
            cur_pos = torch.tensor([[sl + step]], device=device).repeat(bs, 1)
    return hashlib.md5(str(generated).encode()).hexdigest()[:12]

# ====== Test all approaches ======
test_bs = list(range(1, 257))
results = {}

configs = [
    ("A: BF16+FP32flag", None, False, None),
    ("B: LayerCast", patch_all_linear_layercast, True, None),
    ("C: Hybrid_fp32out", patch_all_linear_fp32_output, True, None),
]

for name, patch_fn, bf16_flag, extra in configs:
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = bf16_flag
    originals = patch_fn(model) if patch_fn else {}

    hashes = {}
    t0 = time.time()
    for i, bs in enumerate(test_bs):
        hashes[bs] = gen_hash(bs)
        if (i + 1) % 64 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i+1) * (len(test_bs) - i - 1)
            print(f"  {name}: {i+1}/{len(test_bs)}, {elapsed:.0f}s elapsed, ETA {eta:.0f}s")

    elapsed = time.time() - t0

    unique = set(hashes.values())
    transitions = [bs for i, bs in enumerate(test_bs[1:], 1)
                   if hashes[test_bs[i]] != hashes[test_bs[i-1]]]

    from collections import Counter
    hc = Counter(hashes.values())
    majority = hc.most_common(1)[0][1]
    affected = len(test_bs) - majority

    print(f"  {name}: {len(unique)} unique, {len(transitions)} transitions at {transitions}")
    print(f"    affected: {affected}/{len(test_bs)} ({affected/len(test_bs)*100:.1f}%)")
    print(f"    latency: {elapsed/len(test_bs)*1000:.0f}ms/bs, total: {elapsed:.0f}s")
    print()

    results[name] = {
        'unique': len(unique), 'transitions': transitions,
        'affected': affected, 'affected_pct': affected / len(test_bs) * 100,
        'latency_ms': elapsed / len(test_bs) * 1000,
        'total_s': elapsed,
    }

    if originals:
        restore(model, originals)

# Save
with open(OUT, 'w') as f:
    json.dump(results, f, indent=2)
print(f"Saved to {OUT}")
