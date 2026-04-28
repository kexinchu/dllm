"""M4: Per-layer error accumulation tracking through Llama-3.1-8B."""
import torch, json, types, warnings
warnings.filterwarnings('ignore')
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = '/home/kec23008/docker-sys/Models/Llama-3.1-8B-Instruct'
OUT = '/home/kec23008/docker-sys/dllm/research/exp_m4_layer_accum.json'

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map='auto')
model.eval()
device = next(model.parameters()).device

PROMPT = 'What is deterministic inference in large language models?'
input_ids = tokenizer(PROMPT, return_tensors='pt')['input_ids']
sl = input_ids.shape[1]

results = []

for mode, flag in [('BF16', True), ('FP32', False)]:
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = flag
    print(f"\n=== {mode} mode ===")

    for bs in [2, 8, 32, 64, 128]:
        # Hook every decoder layer's output to capture hidden states
        layer_outputs_ref = {}
        layer_outputs_batch = {}
        hooks = []

        def make_hook(name, storage):
            def hook_fn(module, inp, out):
                # out is a tuple, first element is hidden_states
                h = out[0] if isinstance(out, tuple) else out
                storage[name] = h[0].detach().float().clone()  # sample 0
            return hook_fn

        # Run bs=1
        for name, mod in model.named_modules():
            if hasattr(mod, 'self_attn') and hasattr(mod, 'mlp'):  # decoder layer
                hooks.append(mod.register_forward_hook(make_hook(name, layer_outputs_ref)))

        ids_1 = input_ids.to(device)
        pos_1 = torch.arange(sl, device=device).unsqueeze(0)
        with torch.no_grad():
            ref_logits = model(input_ids=ids_1, position_ids=pos_1).logits[0]

        for h in hooks:
            h.remove()
        hooks = []

        # Run bs=N
        for name, mod in model.named_modules():
            if hasattr(mod, 'self_attn') and hasattr(mod, 'mlp'):
                hooks.append(mod.register_forward_hook(make_hook(name, layer_outputs_batch)))

        ids_b = input_ids.repeat(bs, 1).to(device)
        pos_b = pos_1.repeat(bs, 1)
        with torch.no_grad():
            batch_logits = model(input_ids=ids_b, position_ids=pos_b).logits[0]

        for h in hooks:
            h.remove()

        # Compare per-layer
        logit_diff = (ref_logits.float() - batch_logits.float()).abs()
        logit_max = logit_diff.max().item()
        logit_mean = logit_diff.mean().item()
        argmax_match = (ref_logits.float().argmax(-1) == batch_logits.float().argmax(-1)).float().mean().item()

        layer_diffs = []
        for name in sorted(layer_outputs_ref.keys()):
            if name in layer_outputs_batch:
                d = (layer_outputs_ref[name] - layer_outputs_batch[name]).abs()
                layer_diffs.append({
                    'layer': name,
                    'max_diff': d.max().item(),
                    'mean_diff': d.mean().item(),
                    'l2_diff': d.pow(2).mean().sqrt().item()
                })

        # Extract layer indices for printing
        print(f"  bs={bs:>3}: logit_max={logit_max:.4f} logit_mean={logit_mean:.4e} argmax_match={argmax_match:.4f}")
        print(f"         Layer error growth (max_diff):")
        for i, ld in enumerate(layer_diffs):
            if i % 4 == 0 or i == len(layer_diffs) - 1:
                print(f"           L{i:>2}: {ld['max_diff']:.4e}")

        results.append({
            'mode': mode, 'bs': bs, 'seq_len': sl,
            'logit_max_diff': logit_max, 'logit_mean_diff': logit_mean,
            'argmax_match_rate': argmax_match,
            'per_layer': layer_diffs
        })

with open(OUT, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {OUT}")
