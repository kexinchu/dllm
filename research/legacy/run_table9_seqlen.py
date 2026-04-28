"""Table 9 fix (continuous batching) + sequence length scaling experiments."""
import sys, json, time, hashlib
sys.path.insert(0, '/home/kec23008/docker-sys/dllm')
import torch, warnings
warnings.filterwarnings('ignore')
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = '/home/kec23008/docker-sys/Models/Llama-3.1-8B-Instruct'
OUT = '/home/kec23008/docker-sys/dllm/research/exp_table9_seqlen.json'
results = {}

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map='auto')
model.eval()
device = next(model.parameters()).device

# ======== PART 1: Table 9 Fix — Long Seq Continuous Batching ========
print("\n=== PART 1: Table 9 Fix (Long Sequence, Continuous Batching) ===")

# Create a long prompt (~200 tokens)
long_prompt = ("Explain the theoretical foundations of deterministic inference in large language models. " * 15).strip()
input_ids_single = tokenizer(long_prompt, return_tensors='pt', truncation=True, max_length=256)['input_ids']
seq_len = input_ids_single.shape[1]
print(f"Prompt length: {seq_len} tokens")

table9_results = []
for mode_name, flag_val in [("BF16", True), ("FP32_accum", False)]:
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = flag_val

    # Reference: bs=1
    ids_1 = input_ids_single.to(device)
    pos_1 = torch.arange(seq_len, device=device).unsqueeze(0)
    with torch.no_grad():
        ref_logits = model(input_ids=ids_1, position_ids=pos_1).logits[0]
    ref_argmax = ref_logits.float().argmax(-1)

    for bs in [2, 4, 8, 16, 32]:
        # All sequences are the SAME prompt — no padding, same position_ids
        ids_batch = input_ids_single.repeat(bs, 1).to(device)
        pos_batch = torch.arange(seq_len, device=device).unsqueeze(0).repeat(bs, 1)
        with torch.no_grad():
            batch_logits = model(input_ids=ids_batch, position_ids=pos_batch).logits[0]

        diff = (ref_logits.float() - batch_logits.float()).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        batch_argmax = batch_logits.float().argmax(-1)
        flips = (ref_argmax != batch_argmax).sum().item()
        flip_rate = flips / seq_len

        row = {"mode": mode_name, "bs": bs, "seq_len": seq_len,
               "max_diff": max_diff, "mean_diff": mean_diff,
               "argmax_flips": flips, "flip_rate": flip_rate}
        table9_results.append(row)
        print(f"  {mode_name} bs={bs:>2}: max_diff={max_diff:.4e} flips={flips}/{seq_len} ({flip_rate:.2%})")

results['table9_fix'] = table9_results

# ======== PART 2: Sequence Length Scaling (Op-level GEMM) ========
print("\n=== PART 2: Sequence Length Scaling (GEMM) ===")

seqlen_gemm = []
W = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16)
for seq_len_test in [32, 64, 128, 256, 512, 1024]:
    for mode_name, flag_val in [("BF16", True), ("FP32_accum", False)]:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = flag_val

        torch.manual_seed(42)
        x = torch.randn(1, seq_len_test, 4096, device=device, dtype=torch.bfloat16)
        ref = (x[0:1] @ W.T)  # bs=1 reference

        for bs in [1, 8, 32, 128]:
            x_batch = x.repeat(bs, 1, 1)
            out = x_batch @ W.T
            diff = (ref[0].float() - out[0].float()).abs()
            row = {"seq_len": seq_len_test, "bs": bs, "mode": mode_name,
                   "max_diff": diff.max().item(), "mean_diff": diff.mean().item()}
            seqlen_gemm.append(row)
    print(f"  seq_len={seq_len_test} done")

results['seqlen_gemm'] = seqlen_gemm

# ======== PART 3: Sequence Length Scaling (Model-level) ========
print("\n=== PART 3: Sequence Length Scaling (Model-level) ===")

seqlen_model = []
base_ids = torch.tensor([[128000] + [791, 1917, 374, 2294, 323, 279, 13180, 374, 6437] * 30], device=device)

for seq_target in [32, 64, 128, 200]:
    ids = base_ids[:, :seq_target]
    sl = ids.shape[1]
    pos = torch.arange(sl, device=device).unsqueeze(0)

    for mode_name, flag_val in [("BF16", True), ("FP32_accum", False)]:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = flag_val

        with torch.no_grad():
            ref = model(input_ids=ids, position_ids=pos).logits[0]

        for bs in [2, 8]:
            ids_b = ids.repeat(bs, 1)
            pos_b = pos.repeat(bs, 1)
            with torch.no_grad():
                out = model(input_ids=ids_b, position_ids=pos_b).logits[0]

            diff = (ref.float() - out.float()).abs()
            flips = (ref.float().argmax(-1) != out.float().argmax(-1)).sum().item()
            row = {"seq_len": sl, "bs": bs, "mode": mode_name,
                   "max_diff": diff.max().item(), "mean_diff": diff.mean().item(),
                   "argmax_flips": flips, "flip_rate": flips / sl}
            seqlen_model.append(row)
    print(f"  seq_len={sl} done")

results['seqlen_model'] = seqlen_model

# Save
with open(OUT, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {OUT}")
