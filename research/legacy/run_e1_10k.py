"""E1: 10000-run generation determinism test. Llama-3.1-8B, bs=1..256 random."""
import torch, warnings, hashlib, json, time, random
warnings.filterwarnings('ignore')
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = '/home/kec23008/docker-sys/Models/Llama-3.1-8B-Instruct'
OUT = '/home/kec23008/docker-sys/dllm/research/exp_e1_10k.json'

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map='auto')
model.eval()
device = next(model.parameters()).device

PROMPT = 'What is deterministic inference in large language models?'
input_ids = tokenizer(PROMPT, return_tensors='pt')['input_ids']
sl = input_ids.shape[1]
GEN_LEN = 32
N_RUNS = 10000

# Pre-generate random batch sizes: uniform 1..256
random.seed(42)
batch_sizes = [random.randint(1, 256) for _ in range(N_RUNS)]

# Count occurrences
from collections import Counter
bs_counts = Counter(batch_sizes)
print(f"Batch size range: 1-256, {N_RUNS} runs")
print(f"Unique batch sizes sampled: {len(bs_counts)}")

def generate(bs):
    ids_b = input_ids.repeat(bs, 1).to(device)
    pos_b = torch.arange(sl, device=device).unsqueeze(0).repeat(bs, 1)
    generated = []
    cur_ids = ids_b.clone()
    cur_pos = pos_b.clone()
    with torch.no_grad():
        for step in range(GEN_LEN):
            out = model(input_ids=cur_ids, position_ids=cur_pos).logits
            next_tok = out[0, -1].argmax().item()
            generated.append(next_tok)
            cur_ids = torch.tensor([[next_tok]], device=device).repeat(bs, 1)
            cur_pos = torch.tensor([[sl + step]], device=device).repeat(bs, 1)
    return hashlib.md5(str(generated).encode()).hexdigest()[:12]

results = {}
for mode, flag in [('BF16', True), ('FP32', False)]:
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = flag
    print(f"\n=== {mode}: Running {N_RUNS} generations ===")

    # First: build per-bs hash cache (each bs is deterministic within mode)
    # Only need to run each unique bs once
    bs_hash = {}
    unique_bs = sorted(bs_counts.keys())
    t0 = time.time()
    for i, bs in enumerate(unique_bs):
        bs_hash[bs] = generate(bs)
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(unique_bs) - i - 1)
            print(f"  {i+1}/{len(unique_bs)} unique bs done, elapsed {elapsed:.0f}s, ETA {eta:.0f}s")

    elapsed = time.time() - t0
    print(f"  All {len(unique_bs)} unique bs done in {elapsed:.1f}s")

    # Map each of the N_RUNS to its hash
    run_hashes = [bs_hash[bs] for bs in batch_sizes]

    # Statistics
    unique_hashes = set(run_hashes)
    hash_counter = Counter(run_hashes)

    # Per-hash batch size groups
    hash_to_bs = {}
    for bs, h in bs_hash.items():
        hash_to_bs.setdefault(h, []).append(bs)

    print(f"  Unique outputs: {len(unique_hashes)}")
    print(f"  Hash distribution:")
    for h, count in hash_counter.most_common():
        bs_list = sorted(hash_to_bs[h])
        bs_ranges = f"bs=[{min(bs_list)}..{max(bs_list)}] ({len(bs_list)} values)"
        print(f"    {h}: {count} runs, {bs_ranges}")

    # Find exact transition points
    transitions = []
    prev_h = bs_hash.get(1)
    for bs in range(2, 257):
        if bs in bs_hash and bs_hash[bs] != prev_h:
            transitions.append(bs)
        if bs in bs_hash:
            prev_h = bs_hash[bs]

    print(f"  Kernel transitions: {transitions}")

    results[mode] = {
        'n_runs': N_RUNS,
        'n_unique_bs': len(unique_bs),
        'unique_outputs': len(unique_hashes),
        'hash_distribution': dict(hash_counter),
        'transitions': transitions,
        'hash_to_bs_count': {h: len(bss) for h, bss in hash_to_bs.items()},
        'hash_to_bs_range': {h: [min(bss), max(bss)] for h, bss in hash_to_bs.items()},
        'elapsed_s': elapsed,
        'per_bs_hash': {str(k): v for k, v in bs_hash.items()}
    }

# Summary comparison
print("\n=== SUMMARY ===")
for mode in ['BF16', 'FP32']:
    r = results[mode]
    print(f"{mode}: {r['unique_outputs']} unique outputs, {len(r['transitions'])} transitions, {r['elapsed_s']:.0f}s")

with open(OUT, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {OUT}")
