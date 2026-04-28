"""E2: 1000-run MoE determinism test. DeepSeek-V2-Lite, bs=1..64 random."""
import torch, warnings, hashlib, json, time, random
warnings.filterwarnings('ignore')
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = 'deepseek-ai/DeepSeek-V2-Lite'
OUT = '/home/kec23008/docker-sys/dllm/research/exp_e2_moe_1k.json'

print("Loading DeepSeek-V2-Lite...")
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16, device_map='auto', trust_remote_code=True
)
model.eval()
device = next(model.parameters()).device

PROMPT = 'What is deterministic inference in large language models?'
input_ids = tokenizer(PROMPT, return_tensors='pt')['input_ids']
sl = input_ids.shape[1]
GEN_LEN = 32

# MoE model — smaller batch range (memory limited)
MAX_BS = 64
random.seed(42)
N_RUNS = 1000
batch_sizes = [random.randint(1, MAX_BS) for _ in range(N_RUNS)]

from collections import Counter
bs_counts = Counter(batch_sizes)
print(f"Batch size range: 1-{MAX_BS}, {N_RUNS} runs, {len(bs_counts)} unique bs")

def generate(bs):
    ids_b = input_ids.repeat(bs, 1).to(device)
    pos_b = torch.arange(sl, device=device).unsqueeze(0).repeat(bs, 1)
    generated = []
    cur_ids = ids_b.clone()
    cur_pos = pos_b.clone()
    with torch.no_grad():
        for step in range(GEN_LEN):
            try:
                out = model(input_ids=cur_ids, position_ids=cur_pos, use_cache=False).logits
            except Exception:
                out = model(input_ids=cur_ids, position_ids=cur_pos).logits
            next_tok = out[0, -1].argmax().item()
            generated.append(next_tok)
            cur_ids = torch.tensor([[next_tok]], device=device).repeat(bs, 1)
            cur_pos = torch.tensor([[sl + step]], device=device).repeat(bs, 1)
    return hashlib.md5(str(generated).encode()).hexdigest()[:12]

results = {}
for mode, flag in [('BF16', True), ('FP32', False)]:
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = flag
    print(f"\n=== {mode}: Running unique bs ===")

    bs_hash = {}
    unique_bs = sorted(bs_counts.keys())
    t0 = time.time()
    for i, bs in enumerate(unique_bs):
        try:
            bs_hash[bs] = generate(bs)
        except torch.cuda.OutOfMemoryError:
            print(f"  OOM at bs={bs}, skipping")
            torch.cuda.empty_cache()
            bs_hash[bs] = 'OOM'
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(unique_bs)} done, {elapsed:.0f}s elapsed")

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    run_hashes = [bs_hash.get(bs, 'OOM') for bs in batch_sizes]
    valid_hashes = [h for h in run_hashes if h != 'OOM']
    unique_hashes = set(valid_hashes)
    hash_counter = Counter(valid_hashes)

    hash_to_bs = {}
    for bs, h in bs_hash.items():
        if h != 'OOM':
            hash_to_bs.setdefault(h, []).append(bs)

    print(f"  Unique outputs: {len(unique_hashes)}")
    for h, count in hash_counter.most_common():
        bs_list = sorted(hash_to_bs.get(h, []))
        print(f"    {h}: {count} runs, bs range [{min(bs_list)}..{max(bs_list)}] ({len(bs_list)} values)")

    transitions = []
    prev_h = bs_hash.get(1)
    for bs in range(2, MAX_BS + 1):
        if bs in bs_hash and bs_hash[bs] != 'OOM' and bs_hash[bs] != prev_h:
            transitions.append(bs)
        if bs in bs_hash and bs_hash[bs] != 'OOM':
            prev_h = bs_hash[bs]

    print(f"  Transitions: {transitions}")

    results[mode] = {
        'n_runs': N_RUNS, 'max_bs': MAX_BS,
        'unique_outputs': len(unique_hashes),
        'hash_distribution': dict(hash_counter),
        'transitions': transitions,
        'elapsed_s': elapsed,
        'per_bs_hash': {str(k): v for k, v in bs_hash.items()}
    }

print("\n=== SUMMARY ===")
for mode in ['BF16', 'FP32']:
    r = results[mode]
    print(f"{mode}: {r['unique_outputs']} unique, {len(r['transitions'])} transitions")

with open(OUT, 'w') as f:
    json.dump(results, f, indent=2)
print(f"Saved to {OUT}")
