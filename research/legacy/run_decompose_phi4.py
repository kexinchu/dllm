"""
DetermLLM: Decomposition Experiment — Phi-4 (14B, Microsoft)
=============================================================
Cross-architecture validation. Phi-4 has:
  hidden_dim=5120, intermediate_size=17920, num_heads=40
  K_attn=5120 (< K_max≈7700), K_FFN_down=17920 (> K_max)

Expected: moderate improvement (attn shapes fixed, FFN shapes exceed K_max).

Batch sizes 2–32 (48 GB GPU). GEN_LEN=32 (shorter for speed).

Output: research/exp_decompose_phi4.json
"""

import os, sys, json, hashlib
import torch
import torch.nn.functional as F

DLLM_DIR = os.path.dirname(os.path.abspath(__file__))
FP32_DIR  = os.path.join(DLLM_DIR, '..', 'FP32')
sys.path.insert(0, DLLM_DIR)
sys.path.insert(0, FP32_DIR)

os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

from transformers import AutoModelForCausalLM, AutoTokenizer
import determ_llm

MODEL_PATH  = '/home/kec23008/docker-sys/Models/Phi-4'
OUT_FILE    = os.path.join(DLLM_DIR, 'exp_decompose_phi4.json')

PROMPTS = [
    "The Eiffel Tower is located in",
    "In Python, the function to sort a list is",
    "The chemical formula for water is",
    "Albert Einstein developed the theory of",
    "The capital of Japan is",
    "The largest ocean on Earth is the",
    "The author of Pride and Prejudice is",
    "In mathematics, the value of pi is approximately",
    "The CPU stands for central processing",
    "Neural networks are inspired by the human",
]

GEN_LEN     = 32
BATCH_SIZES = [2, 4, 8, 16, 32]


def generate_tokens(model, tokenizer, prompt, batch_size, gen_len, device):
    enc = tokenizer(prompt, return_tensors='pt')['input_ids']
    ids = enc.repeat(batch_size, 1).to(device)
    token_ids, token_lps = [], []

    with torch.no_grad():
        for _ in range(gen_len):
            out    = model(input_ids=ids)
            logits = out.logits[0, -1]
            lps    = F.log_softmax(logits, dim=-1)
            tok    = lps.argmax().item()
            token_ids.append(tok)
            token_lps.append(lps[tok].item())
            next_col = torch.full((batch_size, 1), tok, dtype=ids.dtype, device=device)
            ids = torch.cat([ids, next_col], dim=1)

    return token_ids, token_lps


def seq_hash(token_ids):
    return hashlib.md5(str(token_ids).encode()).hexdigest()[:12]


def run_condition(model, tokenizer, device, condition_name):
    print(f"\n  Condition: {condition_name}")
    cond_results = {}
    all_match = True

    for p_idx, prompt in enumerate(PROMPTS):
        ref_toks, ref_lps = generate_tokens(model, tokenizer, prompt, 1, GEN_LEN, device)
        ref_hash = seq_hash(ref_toks)
        prompt_res = {'ref_hash': ref_hash, 'batches': {}}

        for bs in BATCH_SIZES:
            bs_toks, bs_lps = generate_tokens(model, tokenizer, prompt, bs, GEN_LEN, device)
            bs_hash   = seq_hash(bs_toks)
            seq_match = (bs_hash == ref_hash)
            tok_mm    = sum(a != b for a, b in zip(ref_toks, bs_toks))
            lp_diffs  = [abs(a - b) for a, b in zip(ref_lps, bs_lps)]
            mean_lp   = sum(lp_diffs) / len(lp_diffs)
            max_lp    = max(lp_diffs)
            first_div = next(
                (i for i,(a,b) in enumerate(zip(ref_toks,bs_toks)) if a!=b), -1)

            prompt_res['batches'][f'bs{bs}'] = {
                'seq_match': seq_match, 'token_mismatches': tok_mm,
                'first_divergence': first_div, 'mean_lp_diff': mean_lp,
                'max_lp_diff': max_lp,
            }
            status = '✓' if seq_match else f'✗ (first_div={first_div}, mm={tok_mm})'
            if bs in [8, 32] or not seq_match:
                print(f"    p{p_idx} bs={bs:>2}: {status}  mean|ΔlogP|={mean_lp:.4f}")
            if not seq_match:
                all_match = False

        mm_p = sum(v['token_mismatches'] for v in prompt_res['batches'].values())
        lp_p = sum(v['mean_lp_diff'] for v in prompt_res['batches'].values()) / len(BATCH_SIZES)
        prompt_res['all_match'] = all(v['seq_match'] for v in prompt_res['batches'].values())
        prompt_res['total_token_mismatches'] = mm_p
        prompt_res['avg_lp_diff'] = lp_p
        cond_results[f'p{p_idx}'] = prompt_res

    avg_lp = sum(cond_results[f'p{i}']['avg_lp_diff']
                 for i in range(len(PROMPTS))) / len(PROMPTS)
    total_tok_mm = sum(cond_results[f'p{i}']['total_token_mismatches']
                       for i in range(len(PROMPTS)))
    print(f"\n  >>> {condition_name}: all_match={all_match}, "
          f"total_tok_mm={total_tok_mm}, avg|ΔlogP|={avg_lp:.4f}")

    return {
        'all_sequences_match': all_match,
        'total_token_mismatches': total_tok_mm,
        'avg_lp_diff': avg_lp,
        'per_prompt': cond_results,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
print(f"Loading {MODEL_PATH} (eager attn)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    device_map='cuda:0',
    attn_implementation='eager',
)
model.eval()
device = next(model.parameters()).device
print(f"  Loaded on {device}")
print(f"  hidden_size={model.config.hidden_size}, "
      f"intermediate_size={model.config.intermediate_size}, "
      f"num_layers={model.config.num_hidden_layers}")

all_results = {}

determ_llm.disable()
torch.use_deterministic_algorithms(False)
all_results['A_BF16_baseline'] = run_condition(model, tokenizer, device, 'A_BF16_baseline')

determ_llm.enable()
all_results['B_linear_only'] = run_condition(model, tokenizer, device, 'B_linear_only')
determ_llm.disable()

print(f"\n{'='*65}")
print("SUMMARY: Decomposition — Phi-4 (14B, Microsoft)")
print(f"{'='*65}")
print(f"{'Condition':<28} {'All Match?':>10} {'Tot MM':>8} {'Avg|ΔlogP|':>12}")
print('-' * 62)
for cond, cres in all_results.items():
    m = '✓' if cres['all_sequences_match'] else '✗'
    print(f"  {cond:<28} {m:>10} {cres['total_token_mismatches']:>8} "
          f"{cres['avg_lp_diff']:>12.4f}")

with open(OUT_FILE, 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved → {OUT_FILE}")
