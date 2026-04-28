"""
DetermLLM: Decomposition Experiment — 50 Prompts (Statistical Credibility)
===========================================================================
Replicates run_det_decompose.py with 50 diverse prompts instead of 10.
Full re-computation (no KV cache), eager attention, Llama-3.2-1B.

Conditions: A (BF16), B (F.linear), D (F.linear + pad_m1)

50 prompts cover: factual recall, math, code, science, history, language,
reasoning, creative writing — representative of real LLM usage.

Output: research/exp_decompose_50p.json
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

MODEL_PATH = '/home/kec23008/docker-sys/Models/Llama-3.2-1B-Instruct'
OUT_FILE   = os.path.join(DLLM_DIR, 'exp_decompose_50p.json')

PROMPTS = [
    # Factual recall
    "The Eiffel Tower is located in",
    "The chemical formula for water is",
    "Albert Einstein developed the theory of",
    "The capital of Japan is",
    "The largest ocean on Earth is the",
    "The author of Pride and Prejudice is",
    "In mathematics, the value of pi is approximately",
    "The CPU stands for central processing",
    "Neural networks are inspired by the human",
    "The speed of light in a vacuum is approximately",
    # Science
    "Photosynthesis is the process by which plants convert",
    "DNA stands for deoxyribonucleic",
    "The periodic table element with atomic number 79 is",
    "Newton's second law of motion states that force equals",
    "The mitochondria is often called the powerhouse of the",
    # History
    "The French Revolution began in the year",
    "World War II ended in",
    "The first president of the United States was",
    "The Roman Empire fell in",
    "The Berlin Wall fell in",
    # Code / tech
    "In Python, the function to sort a list is",
    "To reverse a string in Python you can use",
    "The time complexity of binary search is",
    "SQL stands for Structured Query",
    "The HTTP status code 404 means",
    "In machine learning, overfitting occurs when",
    "The gradient descent algorithm minimizes the",
    "A convolutional neural network is primarily used for",
    "The transformer architecture was introduced in the paper",
    "CUDA stands for Compute Unified Device",
    # Math
    "The derivative of sin(x) is",
    "The integral of 2x with respect to x is",
    "A prime number is a number that is divisible only by",
    "The Fibonacci sequence starts with",
    "The Pythagorean theorem states that",
    # Language
    "The synonym of 'eloquent' is",
    "The antonym of 'benevolent' is",
    "A haiku is a form of poetry with",
    "The Oxford comma refers to",
    "Onomatopoeia is a word that",
    # Geography
    "The longest river in the world is the",
    "Mount Everest is located in the",
    "The Amazon rainforest is located primarily in",
    "The Great Barrier Reef is located off the coast of",
    "The Sahara Desert is located in",
    # Reasoning
    "If all mammals are warm-blooded and a whale is a mammal, then",
    "The main advantage of a hash table over an array is",
    "To reduce model overfitting, one common technique is",
    "The difference between supervised and unsupervised learning is",
    "The reason gradient clipping is used in training is",
]

assert len(PROMPTS) == 50, f"Expected 50 prompts, got {len(PROMPTS)}"

GEN_LEN     = 64
BATCH_SIZES = [2, 4, 8, 16, 32, 64]


def generate_tokens(model, tokenizer, prompt, batch_size, gen_len, device):
    """Full re-computation (no KV cache). M = batch_size × (prompt_len + t)."""
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
    total_mm = 0

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
            if not seq_match:
                all_match = False
                total_mm += tok_mm
                print(f"    p{p_idx} bs={bs:>2}: ✗ (first_div={first_div}, mm={tok_mm})"
                      f"  mean|ΔlogP|={mean_lp:.4f}")

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
    prompts_with_any_mm = sum(
        1 for i in range(len(PROMPTS))
        if not cond_results[f'p{i}']['all_match']
    )
    print(f"\n  >>> {condition_name}: all_match={all_match}, "
          f"total_tok_mm={total_tok_mm}, avg|ΔlogP|={avg_lp:.4f}, "
          f"prompts_with_mm={prompts_with_any_mm}/50")

    return {
        'all_sequences_match': all_match,
        'total_token_mismatches': total_tok_mm,
        'avg_lp_diff': avg_lp,
        'prompts_with_any_mismatch': prompts_with_any_mm,
        'per_prompt': cond_results,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
print(f"Loading {MODEL_PATH} (eager attn)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, dtype=torch.bfloat16, device_map='cuda:0',
    attn_implementation='eager')
model.eval()
device = next(model.parameters()).device
print(f"  Loaded on {device}")

all_results = {}

determ_llm.disable()
all_results['A_BF16_baseline'] = run_condition(
    model, tokenizer, device, 'A_BF16_baseline')

determ_llm.enable(pad_m1=False)
all_results['B_linear_only'] = run_condition(
    model, tokenizer, device, 'B_linear_only')
determ_llm.disable()

determ_llm.enable(pad_m1=True)
all_results['D_linear_pad_m1'] = run_condition(
    model, tokenizer, device, 'D_linear_pad_m1')
determ_llm.disable()

print(f"\n{'='*70}")
print("SUMMARY: Decomposition (full re-computation) — Llama-3.2-1B, 50 prompts")
print(f"{'='*70}")
print(f"{'Condition':<28} {'All Match?':>10} {'Tot MM':>8} {'Avg|ΔlogP|':>12} {'Prompts w/ MM':>14}")
print('-' * 76)
for cond, cres in all_results.items():
    m = '✓' if cres['all_sequences_match'] else '✗'
    print(f"  {cond:<28} {m:>10} {cres['total_token_mismatches']:>8} "
          f"{cres['avg_lp_diff']:>12.4f} {cres['prompts_with_any_mismatch']:>14}/50")

with open(OUT_FILE, 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved → {OUT_FILE}")
