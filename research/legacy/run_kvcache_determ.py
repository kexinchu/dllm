"""
DetermLLM: KV-Cache Determinism Experiment
==========================================
Goal: Test the PRODUCTION serving scenario with KV cache.

Key difference from run_det_decompose.py:
  - use_cache=True: prefill once, then decode token-by-token with cached KV
  - Decode attention is M=1 → exactly the production serving path
  - Tests whether F.linear alone achieves 0 token mismatches in production

Conditions:
  A: BF16 baseline (no patch, use_cache=True)
  B: DetermLLM F.linear only (use_cache=True)
  C: DetermLLM F.linear + attn patch (use_cache=True, attn_implementation='eager')

Models: Llama-3.2-1B-Instruct
Batch sizes: 2, 4, 8, 16, 32, 64
Generation: 64 tokens, greedy

Output: research/exp_kvcache_determ.json
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

MODEL_PATH_1B = '/home/kec23008/docker-sys/Models/Llama-3.2-1B-Instruct'
OUT_FILE = os.path.join(DLLM_DIR, 'exp_kvcache_determ.json')

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

GEN_LEN     = 64
BATCH_SIZES = [2, 4, 8, 16, 32, 64]


def generate_tokens_kvcache(model, tokenizer, prompt, batch_size, gen_len, device):
    """
    KV-cache generation: mirrors real serving.
    - Prefill: process full prompt once → get KV cache
    - Decode: one new token per step, M=1 attention
    Row 0 is always the target request.
    """
    enc = tokenizer(prompt, return_tensors='pt')['input_ids']  # [1, S]
    ids = enc.repeat(batch_size, 1).to(device)

    token_ids, token_lps = [], []

    with torch.no_grad():
        # Step 0: prefill — process the full prompt, get KV cache
        out = model(input_ids=ids, use_cache=True)
        past_kv = out.past_key_values

        # Extract logit for next token from prefill output
        logits = out.logits[0, -1]        # row 0, last token
        lps    = F.log_softmax(logits, dim=-1)
        tok    = lps.argmax().item()
        token_ids.append(tok)
        token_lps.append(lps[tok].item())

        # Decode: only pass the new token + KV cache
        for _ in range(gen_len - 1):
            next_col = torch.full((batch_size, 1), tok, dtype=ids.dtype, device=device)
            out = model(input_ids=next_col, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            logits  = out.logits[0, -1]
            lps     = F.log_softmax(logits, dim=-1)
            tok     = lps.argmax().item()
            token_ids.append(tok)
            token_lps.append(lps[tok].item())

    return token_ids, token_lps


def seq_hash(token_ids):
    return hashlib.md5(str(token_ids).encode()).hexdigest()[:12]


def run_condition(model, tokenizer, device, condition_name):
    print(f"\n  Condition: {condition_name}")
    cond_results = {}
    total_mm = 0
    all_match = True

    for p_idx, prompt in enumerate(PROMPTS):
        ref_toks, ref_lps = generate_tokens_kvcache(
            model, tokenizer, prompt, 1, GEN_LEN, device)
        ref_hash = seq_hash(ref_toks)
        prompt_res = {'ref_hash': ref_hash, 'batches': {}}

        for bs in BATCH_SIZES:
            bs_toks, bs_lps = generate_tokens_kvcache(
                model, tokenizer, prompt, bs, GEN_LEN, device)
            bs_hash   = seq_hash(bs_toks)
            seq_match = (bs_hash == ref_hash)
            tok_mm    = sum(a != b for a, b in zip(ref_toks, bs_toks))
            lp_diffs  = [abs(a - b) for a, b in zip(ref_lps, bs_lps)]
            mean_lp   = sum(lp_diffs) / len(lp_diffs)
            max_lp    = max(lp_diffs)
            first_div = next(
                (i for i, (a,b) in enumerate(zip(ref_toks, bs_toks)) if a != b), -1)

            prompt_res['batches'][f'bs{bs}'] = {
                'seq_match':        seq_match,
                'token_mismatches': tok_mm,
                'first_divergence': first_div,
                'mean_lp_diff':     mean_lp,
                'max_lp_diff':      max_lp,
            }

            status = '✓' if seq_match else \
                f'✗ (first_div={first_div}, mm={tok_mm})'
            if bs in [8, 32, 64] or not seq_match:
                print(f"    p{p_idx} bs={bs:>2}: {status}  mean|ΔlogP|={mean_lp:.4f}")

            if not seq_match:
                all_match = False
                total_mm += tok_mm

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
        'all_sequences_match':    all_match,
        'total_token_mismatches': total_tok_mm,
        'avg_lp_diff':            avg_lp,
        'per_prompt':             cond_results,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

all_results = {}

# ── Conditions A and B: default attn (may use SDPA/FA2) ──────────────────────
print("Loading Llama-3.2-1B-Instruct (default attn)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH_1B, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH_1B,
    dtype=torch.bfloat16,
    device_map='cuda:0',
    # default attn_implementation
)
model.eval()
device = next(model.parameters()).device
print(f"  Loaded on {device}")

# Condition A: BF16 baseline
determ_llm.disable()
torch.use_deterministic_algorithms(False)
all_results['A_BF16_baseline'] = run_condition(
    model, tokenizer, device, 'A_BF16_baseline')

# Condition B: DetermLLM F.linear only
determ_llm.enable()
all_results['B_linear_only'] = run_condition(
    model, tokenizer, device, 'B_linear_only')

determ_llm.disable()
del model
torch.cuda.empty_cache()

# ── Condition C: eager attn + attn patch ─────────────────────────────────────
print("\nLoading Llama-3.2-1B-Instruct (eager attn)...")
model_eager = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH_1B,
    dtype=torch.bfloat16,
    device_map='cuda:0',
    attn_implementation='eager',
)
model_eager.eval()
device = next(model_eager.parameters()).device
print(f"  Loaded on {device}, attn=eager")

# Condition C: F.linear + decode-phase attn patch
determ_llm.enable(attn=True)
all_results['C_linear_and_attn'] = run_condition(
    model_eager, tokenizer, device, 'C_linear_and_attn')

determ_llm.disable()
del model_eager
torch.cuda.empty_cache()

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("SUMMARY: KV-Cache Determinism (production serving path)")
print(f"{'='*65}")
print(f"{'Condition':<30} {'All Match?':>12} {'Tot Tok MM':>12} {'Avg|ΔlogP|':>12}")
print("-" * 70)
for cond, cres in all_results.items():
    m = '✓' if cres['all_sequences_match'] else '✗'
    print(f"  {cond:<30} {m:>12} {cres['total_token_mismatches']:>12} "
          f"{cres['avg_lp_diff']:>12.4f}")

with open(OUT_FILE, 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved → {OUT_FILE}")
