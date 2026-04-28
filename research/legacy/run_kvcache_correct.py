"""
DetermLLM: Correct Production KV-Cache Experiment
===================================================
The CORRECT production serving scenario:
  - Prefill each request at bs=1 (individual prefill, not batched)
  - Duplicate KV cache for decode at batch_size N
  - This matches vLLM continuous batching: prefill is independent of decode batch

Why this is correct:
  With bs=1 prefill, the KV cache is IDENTICAL regardless of decode batch size.
  The only batch-size-varying computation is the DECODE phase:
    - F.linear projections: [N, H, 1, D] → patched, deterministic
    - Attention Q @ K^T: [N, H, 1, L] × [N, H, L, D] → M=1, decode phase
    - attn output @ V: M=1, decode phase

  With DetermLLM F.linear: all decode GEMMs are batch-invariant.
  Prediction: 0 token mismatches.

Conditions:
  A: BF16 baseline (bs=1 prefill, bs=N decode)
  B: DetermLLM F.linear only
  C: DetermLLM F.linear + decode-attn patch (eager)

Output: research/exp_kvcache_correct.json
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
try:
    from transformers import DynamicCache
    HAS_DYNAMIC_CACHE = True
except ImportError:
    HAS_DYNAMIC_CACHE = False
import determ_llm

MODEL_PATH = '/home/kec23008/docker-sys/Models/Llama-3.2-1B-Instruct'
OUT_FILE   = os.path.join(DLLM_DIR, 'exp_kvcache_correct.json')

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


def prefill_bs1(model, tokenizer, prompt, device):
    """Prefill a single sequence, return (first_token_logits, past_kv, prompt_len)."""
    enc = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)
    with torch.no_grad():
        out = model(input_ids=enc, use_cache=True)
    return out.logits[0, -1], out.past_key_values, enc.shape[1]


def duplicate_kv(past_kv, n):
    """
    Duplicate a bs=1 KV cache to bs=n.
    Handles both DynamicCache (newer transformers) and legacy tuple format.
    """
    if HAS_DYNAMIC_CACHE and isinstance(past_kv, DynamicCache):
        # DynamicCache has batch_repeat_interleave(n) — exactly what we need
        import copy
        dup = copy.deepcopy(past_kv)
        dup.batch_repeat_interleave(n)
        return dup
    else:
        # Legacy: tuple of (k, v) per layer, each [1, H, L, D]
        return tuple(
            (k.repeat(n, 1, 1, 1), v.repeat(n, 1, 1, 1))
            for k, v in past_kv
        )


def decode_with_kv(model, tokenizer, first_logits, past_kv_1,
                   batch_size, gen_len, device):
    """
    Decode gen_len tokens using KV cache duplicated from bs=1 prefill.
    All N batch items use the SAME KV cache (from bs=1 prefill).
    Row 0 is compared against bs=1 reference.
    """
    token_ids, token_lps = [], []

    # First token from prefill logits
    lps = F.log_softmax(first_logits, dim=-1)
    tok = lps.argmax().item()
    token_ids.append(tok)
    token_lps.append(lps[tok].item())

    # Duplicate KV for batch_size N
    past_kv = duplicate_kv(past_kv_1, batch_size)

    with torch.no_grad():
        for step in range(gen_len - 1):
            next_col = torch.full(
                (batch_size, 1), tok, dtype=torch.long, device=device)
            out = model(
                input_ids=next_col,
                past_key_values=past_kv,
                use_cache=True,
            )
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
        # bs=1 prefill (always the reference, always deterministic)
        first_logits_1, past_kv_1, _ = prefill_bs1(model, tokenizer, prompt, device)

        ref_toks, ref_lps = decode_with_kv(
            model, tokenizer, first_logits_1, past_kv_1, 1, GEN_LEN, device)
        ref_hash = seq_hash(ref_toks)
        prompt_res = {'ref_hash': ref_hash, 'batches': {}}

        for bs in BATCH_SIZES:
            # Use the SAME bs=1 KV cache duplicated to bs
            bs_toks, bs_lps = decode_with_kv(
                model, tokenizer, first_logits_1, past_kv_1, bs, GEN_LEN, device)
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
        'all_sequences_match': all_match,
        'total_token_mismatches': total_tok_mm,
        'avg_lp_diff': avg_lp,
        'per_prompt': cond_results,
    }


# ── Run conditions ────────────────────────────────────────────────────────────
all_results = {}

# Conditions A & B: default attn
print(f"Loading {MODEL_PATH} (default attn)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, dtype=torch.bfloat16, device_map='cuda:0')
model.eval()
device = next(model.parameters()).device
print(f"  Loaded on {device}")

determ_llm.disable()
all_results['A_BF16_baseline'] = run_condition(
    model, tokenizer, device, 'A_BF16_baseline (bs=1 prefill)')

determ_llm.enable()
all_results['B_linear_only'] = run_condition(
    model, tokenizer, device, 'B_linear_only (bs=1 prefill)')

determ_llm.disable()
del model
torch.cuda.empty_cache()

# Condition C: eager attn + attn patch
print(f"\nLoading {MODEL_PATH} (eager attn)...")
model_eager = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, dtype=torch.bfloat16, device_map='cuda:0',
    attn_implementation='eager')
model_eager.eval()
device = next(model_eager.parameters()).device

determ_llm.enable(attn=True)
all_results['C_linear_and_attn'] = run_condition(
    model_eager, tokenizer, device, 'C_linear_and_attn (bs=1 prefill, eager)')

determ_llm.disable()
del model_eager
torch.cuda.empty_cache()

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("SUMMARY: Correct Production KV-Cache (bs=1 prefill, bs=N decode)")
print(f"{'='*70}")
for cond, cres in all_results.items():
    m = '✓' if cres['all_sequences_match'] else '✗'
    print(f"  {cond:<45} {m} mm={cres['total_token_mismatches']:>4} "
          f"avg|ΔlogP|={cres['avg_lp_diff']:.4f}")

with open(OUT_FILE, 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved → {OUT_FILE}")
