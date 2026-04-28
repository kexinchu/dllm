"""
KV-Cache Diagnostic: Cross-M vs Cross-Row Invariance
=====================================================
Two distinct properties:
  1. Cross-M invariance: row 0 of bs=N decode == row 0 of bs=1 decode (same inputs)
     → Tests whether cuBLAS gives identical results for M=1 vs M=N GEMM
  2. Cross-row invariance: row 0 of bs=N == row 1 of bs=N (different inputs, same
     batch) → The actual production claim DetermLLM makes

This script isolates which property is failing.

Key questions:
  Q1: Does duplicate_kv work correctly? (row 0 == row 1 in KV cache after dup)
  Q2: Is cross-row invariance achieved? (row 0 == row 1 of DECODE output)
  Q3: Is cross-M invariance achieved? (bs=1 == row 0 of bs=N decode)

Output: research/exp_kvcache_diag.json
"""

import os, sys, json, hashlib
import torch
import torch.nn.functional as F
import copy

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
    print("Using DynamicCache API")
except ImportError:
    HAS_DYNAMIC_CACHE = False
    print("Using legacy tuple KV cache")
import determ_llm

MODEL_PATH = '/home/kec23008/docker-sys/Models/Llama-3.2-1B-Instruct'
OUT_FILE   = os.path.join(DLLM_DIR, 'exp_kvcache_diag.json')

PROMPT  = "The Eiffel Tower is located in"
GEN_LEN = 64
BATCH_SIZES = [2, 4, 8, 32, 64]


def prefill_bs1(model, tokenizer, prompt, device):
    enc = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)
    with torch.no_grad():
        out = model(input_ids=enc, use_cache=True)
    return out.logits[0, -1], out.past_key_values, enc.shape[1]


def duplicate_kv(past_kv, n):
    if HAS_DYNAMIC_CACHE and isinstance(past_kv, DynamicCache):
        dup = copy.deepcopy(past_kv)
        dup.batch_repeat_interleave(n)
        return dup
    else:
        return tuple(
            (k.repeat(n, 1, 1, 1), v.repeat(n, 1, 1, 1))
            for k, v in past_kv
        )


def get_kv_row0(past_kv, layer=0):
    """Extract layer 0, key, row 0 as a flat tensor for comparison."""
    if HAS_DYNAMIC_CACHE and isinstance(past_kv, DynamicCache):
        k = past_kv.key_cache[layer] if hasattr(past_kv, 'key_cache') else None
        if k is None:
            return None
        return k[0].float()  # row 0
    else:
        return past_kv[layer][0][0].float()  # layer, key, row 0


def decode_one_step(model, tok, past_kv, batch_size, device):
    """One decode step, return row 0 logits and all rows' top tokens."""
    next_col = torch.full((batch_size, 1), tok, dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(input_ids=next_col, past_key_values=past_kv, use_cache=True)
    # Row 0 logits
    logits_row0 = out.logits[0, -1].float()
    # Top token from each row
    top_tokens = out.logits[:, -1, :].argmax(dim=-1).tolist()
    return logits_row0, top_tokens, out.past_key_values


def decode_sequence_row0(model, tok, past_kv, batch_size, gen_len, device):
    """Decode gen_len tokens, record row 0 tokens."""
    token_ids = []
    for _ in range(gen_len):
        logits_row0, top_tokens, past_kv = decode_one_step(
            model, tok, past_kv, batch_size, device)
        tok = top_tokens[0]
        token_ids.append(tok)
    return token_ids


def run_diagnostics(model, tokenizer, device, label):
    print(f"\n{'='*60}")
    print(f"Condition: {label}")
    print('='*60)
    results = {}

    first_logits_1, past_kv_1, prompt_len = prefill_bs1(
        model, tokenizer, PROMPT, device)

    lps = F.log_softmax(first_logits_1, dim=-1)
    first_tok = lps.argmax().item()
    print(f"  First token (from bs=1 prefill): {first_tok}")
    print(f"  Prompt length: {prompt_len}")

    # ── Sanity check 1: duplicate_kv correctness ──────────────────────────────
    print("\n  [Q1] KV duplication correctness:")
    for n in [1, 2, 4]:
        dup = duplicate_kv(past_kv_1, n)
        # Check if batch dim is n
        if HAS_DYNAMIC_CACHE and isinstance(dup, DynamicCache):
            try:
                k0 = dup.key_cache[0]  # [n, H, L, D]
                bs = k0.shape[0]
                # Are all rows identical?
                if bs > 1:
                    max_diff = max(
                        (k0[0] - k0[i]).abs().max().item()
                        for i in range(1, bs)
                    )
                    print(f"    n={n}: batch_dim={bs}, max_diff(row0 vs others)={max_diff:.6e}")
                else:
                    print(f"    n={n}: batch_dim={bs}, single row OK")
            except AttributeError:
                print(f"    n={n}: key_cache not accessible via .key_cache")
        else:
            k0 = dup[0][0]  # layer 0, key, [n, H, L, D]
            bs = k0.shape[0]
            if bs > 1:
                max_diff = max(
                    (k0[0] - k0[i]).abs().max().item()
                    for i in range(1, bs)
                )
                print(f"    n={n}: batch_dim={bs}, max_diff(row0 vs others)={max_diff:.6e}")
            else:
                print(f"    n={n}: batch_dim={bs}, single row OK")

    # ── Sanity check 2: bs=1 reference reproducibility ────────────────────────
    print("\n  [Q0] bs=1 decode reproducibility (run twice):")
    seq_a = decode_sequence_row0(
        model, first_tok, duplicate_kv(past_kv_1, 1), 1, GEN_LEN, device)
    seq_b = decode_sequence_row0(
        model, first_tok, duplicate_kv(past_kv_1, 1), 1, GEN_LEN, device)
    match_ab = (seq_a == seq_b)
    print(f"    Run 1 == Run 2: {match_ab}  (mm={sum(a!=b for a,b in zip(seq_a,seq_b))})")
    results['bs1_reproducibility'] = {
        'match': match_ab,
        'mm': sum(a != b for a, b in zip(seq_a, seq_b)),
    }

    # ── Q2: Cross-row invariance (row 0 == row 1 within bs=N decode) ──────────
    print("\n  [Q2] Cross-row invariance (row i == row 0 within same batch):")
    q2_results = {}
    for bs in BATCH_SIZES:
        past_kv_n = duplicate_kv(past_kv_1, bs)
        # Decode one step, get ALL rows
        logits_row0, top_toks, _ = decode_one_step(
            model, first_tok, past_kv_n, bs, device)
        all_same = all(t == top_toks[0] for t in top_toks)
        max_diff_rows = 0.0
        if bs > 1:
            next_col = torch.full((bs, 1), first_tok, dtype=torch.long, device=device)
            with torch.no_grad():
                out = model(input_ids=next_col,
                            past_key_values=duplicate_kv(past_kv_1, bs),
                            use_cache=True)
            logits_all = out.logits[:, -1, :].float()  # [bs, vocab]
            max_diff_rows = (logits_all[1:] - logits_all[0:1]).abs().max().item()
        print(f"    bs={bs:>2}: all_rows_same_token={all_same}, "
              f"max_logit_diff_across_rows={max_diff_rows:.6e}")
        q2_results[f'bs{bs}'] = {
            'all_rows_same_token': all_same,
            'max_logit_diff_across_rows': max_diff_rows,
            'top_tokens': top_toks[:4],
        }
    results['cross_row_invariance'] = q2_results

    # ── Q3: Cross-M invariance with per-step logit-diff tracking ─────────────
    print("\n  [Q3] Cross-M invariance (full sequence, row 0 of bs=N vs bs=1):")

    # Reference: bs=1 full decode tracking logits
    def decode_with_logit_track(model, tok, past_kv, batch_size, gen_len, device):
        token_ids = []
        logit_top1_vals = []
        for _ in range(gen_len):
            next_col = torch.full((batch_size, 1), tok, dtype=torch.long, device=device)
            with torch.no_grad():
                out = model(input_ids=next_col, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            logits = out.logits[0, -1].float()
            top1 = logits.argmax().item()
            top1_val = logits[top1].item()
            tok = top1
            token_ids.append(tok)
            logit_top1_vals.append(top1_val)
        return token_ids, logit_top1_vals, past_kv

    ref_seq, ref_logit_vals, _ = decode_with_logit_track(
        model, first_tok, duplicate_kv(past_kv_1, 1), 1, GEN_LEN, device)

    q3_results = {}
    for bs in BATCH_SIZES:
        test_seq, test_logit_vals, _ = decode_with_logit_track(
            model, first_tok, duplicate_kv(past_kv_1, bs), bs, GEN_LEN, device)
        mm = sum(a != b for a, b in zip(ref_seq, test_seq))
        first_div = next(
            (i for i, (a, b) in enumerate(zip(ref_seq, test_seq)) if a != b), -1)
        match = (ref_seq == test_seq)
        # Per-step logit diffs (only where sequences still agree)
        agree_len = first_div if first_div >= 0 else GEN_LEN
        lp_diffs = [abs(a - b) for a, b in
                    zip(ref_logit_vals[:agree_len], test_logit_vals[:agree_len])]
        mean_pre_div = sum(lp_diffs) / len(lp_diffs) if lp_diffs else 0.0
        max_pre_div  = max(lp_diffs) if lp_diffs else 0.0
        print(f"    bs={bs:>2}: match={match}, mm={mm:>3}, first_div={first_div:>3}, "
              f"mean_pre_div_logit_diff={mean_pre_div:.6e}, max={max_pre_div:.6e}")
        q3_results[f'bs{bs}'] = {
            'match': match, 'mm': mm, 'first_div': first_div,
            'mean_pre_div_logit_diff': mean_pre_div,
            'max_pre_div_logit_diff': max_pre_div,
        }
    results['cross_m_invariance'] = {
        'ref_seq': ref_seq,
        'batches': q3_results,
    }

    return results


# ── Load model and run ────────────────────────────────────────────────────────
print(f"Loading {MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, dtype=torch.bfloat16, device_map='cuda:0')
model.eval()
device = next(model.parameters()).device
print(f"  Loaded on {device}")

all_results = {}

determ_llm.disable()
all_results['A_BF16'] = run_diagnostics(model, tokenizer, device, 'A_BF16_baseline')

determ_llm.enable()
all_results['B_linear_only'] = run_diagnostics(model, tokenizer, device, 'B_linear_only')

determ_llm.disable()

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("DIAGNOSTIC SUMMARY")
print('='*60)
for cond, res in all_results.items():
    print(f"\n  {cond}:")
    print(f"    Q0 bs=1 reprod: {res['bs1_reproducibility']}")
    print(f"    Q2 cross-row:   ", end='')
    for bs_k, v in res['cross_row_invariance'].items():
        print(f"bs={bs_k}: same={v['all_rows_same_token']} diff={v['max_logit_diff_across_rows']:.2e}  ",
              end='')
    print()
    print(f"    Q3 cross-M:     ", end='')
    for bs_k, v in res['cross_m_invariance']['batches'].items():
        print(f"bs={bs_k}: match={v['match']} mm={v['mm']}  ", end='')
    print()

with open(OUT_FILE, 'w') as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"\nSaved → {OUT_FILE}")
