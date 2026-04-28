#!/usr/bin/env python3
"""
Validate Avg_Std@top1_prob with our custom decode loop using StaticCache (eager mode).

Compares:
  BF16 + HF generate               (baseline, DynamicCache)
  SRP-FP32 + HF generate           (current path, DynamicCache, ~50% overhead)
  SRP-FP32 + Eager StaticCache     (decode loop on static buffers, no graph)

Goal: confirm that switching to StaticCache preserves Avg_Std=0 and
quantify the overhead reduction without cudagraph.
"""
import sys, os, time, json, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from research.methods import method_BF16, method_SRP_FP32, ALL_SITES
from research.cuda_graph_decode import CudaGraphDecoder

MODEL = '/home/kec23008/docker-sys/Models/DeepSeek-R1-Distill-Qwen-7B'
MATH500 = '/home/kec23008/docker-sys/dllm/research/math500_cached.json'
N_PROB = 5
MAX_NEW = 256
BS_LIST = [1, 2, 4, 8, 16]
SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."

OUT_DIR = '/home/kec23008/docker-sys/dllm/research/exp_validate'
os.makedirs(OUT_DIR, exist_ok=True)


def wrap(tok, problem):
    msgs = [{"role": "user", "content": f"{problem}\n\n{SYSTEM}"}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def first_div(ref, run, max_len=MAX_NEW):
    n = min(len(ref), len(run), max_len)
    for i in range(n):
        if ref[i] != run[i]:
            return i
    return max_len


def hf_greedy(model, tok, ids, max_new):
    """Vanilla HF greedy, capturing per-token top-1 probs via manual decode."""
    L_in = ids.shape[1]
    bs = ids.shape[0]
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True)
        past = out.past_key_values
        logits = out.logits[:, -1]
        p = F.softmax(logits.float(), dim=-1)
        max_p, max_t = p.max(dim=-1)
        toks = [int(max_t[0].item())]
        prbs = [float(max_p[0].item())]
        cur = max_t.unsqueeze(1)
        for _ in range(max_new - 1):
            out = model(input_ids=cur, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[:, -1]
            p = F.softmax(logits.float(), dim=-1)
            max_p, max_t = p.max(dim=-1)
            toks.append(int(max_t[0].item()))
            prbs.append(float(max_p[0].item()))
            cur = max_t.unsqueeze(1)
    return toks, prbs


def run_method(model, tok, problems, name, ctx_factory, decoder_kind):
    """decoder_kind: 'hf' (DynamicCache, hf-style) or 'static' (StaticCache, eager)."""
    print(f"\n=== {name} ({decoder_kind}) ===", flush=True)

    all_per_problem_std = []
    all_pre_div = []
    total_t = 0.0

    with ctx_factory(model):
        max_seq_len = max(
            tok(wrap(tok, p['problem']), return_tensors='pt')['input_ids'].shape[1]
            for p in problems
        ) + MAX_NEW + 8
        decoder = CudaGraphDecoder(model, max_seq_len=max_seq_len) if decoder_kind == 'static' else None

        for pi, prob in enumerate(problems):
            prompt = wrap(tok, prob['problem'])
            enc = tok(prompt, return_tensors='pt')['input_ids'].cuda()

            tokens_by_bs = {}
            probs_by_bs = {}
            problem_t = []
            for bs in BS_LIST:
                ids = enc.repeat(bs, 1).contiguous()
                torch.cuda.synchronize(); t0 = time.perf_counter()
                if decoder_kind == 'hf':
                    t, p = hf_greedy(model, tok, ids, MAX_NEW)
                else:
                    t, p = decoder.greedy(ids, max_new=MAX_NEW, use_graph=False)
                torch.cuda.synchronize(); dt = time.perf_counter() - t0
                tokens_by_bs[bs] = t
                probs_by_bs[bs] = p
                problem_t.append(dt)
                total_t += dt

            ref = tokens_by_bs[1]
            fdis = {bs: first_div(ref, tokens_by_bs[bs]) for bs in BS_LIST if bs != 1}
            pre_div = min([MAX_NEW] + list(fdis.values()))
            all_pre_div.append(pre_div)

            mat = np.array([probs_by_bs[bs] for bs in BS_LIST])
            std_pos = mat.std(axis=0)
            avg_std = float(std_pos[:pre_div].mean()) if pre_div > 0 else None
            all_per_problem_std.append(avg_std)

            print(f"  problem {pi}: pre_div={pre_div}  avg_std={avg_std:.3e}  "
                  f"per-bs t={[f'{x:.1f}' for x in problem_t]}", flush=True)

    valid = [s for s in all_per_problem_std if s is not None]
    avg_std = float(np.mean(valid)) if valid else None
    print(f"  TOTAL: {total_t:.1f}s  Avg_Std@top1={avg_std:.3e}", flush=True)
    return {'name': name, 'decoder_kind': decoder_kind,
            'total_s': total_t, 'avg_std': avg_std,
            'per_problem_std': all_per_problem_std,
            'per_problem_pre_div': all_pre_div}


def main():
    print(f"Loading {MODEL}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map={'':0}, attn_implementation='sdpa').eval()

    problems = json.load(open(MATH500))[:N_PROB]
    print(f"  N_PROB={N_PROB}  BS={BS_LIST}  MAX_NEW={MAX_NEW}", flush=True)

    results = []
    results.append(run_method(model, tok, problems, "BF16",     method_BF16, 'hf'))
    results.append(run_method(model, tok, problems, "BF16",     method_BF16, 'static'))
    results.append(run_method(model, tok, problems, "SRP-FP32", lambda m: method_SRP_FP32(m, ALL_SITES), 'hf'))
    results.append(run_method(model, tok, problems, "SRP-FP32", lambda m: method_SRP_FP32(m, ALL_SITES), 'static'))

    print(f"\n{'='*84}\nFINAL\n{'='*84}")
    print(f"  {'method':<10} {'decoder':<8} {'total':>8} {'Avg_Std':>12}")
    bf16_hf_t = next(r['total_s'] for r in results if r['name']=='BF16' and r['decoder_kind']=='hf')
    for r in results:
        ovh = r['total_s'] / bf16_hf_t * 100 - 100
        print(f"  {r['name']:<10} {r['decoder_kind']:<8} {r['total_s']:>7.1f}s {r['avg_std']:>12.3e}  ({ovh:+.1f}% vs BF16/hf)")

    out = os.path.join(OUT_DIR, 'staticcache_validate.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == '__main__':
    main()
