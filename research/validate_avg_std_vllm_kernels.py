#!/usr/bin/env python3
"""Validate Avg_Std@top1_prob using vendored vLLM batch_invariant kernels.

This drops vLLM's `enable_batch_invariant_mode()` (which overrides aten ops
via torch.library) into our HF stack. Aten overrides are cudagraph- and
dynamo-friendly, so they should give us the cost reduction we couldn't get
with monkey-patched nn.Linear.forward (Step C).

Compares:
  BF16          baseline
  SRP-FP32 (HF) our previous Triton monkey-patch path (~50% overhead)
  vLLM-BI       vLLM batch_invariant aten overrides (target: ≤15% overhead, Avg_Std≈0)
"""
import sys, os, time, json, hashlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from research.methods import method_BF16, method_SRP_FP32, ALL_SITES

MODEL = '/home/kec23008/docker-sys/Models/DeepSeek-R1-Distill-Qwen-7B'
MATH500 = '/home/kec23008/docker-sys/dllm/research/math500_cached.json'
N_PROB = int(os.environ.get('N_PROB', 5))
MAX_NEW = 256
BS_LIST = [1, 2, 4, 8, 16]
SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."

_SUFFIX = f'_n{N_PROB}' if N_PROB != 5 else ''
OUT_JSON = f'/home/kec23008/docker-sys/dllm/research/exp_validate/vllm_kernels_results{_SUFFIX}.json'
OUT_LOG = f'/home/kec23008/docker-sys/dllm/research/exp_validate/vllm_kernels_run{_SUFFIX}.log'


def wrap(tok, problem):
    msgs = [{"role": "user", "content": f"{problem}\n\n{SYSTEM}"}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def first_div(ref, run, max_len=MAX_NEW):
    n = min(len(ref), len(run), max_len)
    for i in range(n):
        if ref[i] != run[i]:
            return i
    return max_len


def manual_decode(model, ids, max_new):
    """Greedy with per-token top-1 prob capture (row 0)."""
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


def run_method(model, tok, problems, name, ctx_factory):
    print(f"\n=== {name} ===", flush=True)
    rows = []
    total_t = 0.0
    with ctx_factory(model):
        for pi, prob in enumerate(problems):
            prompt = wrap(tok, prob['problem'])
            enc = tok(prompt, return_tensors='pt')['input_ids'].cuda()

            tokens_by_bs, probs_by_bs = {}, {}
            problem_t = []
            for bs in BS_LIST:
                ids = enc.repeat(bs, 1).contiguous()
                torch.cuda.synchronize(); t0 = time.perf_counter()
                t, p = manual_decode(model, ids, MAX_NEW)
                torch.cuda.synchronize(); dt = time.perf_counter() - t0
                tokens_by_bs[bs] = t
                probs_by_bs[bs] = p
                problem_t.append(dt)
                total_t += dt

            ref = tokens_by_bs[1]
            fdis = {bs: first_div(ref, tokens_by_bs[bs]) for bs in BS_LIST if bs != 1}
            pre_div = min([MAX_NEW] + list(fdis.values()))
            mat = np.array([probs_by_bs[bs] for bs in BS_LIST])
            std_pos = mat.std(axis=0)
            avg_std = float(std_pos[:pre_div].mean()) if pre_div > 0 else None

            rows.append({'problem_idx': pi, 'pre_div': pre_div, 'avg_std': avg_std,
                         'per_bs_t': problem_t})
            print(f"  problem {pi}: pre_div={pre_div}  avg_std={avg_std:.3e}  "
                  f"t={[f'{x:.1f}' for x in problem_t]}", flush=True)

    valid = [r['avg_std'] for r in rows if r['avg_std'] is not None]
    avg_std = float(np.mean(valid)) if valid else None
    print(f"  TOTAL: {total_t:.1f}s  Avg_Std@top1={avg_std:.3e}", flush=True)
    return {'name': name, 'total_s': total_t, 'avg_std': avg_std, 'rows': rows}


def main():
    print(f"Loading {MODEL}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map={'':0}, attn_implementation='sdpa').eval()

    problems = json.load(open(MATH500))[:N_PROB]
    print(f"  N_PROB={N_PROB}  BS={BS_LIST}  MAX_NEW={MAX_NEW}", flush=True)

    results = []
    # 1) BF16 baseline
    results.append(run_method(model, tok, problems, "BF16", method_BF16))

    # 2) SRP-FP32 (Triton monkey-patch, our existing path)
    results.append(run_method(model, tok, problems, "SRP-FP32 (HF Triton)",
                              lambda m: method_SRP_FP32(m, ALL_SITES)))

    # 3) vLLM batch_invariant aten overrides
    from contextlib import contextmanager
    from research.srp_kernels import batch_invariant_vllm as biv
    @contextmanager
    def vllm_bi_ctx(m):
        biv.enable_batch_invariant_mode()
        try: yield
        finally:
            # batch_invariant has no clean disable; we just leave it on for the rest
            # of the process. (Future: could drop the lib registry.)
            pass
    results.append(run_method(model, tok, problems, "vLLM batch_invariant", vllm_bi_ctx))

    print(f"\n{'='*84}\nFINAL\n{'='*84}")
    bf16_t = results[0]['total_s']
    print(f"  {'method':<25} {'total':>8} {'Avg_Std':>12} {'overhead':>10}")
    for r in results:
        ovh = (r['total_s'] / bf16_t - 1) * 100
        astd = f"{r['avg_std']:.3e}" if r['avg_std'] is not None else "N/A"
        print(f"  {r['name']:<25} {r['total_s']:>7.1f}s {astd:>12} {ovh:>+9.1f}%")

    with open(OUT_JSON, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUT_JSON}")


if __name__ == '__main__':
    main()
