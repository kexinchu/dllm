#!/usr/bin/env python3
"""Test combo: vLLM matmul/norm/softmax overrides + our Triton attention.

Hypothesis: vLLM's matmul_persistent is more tuned than our Triton det_gemm
(closer to cuBLAS speed). Combined with our bit-exact attention kernel, this
might give Avg_Std=0 at lower overhead than either alone.
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from research.methods import method_BF16
from research.srp_kernels import batch_invariant_vllm as biv

MODEL = '/home/kec23008/docker-sys/Models/DeepSeek-R1-Distill-Qwen-7B'
MATH500 = '/home/kec23008/docker-sys/dllm/research/math500_cached.json'
N_PROB = 5
MAX_NEW = 256
BS_LIST = [1, 2, 4, 8, 16]
SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."


def wrap(tok, problem):
    msgs = [{"role": "user", "content": f"{problem}\n\n{SYSTEM}"}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def first_div(ref, run, max_len=MAX_NEW):
    n = min(len(ref), len(run), max_len)
    for i in range(n):
        if ref[i] != run[i]: return i
    return max_len


def manual_decode(model, ids, max_new):
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True)
        past = out.past_key_values
        last = out.logits[:, -1]
        p = F.softmax(last.float(), dim=-1)
        max_p, max_t = p.max(dim=-1)
        toks = [int(max_t[0].item())]; prbs = [float(max_p[0].item())]
        cur = max_t.unsqueeze(1)
        for _ in range(max_new - 1):
            out = model(input_ids=cur, past_key_values=past, use_cache=True)
            past = out.past_key_values
            last = out.logits[:, -1]
            p = F.softmax(last.float(), dim=-1)
            max_p, max_t = p.max(dim=-1)
            toks.append(int(max_t[0].item())); prbs.append(float(max_p[0].item()))
            cur = max_t.unsqueeze(1)
    return toks, prbs


def run(model, tok, problems, name):
    print(f"\n=== {name} ===", flush=True)
    rows = []; total = 0.0
    for pi, p in enumerate(problems):
        prompt = wrap(tok, p['problem'])
        enc = tok(prompt, return_tensors='pt')['input_ids'].cuda()
        toks_by, probs_by = {}, {}
        per = []
        for bs in BS_LIST:
            ids = enc.repeat(bs, 1).contiguous()
            torch.cuda.synchronize(); t0 = time.perf_counter()
            t, pr = manual_decode(model, ids, MAX_NEW)
            torch.cuda.synchronize(); dt = time.perf_counter() - t0
            toks_by[bs] = t; probs_by[bs] = pr
            per.append(dt); total += dt
        ref = toks_by[1]
        fdis = {bs: first_div(ref, toks_by[bs]) for bs in BS_LIST if bs != 1}
        pre = min([MAX_NEW] + list(fdis.values()))
        mat = np.array([probs_by[bs] for bs in BS_LIST])
        std = mat.std(axis=0)
        astd = float(std[:pre].mean()) if pre > 0 else None
        rows.append({'problem_idx': pi, 'pre_div': pre, 'avg_std': astd, 'per_bs_t': per})
        print(f"  problem {pi}: pre_div={pre} avg_std={astd:.3e} t={[f'{x:.1f}' for x in per]}", flush=True)
    valid = [r['avg_std'] for r in rows if r['avg_std'] is not None]
    avg = float(np.mean(valid)) if valid else None
    print(f"  TOTAL: {total:.1f}s  Avg_Std={avg:.3e}", flush=True)
    return {'name': name, 'total_s': total, 'avg_std': avg, 'rows': rows}


def main():
    print(f"Loading {MODEL}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map={'':0}, attn_implementation='sdpa').eval()
    problems = json.load(open(MATH500))[:N_PROB]
    print(f"  N_PROB={N_PROB}", flush=True)

    results = []
    # 1. BF16 baseline
    with method_BF16(model):
        results.append(run(model, tok, problems, "BF16"))

    # 2. vLLM batch_invariant (Linear/Norm/Softmax/BMM aten overrides only)
    biv.enable_batch_invariant_mode()
    results.append(run(model, tok, problems, "vLLM-BI alone (no attention patch)"))

    # 3. vLLM batch_invariant + our Triton FA-style attention
    # Patch attention via the same context manager flow but ONLY attention site
    from research.methods import method_SRP_FP32
    with method_SRP_FP32(model, ('attention',)):
        results.append(run(model, tok, problems, "vLLM-BI + Triton attn"))

    # 4. our full SRP for reference
    # batch_invariant is "sticky" (no clean disable) so we'd keep its overrides;
    # this is essentially "our SRP overlaid on vLLM-BI's overrides"
    with method_SRP_FP32(model, ('linear', 'rmsnorm', 'attention', 'softmax')):
        results.append(run(model, tok, problems, "vLLM-BI + full SRP overlay"))

    print(f"\n{'='*84}\nFINAL\n{'='*84}")
    bf = results[0]['total_s']
    print(f"  {'method':<40} {'total':>8} {'Avg_Std':>12} {'overhead':>10}")
    for r in results:
        ovh = (r['total_s']/bf - 1) * 100
        astd = f"{r['avg_std']:.3e}" if r['avg_std'] is not None else "N/A"
        print(f"  {r['name']:<40} {r['total_s']:>7.1f}s {astd:>12} {ovh:>+9.1f}%")
    json.dump(results, open('/home/kec23008/docker-sys/dllm/research/exp_validate/combo_results.json', 'w'), indent=2)


if __name__ == '__main__':
    main()
