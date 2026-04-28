#!/usr/bin/env python3
"""Validate Avg_Std@top1_prob via the FULL vLLM stack (cudagraph + StaticCache).

Run via the vllm_test conda env:
  LD_LIBRARY_PATH=...:$LD_LIBRARY_PATH \
  VLLM_BATCH_INVARIANT=1 VLLM_ATTENTION_BACKEND=FLASH_ATTN \
  /home/kec23008/miniconda3/envs/vllm_test/bin/python research/validate_avg_std_vllm_native.py

Compares:
  vllm BF16 (no batch_invariant)            — vLLM stack baseline
  vllm BF16 + VLLM_BATCH_INVARIANT=1        — vLLM's official deterministic path
"""
import os, sys, time, json, math
import numpy as np

# Sanity: must be invoked with VLLM_BATCH_INVARIANT and proper LD path.
os.environ.setdefault('VLLM_USE_V1', '1')

from vllm import LLM, SamplingParams
import vllm.envs as venvs
print(f'vllm batch_invariant: {venvs.VLLM_BATCH_INVARIANT}', flush=True)

MODEL = '/home/kec23008/docker-sys/Models/DeepSeek-R1-Distill-Qwen-7B'
MATH500 = '/home/kec23008/docker-sys/dllm/research/math500_cached.json'
N_PROB = int(os.environ.get('N_PROB', 5))
MAX_NEW = 256
BS_LIST = [1, 2, 4, 8, 16]
SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."

_BI_ON = os.environ.get('VLLM_BATCH_INVARIANT', '0') in ('1', 'true', 'TRUE')
_SUFFIX = f'_n{N_PROB}' if N_PROB != 5 else ''
OUT_JSON = (
    f'/home/kec23008/docker-sys/dllm/research/exp_validate/vllm_native_bi_results{_SUFFIX}.json'
    if _BI_ON
    else f'/home/kec23008/docker-sys/dllm/research/exp_validate/vllm_native_bf16_results{_SUFFIX}.json'
)


def first_div(ref, run, max_len=MAX_NEW):
    n = min(len(ref), len(run), max_len)
    for i in range(n):
        if ref[i] != run[i]: return i
    return max_len


def main():
    print(f'Loading {MODEL} ...', flush=True)
    # Pin attention_config to FLASH_ATTN for both BF16 and BATCH_INVARIANT runs
    # so the comparison is apples-to-apples (BI mode requires FLASH_ATTN).
    llm_kwargs = dict(
        model=MODEL, dtype='bfloat16',
        gpu_memory_utilization=0.6,
        max_model_len=2048,
        enforce_eager=False,
        enable_prefix_caching=False,
    )
    try:
        from vllm.config import AttentionConfig
        llm_kwargs['attention_config'] = AttentionConfig(backend='FLASH_ATTN')
    except Exception:
        llm_kwargs['attention_backend'] = 'FLASH_ATTN'
    llm = LLM(**llm_kwargs)
    tok = llm.get_tokenizer()
    problems = json.load(open(MATH500))[:N_PROB]
    print(f'  N_PROB={N_PROB}  BS_LIST={BS_LIST}  MAX_NEW={MAX_NEW}', flush=True)

    sp = SamplingParams(temperature=0.0, max_tokens=MAX_NEW, logprobs=1)

    rows = []
    total_t = 0.0
    for pi, prob in enumerate(problems):
        chat = tok.apply_chat_template(
            [{"role":"user","content":f"{prob['problem']}\n\n{SYSTEM}"}],
            tokenize=False, add_generation_prompt=True)

        tokens_by_bs = {}
        probs_by_bs  = {}
        per_bs_t = []
        for bs in BS_LIST:
            prompts = [chat] * bs
            torch_sync = lambda: __import__('torch').cuda.synchronize()
            torch_sync(); t0 = time.perf_counter()
            outs = llm.generate(prompts, sp, use_tqdm=False)
            torch_sync(); dt = time.perf_counter() - t0
            per_bs_t.append(dt); total_t += dt

            # Extract row 0 tokens + per-step top-1 prob (the chosen token's prob)
            o0 = outs[0].outputs[0]
            toks = list(o0.token_ids)[:MAX_NEW]
            # logprobs is list[Optional[Dict[token, Logprob]]] of length len(token_ids)
            # Top-1 prob = exp(logprobs of the chosen token)
            lps = o0.logprobs
            prbs = []
            for i, t in enumerate(toks):
                lp_dict = lps[i] if lps and i < len(lps) else None
                if lp_dict and t in lp_dict:
                    prbs.append(float(math.exp(lp_dict[t].logprob)))
                else:
                    prbs.append(0.0)
            tokens_by_bs[bs] = toks
            probs_by_bs[bs]  = prbs

        ref = tokens_by_bs[1]
        fdis = {bs: first_div(ref, tokens_by_bs[bs]) for bs in BS_LIST if bs != 1}
        pre_div = min([MAX_NEW] + list(fdis.values()))
        # build prob matrix; pad with 0s if uneven length
        Lcap = MAX_NEW
        mat = np.zeros((len(BS_LIST), Lcap))
        for bi, bs in enumerate(BS_LIST):
            p = probs_by_bs[bs]
            mat[bi, :len(p)] = p[:Lcap]
        std_per_pos = mat.std(axis=0)
        avg_std = float(std_per_pos[:pre_div].mean()) if pre_div > 0 else None

        rows.append({'problem_idx': pi, 'pre_div': pre_div, 'avg_std': avg_std,
                     'per_bs_t': per_bs_t})
        print(f'  problem {pi}: pre_div={pre_div} avg_std={avg_std:.3e} '
              f't={[f"{x:.1f}" for x in per_bs_t]}', flush=True)

    valid = [r['avg_std'] for r in rows if r['avg_std'] is not None]
    avg_std = float(np.mean(valid)) if valid else None
    print(f'\nTOTAL: {total_t:.1f}s  Avg_Std@top1={avg_std:.3e}', flush=True)
    summary = {'mode': 'vllm_native', 'batch_invariant': venvs.VLLM_BATCH_INVARIANT,
               'total_s': total_t, 'avg_std': avg_std, 'rows': rows}
    json.dump(summary, open(OUT_JSON, 'w'), indent=2)
    print(f'Saved: {OUT_JSON}')


if __name__ == '__main__':
    main()
