"""Sanity test: compare hybrid vs triton vs cublaslt on the same prompt.

Goal: understand why hybrid's accuracy (37%) is lower than cublaslt (40%) on
MATH500. Hybrid = cublaslt for N<=4096 + triton for N>4096, so if either
component alone is right, hybrid should be a merge of the two.

Approach: pick 1 prompt, run all 4 methods at bs=8, compare:
  1. First token of response: do cublaslt and hybrid agree?
  2. First divergence index between hybrid and cublaslt
  3. Full token sequences: how many tokens differ?

If hybrid diverges EARLY from cublaslt, it means the triton-selected path
(for N>4096, i.e., FFN/lm_head) is producing different outputs than expected.
"""
import os, sys, json, hashlib
import torch
import torch.nn.functional as F

DLLM = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DLLM)
sys.path.insert(0, os.path.join(DLLM, '..', 'FP32'))

os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

from transformers import AutoModelForCausalLM, AutoTokenizer
import determ_llm
import layercast
from run_math500_eval import apply_chat_template, generate_one, set_method

MODEL_PATH = '/home/kec23008/docker-sys/Models/DeepSeek-R1-Distill-Qwen-7B'

# Use one MATH500 problem
with open(os.path.join(DLLM, 'math500_cached.json')) as f:
    problems = json.load(f)
prob = problems[1]   # a problem where methods disagreed on correctness (p1 in our data)
print(f"Problem: {prob['problem'][:120]}...")
print(f"Gold: {prob['answer']}\n")

print("Loading DeepSeek-R1-Distill-Qwen-7B...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, dtype=torch.bfloat16, device_map='cuda:0',
    attn_implementation='eager',
)
model.eval()
device = next(model.parameters()).device

prompt_text = apply_chat_template(tokenizer, prob['problem'])
gen_len = 200   # small to make it fast
bs = 8

results = {}
for method in ['bf16_baseline', 'dllm_cublaslt', 'dllm_triton', 'dllm_hybrid']:
    set_method(method)
    text, toks, lps = generate_one(model, tokenizer, prompt_text, bs, gen_len, device, seed=0)
    results[method] = {
        'text': text,
        'toks': toks,
        'lps': lps,
        'hash': hashlib.md5(str(toks).encode()).hexdigest()[:8],
    }
    print(f"{method:<20} hash={results[method]['hash']} first_tokens={toks[:10]}")

# Pairwise divergence analysis
print("\n=== Pairwise first divergence token position ===")
methods = list(results.keys())
print(f"{'':<20}", ' '.join(f'{m[:10]:>10}' for m in methods))
for a in methods:
    row = f"{a:<20}"
    for b in methods:
        if a == b:
            row += '     -    '
        else:
            ta = results[a]['toks']
            tb = results[b]['toks']
            div = next((i for i,(x,y) in enumerate(zip(ta, tb)) if x!=y), -1)
            row += f'     {div:>5}'
    print(row)

# Detailed comparison: cublaslt vs hybrid (should be identical for small-N, differ for large-N)
print("\n=== cublaslt vs hybrid: token-by-token first 20 ===")
tc = results['dllm_cublaslt']['toks'][:20]
th = results['dllm_hybrid']['toks'][:20]
lc = results['dllm_cublaslt']['lps'][:20]
lh = results['dllm_hybrid']['lps'][:20]
print(f"{'i':>3} {'cublaslt':<10} {'hybrid':<10} {'match':<6} {'lp_cublaslt':>14} {'lp_hybrid':>14}")
for i in range(min(20, len(tc), len(th))):
    print(f"{i:>3} {tokenizer.decode([tc[i]]):<10} {tokenizer.decode([th[i]]):<10} {'==' if tc[i]==th[i] else '!!':<6} {lc[i]:>14.6f} {lh[i]:>14.6f}")

# Hashes summary
print("\n=== Sequence hashes ===")
for m in methods:
    print(f"  {m:<20} {results[m]['hash']}")

# Key question: does hybrid exactly equal cublaslt+triton composition?
# Hybrid uses cublaslt for N<=4096, triton for N>4096
# In DeepSeek-7B: attn projs (N=3584 or 512) go cublaslt, FFN (N=18944) and lm_head (N=152064) go triton
# So hybrid output should differ from BOTH pure cublaslt and pure triton
# Let's verify this is expected.

print("\n=== Expected routing in hybrid for DeepSeek-7B ===")
print("  attn Q/K/V/O (N=512, 3584)  -> cuBLASLt branch")
print("  FFN gate/up (N=18944)       -> Triton branch")
print("  FFN down (N=3584)           -> cuBLASLt branch")
print("  lm_head (N=152064)          -> Triton branch")
