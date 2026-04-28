"""
Repeat the decomposition experiment with the Triton backend.

Same protocol as run_det_decompose.py but determ_llm.enable(backend='triton').
Goal: verify that the Triton backend gives >=37x reduction on 1B like the
cuBLASLt backend did, and ideally better (bit-exact by construction).
"""
import os, sys, json, hashlib
import torch
import torch.nn.functional as F

DLLM_DIR = os.path.dirname(os.path.abspath(__file__))
FP32_DIR = os.path.join(DLLM_DIR, '..', 'FP32')
sys.path.insert(0, DLLM_DIR)
sys.path.insert(0, FP32_DIR)

os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE']      = '1'

from transformers import AutoModelForCausalLM, AutoTokenizer
import determ_llm

MODEL_PATH = '/home/kec23008/docker-sys/Models/Llama-3.2-1B-Instruct'
OUT_FILE   = os.path.join(DLLM_DIR, 'exp_decompose_triton.json')

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


def generate_tokens(model, tokenizer, prompt, batch_size, gen_len, device):
    enc = tokenizer(prompt, return_tensors='pt')['input_ids']
    ids = enc.repeat(batch_size, 1).to(device)
    token_ids, token_lps = [], []
    with torch.no_grad():
        for _ in range(gen_len):
            out = model(input_ids=ids)
            logits = out.logits[0, -1]
            lps = F.log_softmax(logits, dim=-1)
            tok = lps.argmax().item()
            token_ids.append(tok)
            token_lps.append(lps[tok].item())
            next_col = torch.full((batch_size, 1), tok, dtype=ids.dtype, device=device)
            ids = torch.cat([ids, next_col], dim=1)
    return token_ids, token_lps


def seq_hash(tokens):
    return hashlib.md5(str(tokens).encode()).hexdigest()[:12]


def run_condition(model, tokenizer, device, label):
    print(f"\n  Condition: {label}")
    cond = {}
    all_match = True
    for p_idx, prompt in enumerate(PROMPTS):
        ref_toks, ref_lps = generate_tokens(model, tokenizer, prompt, 1, GEN_LEN, device)
        ref_hash = seq_hash(ref_toks)
        prompt_res = {'ref_hash': ref_hash, 'batches': {}}
        for bs in BATCH_SIZES:
            bs_toks, bs_lps = generate_tokens(model, tokenizer, prompt, bs, GEN_LEN, device)
            seq_match = (seq_hash(bs_toks) == ref_hash)
            tok_mm = sum(a != b for a, b in zip(ref_toks, bs_toks))
            lp_diffs = [abs(a - b) for a, b in zip(ref_lps, bs_lps)]
            prompt_res['batches'][f'bs{bs}'] = {
                'seq_match': seq_match,
                'token_mismatches': tok_mm,
                'mean_lp_diff': sum(lp_diffs) / len(lp_diffs),
                'max_lp_diff': max(lp_diffs),
            }
            if not seq_match:
                all_match = False
        mm_p = sum(v['token_mismatches'] for v in prompt_res['batches'].values())
        lp_p = sum(v['mean_lp_diff'] for v in prompt_res['batches'].values()) / len(BATCH_SIZES)
        prompt_res['total_token_mismatches'] = mm_p
        prompt_res['avg_lp_diff'] = lp_p
        prompt_res['all_match'] = all(v['seq_match'] for v in prompt_res['batches'].values())
        cond[f'p{p_idx}'] = prompt_res
    total_mm = sum(cond[k]['total_token_mismatches'] for k in cond)
    avg_lp   = sum(cond[k]['avg_lp_diff'] for k in cond) / len(cond)
    print(f"  >>> {label}: total_mm={total_mm}, avg|ΔlogP|={avg_lp:.4f}, all_match={all_match}")
    return {
        'all_sequences_match': all_match,
        'total_token_mismatches': total_mm,
        'avg_lp_diff': avg_lp,
        'per_prompt': cond,
    }


print(f"Loading {MODEL_PATH} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, dtype=torch.bfloat16, device_map='cuda:0',
    attn_implementation='eager')
model.eval()
device = next(model.parameters()).device
print(f"  Loaded on {device}")

all_results = {}

# A: BF16 baseline
determ_llm.disable()
all_results['A_BF16_baseline'] = run_condition(model, tokenizer, device, 'A_BF16_baseline')

# B: DetermLLM Triton backend
determ_llm.enable(backend='triton')
all_results['B_dllm_triton'] = run_condition(model, tokenizer, device, 'B_dllm_triton')
determ_llm.disable()

# B_cublaslt: DetermLLM cuBLASLt backend (for comparison)
determ_llm.enable(backend='cublaslt')
all_results['B_dllm_cublaslt'] = run_condition(model, tokenizer, device, 'B_dllm_cublaslt')
determ_llm.disable()

print(f"\n{'='*65}")
print("SUMMARY: Decomposition — Llama-3.2-1B, Triton vs cuBLASLt backends")
print(f"{'='*65}")
print(f"{'Condition':<22} {'All?':>6} {'Tot MM':>8} {'Avg|ΔlogP|':>12}")
print('-' * 55)
for cond, cres in all_results.items():
    m = '✓' if cres['all_sequences_match'] else '✗'
    print(f"  {cond:<20} {m:>6} {cres['total_token_mismatches']:>8} {cres['avg_lp_diff']:>12.4f}")

with open(OUT_FILE, 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved → {OUT_FILE}")
