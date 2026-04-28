"""
Test condition C correctly: determ_llm.enable(attn=True) using production API.
Merges results with exp_det_decompose.json to verify that the attention patch
eliminates the 16 remaining token mismatches from F.linear-only condition.
"""

import os, sys, json, hashlib
import torch

DLLM_DIR = os.path.dirname(os.path.abspath(__file__))
FP32_DIR  = os.path.join(DLLM_DIR, '..', 'FP32')
sys.path.insert(0, DLLM_DIR)
sys.path.insert(0, FP32_DIR)

os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

from transformers import AutoModelForCausalLM, AutoTokenizer
import determ_llm

MODEL_PATH = '/home/kec23008/docker-sys/Models/Llama-3.2-1B-Instruct'
OUT_FILE   = os.path.join(DLLM_DIR, 'exp_det_attn_patch.json')

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
            out    = model(input_ids=ids)
            logits = out.logits[0, -1]
            lps    = torch.nn.functional.log_softmax(logits, dim=-1)
            tok    = lps.argmax().item()
            token_ids.append(tok)
            token_lps.append(lps[tok].item())
            next_col = torch.full((batch_size, 1), tok, dtype=ids.dtype, device=device)
            ids = torch.cat([ids, next_col], dim=1)
    return token_ids, token_lps


def seq_hash(token_ids):
    return hashlib.md5(str(token_ids).encode()).hexdigest()[:12]


print(f"Loading {MODEL_PATH} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map='cuda:0',
    attn_implementation='eager',
)
model.eval()
device = next(model.parameters()).device
print(f"  Loaded on {device}, attn=eager")

# Enable DetermLLM with BOTH linear and attention patches (production API)
determ_llm.enable(attn=True)
print(f"  DetermLLM status: {determ_llm.status()}")

results = {}
total_mismatches = 0
all_match_global = True

for p_idx, prompt in enumerate(PROMPTS):
    ref_toks, ref_lps = generate_tokens(model, tokenizer, prompt, 1, GEN_LEN, device)
    ref_hash = seq_hash(ref_toks)
    prompt_res = {'ref_hash': ref_hash, 'batches': {}}

    for bs in BATCH_SIZES:
        bs_toks, bs_lps = generate_tokens(model, tokenizer, prompt, bs, GEN_LEN, device)
        bs_hash = seq_hash(bs_toks)
        seq_match = (bs_hash == ref_hash)
        tok_mm = sum(a != b for a, b in zip(ref_toks, bs_toks))
        lp_diffs = [abs(a-b) for a, b in zip(ref_lps, bs_lps)]
        mean_lp = sum(lp_diffs) / len(lp_diffs)
        max_lp  = max(lp_diffs)
        first_div = next((i for i,(a,b) in enumerate(zip(ref_toks,bs_toks)) if a!=b), -1)

        prompt_res['batches'][f'bs{bs}'] = {
            'seq_match': seq_match,
            'token_mismatches': tok_mm,
            'first_divergence': first_div,
            'mean_lp_diff': mean_lp,
            'max_lp_diff': max_lp,
        }
        status = '✓' if seq_match else f'✗ (first_div={first_div}, mismatches={tok_mm})'
        if bs in [8, 32, 64] or not seq_match:
            print(f"  p{p_idx} bs={bs:>2}: {status}  mean|ΔlogP|={mean_lp:.4f}")
        if not seq_match:
            all_match_global = False
            total_mismatches += tok_mm

    all_match = all(v['seq_match'] for v in prompt_res['batches'].values())
    tot_mm    = sum(v['token_mismatches'] for v in prompt_res['batches'].values())
    avg_lp    = sum(v['mean_lp_diff'] for v in prompt_res['batches'].values()) / len(BATCH_SIZES)
    prompt_res['all_match'] = all_match
    prompt_res['total_token_mismatches'] = tot_mm
    prompt_res['avg_lp_diff'] = avg_lp
    results[f'p{p_idx}'] = prompt_res

avg_lp_all = sum(results[f'p{i}']['avg_lp_diff'] for i in range(len(PROMPTS))) / len(PROMPTS)
total_tok_mm = sum(results[f'p{i}']['total_token_mismatches'] for i in range(len(PROMPTS)))

print(f"\n>>> C_linear_and_attn (production API): all_seq_match={all_match_global}, "
      f"total_tok_mismatches={total_tok_mm}, avg|ΔlogP|={avg_lp_all:.4f}")

output = {
    'condition': 'C_linear_and_attn_production',
    'all_sequences_match': all_match_global,
    'total_token_mismatches': total_tok_mm,
    'avg_lp_diff': avg_lp_all,
    'per_prompt': results,
}
with open(OUT_FILE, 'w') as f:
    json.dump(output, f, indent=2)
print(f"Saved → {OUT_FILE}")
