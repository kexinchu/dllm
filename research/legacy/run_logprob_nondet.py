"""
Log-probability non-determinism experiment.

BF16 batch non-determinism in logits causes variance in per-token log-probabilities.
This has direct consequences for:
  - RL fine-tuning (PPO/GRPO): reward signal variance from log-prob noise
  - Distillation (KL divergence): teacher KL differs by batch size
  - Beam search: score differences flip beam selection

This script quantifies the log-prob variance across batch sizes 1..32
(simulating continuous batching where M changes each step).

Output: research/exp_logprob_nondet.json
"""

import os, sys, json, torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

MODEL_PATH = '/home/kec23008/docker-sys/Models/Llama-3.2-1B-Instruct'
OUT_FILE   = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'exp_logprob_nondet.json')

from transformers import AutoModelForCausalLM, AutoTokenizer
import determ_llm

PROMPTS = [
    "The Eiffel Tower is located in",
    "In Python, the function to sort a list is",
    "The chemical formula for water is",
    "Albert Einstein developed the theory of",
    "The capital of Japan is",
    "The largest ocean on Earth is the",
    "The author of 'Pride and Prejudice' is",
    "In mathematics, the value of pi is approximately",
    "The CPU stands for central processing",
    "Neural networks are inspired by the human",
]

GEN_LEN = 64
BATCH_SIZES = [1, 2, 4, 8, 16, 32]

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map='cuda:0')
model.eval()
device = next(model.parameters()).device
print(f"Model on {device}")


def compute_logprobs(prompt, bs, mode='bf16'):
    """Generate GEN_LEN tokens, return per-token log-probs at each step."""
    if mode == 'determ':
        determ_llm.enable()
    else:
        determ_llm.disable()

    enc = tokenizer(prompt, return_tensors='pt')['input_ids']
    sl  = enc.shape[1]
    ids = enc.repeat(bs, 1).to(device)
    pos = torch.arange(sl, device=device).unsqueeze(0).repeat(bs, 1)

    token_logprobs = []  # log P(chosen token | context) from row 0
    tokens         = []

    with torch.no_grad():
        for step in range(GEN_LEN):
            logits = model(input_ids=ids, position_ids=pos).logits  # [bs, seq, vocab]
            lp = F.log_softmax(logits[0, -1], dim=-1)  # [vocab]
            tok = lp.argmax().item()
            tokens.append(tok)
            token_logprobs.append(lp[tok].item())
            ids = torch.tensor([[tok]], device=device).repeat(bs, 1)
            pos = torch.tensor([[sl + step]], device=device).repeat(bs, 1)

    determ_llm.disable()
    return tokens, token_logprobs


results = {}

for mode_label, mode_key in [("BF16 baseline", "bf16"), ("DetermLLM", "determ")]:
    print(f"\n{'='*50}")
    print(f"Mode: {mode_label}")
    mode_results = {}

    for p_idx, prompt in enumerate(PROMPTS):
        print(f"  Prompt {p_idx+1}/{len(PROMPTS)}: \"{prompt[:40]}\"")

        # Reference: bs=1
        ref_toks, ref_lp = compute_logprobs(prompt, 1, mode_key)

        prompt_results = {"ref_tokens": ref_toks, "ref_logprobs": ref_lp}
        total_kl = 0.0
        total_argmax_flips = 0

        for bs in BATCH_SIZES[1:]:  # skip bs=1 (that IS the reference)
            _, bs_lp = compute_logprobs(prompt, bs, mode_key)
            # Per-step log-prob difference
            lp_diffs = [abs(r - b) for r, b in zip(ref_lp, bs_lp)]
            argmax_flips = sum(1 for r, b in zip(ref_toks, _) if r != b)
            mean_lp_diff = sum(lp_diffs) / len(lp_diffs)
            max_lp_diff  = max(lp_diffs)
            total_kl    += mean_lp_diff
            total_argmax_flips += argmax_flips
            prompt_results[f"bs{bs}"] = {
                "mean_lp_diff": mean_lp_diff,
                "max_lp_diff": max_lp_diff,
                "argmax_flips": argmax_flips,
            }
            if argmax_flips > 0:
                print(f"    bs={bs}: {argmax_flips} argmax flips! mean_lp_diff={mean_lp_diff:.4f}")

        avg_kl = total_kl / len(BATCH_SIZES[1:])
        print(f"    Avg log-prob diff across batch sizes: {avg_kl:.4f}   "
              f"Total argmax flips: {total_argmax_flips}")
        mode_results[f"p{p_idx}"] = prompt_results

    results[mode_key] = {
        "label": mode_label,
        "per_prompt": mode_results,
    }


# Summary
print(f"\n{'='*50}")
print("SUMMARY: Mean |Δlog P| per token (bs=1 vs bs=32)")
print(f"{'Mode':<20} {'Avg|ΔlogP|':>12} {'Max|ΔlogP|':>12} {'ArgmaxFlips':>12}")
print("-" * 60)

for mkey, mlabel in [("bf16", "BF16 baseline"), ("determ", "DetermLLM")]:
    all_diffs, all_max, all_flips = [], [], 0
    for p_res in results[mkey]["per_prompt"].values():
        for bs in BATCH_SIZES[1:]:
            k = f"bs{bs}"
            if k in p_res:
                all_diffs.append(p_res[k]["mean_lp_diff"])
                all_max.append(p_res[k]["max_lp_diff"])
                all_flips += p_res[k]["argmax_flips"]
    avg = sum(all_diffs)/len(all_diffs) if all_diffs else 0
    mx  = max(all_max) if all_max else 0
    print(f"{mlabel:<20} {avg:>12.4f} {mx:>12.4f} {all_flips:>12}")

with open(OUT_FILE, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {OUT_FILE}")
