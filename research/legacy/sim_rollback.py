"""
LLM-42 Rollback Frequency Simulator

Simulates the decode-verify-rollback protocol by comparing argmax tokens
between bs=1 (reference/deterministic) and bs=N (dynamic batch).

A "rollback" occurs when the argmax token differs between the two.
"""

import torch
import warnings
import hashlib
import json
import time
import argparse
import sys
from collections import defaultdict

warnings.filterwarnings('ignore')


def load_model(model_path, trust_remote=False):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Monkey-patch DynamicCache for DeepSeek compatibility
    from transformers import DynamicCache
    if not hasattr(DynamicCache, 'get_usable_length'):
        DynamicCache.get_usable_length = lambda self, *args, **kwargs: self.get_seq_length()

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=trust_remote)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map='auto', trust_remote_code=trust_remote
    )
    model.eval()
    return model, tokenizer


def simulate_rollback(model, tokenizer, prompt, gen_len, batch_sizes,
                      use_cache=True, fp32_flag=False):
    """
    For each batch size, generate gen_len tokens and compare each step's
    argmax with bs=1 reference. Count mismatches (= rollbacks).

    Returns dict with per-bs rollback stats.
    """
    device = next(model.parameters()).device
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = not fp32_flag

    input_ids = tokenizer(prompt, return_tensors='pt')['input_ids']
    sl = input_ids.shape[1]

    results = {}

    for bs in batch_sizes:
        ids_ref = input_ids.to(device)
        ids_batch = input_ids.repeat(bs, 1).to(device)

        rollbacks = []
        ref_tokens = []
        batch_tokens = []

        cur_ref = ids_ref.clone()
        cur_batch = ids_batch.clone()

        with torch.no_grad():
            for step in range(gen_len):
                # Reference: bs=1
                if use_cache:
                    out_ref = model(input_ids=cur_ref).logits
                else:
                    out_ref = model(input_ids=cur_ref, use_cache=False).logits
                ref_tok = out_ref[0, -1].argmax().item()

                # Batch: bs=N
                if use_cache:
                    out_batch = model(input_ids=cur_batch).logits
                else:
                    out_batch = model(input_ids=cur_batch, use_cache=False).logits
                batch_tok = out_batch[0, -1].argmax().item()

                # Check mismatch
                is_rollback = (ref_tok != batch_tok)
                rollbacks.append(is_rollback)
                ref_tokens.append(ref_tok)
                batch_tokens.append(batch_tok)

                # For next step: use ref_tok as ground truth (LLM-42 rolls back to ref)
                next_tok = torch.tensor([[ref_tok]], device=device)
                cur_ref = torch.cat([cur_ref, next_tok], dim=1) if not use_cache else next_tok
                cur_batch = torch.cat([cur_batch, next_tok.repeat(bs, 1)], dim=1) if not use_cache else next_tok.repeat(bs, 1)

                # For non-cache mode, we always feed full sequence
                if use_cache:
                    cur_ref = next_tok
                    cur_batch = next_tok.repeat(bs, 1)

        n_rollbacks = sum(rollbacks)
        rollback_rate = n_rollbacks / gen_len

        # Find first rollback position
        first_rb = next((i for i, r in enumerate(rollbacks) if r), None)

        results[bs] = {
            'n_rollbacks': n_rollbacks,
            'rollback_rate': rollback_rate,
            'gen_len': gen_len,
            'first_rollback_pos': first_rb,
            'rollback_positions': [i for i, r in enumerate(rollbacks) if r],
        }

        status = f"CLEAN" if n_rollbacks == 0 else f"DIRTY ({n_rollbacks}/{gen_len} = {rollback_rate:.1%})"
        print(f"  bs={bs:>3}: {status}")

    return results


def run_experiment(exp_name, model, tokenizer, prompts, gen_len, batch_sizes,
                   use_cache=True, fp32_flag=False):
    """Run rollback simulation for multiple prompts."""
    print(f"\n{'='*60}")
    print(f"Experiment: {exp_name}")
    print(f"gen_len={gen_len}, batch_sizes={batch_sizes}, fp32={fp32_flag}")
    print(f"{'='*60}")

    all_results = {}
    for i, prompt in enumerate(prompts):
        prompt_preview = prompt[:60] + "..." if len(prompt) > 60 else prompt
        print(f"\nPrompt {i+1}/{len(prompts)}: \"{prompt_preview}\"")

        res = simulate_rollback(
            model, tokenizer, prompt, gen_len, batch_sizes,
            use_cache=use_cache, fp32_flag=fp32_flag
        )
        all_results[f"prompt_{i}"] = {
            'prompt': prompt[:200],
            'results': res
        }

    # Aggregate stats
    agg = {}
    for bs in batch_sizes:
        rates = [all_results[k]['results'][bs]['rollback_rate']
                 for k in all_results if bs in all_results[k]['results']]
        total_rb = sum(all_results[k]['results'][bs]['n_rollbacks']
                      for k in all_results if bs in all_results[k]['results'])
        total_tokens = gen_len * len(prompts)
        agg[bs] = {
            'mean_rollback_rate': sum(rates) / len(rates),
            'max_rollback_rate': max(rates),
            'min_rollback_rate': min(rates),
            'total_rollbacks': total_rb,
            'total_tokens': total_tokens,
        }

    print(f"\n--- Aggregate for {exp_name} ---")
    print(f"{'bs':>5} {'mean_rate':>10} {'max_rate':>10} {'total_rb':>10} {'total_tok':>10}")
    for bs in batch_sizes:
        a = agg[bs]
        print(f"{bs:>5} {a['mean_rollback_rate']:>10.2%} {a['max_rollback_rate']:>10.2%} "
              f"{a['total_rollbacks']:>10} {a['total_tokens']:>10}")

    return {'per_prompt': all_results, 'aggregate': agg, 'config': {
        'exp_name': exp_name, 'gen_len': gen_len, 'batch_sizes': batch_sizes,
        'fp32_flag': fp32_flag, 'n_prompts': len(prompts)
    }}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, required=True,
                        choices=['moe', 'reasoning', 'longgen', 'all'])
    parser.add_argument('--model', type=str, default='llama')
    parser.add_argument('--output', type=str, default='research/exp_d_rollback.json')
    args = parser.parse_args()

    # === Model setup ===
    if args.model == 'deepseek':
        MODEL_PATH = 'deepseek-ai/DeepSeek-V2-Lite'
        model, tokenizer = load_model(MODEL_PATH, trust_remote=True)
        USE_CACHE = False  # DeepSeek cache compatibility issues
    else:
        MODEL_PATH = '/home/kec23008/docker-sys/Models/Llama-3.1-8B-Instruct'
        model, tokenizer = load_model(MODEL_PATH)
        USE_CACHE = False  # Consistent: always no-cache for fair comparison

    BS_LIST = [2, 4, 8, 16, 32, 57, 64, 128]

    # === Prompts ===
    GENERAL_PROMPTS = [
        "What is deterministic inference in large language models?",
        "Explain the concept of batch processing in neural networks.",
        "How does attention mechanism work in transformers?",
    ]

    REASONING_PROMPTS = [
        "Solve step by step: If a train travels at 60 mph for 2.5 hours, then at 80 mph for 1.5 hours, what is the total distance?",
        "Write a Python function to find the longest common subsequence of two strings. Think step by step.",
        "Prove that the square root of 2 is irrational. Show your reasoning.",
        "A farmer has 100 meters of fencing. What dimensions of a rectangular field maximize the area? Solve with calculus.",
        "Debug this code: def fib(n): return fib(n-1) + fib(n-2). What's wrong and how to fix it?",
    ]

    all_experiments = {}

    if args.experiment in ('moe', 'all'):
        if args.model != 'deepseek':
            print("WARNING: MoE experiment should use --model deepseek")
        exp = run_experiment("D1_MoE_general", model, tokenizer,
                           GENERAL_PROMPTS, gen_len=64, batch_sizes=BS_LIST,
                           use_cache=USE_CACHE, fp32_flag=False)
        all_experiments['D1_moe_bf16'] = exp

        exp = run_experiment("D1_MoE_fp32", model, tokenizer,
                           GENERAL_PROMPTS, gen_len=64, batch_sizes=BS_LIST,
                           use_cache=USE_CACHE, fp32_flag=True)
        all_experiments['D1_moe_fp32'] = exp

    if args.experiment in ('reasoning', 'all'):
        exp = run_experiment("D2_reasoning_bf16", model, tokenizer,
                           REASONING_PROMPTS, gen_len=128, batch_sizes=BS_LIST,
                           use_cache=USE_CACHE, fp32_flag=False)
        all_experiments['D2_reasoning_bf16'] = exp

        exp = run_experiment("D2_reasoning_fp32", model, tokenizer,
                           REASONING_PROMPTS, gen_len=128, batch_sizes=BS_LIST,
                           use_cache=USE_CACHE, fp32_flag=True)
        all_experiments['D2_reasoning_fp32'] = exp

    if args.experiment in ('longgen', 'all'):
        exp = run_experiment("D3_longgen_bf16", model, tokenizer,
                           GENERAL_PROMPTS[:2], gen_len=512, batch_sizes=[2, 8, 32, 64],
                           use_cache=USE_CACHE, fp32_flag=False)
        all_experiments['D3_longgen_bf16'] = exp

        exp = run_experiment("D3_longgen_fp32", model, tokenizer,
                           GENERAL_PROMPTS[:2], gen_len=512, batch_sizes=[2, 8, 32, 64],
                           use_cache=USE_CACHE, fp32_flag=True)
        all_experiments['D3_longgen_fp32'] = exp

    with open(args.output, 'w') as f:
        json.dump(all_experiments, f, indent=2)
    print(f"\nAll results saved to {args.output}")
