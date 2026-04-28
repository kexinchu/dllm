"""PROOF D — end-to-end avg_std@top1 with Triton attention kernel enabled.

Extends Proof B with a new configuration:
  4. DetermLLM Triton F.linear + Triton attention, flag=False

Hypothesis: now F.linear AND attention Q@K^T and attn@V are all fixed-plan
with FP32 reduction and bit-exact per row. Remaining non-determinism
sources would be softmax / RMSNorm / casts — all of which are per-row
(no cross-batch reduction) so they're bs-invariant by default.

Expect avg_std@top1 to collapse to ~1e-7 level (matching FP32-full).
"""
import os
import sys
import time
import statistics

os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, "."); sys.path.insert(0, "../FP32")

import torch
import torch.nn.functional as F

import determ_llm

MODEL = "/home/kec23008/docker-sys/Models/DeepSeek-R1-Distill-Qwen-7B"
BS_LIST = [1, 4, 8, 16, 32]
N_PROB = 5
GEN_LEN = 128


def _load_prompts():
    import json
    path = os.path.join(os.path.dirname(__file__), "math500_cached.json")
    with open(path) as f:
        return json.load(f)[:N_PROB]


def _apply_template(tokenizer, problem):
    messages = [{"role": "user", "content":
                 f"{problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def run_config(label, enable_fn, disable_fn, model, tokenizer, prompts):
    enable_fn()
    t0 = time.time()
    per_problem = []
    for p in prompts:
        prompt_text = _apply_template(tokenizer, p["problem"])
        lps_by_bs = {}
        for bs in BS_LIST:
            torch.manual_seed(0)
            enc = tokenizer(prompt_text, return_tensors="pt")["input_ids"].cuda()
            ids = enc.repeat(bs, 1).contiguous()
            lps = []
            with torch.no_grad():
                out = model(input_ids=ids, use_cache=True)
                past = out.past_key_values
                logits = out.logits[0, -1]
                lp = F.log_softmax(logits, dim=-1)
                tk = lp.argmax().item()
                lps.append(lp[tk].item())
                for _ in range(GEN_LEN - 1):
                    nx = torch.full((bs, 1), tk, dtype=ids.dtype, device=ids.device)
                    out = model(input_ids=nx, past_key_values=past, use_cache=True)
                    past = out.past_key_values
                    logits = out.logits[0, -1]
                    lp = F.log_softmax(logits, dim=-1)
                    tk = lp.argmax().item()
                    lps.append(lp[tk].item())
            lps_by_bs[bs] = lps
        per_problem.append(lps_by_bs)
    disable_fn()

    problem_stds = []
    for lps_by_bs in per_problem:
        n_steps = min(len(lps_by_bs[bs]) for bs in BS_LIST)
        pos_stds = []
        for j in range(n_steps):
            vals = [lps_by_bs[bs][j] for bs in BS_LIST]
            pos_stds.append(statistics.pstdev(vals))
        problem_stds.append(statistics.mean(pos_stds))
    avg_std = statistics.mean(problem_stds)

    first_div_list = []
    for lps_by_bs in per_problem:
        n_steps = min(len(lps_by_bs[bs]) for bs in BS_LIST)
        div = n_steps
        for j in range(n_steps):
            vals = set(lps_by_bs[bs][j] for bs in BS_LIST)
            if len(vals) > 1:
                div = j; break
        first_div_list.append(div)

    rt = time.time() - t0
    print(f"[{label:60s}] avg_std@top1={avg_std:.4e}  "
          f"first_div_med={statistics.median(first_div_list):.0f}  "
          f"first_div_mean={statistics.mean(first_div_list):.1f}  "
          f"({rt:.0f}s)")
    return avg_std, first_div_list


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("Loading DeepSeek-7B BF16...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="eager",
    )
    model.eval()
    prompts = _load_prompts()
    print(f"Loaded. N problems={len(prompts)}, gen_len={GEN_LEN}, bs_set={BS_LIST}\n")

    def set_flag(v):
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = v

    # Reference: BF16 baseline, flag=False
    run_config(
        "BF16 baseline, flag=False",
        lambda: set_flag(False),
        lambda: None,
        model, tokenizer, prompts,
    )

    # DetermLLM Triton F.linear only (attn=False)
    run_config(
        "DetermLLM Triton F.linear only",
        lambda: (set_flag(False), determ_llm.enable(backend="triton", attn=False)),
        lambda: determ_llm.disable(),
        model, tokenizer, prompts,
    )

    # DetermLLM Triton F.linear + Triton attention (new)
    run_config(
        "DetermLLM Triton F.linear + Triton attention (NEW)",
        lambda: (set_flag(False), determ_llm.enable(backend="triton", attn=True)),
        lambda: determ_llm.disable(),
        model, tokenizer, prompts,
    )


if __name__ == "__main__":
    main()
