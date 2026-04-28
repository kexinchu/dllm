"""Which specific bs values produce different top1 logprob trajectories?

Under the fully-patched DetermLLM stack (Triton F.linear + Triton attention +
flag=False), run the same prompt at bs ∈ {1, 4, 8, 16, 32} and compare the
top-1 logprob at each decode step. Report the first step where each bs pair
diverges.
"""
import os
import sys

os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, "."); sys.path.insert(0, "../FP32")

import torch
import torch.nn.functional as F
import determ_llm

MODEL = "/home/kec23008/docker-sys/Models/DeepSeek-R1-Distill-Qwen-7B"
BS_LIST = [1, 4, 8, 16, 32]
STEPS = 60


def _load_problem(idx=0):
    import json
    with open(os.path.join(os.path.dirname(__file__), "math500_cached.json")) as f:
        return json.load(f)[idx]["problem"]


def run_gen(model, tokenizer, prompt_text, bs, steps):
    enc = tokenizer(prompt_text, return_tensors="pt")["input_ids"].cuda()
    ids = enc.repeat(bs, 1).contiguous()
    tokens = []
    logprobs = []
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True)
        past = out.past_key_values
        logits = out.logits[0, -1]
        lp = F.log_softmax(logits, dim=-1)
        tk = lp.argmax().item()
        tokens.append(tk); logprobs.append(lp[tk].item())
        for _ in range(steps - 1):
            nx = torch.full((bs, 1), tk, dtype=ids.dtype, device=ids.device)
            out = model(input_ids=nx, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[0, -1]
            lp = F.log_softmax(logits, dim=-1)
            tk = lp.argmax().item()
            tokens.append(tk); logprobs.append(lp[tk].item())
    return tokens, logprobs


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("Loading DeepSeek-7B...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="eager",
    )
    model.eval()

    problem = _load_problem(0)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": f"{problem}\n\nPlease reason step by step, and "
                                      f"put your final answer within \\boxed{{}}."}],
        tokenize=False, add_generation_prompt=True,
    )

    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    determ_llm.enable(backend="triton", attn=True)
    n_patched = determ_llm.patch_rmsnorm(model)
    print(f"Patched {n_patched} RMSNorm modules")
    try:
        results = {}
        for bs in BS_LIST:
            print(f"bs={bs}...")
            toks, lps = run_gen(model, tokenizer, prompt, bs, STEPS)
            results[bs] = (toks, lps)
    finally:
        determ_llm.disable()

    # Pairwise first-diverge
    print("\n=== First-diverge decode step across bs pairs ===")
    print(f"{'pair':<10}{'first_div_tok':>15}{'first_div_lp':>15}{'max_lp_diff':>15}")
    for i, a in enumerate(BS_LIST):
        for b in BS_LIST[i+1:]:
            ta, la = results[a]; tb, lb = results[b]
            n = min(len(ta), len(tb))
            # first differing token
            div_tok = n
            for j in range(n):
                if ta[j] != tb[j]:
                    div_tok = j; break
            # first step where top1_logprob differs (within float epsilon)
            div_lp = n
            max_lp_diff = 0.0
            for j in range(n):
                d = abs(la[j] - lb[j])
                max_lp_diff = max(max_lp_diff, d)
                if d > 0 and div_lp == n:
                    div_lp = j
            print(f"{a:>2}v{b:<6} {div_tok:>15} {div_lp:>15} {max_lp_diff:>15.3e}")


if __name__ == "__main__":
    main()
