#!/usr/bin/env python3
"""
E2 — Top-1 / Top-2 probability gap at divergence points (plan.md §3 / P0).

For each problem on a reasoning model (DeepSeek-R1-Distill-Qwen-7B):
    generate under BF16 at bs=1 (reference) and bs=16 (perturbed)
    find the first position where the greedy tokens differ
    at that position, record the top-5 (token, prob) pairs from each run
    also record (p1 - p2), i.e. the gap between top-1 and top-2

Output a distribution so we can plot a CDF / histogram (Figure 2 of the paper).

For reference we run the same analysis for Ours (DetermLLM+attn) which should
have no divergences.
"""
import sys, os, time, json, gc, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings; warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from research import determ_llm

MATH500_CACHE = "/home/kec23008/docker-sys/dllm/research/math500_cached.json"

MODELS = {
    "deepseek7b": ("/home/kec23008/docker-sys/Models/DeepSeek-R1-Distill-Qwen-7B", 1024, True),
    "llama8b":    ("/home/kec23008/docker-sys/Models/Llama-3.1-8B-Instruct",       512,  False),
}

SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."


def wrap(tok, problem):
    return tok.apply_chat_template(
        [{"role": "user", "content": f"{problem}\n\n{SYSTEM}"}],
        tokenize=False, add_generation_prompt=True,
    )


# ── Step-by-step greedy with logits capture ───────────────────────────────────
def generate_with_logits(model, tok, prompt_text, bs, gen_len, device):
    """Greedy decode via manual loop; record logits at every position for row 0."""
    enc = tok(prompt_text, return_tensors="pt")["input_ids"].to(device)
    ids = enc.repeat(bs, 1).contiguous()

    tokens, probs_top5, token_top5 = [], [], []
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True)
        past = out.past_key_values
        logits = out.logits[0, -1]                          # row 0, last position
        p = F.softmax(logits.float(), dim=-1)
        top5 = torch.topk(p, 5)
        tok_id = top5.indices[0].item()
        tokens.append(tok_id)
        probs_top5.append(top5.values.cpu().tolist())
        token_top5.append(top5.indices.cpu().tolist())

        for _ in range(gen_len - 1):
            next_col = torch.full((bs, 1), tok_id, dtype=ids.dtype, device=device)
            out = model(input_ids=next_col, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[0, -1]
            p = F.softmax(logits.float(), dim=-1)
            top5 = torch.topk(p, 5)
            tok_id_new = top5.indices[0].item()
            tokens.append(tok_id_new)
            probs_top5.append(top5.values.cpu().tolist())
            token_top5.append(top5.indices.cpu().tolist())
            tok_id = tok_id_new
            if tok_id == tok.eos_token_id:
                break

    return tokens, probs_top5, token_top5


# ── Scheme hooks ──────────────────────────────────────────────────────────────
def scheme_bf16():
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True

def scheme_ours_attn():
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    determ_llm.enable(backend="triton", attn=True)

def scheme_teardown():
    try: determ_llm.disable()
    except Exception: pass
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True


def run(model_key, n_problems, bs_ref, bs_perturb, schemes, out_path):
    path, gen_len, _ = MODELS[model_key]
    print(f"\n=== E2: {model_key}  N={n_problems}  bs_ref={bs_ref}  bs_perturb={bs_perturb} ===",
          flush=True)

    tok = AutoTokenizer.from_pretrained(path, use_fast=False)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa",
    )
    model.eval()
    device = next(model.parameters()).device

    with open(MATH500_CACHE) as f:
        problems = json.load(f)[:n_problems]

    results = {"model": model_key, "bs_ref": bs_ref, "bs_perturb": bs_perturb,
               "gen_len": gen_len, "schemes": {}}

    for scheme_name, enter_fn in schemes:
        print(f"\n-- {scheme_name} --", flush=True)
        enter_fn()
        try:
            per_problem = []
            t0 = time.perf_counter()
            for i, ex in enumerate(problems):
                prompt_text = wrap(tok, ex["problem"])
                ref_tok, ref_p5, ref_id5 = generate_with_logits(
                    model, tok, prompt_text, bs_ref, gen_len, device)
                prt_tok, prt_p5, prt_id5 = generate_with_logits(
                    model, tok, prompt_text, bs_perturb, gen_len, device)

                # find first divergence
                div_idx = None
                for j in range(min(len(ref_tok), len(prt_tok))):
                    if ref_tok[j] != prt_tok[j]:
                        div_idx = j; break
                if div_idx is None and len(ref_tok) != len(prt_tok):
                    div_idx = min(len(ref_tok), len(prt_tok))

                entry = {
                    "idx": i,
                    "div_idx": div_idx,
                    "ref_len": len(ref_tok),
                    "prt_len": len(prt_tok),
                }
                if div_idx is not None:
                    entry["ref_top5_probs"]  = ref_p5[div_idx]
                    entry["ref_top5_tokens"] = ref_id5[div_idx]
                    entry["prt_top5_probs"]  = prt_p5[div_idx]
                    entry["prt_top5_tokens"] = prt_id5[div_idx]
                    entry["ref_p1_minus_p2"] = ref_p5[div_idx][0] - ref_p5[div_idx][1]
                    entry["prt_p1_minus_p2"] = prt_p5[div_idx][0] - prt_p5[div_idx][1]
                    entry["ref_top1_text"]   = tok.decode([ref_id5[div_idx][0]])
                    entry["prt_top1_text"]   = tok.decode([prt_id5[div_idx][0]])
                per_problem.append(entry)

                if (i + 1) % 5 == 0:
                    elapsed = time.perf_counter() - t0
                    n_div = sum(1 for e in per_problem if e.get("div_idx") is not None)
                    print(f"   [{i+1:>3}/{len(problems)}] elapsed={elapsed:.0f}s  "
                          f"divs={n_div}", flush=True)

            results["schemes"][scheme_name] = {
                "per_problem": per_problem,
                "n_div":       sum(1 for e in per_problem if e.get("div_idx") is not None),
                "n_total":     len(per_problem),
            }
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
        finally:
            scheme_teardown()
            torch.cuda.empty_cache(); gc.collect()

    # summary
    print("\n" + "=" * 84)
    print(f"  E2 SUMMARY ({model_key})")
    print("=" * 84)
    for sc_name, sc_data in results["schemes"].items():
        divs = [e for e in sc_data["per_problem"] if e.get("div_idx") is not None]
        if divs:
            gaps = [d["ref_p1_minus_p2"] for d in divs]
            gaps.sort()
            print(f"  [{sc_name}] #diverged={len(divs)}/{sc_data['n_total']}  "
                  f"p1-p2 gap: median={gaps[len(gaps)//2]:.4f}  "
                  f"p25={gaps[len(gaps)//4]:.4f}  "
                  f"p75={gaps[3*len(gaps)//4]:.4f}")
        else:
            print(f"  [{sc_name}] no divergences (bit-exact)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS.keys()), default="deepseek7b")
    ap.add_argument("--n-problems", type=int, default=30)
    ap.add_argument("--bs-ref", type=int, default=1)
    ap.add_argument("--bs-perturb", type=int, default=16)
    ap.add_argument("--out", default="/home/kec23008/docker-sys/dllm/research/exp_E2/E2.json")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    print(f"GPU: {torch.cuda.get_device_name(0)}  "
          f"free={torch.cuda.mem_get_info()[0] / 1e9:.1f} GB", flush=True)

    schemes = [("BF16", scheme_bf16), ("DetermLLM+attn", scheme_ours_attn)]
    run(args.model, args.n_problems, args.bs_ref, args.bs_perturb, schemes, args.out)


if __name__ == "__main__":
    main()
