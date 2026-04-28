#!/usr/bin/env python3
"""
E4 — Div_Index distribution (plan.md §3 / Priority P0).  Also covers E6 (bit-exact).

For each (model, scheme, problem):
    reference_tokens = generate at bs=1
    for bs in BS_LIST[1:]:
        tokens_bs = generate at bs (row 0 is target, filler = same prompt repeated)
        div_idx = first position where tokens_bs[i] != reference_tokens[i]
                  (= max_new_tokens if identical → "never diverges")

Per scheme we aggregate:
    - Div_Index distribution over all (problem, bs) pairs
    - % pairs with bit-exact match (Div_Index == max_new_tokens)  → E6 bit-exact
    - %problems with any divergence
    - downstream accuracy (MATH answer correctness) per bs

Batch composition: row 0 = target problem; rows 1..bs-1 = same prompt repeated.
This matches the paper's continuous-batching perturbation: only the kernel-
level batch-invariance property matters, the data is identical.
"""
import sys, os, time, json, gc, hashlib, argparse, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings; warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from research import determ_llm
from motivation.test_layercast_latency import apply_layercast, remove_layercast

MATH500_CACHE = "/home/kec23008/docker-sys/dllm/research/math500_cached.json"
AIME25_PATH   = "/home/kec23008/docker-sys/DynaQuant/calibration_datasets/requests/aime25_available_30.jsonl"

MODELS = {
    "llama8b":   ("/home/kec23008/docker-sys/Models/Llama-3.1-8B-Instruct",   512, False),  # non-reasoning
    "deepseek7b":("/home/kec23008/docker-sys/Models/DeepSeek-R1-Distill-Qwen-7B", 1024, True),  # reasoning
}

BS_LIST = [1, 8, 16]  # reference = bs=1  (3 points keeps wall-time manageable)


def load_dataset(name, n_problems):
    """Return list of dicts with keys {problem, answer, subject(opt)}."""
    if name == "math500":
        with open(MATH500_CACHE) as f:
            data = json.load(f)
        return data[:n_problems]
    elif name == "aime25":
        out = []
        with open(AIME25_PATH) as f:
            for line in f:
                r = json.loads(line)
                out.append({"problem": r["problem"], "answer": r["answer"],
                            "subject": "AIME25"})
        return out[:n_problems]
    else:
        raise ValueError(f"unknown dataset {name}")


# ── prompt templates ─────────────────────────────────────────────────────────
SYSTEM_REASON = "Please reason step by step, and put your final answer within \\boxed{}."

def wrap_prompt(tok, problem, is_reasoning):
    if is_reasoning:
        msgs = [{"role": "user", "content": f"{problem}\n\n{SYSTEM_REASON}"}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    else:
        msgs = [{"role": "user", "content": f"{problem}\n\n{SYSTEM_REASON}"}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


# ── answer extraction ────────────────────────────────────────────────────────
def extract_boxed(text):
    m = "\\boxed{"
    k = text.rfind(m)
    if k < 0:
        return None
    i, d, s = k + len(m), 1, k + len(m)
    while i < len(text) and d > 0:
        if text[i] == "{": d += 1
        elif text[i] == "}":
            d -= 1
            if d == 0: return text[s:i].strip()
        i += 1
    return None

def norm(ans):
    if ans is None: return None
    s = ans.strip().strip("$")
    s = re.sub(r"\s+", " ", s).rstrip(".,")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    try:
        v = float(s.replace(",", ""))
        return str(int(v)) if v == int(v) else f"{v:.6g}"
    except ValueError:
        return s

def match(pred, gold):
    np_, ng = norm(pred), norm(gold)
    if np_ is None or ng is None: return False
    if np_ == ng: return True
    try: return abs(float(np_) - float(ng)) < 1e-6
    except ValueError: return False


# ── scheme hooks ─────────────────────────────────────────────────────────────
def make_schemes(model):
    st = {}
    def bf_e():  torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    def bf_x():  pass
    def fl_e():  torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    def fl_x():  torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    def lc_e():
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        st["lc"] = apply_layercast(model)
    def lc_x():  remove_layercast(model, st.pop("lc"))
    def dt_e():
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        determ_llm.enable(backend="triton")
    def dt_x():  determ_llm.disable()
    def da_e():
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        determ_llm.enable(backend="triton", attn=True)
    def da_x():  determ_llm.disable()
    return [
        ("BF16",           bf_e, bf_x),
        ("FP32flag",       fl_e, fl_x),
        ("LayerCast",      lc_e, lc_x),
        ("DetermLLM",      dt_e, dt_x),
        ("DetermLLM+attn", da_e, da_x),
    ]


# ── generation ───────────────────────────────────────────────────────────────
def generate_batch(model, tok, prompt_text, bs, gen_len, device):
    """Return tokens_row0 (list[int])."""
    enc = tok(prompt_text, return_tensors="pt")["input_ids"].to(device)
    ids = enc.repeat(bs, 1).contiguous()
    mask = torch.ones_like(ids)
    with torch.no_grad():
        out = model.generate(input_ids=ids, attention_mask=mask,
                             max_new_tokens=gen_len, do_sample=False,
                             pad_token_id=tok.pad_token_id)
    return out[0, enc.shape[1]:].cpu().tolist()


# ── Div_Index ────────────────────────────────────────────────────────────────
def div_index(ref_toks, cand_toks, max_len):
    n = min(len(ref_toks), len(cand_toks), max_len)
    for i in range(n):
        if ref_toks[i] != cand_toks[i]:
            return i
    # tails differ only in length?
    if len(ref_toks) != len(cand_toks):
        return n
    return max_len  # identical → never diverges


# ── per-model runner ─────────────────────────────────────────────────────────
def run_model(model_key, dataset_name, n_problems, out_path):
    path, gen_len, is_reason = MODELS[model_key]
    print(f"\n{'=' * 84}\n  MODEL: {model_key}  dataset={dataset_name}  "
          f"gen_len={gen_len}  reasoning={is_reason}\n{'=' * 84}", flush=True)
    tok = AutoTokenizer.from_pretrained(path, use_fast=False)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation="eager",  # eager so DetermLLM's torch.matmul/bmm patch hooks attention
    )
    model.eval()
    device = next(model.parameters()).device

    problems = load_dataset(dataset_name, n_problems)
    print(f"  Loaded {len(problems)} problems from {dataset_name}", flush=True)

    schemes = make_schemes(model)
    per_model = {"model": model_key, "path": path, "gen_len": gen_len,
                 "is_reasoning": is_reason, "bs_list": BS_LIST,
                 "dataset": dataset_name, "n_problems": len(problems),
                 "schemes": []}

    for scheme_name, ein, eout in schemes:
        print(f"\n-- {scheme_name} --", flush=True)
        ein()
        t0 = time.perf_counter()
        scheme_data = {"scheme": scheme_name, "per_problem": []}

        try:
            for p_idx, ex in enumerate(problems):
                prompt_text = wrap_prompt(tok, ex["problem"], is_reason)
                tokens_by_bs = {}
                for bs in BS_LIST:
                    tokens_by_bs[bs] = generate_batch(model, tok, prompt_text, bs, gen_len, device)

                ref = tokens_by_bs[BS_LIST[0]]
                per_bs = {}
                for bs in BS_LIST:
                    toks = tokens_by_bs[bs]
                    div_idx = div_index(ref, toks, gen_len)
                    text = tok.decode(toks, skip_special_tokens=True)
                    pred = extract_boxed(text)
                    correct = match(pred, ex["answer"])
                    per_bs[bs] = {
                        "output_len": len(toks),
                        "div_idx":    div_idx,
                        "bit_exact":  (div_idx == gen_len and len(toks) == len(ref)),
                        "pred":       pred,
                        "correct":    correct,
                        "hash":       hashlib.md5(str(toks).encode()).hexdigest()[:12],
                        # save full tokens (typ. ≤2 KB each) so we can decode concrete answer-flip
                        # examples for Fig 1B without re-running.
                        "tokens":     toks,
                    }
                scheme_data["per_problem"].append({
                    "idx":         p_idx,
                    "subject":     ex.get("subject", ""),
                    "gold":        ex["answer"],
                    "per_bs":      per_bs,
                })
                if (p_idx + 1) % 5 == 0:
                    elapsed = time.perf_counter() - t0
                    diverges = sum(1 for q in scheme_data["per_problem"]
                                   if any(r["div_idx"] < gen_len for r in q["per_bs"].values()))
                    print(f"   [{p_idx+1:>3}/{len(problems)}] elapsed={elapsed:.0f}s  "
                          f"diverge_so_far={diverges}/{p_idx+1}", flush=True)
        finally:
            eout(); torch.cuda.empty_cache(); gc.collect()

        # aggregate
        n_prob = len(scheme_data["per_problem"])
        n_pairs = n_prob * (len(BS_LIST) - 1)
        n_bitexact = sum(1 for q in scheme_data["per_problem"] for bs in BS_LIST[1:]
                         if q["per_bs"][bs]["bit_exact"])
        n_problem_diverge = sum(1 for q in scheme_data["per_problem"]
                                if any(not q["per_bs"][bs]["bit_exact"] for bs in BS_LIST[1:]))
        accs_by_bs = {bs: sum(1 for q in scheme_data["per_problem"] if q["per_bs"][bs]["correct"]) / n_prob
                      for bs in BS_LIST}
        div_vals = [q["per_bs"][bs]["div_idx"] for q in scheme_data["per_problem"] for bs in BS_LIST[1:]]
        div_vals.sort()
        scheme_data["aggregate"] = {
            "n_problems":         n_prob,
            "n_pairs":            n_pairs,
            "bit_exact_rate":     n_bitexact / n_pairs,
            "problem_diverge_rate": n_problem_diverge / n_prob,
            "accuracy_by_bs":     accs_by_bs,
            "div_index_median":   div_vals[len(div_vals)//2] if div_vals else None,
            "div_index_p25":      div_vals[len(div_vals)//4] if div_vals else None,
            "div_index_p75":      div_vals[3*len(div_vals)//4] if div_vals else None,
            "runtime_s":          time.perf_counter() - t0,
        }
        agg = scheme_data["aggregate"]
        print(f"   [{scheme_name}] bit-exact={agg['bit_exact_rate']*100:.1f}%  "
              f"diverge_problems={agg['problem_diverge_rate']*100:.1f}%  "
              f"div_med={agg['div_index_median']}  "
              f"acc(bs={BS_LIST[0]})={accs_by_bs[BS_LIST[0]]*100:.1f}%  "
              f"acc(bs={BS_LIST[-1]})={accs_by_bs[BS_LIST[-1]]*100:.1f}%  "
              f"time={agg['runtime_s']:.0f}s", flush=True)

        per_model["schemes"].append(scheme_data)
        with open(out_path, "w") as f:
            json.dump(per_model, f, indent=2)

    return per_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS.keys()), required=True)
    ap.add_argument("--dataset", choices=["math500", "aime25"], default="math500")
    ap.add_argument("--n-problems", type=int, default=50)
    ap.add_argument("--out-dir", default="/home/kec23008/docker-sys/dllm/research/exp_E4")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"E4_{args.model}_{args.dataset}.json")
    print(f"GPU: {torch.cuda.get_device_name(0)}  "
          f"free={torch.cuda.mem_get_info()[0] / 1e9:.1f} GB", flush=True)

    run_model(args.model, args.dataset, args.n_problems, out)
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()
