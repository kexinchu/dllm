#!/usr/bin/env python3
"""
Core evaluation runner aligned to NIPS-2027 plan (E1 / E2 / E5 / E6 / E8).

For each (model, dataset, method, bs):
  - generate at bs=1   (reference)
  - generate at bs>1   (perturbed)
  - compute div_idx, exact_match, accuracy, output length
  - record TTFT, TPOT, throughput, peak memory

Methods (research/methods.py):
  BF16, LayerCast, SRP-FP32, plus site-ablation
  (SRP-FP32-{linear,rmsnorm,attention,softmax}).

  FP32-all is a separate method that requires reloading the model in
  ``dtype=torch.float32`` — pass --fp32-all to load once that way.

  SRP-FP64 was removed 2026-04-25: A6000/Ampere lacks BF16→FP64 tensor core
  path, so FP64 cost is dominated by ALU rather than reduction precision.

Output JSON: list of records with the schema in plan.md §5.1.
"""
import sys, os, time, json, gc, hashlib, argparse, re
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from research.methods import (
    method_BF16, method_LayerCast,
    method_SRP_FP32, method_FP32_all,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
MATH500_CACHE = os.path.join(_ROOT, "research", "math500_cached.json")
AIME25_PATH   = "/home/kec23008/docker-sys/DynaQuant/calibration_datasets/requests/aime25_available_30.jsonl"

MODELS = {
    "llama8b":     ("/home/kec23008/docker-sys/Models/Llama-3.1-8B-Instruct",       512, False),
    "deepseek7b":  ("/home/kec23008/docker-sys/Models/DeepSeek-R1-Distill-Qwen-7B", 1024, True),
    "llama1b":     ("/home/kec23008/docker-sys/Models/Llama-3.2-1B-Instruct",       512, False),
    "phi4":        ("/home/kec23008/docker-sys/Models/Phi-4",                       1024, False),
}


# ── Datasets ──────────────────────────────────────────────────────────────────
def load_dataset(name, n_problems):
    if name == "math500":
        with open(MATH500_CACHE) as f: data = json.load(f)
        return data[:n_problems]
    elif name == "aime25":
        out = []
        with open(AIME25_PATH) as f:
            for line in f:
                r = json.loads(line)
                out.append({"problem": r["problem"], "answer": r["answer"],
                            "subject": "AIME25", "level": None})
        return out[:n_problems]
    else:
        raise ValueError(f"unknown dataset {name}")


# ── Prompt + answer extraction (shared with old E4) ──────────────────────────
SYSTEM_REASON = "Please reason step by step, and put your final answer within \\boxed{}."

def wrap_prompt(tok, problem):
    msgs = [{"role": "user", "content": f"{problem}\n\n{SYSTEM_REASON}"}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def extract_boxed(text):
    m = "\\boxed{"
    k = text.rfind(m)
    if k < 0: return None
    i, d, s = k + len(m), 1, k + len(m)
    while i < len(text) and d > 0:
        if   text[i] == "{": d += 1
        elif text[i] == "}":
            d -= 1
            if d == 0: return text[s:i].strip()
        i += 1
    return None

def norm_ans(ans):
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
    np_, ng = norm_ans(pred), norm_ans(gold)
    if np_ is None or ng is None: return False
    if np_ == ng: return True
    try: return abs(float(np_) - float(ng)) < 1e-6
    except ValueError: return False


# ── Method dispatch ───────────────────────────────────────────────────────────
def get_method_cm(name, sites=None):
    """Return a context-manager factory `cm(model)` for `name`."""
    if name == "BF16":          return method_BF16
    if name == "LayerCast":     return method_LayerCast
    if name == "FP32-all":      return method_FP32_all
    if name == "SRP-FP32":      return lambda m: method_SRP_FP32(m, sites or ("linear","rmsnorm","attention","softmax"))
    if name == "SRP-FP32-Critical":
        from research.methods import CRITICAL_SITES
        return lambda m: method_SRP_FP32(m, CRITICAL_SITES)
    # Site ablation: name like "SRP-FP32-linear"
    if name.startswith("SRP-FP32-"):
        site = name[len("SRP-FP32-"):]
        return lambda m: method_SRP_FP32(m, (site,))
    raise ValueError(f"unknown method: {name}")


# ── Generation primitives ─────────────────────────────────────────────────────
def generate_batch(model, tok, prompt_text, bs, gen_len, device,
                   measure_ttft=False):
    """Generate at given bs. Always measures wall_ms / peak_mem_gb / tok_s.
    If `measure_ttft` is True, additionally times a separate generate(max_new=1)
    so TTFT/TPOT can be derived (this nearly doubles cost — only for E8).
    """
    enc = tok(prompt_text, return_tensors="pt")["input_ids"].to(device)
    L_in = enc.shape[1]
    ids = enc.repeat(bs, 1).contiguous()
    mask = torch.ones_like(ids)

    timing = {}

    if measure_ttft:
        torch.cuda.synchronize()
        t_ttft = time.perf_counter()
        with torch.no_grad():
            model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=1,
                           do_sample=False, pad_token_id=tok.pad_token_id)
        torch.cuda.synchronize()
        timing["ttft_ms"] = (time.perf_counter() - t_ttft) * 1000

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(input_ids=ids, attention_mask=mask,
                             max_new_tokens=gen_len, do_sample=False,
                             pad_token_id=tok.pad_token_id)
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - t0) * 1000
    peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
    new_len = out.shape[1] - L_in

    timing.update(
        wall_ms=wall_ms, peak_mem_gb=peak_gb, new_len=new_len,
        tok_s=(bs * new_len / (wall_ms / 1000.0)) if wall_ms > 0 else 0,
    )
    if "ttft_ms" in timing:
        timing["tpot_ms"] = (wall_ms - timing["ttft_ms"]) / max(new_len - 1, 1)

    return out[0, L_in:].cpu().tolist(), timing


def div_index(ref_toks, cand_toks, max_len):
    n = min(len(ref_toks), len(cand_toks), max_len)
    for i in range(n):
        if ref_toks[i] != cand_toks[i]:
            return i
    if len(ref_toks) != len(cand_toks):
        return n
    return max_len


# ── Main runner: one (model, dataset, methods, bs_list) experiment ────────────
def run_eval(model_key, dataset_name, methods, bs_list, n_problems,
             out_path, fp32_all=False, gen_len_override=None,
             include_timing=True):
    path, default_gen_len, _is_reason = MODELS[model_key]
    gen_len = gen_len_override or default_gen_len

    print(f"\n{'='*92}")
    print(f"  MODEL:{model_key}  DATASET:{dataset_name}  bs={bs_list}  "
          f"gen_len={gen_len}  N={n_problems}  fp32_all={fp32_all}")
    print(f"{'='*92}", flush=True)

    tok = AutoTokenizer.from_pretrained(path, use_fast=False)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    dtype = torch.float32 if fp32_all else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=dtype, device_map={"": 0},
        attn_implementation="sdpa",  # SRP attention patch hooks F.scaled_dot_product_attention
    )
    model.eval()
    device = next(model.parameters()).device

    problems = load_dataset(dataset_name, n_problems)
    print(f"  Loaded {len(problems)} problems from {dataset_name}", flush=True)

    out = {
        "model": model_key, "model_path": path, "dataset": dataset_name,
        "n_problems": len(problems), "gen_len": gen_len,
        "bs_list": list(bs_list), "fp32_all": fp32_all,
        "records": [],
    }

    for method_name in methods:
        cm_factory = get_method_cm(method_name)
        print(f"\n-- method: {method_name} --", flush=True)

        with cm_factory(model):
            t0 = time.perf_counter()
            for p_idx, ex in enumerate(problems):
                prompt_text = wrap_prompt(tok, ex["problem"])

                tokens_by_bs = {}
                timing_by_bs = {}
                for bs in bs_list:
                    toks, timing = generate_batch(model, tok, prompt_text, bs, gen_len, device,
                                                  measure_ttft=include_timing)
                    tokens_by_bs[bs] = toks
                    timing_by_bs[bs] = timing

                ref_toks = tokens_by_bs[bs_list[0]]
                ref_text = tok.decode(ref_toks, skip_special_tokens=True)
                ref_pred = extract_boxed(ref_text)
                ref_correct = match(ref_pred, ex["answer"])

                for bs in bs_list:
                    toks = tokens_by_bs[bs]
                    text = tok.decode(toks, skip_special_tokens=True)
                    pred = extract_boxed(text)
                    correct = match(pred, ex["answer"])
                    div_idx = div_index(ref_toks, toks, gen_len)
                    bit_exact = (div_idx == gen_len and len(toks) == len(ref_toks))

                    rec = {
                        "model":      model_key, "dataset": dataset_name,
                        "method":     method_name, "bs": bs,
                        "problem_idx": p_idx, "level": ex.get("level"),
                        "subject":    ex.get("subject", ""),
                        "gold":       ex["answer"], "pred": pred, "correct": correct,
                        "output_len": len(toks), "div_idx": div_idx,
                        "bit_exact":  bit_exact,
                        "hash":       hashlib.md5(str(toks).encode()).hexdigest()[:12],
                        # store first 64 tokens for downstream analysis (logit margins, etc.)
                        "first_64":   toks[:64],
                    }
                    if bs in timing_by_bs:
                        rec.update({k: v for k, v in timing_by_bs[bs].items()
                                    if k in ("ttft_ms","tpot_ms","tok_s","peak_mem_gb","wall_ms")})
                    out["records"].append(rec)

                if (p_idx + 1) % 5 == 0:
                    elapsed = time.perf_counter() - t0
                    n_div = sum(1 for r in out["records"]
                                if r["method"] == method_name and not r["bit_exact"]
                                and r["bs"] != bs_list[0])
                    n_pairs = (p_idx + 1) * (len(bs_list) - 1)
                    print(f"   [{p_idx+1:>3}/{len(problems)}] elapsed={elapsed:.0f}s  "
                          f"div={n_div}/{n_pairs}", flush=True)

            # incremental save after each method
            with open(out_path, "w") as f:
                json.dump(out, f, indent=2)

        torch.cuda.empty_cache(); gc.collect()

    print(f"\n  Saved: {out_path}")
    return out


# ── Quick aggregate per (method, bs) ──────────────────────────────────────────
def aggregate(records, ref_bs):
    """Build per (method, bs) summary table."""
    from collections import defaultdict
    buckets = defaultdict(list)
    for r in records:
        buckets[(r["method"], r["bs"])].append(r)
    rows = []
    for (m, bs), rs in sorted(buckets.items()):
        n = len(rs)
        n_correct = sum(1 for r in rs if r["correct"])
        n_bitexact = sum(1 for r in rs if r["bit_exact"]) if bs != ref_bs else n
        div_idxs = [r["div_idx"] for r in rs] if bs != ref_bs else []
        ttfts  = [r["ttft_ms"]    for r in rs if "ttft_ms"    in r]
        tpots  = [r["tpot_ms"]    for r in rs if "tpot_ms"    in r]
        toks_s = [r["tok_s"]      for r in rs if "tok_s"      in r]
        peaks  = [r["peak_mem_gb"] for r in rs if "peak_mem_gb" in r]
        out_lens = [r["output_len"] for r in rs]
        rows.append({
            "method": m, "bs": bs, "n": n,
            "accuracy":      n_correct / n,
            "exact_match":   n_bitexact / n,            # NIPS' Exact Match Consistency
            "div_percent":   1 - n_bitexact / n,
            "avg_first_div_idx": float(np.mean(div_idxs)) if div_idxs else None,
            "avg_output_len":  float(np.mean(out_lens)),
            "std_len":         float(np.std(out_lens)),
            "TTFT_ms":         float(np.mean(ttfts)) if ttfts else None,
            "TPOT_ms":         float(np.mean(tpots)) if tpots else None,
            "tok_s":           float(np.mean(toks_s)) if toks_s else None,
            "peak_mem_GB":     float(np.mean(peaks)) if peaks else None,
        })
    return rows


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",   choices=list(MODELS.keys()), required=True)
    ap.add_argument("--dataset", choices=["math500","aime25"], required=True)
    ap.add_argument("--methods", nargs="+", required=True,
                    help="space-separated method names, e.g. BF16 LayerCast SRP-FP32")
    ap.add_argument("--bs",      nargs="+", type=int, default=[1, 8, 16])
    ap.add_argument("--n-problems", type=int, default=30)
    ap.add_argument("--gen-len", type=int, default=None)
    ap.add_argument("--fp32-all", action="store_true")
    ap.add_argument("--no-timing", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    print(f"GPU: {torch.cuda.get_device_name(0)}  "
          f"free={torch.cuda.mem_get_info()[0]/1e9:.1f} GB", flush=True)

    out = run_eval(
        model_key=args.model, dataset_name=args.dataset,
        methods=args.methods, bs_list=args.bs,
        n_problems=args.n_problems, out_path=args.out,
        fp32_all=args.fp32_all, gen_len_override=args.gen_len,
        include_timing=not args.no_timing,
    )

    # Print summary
    rows = aggregate(out["records"], ref_bs=args.bs[0])
    print("\n" + "="*108)
    print(f"  AGGREGATE  ({args.model} / {args.dataset}, N={args.n_problems})")
    print("="*108)
    print(f"  {'method':<22} {'bs':>3} {'acc':>5} {'EM':>5} {'div%':>5} "
          f"{'div_med':>7} {'TTFT':>7} {'TPOT':>7} {'tok/s':>7} {'mem GB':>7}")
    for r in rows:
        em = f"{r['exact_match']*100:>4.0f}%"
        dv = f"{r['div_percent']*100:>4.0f}%"
        print(f"  {r['method']:<22} {r['bs']:>3} {r['accuracy']*100:>4.0f}% {em} {dv} "
              f"{(r['avg_first_div_idx'] or 0):>7.0f} "
              f"{(r['TTFT_ms'] or 0):>7.0f} "
              f"{(r['TPOT_ms'] or 0):>7.1f} "
              f"{(r['tok_s'] or 0):>7.1f} "
              f"{(r['peak_mem_GB'] or 0):>7.2f}")


if __name__ == "__main__":
    main()
