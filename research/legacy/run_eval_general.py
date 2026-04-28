"""
Multi-model multi-dataset determinism evaluation.

Supports MATH500 and GSM8K as datasets; any HF causal LM as model.
Per-problem output includes: token sequence, logprobs, extracted answer,
correctness, token hash for cross-config comparison.

Usage:
  python run_eval_general.py \
      --model-path /path/to/Model \
      --dataset math500|gsm8k \
      --method bf16_baseline|layercast|dllm_hybrid|... \
      --batch-size 8 \
      --seed 0 \
      --n-problems 20 \
      --gen-len 1024 \
      --out result.json
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time

import torch
import torch.nn.functional as F

DLLM_DIR = os.path.dirname(os.path.abspath(__file__))
FP32_DIR = os.path.join(DLLM_DIR, "..", "FP32")
sys.path.insert(0, DLLM_DIR)
sys.path.insert(0, FP32_DIR)

import determ_llm
import layercast
import layercast_true

# ── Dataset loaders ──────────────────────────────────────────────────────────
DATASET_CACHE = {
    "math500": os.path.join(DLLM_DIR, "math500_cached.json"),
    "gsm8k":   os.path.join(DLLM_DIR, "gsm8k_cached.json"),
}

# Prompt suffix used to induce \boxed{} answer format
BOXED_SUFFIX = "Please reason step by step, and put your final answer within \\boxed{}."


def load_dataset_cached(name, n):
    path = DATASET_CACHE.get(name)
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Dataset cache not found: {path}")
    with open(path) as f:
        data = json.load(f)
    return data[:n]


# ── Answer extraction ────────────────────────────────────────────────────────
def extract_boxed_answer(text):
    marker = "\\boxed{"
    last = text.rfind(marker)
    if last == -1:
        return None
    i = last + len(marker); depth = 1; start = i
    while i < len(text) and depth > 0:
        c = text[i]
        if c == "{": depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i].strip()
        i += 1
    return None


def extract_final_number(text):
    """For GSM8K-style direct numeric answers. Look for last number in text."""
    # First prefer explicit 'answer is N' pattern
    m = re.search(r'(?:answer is|final answer.*is|=\s*)\s*(-?\d+(?:\.\d+)?(?:/\d+)?)',
                  text, re.IGNORECASE)
    if m:
        return m.group(1)
    # Else pick last number in text
    nums = re.findall(r'(?<![\w.])-?\d+(?:\.\d+)?(?:/\d+)?', text)
    return nums[-1] if nums else None


def normalize_num(s):
    if s is None: return None
    s = s.strip().strip("$").rstrip(".,")
    s = s.replace(",", "")
    try:
        v = float(s)
        return str(int(v)) if v == int(v) else f"{v:.6g}"
    except ValueError:
        pass
    # Try fraction
    if "/" in s:
        try:
            a, b = s.split("/")
            return f"{float(a)/float(b):.6g}"
        except (ValueError, ZeroDivisionError):
            pass
    return s


def answer_matches(pred, gold, dataset):
    if pred is None or gold is None:
        return False
    np_ = normalize_num(pred)
    ng  = normalize_num(gold)
    if np_ == ng and np_ is not None:
        return True
    try:
        return abs(float(np_) - float(ng)) < 1e-6
    except (ValueError, TypeError):
        return False


# ── Generation ───────────────────────────────────────────────────────────────
def apply_chat_template(tokenizer, problem, system_suffix=""):
    user_msg = problem if not system_suffix else f"{problem}\n\n{system_suffix}"
    messages = [{"role": "user", "content": user_msg}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def generate_one(model, tokenizer, prompt_text, batch_size, gen_len, device, seed):
    torch.manual_seed(seed)
    enc = tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(device)
    ids = enc.repeat(batch_size, 1).contiguous()

    token_ids = []; token_lps = []
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True)
        past = out.past_key_values
        logits = out.logits[0, -1]
        lps = F.log_softmax(logits, dim=-1)
        tok = lps.argmax().item()
        token_ids.append(tok); token_lps.append(lps[tok].item())

        for _ in range(gen_len - 1):
            next_col = torch.full((batch_size, 1), tok, dtype=ids.dtype, device=device)
            out = model(input_ids=next_col, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[0, -1]
            lps = F.log_softmax(logits, dim=-1)
            tok_new = lps.argmax().item()
            token_ids.append(tok_new); token_lps.append(lps[tok_new].item())
            tok = tok_new
            if tok == tokenizer.eos_token_id:
                break
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    return text, token_ids, token_lps


# ── Method dispatch ──────────────────────────────────────────────────────────
def set_method(method):
    try: determ_llm.disable()
    except Exception: pass
    try: layercast.disable()
    except Exception: pass
    try: layercast_true.disable()
    except Exception: pass
    # FP32 reduction accumulator for BF16 matmul. Setting this to True
    # (the previous value) actively *enables* BF16 reduction, defeating
    # the determinism guarantee we depend on.
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

    if method == "bf16_baseline":
        pass
    elif method == "layercast":
        layercast.enable()
    elif method == "layercast_true":
        layercast_true.enable()
    elif method == "dllm_cublaslt":
        determ_llm.enable(backend="cublaslt")
    elif method == "dllm_triton":
        determ_llm.enable(backend="triton")
    elif method == "dllm_hybrid":
        determ_llm.enable(backend="hybrid")
    elif method == "dllm_triton":
        # F.linear: Triton fixed-plan kernel (BF16 in/out, FP32 reduction,
        # bs-invariant by construction). Attention stays on cuBLAS BF16.
        determ_llm.enable(backend="triton")
    else:
        raise ValueError(f"Unknown method: {method}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--dataset", choices=["math500", "gsm8k"], required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--batch-size", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-problems", type=int, default=20)
    ap.add_argument("--gen-len", type=int, default=1024)
    ap.add_argument("--out", required=True)
    ap.add_argument("--attn-impl", default="eager",
                    help="attn_implementation for model.from_pretrained")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float32", "float16"],
                    help="model weight dtype (float32 loads the full model in FP32)")
    args = ap.parse_args()

    model_name = os.path.basename(args.model_path.rstrip("/"))
    print(f"[{model_name} | {args.dataset} | {args.method} | bs={args.batch_size} | "
          f"seed={args.seed} | n={args.n_problems} | gen={args.gen_len}]", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    t_load = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype_map = {"bfloat16": torch.bfloat16, "float32": torch.float32, "float16": torch.float16}
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=dtype_map[args.dtype],
        device_map="cuda:0",
        attn_implementation=args.attn_impl,
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"  model loaded in {time.time()-t_load:.1f}s on {device}", flush=True)

    set_method(args.method)

    problems = load_dataset_cached(args.dataset, args.n_problems)
    print(f"  dataset {args.dataset}: {len(problems)} problems loaded", flush=True)

    results = {
        "meta": {
            "model": args.model_path,
            "model_short": model_name,
            "dataset": args.dataset,
            "method": args.method,
            "dtype": args.dtype,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "n_problems": args.n_problems,
            "gen_len": args.gen_len,
        },
        "per_problem": [],
    }

    t_start = time.time()
    for i, ex in enumerate(problems):
        suffix = BOXED_SUFFIX   # always use boxed for consistency
        prompt_text = apply_chat_template(tokenizer, ex["problem"], suffix)
        text, toks, lps = generate_one(
            model, tokenizer, prompt_text,
            args.batch_size, args.gen_len, device, args.seed
        )
        # Answer extraction: try \boxed{}, then fallback to final-number for GSM8K
        pred = extract_boxed_answer(text)
        if pred is None and args.dataset == "gsm8k":
            pred = extract_final_number(text)
        correct = answer_matches(pred, ex["answer"], args.dataset)
        th = hashlib.md5(str(toks).encode()).hexdigest()[:12]

        results["per_problem"].append({
            "idx": i,
            "gold": ex["answer"],
            "pred": pred,
            "correct": correct,
            "output_len": len(toks),
            "token_hash": th,
            "token_ids": toks,
            "top1_logprobs": lps,
            "first_100_tokens": toks[:100],
            "response_preview": text[:500],
        })

        elapsed = time.time() - t_start
        acc = sum(1 for p in results["per_problem"] if p["correct"]) / (i+1)
        if (i+1) % 5 == 0 or i+1 == len(problems):
            print(f"  [{i+1:>3}/{len(problems)}] acc={acc:.1%} elapsed={elapsed:.0f}s", flush=True)

    n_correct = sum(1 for p in results["per_problem"] if p["correct"])
    avg_len = sum(p["output_len"] for p in results["per_problem"]) / len(results["per_problem"])

    results["aggregate"] = {
        "accuracy": n_correct / len(results["per_problem"]),
        "n_correct": n_correct,
        "n_total": len(results["per_problem"]),
        "avg_output_length": avg_len,
        "total_runtime_seconds": time.time() - t_start,
    }
    print(f"\n  RESULT: acc={results['aggregate']['accuracy']:.1%} "
          f"avg_len={avg_len:.0f} rt={results['aggregate']['total_runtime_seconds']:.0f}s",
          flush=True)

    with open(args.out, "w") as f:
        json.dump(results, f)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
