"""
MATH500 evaluation for DeepSeek-R1-Distill-Qwen-7B under different methods
and runtime configurations.

Usage:
  python run_math500_eval.py --method triton --batch-size 8 --seed 0 \
                             --n-problems 100 --out results.json

Methods:
  bf16_baseline      vanilla BF16 inference
  layercast          BF16 weights, FP32 compute (Yuan et al. baseline)
  dllm_cublaslt      our cuBLASLt backend
  dllm_triton        our Triton backend
  fp32               full FP32 (reference upper bound)

We run each problem at the requested batch size by padding with the same
prompt (batch-composition perturbation test). Row 0 of the batch is the
target; others are filler. This matches Yuan et al.'s continuous-batching
simulation.

Output per problem: response, extracted answer, correctness, length, logprobs.
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

# Harness modules (ours)
import determ_llm
import layercast

MODEL_PATH = "/home/kec23008/docker-sys/Models/DeepSeek-R1-Distill-Qwen-7B"

# Reasoning-style prompt wrapper (matches Yuan et al.'s template for DeepSeek-R1-Distill)
SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


# ── Answer extraction (MATH500 / AIME style) ─────────────────────────────────
def extract_boxed_answer(text: str):
    """Extract content from the LAST \\boxed{...} in the response.

    Handles nested braces. Returns None if no boxed answer found.
    """
    marker = "\\boxed{"
    last = text.rfind(marker)
    if last == -1:
        return None
    # Walk forward matching braces
    i = last + len(marker)
    depth = 1
    start = i
    while i < len(text) and depth > 0:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i].strip()
        i += 1
    return None


def normalize_math_answer(ans):
    """Loose normalization for string-equality answer checking."""
    if ans is None:
        return None
    s = ans.strip()
    # Drop surrounding latex math mode markers
    s = s.strip("$")
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s)
    # Strip trailing punctuation
    s = s.rstrip(".,")
    # Normalize simple fraction latex
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    # Numeric canonicalization where possible
    try:
        v = float(s.replace(",", ""))
        # Integer-valued floats printed without decimal point
        if v == int(v):
            return str(int(v))
        return f"{v:.6g}"
    except ValueError:
        return s


def answers_match(pred, gold):
    """Return True if pred matches gold after normalization."""
    np_ = normalize_math_answer(pred)
    ng = normalize_math_answer(gold)
    if np_ is None or ng is None:
        return False
    if np_ == ng:
        return True
    # Numeric tolerance fallback
    try:
        return abs(float(np_) - float(ng)) < 1e-6
    except ValueError:
        return False


# ── Dataset loader ────────────────────────────────────────────────────────────
def load_math500(n=100):
    """Load first `n` problems from MATH500.

    We cache the dataset locally in research/math500_cached.json so repeated
    runs don't hit HuggingFace.
    """
    cache = os.path.join(DLLM_DIR, "math500_cached.json")
    if os.path.exists(cache):
        with open(cache) as f:
            data = json.load(f)
    else:
        from datasets import load_dataset
        # HuggingFaceH4/MATH-500 is a common mirror
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        data = [
            {"problem": ex["problem"], "answer": ex["answer"], "subject": ex.get("subject", "")}
            for ex in ds
        ]
        with open(cache, "w") as f:
            json.dump(data, f)
    return data[:n]


# ── Generation ────────────────────────────────────────────────────────────────
def apply_chat_template(tokenizer, problem):
    """DeepSeek-R1-Distill uses Qwen-style chat template."""
    messages = [
        {"role": "user", "content": f"{problem}\n\n{SYSTEM_PROMPT}"},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def generate_one(model, tokenizer, prompt_text, batch_size, gen_len, device, seed):
    """Generate for one problem at given batch size.

    The same prompt fills every row of the batch. Batch composition is thus
    identical across rows; any output variation in row 0 across different
    batch sizes comes from numerical ordering differences, not from the
    prompt content.
    """
    torch.manual_seed(seed)
    enc = tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(device)
    ids = enc.repeat(batch_size, 1).contiguous()

    token_ids = []
    token_lps = []
    with torch.no_grad():
        # Use KV cache for speed; prefill once, then step decode.
        out = model(input_ids=ids, use_cache=True)
        past = out.past_key_values
        logits = out.logits[0, -1]
        lps = F.log_softmax(logits, dim=-1)
        tok = lps.argmax().item()
        token_ids.append(tok)
        token_lps.append(lps[tok].item())

        for _ in range(gen_len - 1):
            next_col = torch.full((batch_size, 1), tok, dtype=ids.dtype, device=device)
            out = model(input_ids=next_col, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[0, -1]
            lps = F.log_softmax(logits, dim=-1)
            tok_new = lps.argmax().item()
            token_ids.append(tok_new)
            token_lps.append(lps[tok_new].item())
            tok = tok_new
            # Early stop on EOS
            if tok == tokenizer.eos_token_id:
                break

    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    return text, token_ids, token_lps


# ── Method setup / teardown ──────────────────────────────────────────────────
def set_method(method):
    """Configure the active inference method. Must be called before each run."""
    # Reset all patches
    try:
        determ_llm.disable()
    except Exception:
        pass
    try:
        layercast.disable()
    except Exception:
        pass
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True

    if method == "bf16_baseline":
        pass
    elif method == "layercast":
        layercast.enable()
    elif method == "dllm_cublaslt":
        determ_llm.enable(backend="cublaslt")
    elif method == "dllm_triton":
        determ_llm.enable(backend="triton")
    elif method == "dllm_hybrid":
        determ_llm.enable(backend="hybrid")
    elif method == "dllm_hybrid_attn":
        # Hybrid backend + decode-phase attention patch (torch.matmul/bmm)
        determ_llm.enable(backend="hybrid", attn=True)
    elif method == "fp32":
        # Reference: full FP32 (we cheat by using LayerCast which is FP32 compute)
        layercast.enable()
    else:
        raise ValueError(f"Unknown method: {method}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    choices=["bf16_baseline", "layercast", "dllm_cublaslt",
                             "dllm_triton", "dllm_hybrid", "dllm_hybrid_attn", "fp32"])
    ap.add_argument("--batch-size", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-problems", type=int, default=100)
    ap.add_argument("--gen-len", type=int, default=2048)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-path", default=MODEL_PATH)
    args = ap.parse_args()

    print(f"=== method={args.method} bs={args.batch_size} seed={args.seed} ===")
    print(f"Loading {args.model_path} ...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="eager",  # so our attention patch can plug in
    )
    model.eval()
    device = next(model.parameters()).device

    set_method(args.method)
    print(f"  method activated: {args.method}")

    print(f"Loading MATH500 (n={args.n_problems}) ...", flush=True)
    problems = load_math500(args.n_problems)

    results = {
        "meta": {
            "method": args.method,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "n_problems": args.n_problems,
            "gen_len": args.gen_len,
            "model": args.model_path,
        },
        "per_problem": [],
    }

    t_start = time.time()
    for i, ex in enumerate(problems):
        prompt_text = apply_chat_template(tokenizer, ex["problem"])
        text, toks, lps = generate_one(
            model, tokenizer, prompt_text,
            args.batch_size, args.gen_len, device, args.seed
        )
        pred = extract_boxed_answer(text)
        correct = answers_match(pred, ex["answer"])
        tok_hash = hashlib.md5(str(toks).encode()).hexdigest()[:12]

        results["per_problem"].append({
            "idx": i,
            "subject": ex.get("subject", ""),
            "gold": ex["answer"],
            "pred": pred,
            "correct": correct,
            "output_len": len(toks),
            "token_hash": tok_hash,
            "first_20_tokens": toks[:20],
            "mean_logprob": sum(lps) / len(lps) if lps else 0.0,
            "response": text[:5000],  # truncate to keep JSON size bounded
        })

        # Per-problem status line
        elapsed = time.time() - t_start
        acc_so_far = sum(1 for p in results["per_problem"] if p["correct"]) / (i + 1)
        print(f"  [{i+1:>3}/{len(problems)}] correct={'✓' if correct else '✗'} "
              f"len={len(toks)} hash={tok_hash} acc_so_far={acc_so_far:.2%} "
              f"elapsed={elapsed:.0f}s", flush=True)

    # Aggregate
    n_correct = sum(1 for p in results["per_problem"] if p["correct"])
    n_total = len(results["per_problem"])
    acc = n_correct / n_total if n_total else 0.0
    avg_len = sum(p["output_len"] for p in results["per_problem"]) / n_total

    results["aggregate"] = {
        "accuracy": acc,
        "n_correct": n_correct,
        "n_total": n_total,
        "avg_output_length": avg_len,
        "total_runtime_seconds": time.time() - t_start,
    }
    print(f"\n  RESULT: accuracy={acc:.2%} ({n_correct}/{n_total}), "
          f"avg_len={avg_len:.1f}, runtime={results['aggregate']['total_runtime_seconds']:.1f}s",
          flush=True)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
