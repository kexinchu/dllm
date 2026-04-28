#!/usr/bin/env python3
"""
Batch-invariance calibration (Table 1 for paper).

Fixed prompt, 256 new tokens, N runs cycling through batch sizes in BS_CYCLE.
For each run at bs>1 the target sequence (index 0) is compared against the
bs=1 within-scheme reference. Metrics per scheme:
  - unique output count (hash of 256-token sequence)
  - #runs differing from bs=1 reference / N
  - first diverging token index (min across runs)
  - deterministic (unique == 1)
  - total wall-clock time; overhead = t_scheme / t_bf16

Schemes:
  1. BF16 baseline                     (allow_bf16_reduced_precision_reduction=True)
  2. FP32 flag (cuBLAS)                (allow_bf16_reduced_precision_reduction=False)
  3. LayerCast  (BF16 store, FP32 compute per Linear)
  4. DetermLLM  (Triton fixed-plan GEMM)
"""
import sys, os, time, hashlib, json, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings; warnings.filterwarnings("ignore")

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from research import determ_llm
from motivation.test_layercast_latency import apply_layercast, remove_layercast

MODEL_PATH = "/home/kec23008/docker-sys/Models/Llama-3.1-8B-Instruct"
N_RUNS    = 1000
MAX_NEW   = 256
BS_CYCLE  = [1, 2, 4, 8, 16]  # 200 runs per bs

TARGET_PROMPT = "What is deterministic inference in large language models?"

FILLERS = [
    "What is the capital of France and why is it important?",
    "Explain quantum computing in simple terms for beginners.",
    "Write a short poem about mountains and rivers in spring.",
    "How does photosynthesis work in C3 and C4 plants today?",
    "What is the meaning of life according to philosophy here?",
    "Tell me a joke about a programmer and a rubber duck now.",
    "Describe the process of nuclear fusion happening in the sun.",
    "What is machine learning and how does it differ from AI?",
    "How do modern CPUs achieve instruction level parallelism now?",
    "What is the relationship between entropy and information?",
    "Explain the double slit experiment and wave particle duality.",
    "Describe the architecture of a modern transformer network.",
    "How does CRISPR gene editing technology work step by step?",
    "What are the fundamental forces of nature and interactions?",
    "Explain public key cryptography and the RSA algorithm now.",
]


def make_equal_length_batch(tok, target_prompt, filler_prompts, target_len, device):
    """Continuous-batching sim: all sequences same length, same position_ids[0..L-1]."""
    all_prompts = [target_prompt] + filler_prompts
    all_ids = []
    for p in all_prompts:
        ids = tok.encode(p, add_special_tokens=True)
        if len(ids) >= target_len:
            ids = ids[:target_len]
        else:
            ids = ids + [tok.pad_token_id] * (target_len - len(ids))
        all_ids.append(ids)
    input_ids = torch.tensor(all_ids, device=device)
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(target_len, device=device).unsqueeze(0).expand(len(all_prompts), -1)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids}


def run_scheme(model, tok, name, enter_fn, exit_fn, target_prompt, target_len, device):
    print(f"\n=== {name} ===", flush=True)
    enter_fn()
    try:
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        # Reference bs=1
        ref_inp = make_equal_length_batch(tok, target_prompt, [], target_len, device)
        with torch.no_grad():
            ref_out = model.generate(**ref_inp, max_new_tokens=MAX_NEW,
                                     do_sample=False, pad_token_id=tok.pad_token_id)
        ref_tokens = ref_out[0, target_len:target_len+MAX_NEW].cpu().tolist()
        ref_hash   = hashlib.sha256(str(ref_tokens).encode()).hexdigest()[:16]

        hashes         = []
        first_diff_any = []
        per_bs_diff    = {bs: 0 for bs in BS_CYCLE}
        per_bs_total   = {bs: 0 for bs in BS_CYCLE}

        for i in range(N_RUNS):
            bs = BS_CYCLE[i % len(BS_CYCLE)]
            fillers = FILLERS[:max(bs-1, 0)]
            inp = make_equal_length_batch(tok, target_prompt, fillers, target_len, device)
            with torch.no_grad():
                out = model.generate(**inp, max_new_tokens=MAX_NEW,
                                     do_sample=False, pad_token_id=tok.pad_token_id)
            tokens = out[0, target_len:target_len+MAX_NEW].cpu().tolist()
            h = hashlib.sha256(str(tokens).encode()).hexdigest()[:16]
            hashes.append(h)
            per_bs_total[bs] += 1
            if h != ref_hash:
                per_bs_diff[bs] += 1
                # first-diff index
                for idx, (a, b) in enumerate(zip(ref_tokens, tokens)):
                    if a != b:
                        first_diff_any.append(idx); break
                else:
                    # lengths differ but no element-wise mismatch — shouldn't happen here
                    first_diff_any.append(min(len(ref_tokens), len(tokens)))

            if (i + 1) % 50 == 0:
                elapsed = time.perf_counter() - t0
                uniq = len(set(hashes))
                print(f"    [{i+1:>4}/{N_RUNS}] {elapsed:6.0f}s  uniq={uniq}  diff={len(first_diff_any)}", flush=True)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        unique   = len(set(hashes))
        diff_cnt = sum(1 for h in hashes if h != ref_hash)
        first_di = min(first_diff_any) if first_diff_any else None

        return {
            "name":            name,
            "N":               N_RUNS,
            "unique":          unique,
            "diff_count":      diff_cnt,
            "deterministic":   unique == 1,
            "first_diff_idx":  first_di,
            "total_time_s":    elapsed,
            "per_run_ms":      elapsed / N_RUNS * 1000,
            "ref_hash":        ref_hash,
            "per_bs_diff":     per_bs_diff,
            "per_bs_total":    per_bs_total,
        }
    finally:
        exit_fn()
        torch.cuda.empty_cache(); gc.collect()


def main():
    torch.cuda.empty_cache()
    device = "cuda:0"

    print("=" * 80)
    print("  BATCH-INVARIANCE CALIBRATION")
    print(f"  model={MODEL_PATH}")
    print(f"  N={N_RUNS}  max_new_tokens={MAX_NEW}  bs_cycle={BS_CYCLE}")
    print("=" * 80, flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation="sdpa",
    )
    model.eval()

    target_ids = tok.encode(TARGET_PROMPT, add_special_tokens=True)
    target_len = len(target_ids)
    print(f"  target_len={target_len} tokens  prompt=\"{TARGET_PROMPT}\"\n", flush=True)

    # -------- scheme enter/exit hooks --------
    def bf16_enter():
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    def bf16_exit():
        pass

    def fp32flag_enter():
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    def fp32flag_exit():
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True

    _lc_state = {}
    def lc_enter():
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        _lc_state["orig"] = apply_layercast(model)
    def lc_exit():
        remove_layercast(model, _lc_state["orig"])

    def det_enter():
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        determ_llm.enable(backend='triton')
    def det_exit():
        determ_llm.disable()

    schemes = [
        ("BF16 baseline",        bf16_enter,    bf16_exit),
        ("FP32 flag (cuBLAS)",   fp32flag_enter, fp32flag_exit),
        ("LayerCast",            lc_enter,      lc_exit),
        ("DetermLLM",            det_enter,     det_exit),
    ]

    out_path = "/home/kec23008/docker-sys/dllm/motivation/batch_invariance_calibration.json"
    all_results = []
    for name, ein, eout in schemes:
        r = run_scheme(model, tok, name, ein, eout, TARGET_PROMPT, target_len, device)
        print(f"  [{name}] unique={r['unique']}  diff={r['diff_count']}/{r['N']}  "
              f"first_diff_idx={r['first_diff_idx']}  det={r['deterministic']}  "
              f"time={r['total_time_s']:.1f}s  per_run={r['per_run_ms']:.0f}ms", flush=True)
        all_results.append(r)
        # incremental save
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)

    # -------- final table --------
    bf16_t = all_results[0]["total_time_s"]
    print("\n" + "=" * 96)
    print("  FINAL TABLE  (Llama-3.1-8B-Instruct, N=1000, 256 new tokens)")
    print("=" * 96)
    print(f"  {'Scheme':<22} {'unique':>7} {'diff/1000':>10} {'1st diff':>10} {'det':>5} {'t (s)':>9} {'overhead':>9}")
    print("  " + "-" * 78)
    for r in all_results:
        ovh = r["total_time_s"] / bf16_t
        fd  = str(r["first_diff_idx"]) if r["first_diff_idx"] is not None else "-"
        det = "YES" if r["deterministic"] else "NO"
        print(f"  {r['name']:<22} {r['unique']:>7} {r['diff_count']:>10} {fd:>10} {det:>5} "
              f"{r['total_time_s']:>9.1f} {ovh:>8.2f}x")
    print("=" * 96)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
