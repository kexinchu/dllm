#!/usr/bin/env python3
"""
MoE Determinism on DeepSeek-V2-Lite (BF16, 64 experts, top-6, 30GB).
Fits on 1x A6000. Continuous batching simulation (no padding).
"""
import sys, os, time, hashlib, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings; warnings.filterwarnings("ignore")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/home/kec23008/docker-sys/Models/DeepSeek-V2-Lite"


def header(t):
    print(f"\n{'='*80}\n  {t}\n{'='*80}")


def make_equal_length_batch(tok, target, fillers, tlen, device):
    all_p = [target] + fillers
    all_ids = []
    for p in all_p:
        ids = tok.encode(p, add_special_tokens=True)
        ids = ids[:tlen] if len(ids) >= tlen else ids + [tok.pad_token_id] * (tlen - len(ids))
        all_ids.append(ids)
    input_ids = torch.tensor(all_ids, device=device)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "position_ids": torch.arange(tlen, device=device).unsqueeze(0).expand(len(all_p), -1),
    }


def main():
    print("=" * 80)
    print("  DEEPSEEK-V2-LITE MOE DETERMINISM")
    print("=" * 80)
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

    print("Loading model...")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    model.eval()
    mem = torch.cuda.memory_allocated(0) / 1e9
    print(f"  Loaded. GPU memory: {mem:.1f} GB")

    PROMPT = "Explain the concept of mixture of experts in neural networks."
    fillers = [
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
    target_ids = tok.encode(PROMPT, add_special_tokens=True)
    target_len = len(target_ids)
    device = next(model.parameters()).device

    # ---- Test 1: Generation Determinism (200 runs) ----
    header("TEST 1: GENERATION DETERMINISM (200 runs)")
    batch_cycle = [1, 2, 4, 8, 16]
    N = 200
    results = {}

    for mode, flag_val in [("BF16 (default)", True), ("FP32 accum", False)]:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = flag_val
        hashes = []
        hash_by_bs = {}
        t0 = time.perf_counter()
        for i in range(N):
            bs = batch_cycle[i % len(batch_cycle)]
            inp = make_equal_length_batch(tok, PROMPT, fillers[:max(bs-1,0)], target_len, device)
            with torch.no_grad():
                logits = model(**inp, use_cache=False).logits[0]  # [seq_len, vocab]
            # Hash the argmax token predictions (proxy for generation)
            argmax_tokens = logits.float().argmax(dim=-1).cpu().tolist()
            h = hashlib.sha256(str(argmax_tokens).encode()).hexdigest()[:16]
            hashes.append(h)
            hash_by_bs.setdefault(bs, []).append(h)
            if (i+1) % 50 == 0:
                elapsed = time.perf_counter() - t0
                print(f"    [{i+1}/{N}] {elapsed:.0f}s unique={len(set(hashes))}")

        unique = len(set(hashes))
        elapsed = time.perf_counter() - t0
        det = "YES" if unique == 1 else "NO"
        print(f"  {mode}: {unique} unique / {N} runs  deterministic={det}  ({elapsed:.0f}s)")
        if unique > 1:
            from collections import Counter
            print(f"    Distribution: {dict(Counter(hashes).most_common(5))}")
            for bs in sorted(hash_by_bs):
                print(f"    bs={bs}: {len(set(hash_by_bs[bs]))} unique / {len(hash_by_bs[bs])} runs")
        results[mode] = {"unique": unique, "N": N, "deterministic": unique == 1}
        print()

    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True

    # ---- Test 2: Logits Batch Variance ----
    header("TEST 2: LOGITS BATCH VARIANCE")
    for mode, flag_val in [("BF16", True), ("FP32 accum", False)]:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = flag_val
        ref_inp = make_equal_length_batch(tok, PROMPT, [], target_len, device)
        with torch.no_grad():
            ref_logits = model(**ref_inp, use_cache=False).logits[0]
        print(f"  {mode}:")
        for bs in [2, 4, 8, 16]:
            inp = make_equal_length_batch(tok, PROMPT, fillers[:bs-1], target_len, device)
            with torch.no_grad():
                logits = model(**inp, use_cache=False).logits[0]
            diff = (ref_logits.float() - logits.float()).abs()
            md = diff.max().item()
            argmax_ok = (ref_logits.float().argmax(-1) == logits.float().argmax(-1)).all().item()
            print(f"    bs={bs:>2}: max_diff={md:.4e} argmax={'MATCH' if argmax_ok else 'FLIP'}")
        print()
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True

    # ---- Test 3: Latency ----
    header("TEST 3: LATENCY")
    inp = make_equal_length_batch(tok, PROMPT, [], target_len, device)
    for mode, flag_val in [("BF16", True), ("FP32 accum", False)]:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = flag_val
        for _ in range(2):
            with torch.no_grad():
                model(**inp, use_cache=False)
        torch.cuda.synchronize()
        times = []
        for _ in range(10):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                model(**inp, use_cache=False)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
        mean_ms = sum(times) / len(times)
        print(f"  {mode}: {mean_ms:.1f} ms")
        results[f"latency_{mode}"] = mean_ms
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True

    # Save
    header("SUMMARY")
    for mode in ["BF16 (default)", "FP32 accum"]:
        if mode in results:
            print(f"  {mode}: {results[mode]['unique']} unique / {results[mode]['N']} runs → {'DET' if results[mode]['deterministic'] else 'NON-DET'}")

    out_path = os.path.join(os.path.dirname(__file__), "exp_moe.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    out_md = os.path.join(os.path.dirname(__file__), "exp_moe.md")
    with open(out_md, "w") as f:
        f.write("# DeepSeek-V2-Lite MoE Determinism Results\n\n")
        f.write("**Model**: DeepSeek-V2-Lite (BF16, 15.7B total, 64 experts, top-6)\n")
        f.write("**Hardware**: NVIDIA RTX A6000\n\n")
        f.write("## Generation Determinism (200 runs, continuous batching sim)\n\n")
        f.write("| Mode | Unique Outputs | Deterministic? |\n|------|---:|:---:|\n")
        for mode in ["BF16 (default)", "FP32 accum"]:
            if mode in results:
                f.write(f"| {mode} | {results[mode]['unique']} | {'Yes' if results[mode]['deterministic'] else 'No'} |\n")
        f.write(f"\n## Latency\n\n")
        f.write(f"| BF16 | FP32 accum | Overhead |\n|---:|---:|---:|\n")
        bf16 = results.get("latency_BF16", 0)
        fp32 = results.get("latency_FP32 accum", 0)
        f.write(f"| {bf16:.1f} ms | {fp32:.1f} ms | {fp32/bf16:.2f}x |\n")

    print(f"\n  Saved to {out_path} and {out_md}")


if __name__ == "__main__":
    main()
