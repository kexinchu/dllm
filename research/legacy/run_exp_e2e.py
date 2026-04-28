#!/usr/bin/env python3
"""
End-to-End Batch-Variance Experiments
======================================
Tests how batch-size variation affects full model generation determinism,
argmax flip rates, latency, and long-sequence fidelity.

Experiments:
  1. Llama-3.1-8B 1000-run generation determinism (BF16 vs FP32 accum)
  2. Llama-3.1-8B argmax flip rate (10 prompts x 4 batch sizes)
  3. Qwen3-30B-A3B MoE generation determinism (200 runs)
  4. Llama-3.1-8B latency comparison (BF16 vs FP32 accum)
  5. Llama-3.1-8B long-sequence (200+ tokens) logits comparison
"""

import json
import os
import sys
import time
import hashlib
import traceback
from collections import Counter

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Paths ───────────────────────────────────────────────────────────────────
LLAMA_PATH = "/home/kec23008/docker-sys/Models/Llama-3.1-8B-Instruct"
QWEN_PATH = "/home/kec23008/docker-sys/Models/Qwen3-30B-A3B-Instruct-2507-int4-mixed-AutoRound"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Reproducibility ─────────────────────────────────────────────────────────
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# ── Results container ────────────────────────────────────────────────────────
results = {
    "metadata": {
        "gpu": torch.cuda.get_device_name(0),
        "num_gpus": torch.cuda.device_count(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    },
    "exp1_generation_determinism": {},
    "exp2_argmax_flip_rate": {},
    "exp3_qwen_moe_determinism": {},
    "exp4_latency": {},
    "exp5_long_sequence": {},
}


def header(title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    sys.stdout.flush()


def make_equal_length_batch(tok, target_prompt, filler_prompts, target_len, device):
    """
    Tokenize all prompts and truncate/pad to exactly target_len tokens.
    Simulates continuous batching: no left-padding, no RoPE position shift.
    """
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

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }


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


# ═════════════════════════════════════════════════════════════════════════════
# Load Llama model
# ═════════════════════════════════════════════════════════════════════════════
def load_llama():
    print("Loading Llama-3.1-8B-Instruct...")
    sys.stdout.flush()
    tok = AutoTokenizer.from_pretrained(LLAMA_PATH, use_fast=False)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        LLAMA_PATH, dtype=torch.bfloat16, device_map={"": 0}
    )
    model.eval()
    print(f"  Model loaded on {next(model.parameters()).device}")
    sys.stdout.flush()
    return model, tok


# ═════════════════════════════════════════════════════════════════════════════
# Experiment 1: 1000-run generation determinism
# ═════════════════════════════════════════════════════════════════════════════
def experiment1(model, tok):
    header("EXPERIMENT 1: Llama-3.1-8B 1000-run Generation Determinism")
    device = next(model.parameters()).device

    target_prompt = "What is deterministic inference in large language models?"
    target_ids = tok.encode(target_prompt, add_special_tokens=True)
    target_len = len(target_ids)
    batch_cycle = [1, 2, 3, 4, 5, 7, 8, 9, 15, 16]
    N = 1000
    MAX_NEW = 32

    print(f"  Prompt: \"{target_prompt}\"")
    print(f"  Token length: {target_len}")
    print(f"  Runs: {N}, batch cycle: {batch_cycle}")
    print(f"  Max new tokens: {MAX_NEW}, greedy decoding")
    sys.stdout.flush()

    exp1_results = {}

    for mode_label, flag_val in [("bf16_default", True), ("fp32_accum", False)]:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = flag_val
        mode_display = "BF16 (default)" if flag_val else "FP32 accum (cuBLAS flag)"
        print(f"\n  --- {mode_display} ---")
        sys.stdout.flush()

        hashes = []
        hash_by_bs = {}
        outputs_by_hash = {}
        t0 = time.perf_counter()

        for i in range(N):
            bs = batch_cycle[i % len(batch_cycle)]
            inp = make_equal_length_batch(tok, target_prompt, FILLERS[:max(bs-1, 0)], target_len, device)
            with torch.no_grad():
                out = model.generate(
                    **inp, max_new_tokens=MAX_NEW, do_sample=False,
                    temperature=1.0,
                    pad_token_id=tok.pad_token_id,
                )
            tokens = out[0].cpu().tolist()
            h = hashlib.sha256(str(tokens).encode()).hexdigest()[:16]
            hashes.append(h)
            hash_by_bs.setdefault(bs, []).append(h)

            # Store one decoded sample per hash
            if h not in outputs_by_hash:
                outputs_by_hash[h] = tok.decode(out[0][target_len:], skip_special_tokens=True)[:120]

            if (i + 1) % 100 == 0:
                elapsed = time.perf_counter() - t0
                unique = len(set(hashes))
                print(f"    [{i+1}/{N}] {elapsed:.0f}s elapsed, unique_outputs={unique}")
                sys.stdout.flush()

        elapsed = time.perf_counter() - t0
        unique = len(set(hashes))
        det = unique == 1

        print(f"  Result: {unique} unique outputs / {N} runs  deterministic={'YES' if det else 'NO'}  ({elapsed:.1f}s)")

        # Hash distribution
        counter = Counter(hashes)
        hash_dist = dict(counter.most_common(10))

        # Per-batch-size uniqueness
        bs_uniqueness = {}
        for bs in sorted(hash_by_bs):
            bs_unique = len(set(hash_by_bs[bs]))
            bs_total = len(hash_by_bs[bs])
            bs_uniqueness[str(bs)] = {"unique": bs_unique, "total": bs_total}
            print(f"    bs={bs:>2}: {bs_unique} unique / {bs_total} runs")

        if det:
            sample = list(outputs_by_hash.values())[0]
            print(f"    Output: \"{sample[:80]}...\"")

        exp1_results[mode_label] = {
            "total_runs": N,
            "unique_outputs": unique,
            "deterministic": det,
            "elapsed_seconds": round(elapsed, 1),
            "hash_distribution": hash_dist,
            "per_bs_uniqueness": bs_uniqueness,
            "sample_outputs": outputs_by_hash,
        }
        sys.stdout.flush()

    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    results["exp1_generation_determinism"] = exp1_results


# ═════════════════════════════════════════════════════════════════════════════
# Experiment 2: Argmax flip rate
# ═════════════════════════════════════════════════════════════════════════════
def experiment2(model, tok):
    header("EXPERIMENT 2: Llama-3.1-8B Argmax Flip Rate")
    device = next(model.parameters()).device

    # 10 prompts of varying length (30-80 tokens)
    prompts = [
        "Please provide a detailed explanation of quantum computing and how qubits differ from classical bits in practice.",
        "Write a comprehensive essay about the history of mathematics from ancient civilizations through the modern era.",
        "Explain in detail how modern operating systems manage memory processes and file systems with virtual memory.",
        "Describe the complete lifecycle of a star from nebula formation through main sequence and eventual death.",
        "Explain the principles of thermodynamics including the zeroth first second and third laws with examples.",
        "Discuss the major breakthroughs in physics during the twentieth century including relativity and quantum mechanics.",
        "Describe how the internet works from physical cables to application layer protocols including TCP IP.",
        "Explain the central dogma of molecular biology covering DNA replication transcription and translation.",
        "What is the relationship between entropy information theory and thermodynamics in physics and computer science?",
        "How do modern GPUs achieve massive parallelism for deep learning workloads and what are the key architectural features?",
    ]

    batch_sizes = [2, 4, 8, 16]

    print(f"  Prompts: {len(prompts)}")
    print(f"  Batch sizes: {batch_sizes}")
    sys.stdout.flush()

    exp2_results = {}

    for mode_label, flag_val in [("bf16_default", True), ("fp32_accum", False)]:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = flag_val
        mode_display = "BF16 (default)" if flag_val else "FP32 accum"
        print(f"\n  --- {mode_display} ---")
        sys.stdout.flush()

        total_tokens = 0
        total_flips = 0
        per_prompt_data = []

        for p_idx, prompt in enumerate(prompts):
            target_ids = tok.encode(prompt, add_special_tokens=True)
            target_len = len(target_ids)

            # Reference: bs=1
            ref_inp = make_equal_length_batch(tok, prompt, [], target_len, device)
            with torch.no_grad():
                ref_logits = model(**ref_inp).logits[0]
            ref_argmax = ref_logits.float().argmax(dim=-1)

            prompt_flips = 0
            prompt_tokens = 0

            for bs in batch_sizes:
                inp = make_equal_length_batch(tok, prompt, FILLERS[:bs-1], target_len, device)
                with torch.no_grad():
                    logits = model(**inp).logits[0]
                batch_argmax = logits.float().argmax(dim=-1)

                flips = (ref_argmax != batch_argmax).sum().item()
                max_diff = (ref_logits.float() - logits.float()).abs().max().item()
                mean_diff = (ref_logits.float() - logits.float()).abs().mean().item()

                prompt_flips += flips
                prompt_tokens += target_len
                total_flips += flips
                total_tokens += target_len

            per_prompt_data.append({
                "prompt_idx": p_idx,
                "token_length": target_len,
                "flips": prompt_flips,
                "tokens_checked": prompt_tokens,
                "flip_rate": prompt_flips / max(prompt_tokens, 1),
            })

            print(f"    prompt {p_idx} ({target_len} tok): cumulative {total_flips}/{total_tokens} "
                  f"({total_flips/max(total_tokens,1)*100:.4f}%)")
            sys.stdout.flush()

        rate = total_flips / max(total_tokens, 1) * 100
        print(f"  {mode_display} TOTAL: {total_flips} flips / {total_tokens} tokens = {rate:.4f}%")

        exp2_results[mode_label] = {
            "total_flips": total_flips,
            "total_tokens": total_tokens,
            "flip_rate_pct": round(rate, 6),
            "per_prompt": per_prompt_data,
        }

    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    results["exp2_argmax_flip_rate"] = exp2_results


# ═════════════════════════════════════════════════════════════════════════════
# Experiment 3: Qwen3-30B-A3B MoE
# ═════════════════════════════════════════════════════════════════════════════
def experiment3():
    header("EXPERIMENT 3: Qwen3-30B-A3B MoE Generation Determinism")

    try:
        print("  Attempting to load Qwen3-30B-A3B (INT4) on 2 GPUs...")
        sys.stdout.flush()
        t0 = time.perf_counter()

        tok_q = AutoTokenizer.from_pretrained(QWEN_PATH, use_fast=False)
        tok_q.pad_token = tok_q.eos_token if tok_q.eos_token else tok_q.pad_token
        tok_q.padding_side = "left"

        model_q = AutoModelForCausalLM.from_pretrained(
            QWEN_PATH, device_map="auto"
        )
        model_q.eval()
        load_time = time.perf_counter() - t0
        print(f"  Loaded in {load_time:.1f}s")
        sys.stdout.flush()

        device = next(model_q.parameters()).device

        target_prompt = "What is deterministic inference in large language models?"
        target_ids = tok_q.encode(target_prompt, add_special_tokens=True)
        target_len = len(target_ids)
        batch_cycle = [1, 2, 3, 4, 5, 7, 8]
        N = 200
        MAX_NEW = 32

        print(f"  Prompt token length: {target_len}")
        print(f"  Runs: {N}, batch cycle: {batch_cycle}")
        sys.stdout.flush()

        qwen_fillers = FILLERS[:7]  # Use fewer fillers for smaller batch sizes
        exp3_results = {}

        for mode_label, flag_val in [("bf16_default", True), ("fp32_accum", False)]:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = flag_val
            mode_display = "BF16 (default)" if flag_val else "FP32 accum"
            print(f"\n  --- {mode_display} ---")
            sys.stdout.flush()

            hashes = []
            hash_by_bs = {}
            outputs_by_hash = {}
            t0 = time.perf_counter()

            for i in range(N):
                bs = batch_cycle[i % len(batch_cycle)]
                inp = make_equal_length_batch(tok_q, target_prompt, qwen_fillers[:max(bs-1, 0)], target_len, device)
                with torch.no_grad():
                    out = model_q.generate(
                        **inp, max_new_tokens=MAX_NEW, do_sample=False,
                        temperature=1.0,
                        pad_token_id=tok_q.pad_token_id,
                    )
                tokens = out[0].cpu().tolist()
                h = hashlib.sha256(str(tokens).encode()).hexdigest()[:16]
                hashes.append(h)
                hash_by_bs.setdefault(bs, []).append(h)

                if h not in outputs_by_hash:
                    outputs_by_hash[h] = tok_q.decode(out[0][target_len:], skip_special_tokens=True)[:120]

                if (i + 1) % 50 == 0:
                    elapsed = time.perf_counter() - t0
                    unique = len(set(hashes))
                    print(f"    [{i+1}/{N}] {elapsed:.0f}s elapsed, unique_outputs={unique}")
                    sys.stdout.flush()

            elapsed = time.perf_counter() - t0
            unique = len(set(hashes))
            det = unique == 1
            counter = Counter(hashes)
            hash_dist = dict(counter.most_common(10))

            bs_uniqueness = {}
            for bs in sorted(hash_by_bs):
                bs_unique = len(set(hash_by_bs[bs]))
                bs_total = len(hash_by_bs[bs])
                bs_uniqueness[str(bs)] = {"unique": bs_unique, "total": bs_total}

            print(f"  Result: {unique} unique outputs / {N} runs  deterministic={'YES' if det else 'NO'}  ({elapsed:.1f}s)")

            exp3_results[mode_label] = {
                "total_runs": N,
                "unique_outputs": unique,
                "deterministic": det,
                "elapsed_seconds": round(elapsed, 1),
                "hash_distribution": hash_dist,
                "per_bs_uniqueness": bs_uniqueness,
                "sample_outputs": outputs_by_hash,
            }

        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        results["exp3_qwen_moe_determinism"] = exp3_results

        # Cleanup Qwen model
        del model_q, tok_q
        torch.cuda.empty_cache()

    except Exception as e:
        tb = traceback.format_exc()
        error_msg = f"Failed to load/run Qwen3 MoE: {str(e)}"
        print(f"  ERROR: {error_msg}")
        print(f"  Traceback:\n{tb}")
        results["exp3_qwen_moe_determinism"] = {
            "status": "SKIPPED",
            "error": error_msg,
            "traceback": tb,
        }
        # Cleanup on failure
        torch.cuda.empty_cache()


# ═════════════════════════════════════════════════════════════════════════════
# Experiment 4: Latency comparison
# ═════════════════════════════════════════════════════════════════════════════
def experiment4(model, tok):
    header("EXPERIMENT 4: Llama-3.1-8B Latency Comparison")
    device = next(model.parameters()).device

    target_prompt = "What is deterministic inference in large language models?"
    target_ids = tok.encode(target_prompt, add_special_tokens=True)
    target_len = len(target_ids)
    MAX_NEW = 32
    N_WARMUP = 3
    N_MEASURE = 20

    print(f"  Warmup: {N_WARMUP} runs, Measurement: {N_MEASURE} runs")
    print(f"  Max new tokens: {MAX_NEW}")
    sys.stdout.flush()

    exp4_results = {}

    for mode_label, flag_val in [("bf16_default", True), ("fp32_accum", False)]:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = flag_val
        mode_display = "BF16 (default)" if flag_val else "FP32 accum"
        print(f"\n  --- {mode_display} ---")
        sys.stdout.flush()

        # Warmup
        for _ in range(N_WARMUP):
            inp = make_equal_length_batch(tok, target_prompt, [], target_len, device)
            with torch.no_grad():
                model.generate(**inp, max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=tok.pad_token_id)
            torch.cuda.synchronize()

        # Measure
        latencies = []
        for i in range(N_MEASURE):
            inp = make_equal_length_batch(tok, target_prompt, [], target_len, device)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                model.generate(**inp, max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=tok.pad_token_id)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append(t1 - t0)

        mean_lat = sum(latencies) / len(latencies)
        std_lat = (sum((x - mean_lat)**2 for x in latencies) / len(latencies)) ** 0.5
        min_lat = min(latencies)
        max_lat = max(latencies)

        print(f"    Mean: {mean_lat*1000:.1f} ms")
        print(f"    Std:  {std_lat*1000:.1f} ms")
        print(f"    Min:  {min_lat*1000:.1f} ms")
        print(f"    Max:  {max_lat*1000:.1f} ms")

        exp4_results[mode_label] = {
            "n_runs": N_MEASURE,
            "mean_ms": round(mean_lat * 1000, 2),
            "std_ms": round(std_lat * 1000, 2),
            "min_ms": round(min_lat * 1000, 2),
            "max_ms": round(max_lat * 1000, 2),
            "all_latencies_ms": [round(x * 1000, 2) for x in latencies],
        }

    # Compute overhead ratio
    if "bf16_default" in exp4_results and "fp32_accum" in exp4_results:
        bf16_mean = exp4_results["bf16_default"]["mean_ms"]
        fp32_mean = exp4_results["fp32_accum"]["mean_ms"]
        overhead = (fp32_mean - bf16_mean) / bf16_mean * 100 if bf16_mean > 0 else 0
        exp4_results["overhead_ratio"] = {
            "fp32_accum_vs_bf16_pct": round(overhead, 2),
            "description": f"FP32 accum is {overhead:+.2f}% vs BF16 default",
        }
        print(f"\n  Overhead: FP32 accum is {overhead:+.2f}% vs BF16 default")

    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    results["exp4_latency"] = exp4_results


# ═════════════════════════════════════════════════════════════════════════════
# Experiment 5: Long sequence (200+ tokens)
# ═════════════════════════════════════════════════════════════════════════════
def experiment5(model, tok):
    header("EXPERIMENT 5: Llama-3.1-8B Long Sequence (200+ tokens)")
    device = next(model.parameters()).device

    # Build a long prompt (~200+ tokens)
    long_prompt = (
        "Please provide a comprehensive and detailed analysis of the following interconnected topics, "
        "covering their historical development, current state of the art, and future directions. "
        "First, discuss the evolution of artificial intelligence from the Dartmouth conference in 1956 "
        "through expert systems, the AI winter, and the modern deep learning revolution. Second, explain "
        "how transformer architectures revolutionized natural language processing, starting from the "
        "attention mechanism in sequence-to-sequence models, through BERT and GPT, to modern large "
        "language models with billions of parameters. Third, analyze the computational challenges of "
        "training and serving these models, including distributed training strategies, mixed precision "
        "arithmetic, quantization techniques, and efficient inference frameworks like vLLM and TensorRT. "
        "Fourth, discuss the implications of non-deterministic floating-point arithmetic in GPU computing "
        "for model reproducibility, covering sources of variance in cuBLAS GEMM operations, FlashAttention "
        "split-KV strategies, and batch-dependent kernel selection."
    )

    target_ids = tok.encode(long_prompt, add_special_tokens=True)
    target_len = len(target_ids)
    print(f"  Prompt token length: {target_len}")
    sys.stdout.flush()

    batch_sizes_to_test = [1, 2, 4, 8, 16]

    exp5_results = {}

    for mode_label, flag_val in [("bf16_default", True), ("fp32_accum", False)]:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = flag_val
        mode_display = "BF16 (default)" if flag_val else "FP32 accum"
        print(f"\n  --- {mode_display} ---")
        sys.stdout.flush()

        # Reference: bs=1
        ref_inp = make_equal_length_batch(tok, long_prompt, [], target_len, device)
        with torch.no_grad():
            ref_logits = model(**ref_inp).logits[0]
        ref_argmax = ref_logits.float().argmax(dim=-1)

        comparisons = []
        for bs in batch_sizes_to_test:
            if bs == 1:
                inp = ref_inp
            else:
                inp = make_equal_length_batch(tok, long_prompt, FILLERS[:bs-1], target_len, device)

            with torch.no_grad():
                logits = model(**inp).logits[0]

            diff = (ref_logits.float() - logits.float()).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            batch_argmax = logits.float().argmax(dim=-1)
            flips = (ref_argmax != batch_argmax).sum().item()
            flip_rate = flips / target_len * 100

            status = "MATCH" if flips == 0 else f"FLIP({flips})"
            print(f"    bs={bs:>2}: max_diff={max_diff:.4e}  mean_diff={mean_diff:.4e}  "
                  f"argmax={status}  flip_rate={flip_rate:.3f}%")

            comparisons.append({
                "batch_size": bs,
                "max_diff": max_diff,
                "mean_diff": mean_diff,
                "argmax_flips": flips,
                "total_positions": target_len,
                "flip_rate_pct": round(flip_rate, 4),
            })

        exp5_results[mode_label] = {
            "prompt_token_length": target_len,
            "comparisons": comparisons,
        }

    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    results["exp5_long_sequence"] = exp5_results


# ═════════════════════════════════════════════════════════════════════════════
# Write markdown report
# ═════════════════════════════════════════════════════════════════════════════
def write_markdown():
    md_path = os.path.join(OUT_DIR, "exp_e2e.md")
    r = results
    meta = r["metadata"]

    lines = []
    lines.append("# End-to-End Batch Variance Experiments")
    lines.append("")
    lines.append(f"**Hardware:** {meta['gpu']} x {meta['num_gpus']}")
    lines.append(f"**Software:** PyTorch {meta['torch_version']}, CUDA {meta['cuda_version']}")
    lines.append(f"**Method:** Continuous batching simulation (equal-length sequences, no padding, explicit position_ids)")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── Experiment 1 ──
    lines.append("## 1. Llama-3.1-8B: 1000-Run Generation Determinism")
    lines.append("")
    lines.append("**Setup:** Single prompt, 1000 runs cycling through batch_sizes=[1,2,3,4,5,7,8,9,15,16],")
    lines.append("32 new tokens, greedy decoding (do_sample=False, temperature=1.0).")
    lines.append("")

    exp1 = r.get("exp1_generation_determinism", {})
    for mode_key, mode_display in [("bf16_default", "BF16 (default)"), ("fp32_accum", "FP32 accum (cuBLAS flag)")]:
        data = exp1.get(mode_key, {})
        if not data:
            continue
        lines.append(f"### 1.{1 if mode_key=='bf16_default' else 2} {mode_display}")
        lines.append("")
        lines.append(f"- **Total runs:** {data.get('total_runs', 'N/A')}")
        lines.append(f"- **Unique outputs:** {data.get('unique_outputs', 'N/A')}")
        lines.append(f"- **Deterministic:** {'YES' if data.get('deterministic') else 'NO'}")
        lines.append(f"- **Elapsed:** {data.get('elapsed_seconds', 'N/A')}s")
        lines.append("")

        # Hash distribution
        hash_dist = data.get("hash_distribution", {})
        if hash_dist:
            lines.append("**Hash distribution (top 10):**")
            lines.append("")
            lines.append("| Hash | Count |")
            lines.append("|------|------:|")
            for h, cnt in hash_dist.items():
                lines.append(f"| `{h}` | {cnt} |")
            lines.append("")

        # Per-BS uniqueness
        bs_uniq = data.get("per_bs_uniqueness", {})
        if bs_uniq:
            lines.append("**Per-batch-size uniqueness:**")
            lines.append("")
            lines.append("| Batch Size | Unique | Total |")
            lines.append("|-----------:|-------:|------:|")
            for bs_key in sorted(bs_uniq, key=lambda x: int(x)):
                u = bs_uniq[bs_key]
                lines.append(f"| {bs_key} | {u['unique']} | {u['total']} |")
            lines.append("")

        # Sample outputs
        samples = data.get("sample_outputs", {})
        if samples:
            lines.append("**Sample outputs:**")
            lines.append("")
            for h, text in list(samples.items())[:5]:
                lines.append(f"- `{h}`: \"{text[:100]}...\"")
            lines.append("")

    lines.append("---")
    lines.append("")

    # ── Experiment 2 ──
    lines.append("## 2. Llama-3.1-8B: Argmax Flip Rate")
    lines.append("")
    lines.append("**Setup:** 10 prompts of varying length, batch_sizes=[2,4,8,16], compare argmax vs bs=1 reference.")
    lines.append("")

    exp2 = r.get("exp2_argmax_flip_rate", {})
    for mode_key, mode_display in [("bf16_default", "BF16 (default)"), ("fp32_accum", "FP32 accum")]:
        data = exp2.get(mode_key, {})
        if not data:
            continue
        lines.append(f"### 2.{1 if mode_key=='bf16_default' else 2} {mode_display}")
        lines.append("")
        lines.append(f"- **Total flips:** {data.get('total_flips', 'N/A')}")
        lines.append(f"- **Total tokens:** {data.get('total_tokens', 'N/A')}")
        lines.append(f"- **Flip rate:** {data.get('flip_rate_pct', 'N/A')}%")
        lines.append("")

        per_prompt = data.get("per_prompt", [])
        if per_prompt:
            lines.append("| Prompt | Token Length | Flips | Tokens Checked | Flip Rate |")
            lines.append("|-------:|------------:|------:|---------------:|----------:|")
            for p in per_prompt:
                lines.append(f"| {p['prompt_idx']} | {p['token_length']} | {p['flips']} | {p['tokens_checked']} | {p['flip_rate']*100:.4f}% |")
            lines.append("")

    lines.append("---")
    lines.append("")

    # ── Experiment 3 ──
    lines.append("## 3. Qwen3-30B-A3B MoE: Generation Determinism")
    lines.append("")

    exp3 = r.get("exp3_qwen_moe_determinism", {})
    if exp3.get("status") == "SKIPPED":
        lines.append(f"**Status:** SKIPPED")
        lines.append(f"**Error:** {exp3.get('error', 'Unknown')}")
        lines.append("")
    else:
        lines.append("**Setup:** 200 runs cycling through batch_sizes=[1,2,3,4,5,7,8], 32 new tokens, greedy.")
        lines.append("")
        for mode_key, mode_display in [("bf16_default", "BF16 (default)"), ("fp32_accum", "FP32 accum")]:
            data = exp3.get(mode_key, {})
            if not data:
                continue
            lines.append(f"### 3.{1 if mode_key=='bf16_default' else 2} {mode_display}")
            lines.append("")
            lines.append(f"- **Total runs:** {data.get('total_runs', 'N/A')}")
            lines.append(f"- **Unique outputs:** {data.get('unique_outputs', 'N/A')}")
            lines.append(f"- **Deterministic:** {'YES' if data.get('deterministic') else 'NO'}")
            lines.append(f"- **Elapsed:** {data.get('elapsed_seconds', 'N/A')}s")
            lines.append("")

            hash_dist = data.get("hash_distribution", {})
            if hash_dist:
                lines.append("**Hash distribution:**")
                lines.append("")
                lines.append("| Hash | Count |")
                lines.append("|------|------:|")
                for h, cnt in hash_dist.items():
                    lines.append(f"| `{h}` | {cnt} |")
                lines.append("")

            bs_uniq = data.get("per_bs_uniqueness", {})
            if bs_uniq:
                lines.append("**Per-batch-size uniqueness:**")
                lines.append("")
                lines.append("| Batch Size | Unique | Total |")
                lines.append("|-----------:|-------:|------:|")
                for bs_key in sorted(bs_uniq, key=lambda x: int(x)):
                    u = bs_uniq[bs_key]
                    lines.append(f"| {bs_key} | {u['unique']} | {u['total']} |")
                lines.append("")

    lines.append("---")
    lines.append("")

    # ── Experiment 4 ──
    lines.append("## 4. Llama-3.1-8B: Latency Comparison")
    lines.append("")
    lines.append("**Setup:** 20 generation runs (3 warmup), bs=1, 32 new tokens.")
    lines.append("")

    exp4 = r.get("exp4_latency", {})

    lines.append("| Mode | Mean (ms) | Std (ms) | Min (ms) | Max (ms) |")
    lines.append("|------|----------:|---------:|---------:|---------:|")
    for mode_key, mode_display in [("bf16_default", "BF16 (default)"), ("fp32_accum", "FP32 accum")]:
        data = exp4.get(mode_key, {})
        if data:
            lines.append(f"| {mode_display} | {data['mean_ms']:.1f} | {data['std_ms']:.1f} | {data['min_ms']:.1f} | {data['max_ms']:.1f} |")
    lines.append("")

    overhead = exp4.get("overhead_ratio", {})
    if overhead:
        lines.append(f"**Overhead:** {overhead.get('description', 'N/A')}")
        lines.append("")

    lines.append("---")
    lines.append("")

    # ── Experiment 5 ──
    lines.append("## 5. Llama-3.1-8B: Long Sequence (200+ tokens)")
    lines.append("")

    exp5 = r.get("exp5_long_sequence", {})
    for mode_key, mode_display in [("bf16_default", "BF16 (default)"), ("fp32_accum", "FP32 accum")]:
        data = exp5.get(mode_key, {})
        if not data:
            continue
        lines.append(f"### 5.{1 if mode_key=='bf16_default' else 2} {mode_display}")
        lines.append(f"**Prompt length:** {data.get('prompt_token_length', 'N/A')} tokens")
        lines.append("")

        comps = data.get("comparisons", [])
        if comps:
            lines.append("| Batch Size | Max Diff | Mean Diff | Argmax Flips | Flip Rate |")
            lines.append("|-----------:|---------:|----------:|-------------:|----------:|")
            for c in comps:
                lines.append(f"| {c['batch_size']} | {c['max_diff']:.4e} | {c['mean_diff']:.4e} | {c['argmax_flips']} | {c['flip_rate_pct']:.4f}% |")
            lines.append("")

    lines.append("---")
    lines.append("")

    # ── Summary ──
    lines.append("## Summary of Findings")
    lines.append("")

    exp1_bf16 = exp1.get("bf16_default", {})
    exp1_fp32 = exp1.get("fp32_accum", {})
    exp2_bf16 = exp2.get("bf16_default", {})
    exp2_fp32 = exp2.get("fp32_accum", {})

    lines.append("| Experiment | BF16 (default) | FP32 accum |")
    lines.append("|------------|----------------|------------|")

    # Exp 1
    bf16_det = "YES" if exp1_bf16.get("deterministic") else f"NO ({exp1_bf16.get('unique_outputs', '?')} unique)"
    fp32_det = "YES" if exp1_fp32.get("deterministic") else f"NO ({exp1_fp32.get('unique_outputs', '?')} unique)"
    lines.append(f"| 1000-run generation determinism | {bf16_det} | {fp32_det} |")

    # Exp 2
    bf16_flip = f"{exp2_bf16.get('flip_rate_pct', '?')}%"
    fp32_flip = f"{exp2_fp32.get('flip_rate_pct', '?')}%"
    lines.append(f"| Argmax flip rate | {bf16_flip} | {fp32_flip} |")

    # Exp 3
    if exp3.get("status") == "SKIPPED":
        lines.append(f"| Qwen3 MoE determinism | SKIPPED | SKIPPED |")
    else:
        e3_bf16 = exp3.get("bf16_default", {})
        e3_fp32 = exp3.get("fp32_accum", {})
        e3_bf16_det = "YES" if e3_bf16.get("deterministic") else f"NO ({e3_bf16.get('unique_outputs', '?')} unique)"
        e3_fp32_det = "YES" if e3_fp32.get("deterministic") else f"NO ({e3_fp32.get('unique_outputs', '?')} unique)"
        lines.append(f"| Qwen3 MoE determinism (200 runs) | {e3_bf16_det} | {e3_fp32_det} |")

    # Exp 4
    bf16_lat = exp4.get("bf16_default", {}).get("mean_ms", "?")
    fp32_lat = exp4.get("fp32_accum", {}).get("mean_ms", "?")
    overhead_pct = exp4.get("overhead_ratio", {}).get("fp32_accum_vs_bf16_pct", "?")
    lines.append(f"| Latency (mean ms) | {bf16_lat} ms | {fp32_lat} ms ({overhead_pct:+.1f}% overhead) |" if isinstance(overhead_pct, (int, float)) else f"| Latency (mean ms) | {bf16_lat} ms | {fp32_lat} ms |")

    lines.append("")

    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nMarkdown report saved to {md_path}")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════
def main():
    t_global = time.perf_counter()

    print("=" * 80)
    print("  END-TO-END BATCH VARIANCE EXPERIMENTS")
    print(f"  GPU: {torch.cuda.get_device_name(0)} x {torch.cuda.device_count()}")
    print(f"  PyTorch: {torch.__version__}  CUDA: {torch.version.cuda}")
    print("=" * 80)
    sys.stdout.flush()

    # Load Llama model
    model, tok = load_llama()

    # Experiment 1: 1000-run generation determinism
    experiment1(model, tok)
    torch.cuda.empty_cache()

    # Experiment 2: Argmax flip rate
    experiment2(model, tok)
    torch.cuda.empty_cache()

    # Experiment 4: Latency comparison (run before unloading llama)
    experiment4(model, tok)
    torch.cuda.empty_cache()

    # Experiment 5: Long sequence
    experiment5(model, tok)
    torch.cuda.empty_cache()

    # Free Llama before loading Qwen
    del model, tok
    torch.cuda.empty_cache()

    # Experiment 3: Qwen3 MoE
    experiment3()
    torch.cuda.empty_cache()

    # Save results
    json_path = os.path.join(OUT_DIR, "exp_e2e.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON results saved to {json_path}")

    # Write markdown
    write_markdown()

    elapsed_total = time.perf_counter() - t_global
    print(f"\nTotal elapsed time: {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)")
    print("=" * 80)
    print("All experiments complete.")


if __name__ == "__main__":
    main()
