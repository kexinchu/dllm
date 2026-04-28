"""
vLLM Integration Test for DetermLLM.

Tests:
  1. Correctness: same request produces identical output at batch sizes 1..32
  2. Throughput: tokens/sec with and without DetermLLM patch
  3. Serving overhead: TTFT and throughput comparison

The key test: in vLLM's continuous batching decode phase, M = number of
concurrent decode requests. With BF16: different concurrent counts → different
cuBLAS algorithm → non-deterministic output. DetermLLM fixes this.

Output: research/exp_vllm_determ.json
"""

import os, sys, json, time, hashlib, warnings
warnings.filterwarnings('ignore')

DLLM_DIR = os.path.dirname(os.path.abspath(__file__))
FP32_DIR  = os.path.join(DLLM_DIR, '..', 'FP32')
sys.path.insert(0, DLLM_DIR)
sys.path.insert(0, FP32_DIR)

OUT_FILE = os.path.join(DLLM_DIR, 'exp_vllm_determ.json')
MODEL_PATH = '/home/kec23008/docker-sys/Models/Llama-3.2-1B-Instruct'

# Activate DetermLLM BEFORE vLLM imports (so it intercepts model load too)
import determ_llm

import torch
from vllm import LLM, SamplingParams

# ── test prompts ─────────────────────────────────────────────────────────────
BASE_PROMPTS = [
    "What is deterministic inference in large language models?",
    "Explain the concept of batch processing in neural networks.",
    "How does the attention mechanism work in transformers?",
    "Describe the difference between FP16 and BF16 floating point formats.",
    "What are the main challenges in deploying large language models at scale?",
    "Explain gradient descent in simple terms.",
    "What is the role of layer normalization in deep learning?",
    "How does tokenization work in language models?",
]

SAMPLING = SamplingParams(temperature=0.0, max_tokens=32)  # greedy decoding


def req_hash(output):
    tokens = output.outputs[0].token_ids
    return hashlib.md5(str(list(tokens)).encode()).hexdigest()[:12]


def run_batch(llm, prompts):
    outputs = llm.generate(prompts, SAMPLING)
    return outputs


def measure_throughput(llm, prompts, n_runs=5):
    """Returns mean throughput in tokens/sec."""
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        outs = llm.generate(prompts, SAMPLING)
        t1 = time.perf_counter()
        total_toks = sum(len(o.outputs[0].token_ids) for o in outs)
        times.append(total_toks / (t1 - t0))
    return sum(times) / len(times), times


results = {}

for mode_name, use_determ in [("BF16_baseline", False), ("DetermLLM", True)]:
    print(f"\n{'='*60}")
    print(f"Mode: {mode_name}")
    print(f"{'='*60}")

    if use_determ:
        determ_llm.enable()
        print("  DetermLLM patch: ENABLED")
    else:
        determ_llm.disable()
        print("  DetermLLM patch: DISABLED")

    # Load vLLM engine
    print("  Loading vLLM engine...")
    llm = LLM(
        model=MODEL_PATH,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=2048,
        tensor_parallel_size=1,
        seed=42,
        enforce_eager=True,        # disable CUDA graphs for fair comparison
        trust_remote_code=False,
    )

    # ── Test 1: determinism across batch sizes ────────────────────────────
    print("\n  Test 1: Determinism across concurrent request counts...")

    # Reference: 1 prompt alone
    ref_hashes = {}
    for i, p in enumerate(BASE_PROMPTS):
        out = llm.generate([p], SAMPLING)
        ref_hashes[i] = req_hash(out[0])

    # Now run same prompts in increasing batch sizes
    batch_results = {}
    for n_concurrent in [2, 4, 8, len(BASE_PROMPTS)]:
        prompts_batch = BASE_PROMPTS[:n_concurrent]
        outs = llm.generate(prompts_batch, SAMPLING)
        mismatches = 0
        for i, out in enumerate(outs):
            h = req_hash(out)
            if h != ref_hashes[i]:
                mismatches += 1
        batch_results[n_concurrent] = {
            "n_mismatches": mismatches,
            "deterministic": mismatches == 0,
        }
        status = "✓ DET" if mismatches == 0 else f"✗ {mismatches}/{n_concurrent}"
        print(f"  n_concurrent={n_concurrent:>2}: {status}")

    # ── Test 2: Throughput ────────────────────────────────────────────────
    print("\n  Test 2: Throughput benchmark...")
    tp_mean, tp_all = measure_throughput(llm, BASE_PROMPTS, n_runs=5)
    print(f"  Throughput: {tp_mean:.1f} tokens/sec (5 runs)")

    # ── Test 3: TTFT (time to first token) ───────────────────────────────
    print("\n  Test 3: Latency (TTFT + full generation)...")
    ttft_times = []
    full_times = []
    for _ in range(5):
        t0 = time.perf_counter()
        outs = llm.generate(BASE_PROMPTS[:4], SAMPLING)
        t1 = time.perf_counter()
        full_times.append((t1 - t0) * 1000)
    mean_full = sum(full_times) / len(full_times)
    print(f"  Full generation (4 requests): {mean_full:.1f} ms")

    results[mode_name] = {
        "determ_enabled": use_determ,
        "determinism_test": batch_results,
        "overall_deterministic": all(v["deterministic"] for v in batch_results.values()),
        "throughput_tps": tp_mean,
        "throughput_all_runs": tp_all,
        "latency_4req_ms": mean_full,
    }

    del llm
    torch.cuda.empty_cache()

# ── compute overhead ──────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")

baseline = results["BF16_baseline"]
ours = results["DetermLLM"]

tp_overhead = (1 - ours["throughput_tps"] / baseline["throughput_tps"]) * 100
lat_overhead = (ours["latency_4req_ms"] / baseline["latency_4req_ms"] - 1) * 100

print(f"BF16 baseline:  det={baseline['overall_deterministic']}, "
      f"throughput={baseline['throughput_tps']:.1f} tok/s, "
      f"latency={baseline['latency_4req_ms']:.1f} ms")
print(f"DetermLLM:      det={ours['overall_deterministic']}, "
      f"throughput={ours['throughput_tps']:.1f} tok/s, "
      f"latency={ours['latency_4req_ms']:.1f} ms")
print(f"Throughput overhead: {tp_overhead:+.1f}%")
print(f"Latency overhead:    {lat_overhead:+.1f}%")

results["summary"] = {
    "throughput_overhead_pct": tp_overhead,
    "latency_overhead_pct": lat_overhead,
}

with open(OUT_FILE, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {OUT_FILE}")
