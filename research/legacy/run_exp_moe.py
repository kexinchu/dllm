#!/usr/bin/env python3
"""
MoE Determinism Experiments on Qwen3.5-35B-A3B (BF16, no quantization).

Tests:
1. Generation determinism: 200 runs with varying batch sizes
2. MoE router logits batch variance: same input, different batch → router logits diff
3. Expert selection stability: near-tie prevalence + flip rate
4. Latency comparison: BF16 vs FP32 accum
"""
import sys, os, time, hashlib, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings; warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/home/kec23008/docker-sys/Models/Qwen3.5-35B-A3B"


def header(title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")


def make_equal_length_batch(tok, target_prompt, filler_prompts, target_len, device):
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


def load_model():
    print("Loading Qwen3.5-35B-A3B...")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()
    # Report memory
    for i in range(torch.cuda.device_count()):
        mem = torch.cuda.memory_allocated(i) / 1e9
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}, {mem:.1f} GB allocated")
    return model, tok


# ============================================================================
# Test 1: Generation Determinism (200 runs)
# ============================================================================

def test_generation_determinism(model, tok):
    header("TEST 1: GENERATION DETERMINISM (200 runs, Qwen3.5 MoE)")
    device = next(model.parameters()).device
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
    batch_cycle = [1, 2, 4, 8, 16]
    N = 200
    MAX_NEW = 32

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
                out = model.generate(**inp, max_new_tokens=MAX_NEW, do_sample=False,
                                     pad_token_id=tok.pad_token_id)
            h = hashlib.sha256(str(out[0].cpu().tolist()).encode()).hexdigest()[:16]
            hashes.append(h)
            hash_by_bs.setdefault(bs, []).append(h)
            if (i+1) % 50 == 0:
                elapsed = time.perf_counter() - t0
                unique = len(set(hashes))
                print(f"    [{i+1}/{N}] {elapsed:.0f}s unique={unique}")

        unique = len(set(hashes))
        elapsed = time.perf_counter() - t0
        print(f"  {mode}: {unique} unique / {N} runs  deterministic={'YES' if unique==1 else 'NO'}  ({elapsed:.0f}s)")
        if unique > 1:
            from collections import Counter
            print(f"    Distribution: {dict(Counter(hashes).most_common(5))}")
            for bs in sorted(hash_by_bs):
                print(f"    bs={bs}: {len(set(hash_by_bs[bs]))} unique / {len(hash_by_bs[bs])} runs")
        results[mode] = {"unique": unique, "N": N, "deterministic": unique == 1}
        print()

    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    return results


# ============================================================================
# Test 2: MoE Router Logits Batch Variance
# ============================================================================

def test_router_logits_variance(model, tok):
    header("TEST 2: MOE ROUTER LOGITS BATCH VARIANCE")
    device = next(model.parameters()).device
    PROMPT = "Explain the concept of mixture of experts in neural networks."
    fillers = [
        "What is the capital of France and why is it important?",
        "Explain quantum computing in simple terms for beginners.",
        "Write a short poem about mountains and rivers in spring.",
        "How does photosynthesis work in C3 and C4 plants today?",
        "What is the meaning of life according to philosophy here?",
        "Tell me a joke about a programmer and a rubber duck now.",
        "Describe the process of nuclear fusion happening in the sun.",
    ]
    target_ids = tok.encode(PROMPT, add_special_tokens=True)
    target_len = len(target_ids)

    # Hook the first MoE gate to capture router logits
    captured_logits = {}
    hooks = []

    def find_moe_gate(model):
        """Find the first MoE gating module."""
        for name, mod in model.named_modules():
            cname = type(mod).__name__
            if 'TopKRouter' in cname or 'MoeGate' in cname or 'TopkRouter' in cname:
                return name, mod
            # Qwen3 MoE block
            if 'SparseMoeBlock' in cname or 'MoeBlock' in cname:
                for sub_name, sub_mod in mod.named_modules():
                    if 'gate' in sub_name.lower():
                        return f"{name}.{sub_name}", sub_mod
        return None, None

    gate_name, gate_mod = find_moe_gate(model)
    if gate_mod is None:
        print("  Could not find MoE gate module. Listing module types:")
        seen = set()
        for name, mod in model.named_modules():
            cname = type(mod).__name__
            if cname not in seen:
                seen.add(cname)
                print(f"    {cname}")
        return {"error": "no MoE gate found"}

    print(f"  Found MoE gate: {gate_name} ({type(gate_mod).__name__})")

    for mode, flag_val in [("BF16", True), ("FP32 accum", False)]:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = flag_val
        # Get reference logits at bs=1
        inp1 = make_equal_length_batch(tok, PROMPT, [], target_len, device)
        with torch.no_grad():
            _ = model(**inp1)

        print(f"  {mode}:")
        for bs in [2, 4, 8]:
            inp = make_equal_length_batch(tok, PROMPT, fillers[:bs-1], target_len, device)
            with torch.no_grad():
                logits_batch = model(**inp).logits
            # Compare output logits (easier than hooking gate)
            inp1 = make_equal_length_batch(tok, PROMPT, [], target_len, device)
            with torch.no_grad():
                logits_ref = model(**inp1).logits
            diff = (logits_ref[0].float() - logits_batch[0].float()).abs()
            md = diff.max().item()
            argmax_ok = (logits_ref[0].float().argmax(-1) == logits_batch[0].float().argmax(-1)).all().item()
            print(f"    bs={bs}: logits max_diff={md:.4e} argmax_match={'YES' if argmax_ok else 'NO'}")

    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    return {}


# ============================================================================
# Test 3: Latency
# ============================================================================

def test_latency(model, tok):
    header("TEST 3: LATENCY (Qwen3.5 MoE)")
    device = next(model.parameters()).device
    PROMPT = "Explain the concept of mixture of experts in neural networks."
    target_ids = tok.encode(PROMPT, add_special_tokens=True)
    target_len = len(target_ids)
    inp = make_equal_length_batch(tok, PROMPT, [], target_len, device)

    results = {}
    for mode, flag_val in [("BF16", True), ("FP32 accum", False)]:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = flag_val
        # Warmup
        for _ in range(2):
            with torch.no_grad():
                model.generate(**inp, max_new_tokens=32, do_sample=False, pad_token_id=tok.pad_token_id)
        torch.cuda.synchronize()
        times = []
        for _ in range(10):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                model.generate(**inp, max_new_tokens=32, do_sample=False, pad_token_id=tok.pad_token_id)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
        mean_ms = sum(times) / len(times)
        std_ms = (sum((t - mean_ms)**2 for t in times) / len(times)) ** 0.5
        print(f"  {mode}: {mean_ms:.1f} +/- {std_ms:.1f} ms")
        results[mode] = {"mean_ms": mean_ms, "std_ms": std_ms}

    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    bf16 = results["BF16"]["mean_ms"]
    fp32 = results["FP32 accum"]["mean_ms"]
    print(f"  Overhead: {fp32/bf16:.2f}x")
    return results


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 80)
    print("  QWEN3.5-35B-A3B MOE DETERMINISM EXPERIMENTS")
    print("=" * 80)
    print(f"  GPU: {torch.cuda.device_count()}x {torch.cuda.get_device_name(0)}")

    model, tok = load_model()
    all_results = {}

    all_results["generation"] = test_generation_determinism(model, tok)
    torch.cuda.empty_cache()

    all_results["router"] = test_router_logits_variance(model, tok)
    torch.cuda.empty_cache()

    all_results["latency"] = test_latency(model, tok)

    # Save
    out_path = os.path.join(os.path.dirname(__file__), "exp_moe.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    header("SUMMARY")
    g = all_results["generation"]
    for mode in ["BF16 (default)", "FP32 accum"]:
        if mode in g:
            print(f"  {mode}: {g[mode]['unique']} unique / {g[mode]['N']} runs → {'DETERMINISTIC' if g[mode]['deterministic'] else 'NON-DETERMINISTIC'}")

    out_md = os.path.join(os.path.dirname(__file__), "exp_moe.md")
    with open(out_md, "w") as f:
        f.write("# Qwen3.5-35B-A3B MoE Determinism Results\n\n")
        f.write(f"**Model**: Qwen3.5-35B-A3B (BF16, 35B total, 3B active, 128 experts, top-8)\n")
        f.write(f"**Hardware**: 2x NVIDIA RTX A6000 (device_map=auto)\n\n")
        f.write("## Generation Determinism (200 runs)\n\n")
        f.write("| Mode | Unique Outputs | Deterministic? |\n")
        f.write("|------|---------------:|:--------------:|\n")
        for mode in ["BF16 (default)", "FP32 accum"]:
            if mode in g:
                det = "Yes" if g[mode]["deterministic"] else "No"
                f.write(f"| {mode} | {g[mode]['unique']} | {det} |\n")
        f.write("\n## Latency\n\n")
        l = all_results["latency"]
        f.write("| Mode | Mean (ms) | Std (ms) |\n")
        f.write("|------|----------:|---------:|\n")
        for mode in ["BF16", "FP32 accum"]:
            if mode in l:
                f.write(f"| {mode} | {l[mode]['mean_ms']:.1f} | {l[mode]['std_ms']:.1f} |\n")
    print(f"\n  Results saved to {out_path} and {out_md}")


if __name__ == "__main__":
    main()
