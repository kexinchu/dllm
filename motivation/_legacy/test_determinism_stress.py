#!/usr/bin/env python3
"""
Stress test: find scenarios where BF16 is non-deterministic but FP32 accum is deterministic.

Strategy:
1. Op-level: same input row embedded in different batch sizes → check output row changes
2. Model-level: same prompt with different batch compositions → check logits/tokens change
3. cuBLAS algorithm sensitivity: trigger different GEMM algorithms via shape changes
4. Long sequence: more tokens = more reduction = more chance of divergence
5. Repeated runs with torch.use_deterministic_algorithms(False)
"""
import sys, os, time, hashlib, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F


def header(title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")


# ============================================================================
# Test 1: GEMM batch-variance at op level
# ============================================================================

def test_gemm_batch_variance():
    """
    Same input row in GEMM with different M (batch) dimensions.
    cuBLAS may choose different algorithms/tile configs for different M.
    """
    header("TEST 1: GEMM BATCH VARIANCE (op-level)")
    device = torch.device("cuda")
    torch.manual_seed(42)

    K, N = 4096, 4096
    W = torch.randn(N, K, device=device, dtype=torch.bfloat16)  # weight [out, in]
    x_target = torch.randn(1, K, device=device, dtype=torch.bfloat16)  # our target row

    M_values = [1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 33, 63, 64, 65, 127, 128, 256, 512, 1024]

    print(f"  Weight shape: [{N}, {K}], target row: [1, {K}]")
    print(f"  Testing M (batch) values: {M_values}")
    print()

    # BF16 mode
    ref_bf16 = F.linear(x_target, W)  # [1, N]
    bf16_diffs = {}
    for M in M_values:
        filler = torch.randn(M - 1, K, device=device, dtype=torch.bfloat16)
        batch = torch.cat([x_target, filler], dim=0)  # [M, K]
        out = F.linear(batch, W)
        diff = (out[0:1].float() - ref_bf16.float()).abs().max().item()
        bf16_diffs[M] = diff

    # FP32 accum mode
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    ref_fp32 = F.linear(x_target, W)
    fp32_diffs = {}
    for M in M_values:
        filler = torch.randn(M - 1, K, device=device, dtype=torch.bfloat16)
        batch = torch.cat([x_target, filler], dim=0)
        out = F.linear(batch, W)
        diff = (out[0:1].float() - ref_fp32.float()).abs().max().item()
        fp32_diffs[M] = diff
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True

    print(f"  {'M':>6} | {'BF16 max_diff':>14} | {'FP32 max_diff':>14} | {'BF16 changed':>12}")
    print(f"  {'-'*6}-+-{'-'*14}-+-{'-'*14}-+-{'-'*12}")
    bf16_nonzero = 0
    fp32_nonzero = 0
    for M in M_values:
        bd = bf16_diffs[M]
        fd = fp32_diffs[M]
        changed = "YES" if bd > 0 else ""
        if bd > 0: bf16_nonzero += 1
        if fd > 0: fp32_nonzero += 1
        print(f"  {M:>6} | {bd:>14.4e} | {fd:>14.4e} | {changed:>12}")

    print(f"\n  BF16 non-zero diffs: {bf16_nonzero}/{len(M_values)}")
    print(f"  FP32 non-zero diffs: {fp32_nonzero}/{len(M_values)}")

    return {"bf16": bf16_diffs, "fp32": fp32_diffs,
            "bf16_variant_count": bf16_nonzero, "fp32_variant_count": fp32_nonzero}


# ============================================================================
# Test 2: GEMM with 3D tensors (model-realistic shapes)
# ============================================================================

def test_gemm_3d_batch_variance():
    """
    Simulate transformer linear layers: [batch, seq_len, hidden] @ weight.
    Different batch sizes may trigger different cuBLAS strategies because
    the effective M = batch * seq_len changes.
    """
    header("TEST 2: GEMM 3D BATCH VARIANCE (transformer-realistic)")
    device = torch.device("cuda")
    torch.manual_seed(42)

    hidden = 4096
    seq_len = 64  # longer sequence to increase M
    W = torch.randn(hidden, hidden, device=device, dtype=torch.bfloat16)
    x_target = torch.randn(1, seq_len, hidden, device=device, dtype=torch.bfloat16)

    batch_sizes = [1, 2, 4, 8, 16, 32]
    print(f"  Shape: [B, {seq_len}, {hidden}] @ [{hidden}, {hidden}]")
    print(f"  Effective M = B * seq_len")
    print()

    for mode, flag_val in [("BF16", True), ("FP32 accum", False)]:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = flag_val
        ref = F.linear(x_target.reshape(-1, hidden), W).reshape(1, seq_len, hidden)
        print(f"  {mode}:")
        for bs in batch_sizes:
            filler = torch.randn(bs - 1, seq_len, hidden, device=device, dtype=torch.bfloat16)
            batch = torch.cat([x_target, filler], dim=0)
            out = F.linear(batch.reshape(-1, hidden), W).reshape(bs, seq_len, hidden)
            diff = (out[0:1].float() - ref.float()).abs().max().item()
            eff_M = bs * seq_len
            print(f"    bs={bs:>3} (M={eff_M:>5}): max_diff={diff:.4e}")
        print()

    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True


# ============================================================================
# Test 3: Model-level repeated generation with varying batch
# ============================================================================

def test_model_determinism_stress(model, tokenizer):
    """
    Same prompt, 200 runs with varying batch sizes.
    Compare BF16 vs FP32 accum at token level.
    """
    header("TEST 3: MODEL-LEVEL DETERMINISM (200 runs, varying batch)")
    device = next(model.parameters()).device

    PROMPT = "What is deterministic inference in large language models?"
    fillers = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a haiku about mountains and rivers.",
        "How does photosynthesis work in plants?",
        "What is 2 + 2 and explain why?",
        "Tell me a joke about programming.",
        "Describe the solar system from inner to outer.",
        "What is machine learning used for today?",
        "How do airplanes generate lift to fly?",
        "What is gravity and how does it work?",
        "Explain neural networks to a five year old.",
        "What is the speed of light in a vacuum?",
        "How does DNA replication work exactly?",
        "What is entropy in thermodynamics?",
        "Explain the Turing test and its significance.",
    ]

    def run_gen(prompt, bs, max_new=32):
        if bs == 1:
            inp = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
        else:
            prompts = [prompt] + fillers[:bs-1]
            inp = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
        return out[0].cpu().tolist()

    batch_cycle = [1, 2, 4, 8, 16, 1, 3, 7, 15, 1]
    N = 200

    for mode, flag_val in [("BF16 (default)", True), ("FP32 accum (flag)", False)]:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = flag_val
        hashes = []
        hash_by_bs = {}
        t0 = time.perf_counter()

        for i in range(N):
            bs = batch_cycle[i % len(batch_cycle)]
            tokens = run_gen(PROMPT, bs)
            h = hashlib.sha256(str(tokens).encode()).hexdigest()[:16]
            hashes.append(h)
            hash_by_bs.setdefault(bs, []).append(h)

            if (i+1) % 50 == 0:
                elapsed = time.perf_counter() - t0
                unique = len(set(hashes))
                print(f"    [{i+1}/{N}] {elapsed:.0f}s unique={unique}")

        unique = len(set(hashes))
        det = "YES" if unique == 1 else "NO"
        print(f"  {mode}: {unique} unique / {N} runs  deterministic={det}")

        if unique > 1:
            from collections import Counter
            c = Counter(hashes)
            print(f"    Hash distribution: {c.most_common(5)}")
            print(f"    Per batch-size:")
            for bs in sorted(hash_by_bs):
                bs_unique = len(set(hash_by_bs[bs]))
                print(f"      bs={bs}: {bs_unique} unique / {len(hash_by_bs[bs])} runs")
        print()

    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True


# ============================================================================
# Test 4: Logits-level sensitivity with long sequences
# ============================================================================

def test_long_sequence_sensitivity(model, tokenizer):
    """
    Longer sequences = larger effective M in GEMM = more likely to trigger
    different cuBLAS algorithm. Also more attention reduction steps.
    """
    header("TEST 4: LONG SEQUENCE SENSITIVITY")
    device = next(model.parameters()).device

    # Construct a long prompt
    long_prompt = (
        "Please provide a detailed explanation of the following topics: "
        "quantum computing, artificial intelligence, machine learning, "
        "deep learning, natural language processing, computer vision, "
        "reinforcement learning, generative models, transformer architecture, "
        "attention mechanisms, gradient descent, backpropagation, "
        "convolutional neural networks, recurrent neural networks, "
        "long short-term memory networks, and transfer learning. "
        "For each topic, explain the key concepts, applications, and recent advances."
    )

    fillers = [
        "Summarize the history of computing from Charles Babbage to modern GPUs.",
        "Explain how modern processors achieve parallelism through pipelining and superscalar execution.",
        "Describe the evolution of programming languages from assembly to modern high-level languages.",
    ]

    inp_single = tokenizer(long_prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    seq_len_single = inp_single["input_ids"].shape[1]
    print(f"  Prompt length: {seq_len_single} tokens")

    for mode, flag_val in [("BF16", True), ("FP32 accum", False)]:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = flag_val

        with torch.no_grad():
            ref = model(**inp_single).logits[0]  # [seq_len, vocab]

        print(f"\n  {mode}:")
        for bs in [2, 4]:
            prompts = [long_prompt] + fillers[:bs-1]
            inp = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

            with torch.no_grad():
                logits = model(**inp).logits

            # Align (left-padding)
            nonpad = inp["attention_mask"][0].sum().item()
            pad_len = inp["attention_mask"].shape[1] - nonpad
            batch_0 = logits[0, pad_len:]
            ml = min(ref.shape[0], batch_0.shape[0])
            diff = (ref[:ml].float() - batch_0[:ml].float()).abs()
            md = diff.max().item()
            mean_d = diff.mean().item()
            argmax_ok = (ref[:ml].float().argmax(-1) == batch_0[:ml].float().argmax(-1)).all().item()
            argmax_flips = (ref[:ml].float().argmax(-1) != batch_0[:ml].float().argmax(-1)).sum().item()
            print(f"    bs={bs}: max_diff={md:.4e} mean={mean_d:.4e} argmax_match={'YES' if argmax_ok else 'NO'} flips={argmax_flips}/{ml}")

    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True


# ============================================================================
# Test 5: Direct cuBLAS algorithm sensitivity
# ============================================================================

def test_cublas_algorithm_sensitivity():
    """
    cuBLAS selects algorithms based on M,N,K. Near algorithm-switch boundaries,
    same logical computation may use different code paths.
    Test many M values around powers of 2 (common switch points).
    """
    header("TEST 5: cuBLAS ALGORITHM SWITCH SENSITIVITY")
    device = torch.device("cuda")
    torch.manual_seed(42)

    K, N = 4096, 11008  # Llama-3.1-8B gate_proj shape
    W = torch.randn(N, K, device=device, dtype=torch.bfloat16)
    x_row = torch.randn(1, K, device=device, dtype=torch.bfloat16)

    # Test M values around algorithm switch boundaries
    test_Ms = []
    for base in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        for offset in [-1, 0, 1]:
            m = base + offset
            if m > 0 and m not in test_Ms:
                test_Ms.append(m)
    test_Ms.sort()

    bf16_variant_count = 0
    fp32_variant_count = 0

    for mode, flag_val in [("BF16", True), ("FP32 accum", False)]:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = flag_val
        ref = F.linear(x_row, W)
        variants = 0
        variant_Ms = []

        for M in test_Ms:
            filler = torch.randn(M - 1, K, device=device, dtype=torch.bfloat16)
            batch = torch.cat([x_row, filler], dim=0)
            out = F.linear(batch, W)
            diff = (out[0:1].float() - ref.float()).abs().max().item()
            if diff > 0:
                variants += 1
                variant_Ms.append(M)

        if mode == "BF16":
            bf16_variant_count = variants
        else:
            fp32_variant_count = variants

        print(f"  {mode}: {variants}/{len(test_Ms)} M values caused output change")
        if variant_Ms:
            print(f"    Variant Ms: {variant_Ms[:20]}{'...' if len(variant_Ms)>20 else ''}")

    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True

    return {"bf16_variants": bf16_variant_count, "fp32_variants": fp32_variant_count}


# ============================================================================
# Test 6: Run-to-run variance (same shape, multiple runs)
# ============================================================================

def test_run_to_run_variance():
    """
    Same input, same shape, 100 runs. Check if output ever changes.
    This tests whether cuBLAS is internally non-deterministic.
    """
    header("TEST 6: RUN-TO-RUN VARIANCE (same shape, 100 runs)")
    device = torch.device("cuda")
    torch.manual_seed(42)

    configs = [
        (32, 4096, 4096, "Llama q_proj (M=32)"),
        (32, 4096, 11008, "Llama gate_proj (M=32)"),
        (1, 4096, 4096, "Decode q_proj (M=1)"),
        (128, 4096, 4096, "Long prefill (M=128)"),
    ]

    N_RUNS = 100
    for M, K, N, desc in configs:
        A = torch.randn(M, K, device=device, dtype=torch.bfloat16)
        B = torch.randn(N, K, device=device, dtype=torch.bfloat16)  # [N, K] for F.linear

        for mode, flag_val in [("BF16", True), ("FP32", False)]:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = flag_val
            ref = F.linear(A, B)
            n_diff = 0
            max_diff = 0.0
            for _ in range(N_RUNS):
                out = F.linear(A, B)
                d = (out.float() - ref.float()).abs().max().item()
                if d > 0:
                    n_diff += 1
                    max_diff = max(max_diff, d)

            status = f"{n_diff}/{N_RUNS} differ" if n_diff > 0 else "deterministic"
            print(f"  {desc:<30} {mode:<5}: {status}" + (f" (max={max_diff:.2e})" if n_diff > 0 else ""))

    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 80)
    print("  DETERMINISM STRESS TEST")
    print("=" * 80)
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  allow_bf16_reduced_precision_reduction: {torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}")

    results = {}

    # Op-level tests (no model needed)
    results["gemm_2d"] = test_gemm_batch_variance()
    test_gemm_3d_batch_variance()
    results["cublas_algo"] = test_cublas_algorithm_sensitivity()
    test_run_to_run_variance()

    # Model-level tests
    from transformers import AutoModelForCausalLM, AutoTokenizer
    MODEL_PATH = "/home/kec23008/docker-sys/Models/Llama-3.1-8B-Instruct"
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16, device_map="auto")
    model.eval()

    test_long_sequence_sensitivity(model, tokenizer)
    test_model_determinism_stress(model, tokenizer)

    # Summary
    header("SUMMARY")
    g = results["gemm_2d"]
    c = results["cublas_algo"]
    print(f"  GEMM 2D batch variance:  BF16={g['bf16_variant_count']} variants, FP32={g['fp32_variant_count']} variants")
    print(f"  cuBLAS algo sensitivity: BF16={c['bf16_variants']} variants, FP32={c['fp32_variants']} variants")

    out_path = os.path.join(os.path.dirname(__file__), "determinism_stress_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
