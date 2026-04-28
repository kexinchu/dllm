#!/usr/bin/env python3
"""
Continuous-batching simulation for determinism testing.

In serving engines (vLLM/SGLang), each sequence has its own position_ids
and KV-cache — no padding, no RoPE position shift. The non-determinism
comes purely from kernel-level batch variance (split-K, split-KV).

This test simulates continuous batching by:
1. Running each sequence independently (bs=1) to get reference logits
2. Running a "fused batch" where all sequences share one forward pass
   but with per-sequence position_ids (no padding, concatenated along seq dim)
3. Comparing logits of the target sequence between the two modes

Since HuggingFace doesn't natively support continuous batching, we simulate it by:
- Ensuring ALL sequences in the batch have the SAME length (no padding needed)
- Using identical position_ids [0..L-1] for each sequence
- This way, the only variable is the batch dimension M affecting kernel selection
"""
import sys, os, time, hashlib, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings; warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/home/kec23008/docker-sys/Models/Llama-3.1-8B-Instruct"


def header(title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")


def load_model():
    print("Loading model...")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16, device_map="auto")
    model.eval()
    return model, tok


def make_equal_length_batch(tok, target_prompt, filler_prompts, target_len, device):
    """
    Tokenize all prompts and truncate/pad to exactly target_len tokens.
    This simulates continuous batching where all sequences happen to be same length.
    No left-padding, no position shift — pure batch-size variation.
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
    # All positions are real (no padding mask difference)
    attention_mask = torch.ones_like(input_ids)
    # Explicit position_ids: all sequences use [0, 1, ..., L-1]
    position_ids = torch.arange(target_len, device=device).unsqueeze(0).expand(len(all_prompts), -1)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }


# ============================================================================
# Test 1: Logits batch invariance (continuous batching simulation)
# ============================================================================

def test_logits_continuous_batching(model, tok):
    header("TEST 1: LOGITS BATCH INVARIANCE (continuous batching sim)")
    device = next(model.parameters()).device

    target_prompt = "What is deterministic inference in large language models?"
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

    # Determine target length from tokenized prompt
    target_ids = tok.encode(target_prompt, add_special_tokens=True)
    target_len = len(target_ids)
    print(f"  Target prompt: \"{target_prompt}\"")
    print(f"  Target length: {target_len} tokens")
    print(f"  All sequences truncated/padded to {target_len} tokens (no left-padding)")
    print(f"  All sequences use position_ids [0..{target_len-1}] (no RoPE shift)")
    print()

    batch_sizes = [1, 2, 3, 4, 5, 7, 8, 9, 15, 16]

    for mode, flag_val in [("BF16 (default)", True), ("FP32 accum (cuBLAS flag)", False)]:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = flag_val
        print(f"  --- {mode} ---")

        # Reference: bs=1
        ref_inp = make_equal_length_batch(tok, target_prompt, [], target_len, device)
        with torch.no_grad():
            ref_logits = model(**ref_inp).logits[0]  # [target_len, vocab]

        for bs in batch_sizes:
            if bs == 1:
                inp = ref_inp
            else:
                inp = make_equal_length_batch(tok, target_prompt, fillers[:bs-1], target_len, device)

            with torch.no_grad():
                logits = model(**inp).logits[0]  # first sequence = target

            diff = (ref_logits.float() - logits.float()).abs()
            md = diff.max().item()
            mean_d = diff.mean().item()
            argmax_match = (ref_logits.float().argmax(-1) == logits.float().argmax(-1)).all().item()
            flips = (ref_logits.float().argmax(-1) != logits.float().argmax(-1)).sum().item()

            status = "MATCH" if argmax_match else f"FLIP({flips})"
            print(f"    bs={bs:>2}: max_diff={md:.4e} mean={mean_d:.4e} argmax={status}")

        print()

    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True


# ============================================================================
# Test 2: Token-level determinism (generation, continuous batching sim)
# ============================================================================

def test_generation_determinism(model, tok):
    header("TEST 2: GENERATION DETERMINISM (500 runs, continuous batching sim)")
    device = next(model.parameters()).device

    target_prompt = "What is deterministic inference in large language models?"
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

    target_ids = tok.encode(target_prompt, add_special_tokens=True)
    target_len = len(target_ids)
    batch_cycle = [1, 2, 3, 4, 5, 7, 8, 9, 15, 16]
    N = 500
    MAX_NEW = 32

    for mode, flag_val in [("BF16 (default)", True), ("FP32 accum (cuBLAS flag)", False)]:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = flag_val
        hashes = []
        hash_by_bs = {}
        t0 = time.perf_counter()

        for i in range(N):
            bs = batch_cycle[i % len(batch_cycle)]
            inp = make_equal_length_batch(tok, target_prompt, fillers[:max(bs-1, 0)], target_len, device)
            with torch.no_grad():
                out = model.generate(
                    **inp, max_new_tokens=MAX_NEW, do_sample=False,
                    pad_token_id=tok.pad_token_id,
                )
            tokens = out[0].cpu().tolist()
            h = hashlib.sha256(str(tokens).encode()).hexdigest()[:16]
            hashes.append(h)
            hash_by_bs.setdefault(bs, []).append(h)

            if (i + 1) % 100 == 0:
                elapsed = time.perf_counter() - t0
                unique = len(set(hashes))
                print(f"    [{i+1}/{N}] {elapsed:.0f}s unique={unique}")

        unique = len(set(hashes))
        det = "YES" if unique == 1 else "NO"
        elapsed = time.perf_counter() - t0
        print(f"  {mode}: {unique} unique / {N} runs  deterministic={det}  ({elapsed:.0f}s)")

        if unique > 1:
            from collections import Counter
            c = Counter(hashes)
            print(f"    Hash distribution: {dict(c.most_common(5))}")
            print(f"    Per batch-size uniqueness:")
            for bs in sorted(hash_by_bs):
                bs_unique = len(set(hash_by_bs[bs]))
                print(f"      bs={bs:>2}: {bs_unique} unique / {len(hash_by_bs[bs])} runs")
        else:
            # Show the deterministic output
            sample = tok.decode(out[0][target_len:], skip_special_tokens=True)
            print(f"    Output: \"{sample[:80]}...\"")

        print()

    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True


# ============================================================================
# Test 3: Large-scale argmax flip test (no padding, many prompts)
# ============================================================================

def test_argmax_flips_no_padding(model, tok):
    header("TEST 3: ARGMAX FLIP RATE (no padding, ~5000 tokens)")
    device = next(model.parameters()).device

    prompts = [
        "Please provide a detailed explanation of the following topics: "
        "quantum computing, artificial intelligence, machine learning, "
        "deep learning, natural language processing, computer vision, "
        "reinforcement learning, generative models, transformer architecture, "
        "attention mechanisms, gradient descent, backpropagation.",
        "Write a comprehensive essay about the history of mathematics from "
        "ancient civilizations through the modern era, covering key figures "
        "such as Euclid, Archimedes, Newton, Leibniz, Euler, Gauss, Riemann.",
        "Explain in detail how modern operating systems manage memory, "
        "processes, file systems, and I/O devices. Cover virtual memory, "
        "page tables, context switching, and scheduling algorithms.",
        "Describe the complete lifecycle of a star from nebula formation "
        "through main sequence, red giant, and eventual death as white dwarf "
        "supernova or black hole depending on initial mass.",
        "Explain the principles of thermodynamics including the zeroth first "
        "second and third laws with examples from everyday life and engineering.",
        "Discuss the major breakthroughs in physics during the twentieth century "
        "including special and general relativity, quantum mechanics, and the "
        "standard model of particle physics.",
        "Describe how the internet works from physical cables to application "
        "layer protocols including TCP IP DNS HTTP TLS and routing algorithms.",
        "Explain the central dogma of molecular biology covering DNA replication "
        "transcription translation protein folding and gene regulation mechanisms.",
    ]

    fillers = [
        "What is the capital of France and why is it important for Europe?",
        "Explain quantum computing in simple terms for complete beginners.",
        "Write a short poem about mountains and rivers in the spring time.",
        "How does photosynthesis work in C3 and C4 plants in great detail?",
        "What is the meaning of life according to different world philosophies?",
        "Tell me a long joke about a programmer and a rubber duck debugging.",
        "Describe the process of nuclear fusion happening inside our own sun.",
        "What is machine learning and how does it differ from traditional AI?",
        "How do modern CPUs achieve instruction level parallelism effectively?",
        "What is the relationship between entropy and information theory now?",
        "Explain the double slit experiment and wave particle duality clearly.",
        "Describe the architecture of a modern transformer neural network here.",
        "How does CRISPR gene editing technology work step by step in detail?",
        "What are the fundamental forces of nature and how do they interact?",
        "Explain public key cryptography and the RSA algorithm in full detail.",
    ]

    batch_sizes = [2, 4, 8, 16]

    for mode, flag_val in [("BF16", True), ("FP32 accum", False)]:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = flag_val
        total_tokens = 0
        total_flips = 0

        for p_idx, prompt in enumerate(prompts):
            target_ids = tok.encode(prompt, add_special_tokens=True)
            target_len = len(target_ids)

            # Reference: bs=1
            ref_inp = make_equal_length_batch(tok, prompt, [], target_len, device)
            with torch.no_grad():
                ref_logits = model(**ref_inp).logits[0]
            ref_argmax = ref_logits.float().argmax(dim=-1)

            for bs in batch_sizes:
                inp = make_equal_length_batch(tok, prompt, fillers[:bs-1], target_len, device)
                with torch.no_grad():
                    logits = model(**inp).logits[0]
                batch_argmax = logits.float().argmax(dim=-1)

                flips = (ref_argmax != batch_argmax).sum().item()
                total_tokens += target_len
                total_flips += flips

            print(f"    prompt {p_idx} ({target_len} tok): cumulative {total_flips}/{total_tokens} ({total_flips/max(total_tokens,1)*100:.3f}%)")

        rate = total_flips / max(total_tokens, 1) * 100
        print(f"  {mode} TOTAL: {total_flips} flips / {total_tokens} tokens = {rate:.4f}%\n")

    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 80)
    print("  CONTINUOUS BATCHING DETERMINISM TEST")
    print("  (no padding, no RoPE shift, pure kernel batch-variance)")
    print("=" * 80)
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

    model, tok = load_model()

    test_logits_continuous_batching(model, tok)
    torch.cuda.empty_cache()

    test_argmax_flips_no_padding(model, tok)
    torch.cuda.empty_cache()

    test_generation_determinism(model, tok)

    header("DONE")


if __name__ == "__main__":
    main()
