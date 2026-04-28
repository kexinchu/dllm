"""
KV-Cache Layout Determinism Test — SGLang
==========================================
All tests batch_size=1, orthogonal to batch invariance.
Tests whether prefix caching (radix cache) and chunked prefill
affect the first generated token.

Scenario A: Prefix Cache Hit vs Fresh Prefill
  - Engine 1 (radix cache disabled):  send full prompt → first token
  - Engine 2 (radix cache enabled):   warm cache with prefix, then send
                                       full prompt → prefix hits cache → first token
  - Compare first tokens

Scenario B: Chunked Prefill
  - Engine 1: baseline (default page_size)
  - Engine 2: different page_size values
  - Compare first tokens
"""

import os
import gc
import json
import time
import itertools
import torch
from collections import defaultdict

MODEL_PATH = "/home/kec23008/docker-sys/Models/Llama-3.1-8B-Instruct"

# Reuse same prompt sets from vllm test
PREFIXES = [
    "You are a helpful AI assistant.\n\nQuestion: ",
    "You are a coding assistant.\n\nTask: ",
    "Answer the following question concisely.\n\nQ: ",
    "You are an expert financial analyst with deep knowledge of global markets and investment strategies. Provide detailed analysis.\n\nQuestion: ",
    "You are a medical knowledge assistant. Provide accurate health information based on current evidence.\n\nQuestion: ",
    "Context: The transformer architecture was introduced in the paper 'Attention is All You Need' by Vaswani et al. in 2017. It replaced recurrent neural networks with self-attention mechanisms, enabling much more efficient parallelization during training. The key innovation was the multi-head attention mechanism, which allows the model to jointly attend to information from different representation subspaces. The architecture consists of an encoder and decoder, each composed of layers of multi-head attention and feed-forward networks. Layer normalization and residual connections are used throughout.\n\nBased on the context above, answer: ",
    "Context: Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data. Supervised learning uses labeled data to train models, while unsupervised learning finds patterns in unlabeled data. Reinforcement learning trains agents through reward signals. Deep learning uses neural networks with many layers to learn hierarchical representations.\n\nQuestion: ",
    "You are a systems architect. Analyze the following.\n\nQuestion: ",
]

SUFFIXES = [
    "What is the capital of France?",
    "Explain quantum computing briefly.",
    "What are the differences between Python and Java?",
    "How does photosynthesis work?",
    "What is the theory of relativity?",
    "How do neural networks learn?",
    "What causes earthquakes?",
    "How does TCP/IP work?",
    "What is the Pythagorean theorem?",
    "How do vaccines work?",
    "What is blockchain technology?",
    "How does encryption work?",
    "What is natural selection?",
    "How do black holes form?",
    "What is supply and demand?",
    "How does a compiler work?",
    "What is the water cycle?",
    "How do batteries work?",
    "What is DNA replication?",
    "How does GPS navigation work?",
    "What is machine learning?",
    "How do search engines work?",
    "What is the greenhouse effect?",
    "How does the immune system work?",
    "What is object oriented programming?",
    "How do operating systems manage memory?",
    "What is the Doppler effect?",
    "How does a transistor work?",
    "What is the Turing test?",
    "How do databases handle concurrency?",
]

LONG_CONTEXTS = [
    "The transformer architecture revolutionized natural language processing when it was introduced in 2017. "
    "Unlike recurrent neural networks, transformers process all tokens in parallel using self-attention mechanisms. "
    "This parallelism enables much faster training on modern GPU hardware. The key components include multi-head "
    "attention, feed-forward networks, layer normalization, and residual connections. Positional encodings provide "
    "sequence order information. Since then, models like BERT, GPT, T5, and LLaMA have built upon this foundation, "
    "scaling to billions of parameters. Recent advances include mixture-of-experts architectures, flash attention "
    "for efficient memory usage, and various quantization techniques for deployment.",

    "Machine learning systems face numerous challenges in production deployment. Model serving requires careful "
    "management of computational resources, memory allocation, and request batching. Continuous batching allows "
    "dynamic insertion and removal of requests during inference, dramatically improving throughput. PagedAttention "
    "manages KV cache memory using virtual memory concepts, reducing fragmentation. Speculative decoding uses "
    "draft models to propose tokens verified in parallel. Quantization techniques like GPTQ, AWQ, and FP8 reduce "
    "memory footprint at the cost of some accuracy.",

    "The human immune system is a complex network of cells, tissues, and organs that work together to defend "
    "the body against harmful pathogens. The innate immune system provides immediate, non-specific defense through "
    "physical barriers like skin, chemical barriers like stomach acid, and cellular defenses like phagocytes. "
    "The adaptive immune system develops targeted responses through T cells and B cells, creating immunological "
    "memory for faster future responses. Vaccines work by training the adaptive immune system to recognize specific "
    "antigens without causing disease.",

    "Distributed systems are computer systems whose components are located on different networked computers that "
    "communicate and coordinate their actions by passing messages. Key challenges include handling partial failures, "
    "network partitions, and maintaining consistency across replicas. The CAP theorem states that a distributed "
    "system can only guarantee two of three properties: consistency, availability, and partition tolerance. "
    "Consensus algorithms like Raft and Paxos enable agreement among distributed nodes.",

    "Climate change refers to long-term shifts in global temperatures and weather patterns. Human activities, "
    "primarily burning fossil fuels, have been the main driver since the 1800s, releasing greenhouse gases like "
    "carbon dioxide and methane. The global average temperature has risen by about 1.1 degrees Celsius since "
    "pre-industrial times. Effects include rising sea levels, more frequent extreme weather events, ocean "
    "acidification, and biodiversity loss.",
]

LONG_QUESTIONS = [
    "What are the key challenges mentioned?",
    "Summarize the main points in three sentences.",
    "What is the most important concept discussed?",
    "How do the different components interact?",
    "What are the practical implications?",
    "What historical context is relevant?",
    "What are the trade-offs involved?",
    "How might this evolve in the future?",
    "What are the limitations of current approaches?",
    "Explain the cause and effect relationships.",
    "What is the central argument?",
    "How does this compare to alternative approaches?",
    "What evidence supports the claims made?",
    "What are the most important takeaways?",
    "How would you explain this to a beginner?",
    "What additional context would be helpful?",
    "What assumptions are being made?",
    "How does this relate to real world applications?",
    "What are the potential risks?",
    "What questions remain unanswered?",
]


def build_prompt_pairs(num_runs):
    combos = list(itertools.product(PREFIXES, SUFFIXES))
    pairs = []
    for i in range(num_runs):
        prefix, suffix = combos[i % len(combos)]
        pairs.append({"prefix": prefix, "suffix": suffix, "full": prefix + suffix})
    return pairs


def build_long_prompts(num_runs):
    combos = list(itertools.product(LONG_CONTEXTS, LONG_QUESTIONS))
    prompts = []
    for i in range(num_runs):
        ctx, q = combos[i % len(combos)]
        variant = f" (variant {i // len(combos)})" if i >= len(combos) else ""
        prompts.append(f"Context: {ctx}{variant}\n\nQuestion: {q}\n\nAnswer:")
    return prompts


def cleanup_engine():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Scenario A: Prefix Cache Hit vs Fresh Prefill
# ---------------------------------------------------------------------------

def run_scenario_a_sglang(num_runs=1000):
    import sglang as sgl

    print("\n" + "=" * 70)
    print("SGLang Scenario A: Prefix Cache Hit vs Fresh Prefill")
    print("=" * 70)

    pairs = build_prompt_pairs(num_runs)
    full_prompts = [p["full"] for p in pairs]
    sp = {"temperature": 0.0, "max_new_tokens": 1}

    # --- Phase 1: Radix cache disabled (no prefix caching) ---
    print("  Phase 1: Radix cache DISABLED...")
    engine_nocache = sgl.Engine(
        model_path=MODEL_PATH,
        dtype="bfloat16",
        disable_radix_cache=True,
        mem_fraction_static=0.5,
        attention_backend="triton",
        disable_cuda_graph=True,
    )

    outputs_nocache = engine_nocache.generate(
        prompt=full_prompts,
        sampling_params=sp,
    )

    tokens_nocache = []
    for out in outputs_nocache:
        tokens_nocache.append({
            "text": out["text"],
        })

    engine_nocache.shutdown()
    cleanup_engine()

    # --- Phase 2: Radix cache enabled ---
    print("  Phase 2: Radix cache ENABLED...")
    engine_cache = sgl.Engine(
        model_path=MODEL_PATH,
        dtype="bfloat16",
        disable_radix_cache=False,
        mem_fraction_static=0.5,
        attention_backend="triton",
        disable_cuda_graph=True,
    )

    # Warm cache with unique prefixes
    unique_prefixes = list(set(p["prefix"] for p in pairs))
    print(f"    Warming cache with {len(unique_prefixes)} unique prefixes...")
    warmup_prompts = [p + "Hello" for p in unique_prefixes]
    _ = engine_cache.generate(prompt=warmup_prompts, sampling_params=sp)

    # Now send real prompts — prefix should hit radix cache
    print(f"    Generating {num_runs} prompts (prefix cache should hit)...")
    outputs_cache = engine_cache.generate(
        prompt=full_prompts,
        sampling_params=sp,
    )

    tokens_cache = []
    for out in outputs_cache:
        tokens_cache.append({
            "text": out["text"],
        })

    engine_cache.shutdown()
    cleanup_engine()

    # --- Compare ---
    flips = 0
    flip_examples = []
    for i in range(num_runs):
        if tokens_nocache[i]["text"] != tokens_cache[i]["text"]:
            flips += 1
            flip_examples.append({
                "idx": i,
                "suffix": pairs[i]["suffix"][:50],
                "prefix_len": len(pairs[i]["prefix"]),
                "token_nocache": tokens_nocache[i]["text"],
                "token_cache": tokens_cache[i]["text"],
            })

    print(f"\n  RESULT: first_token_flip = {flips}/{num_runs} ({flips/num_runs*100:.2f}%)")
    if flip_examples:
        print(f"  Flip examples ({len(flip_examples)} total):")
        for ex in flip_examples[:15]:
            print(f"    \"{ex['suffix']}\" | no_cache → \"{ex['token_nocache']}\"  "
                  f"cached → \"{ex['token_cache']}\"")

    return {
        "total": num_runs,
        "first_token_flip": flips,
        "flip_pct": f"{flips/num_runs*100:.2f}%",
        "flip_examples": flip_examples[:20],
    }


# ---------------------------------------------------------------------------
# Scenario B: Different page_size (chunked KV layout)
# ---------------------------------------------------------------------------

def run_scenario_b_sglang(num_runs=1000):
    import sglang as sgl

    print("\n" + "=" * 70)
    print("SGLang Scenario B: Chunked Prefill — Different Configurations")
    print("=" * 70)

    prompts = build_long_prompts(num_runs)
    sp = {"temperature": 0.0, "max_new_tokens": 1}

    # --- Baseline: default page_size, no chunked prefix cache ---
    print("  Baseline: default config, radix cache disabled...")
    engine_base = sgl.Engine(
        model_path=MODEL_PATH,
        dtype="bfloat16",
        disable_radix_cache=True,
        mem_fraction_static=0.5,
        attention_backend="triton",
        disable_cuda_graph=True,
    )
    outputs_base = engine_base.generate(prompt=prompts, sampling_params=sp)
    tokens_base = [out["text"] for out in outputs_base]
    engine_base.shutdown()
    cleanup_engine()

    # --- Variants: different chunked_prefill_size ---
    chunk_configs = [128, 64, 32]
    results = {}

    for chunk_size in chunk_configs:
        print(f"  Chunked prefill size={chunk_size}...")
        engine_chunk = sgl.Engine(
            model_path=MODEL_PATH,
            dtype="bfloat16",
            disable_radix_cache=True,
            chunked_prefill_size=chunk_size,
            mem_fraction_static=0.5,
        )
        outputs_chunk = engine_chunk.generate(prompt=prompts, sampling_params=sp)

        flips = 0
        flip_examples = []
        for i, out in enumerate(outputs_chunk):
            tok = out["text"]
            if tok != tokens_base[i]:
                flips += 1
                flip_examples.append({
                    "idx": i,
                    "prompt_tail": prompts[i][-60:],
                    "token_base": tokens_base[i],
                    "token_chunked": tok,
                })

        print(f"    first_token_flip = {flips}/{num_runs} ({flips/num_runs*100:.2f}%)")
        if flip_examples:
            for ex in flip_examples[:5]:
                print(f"      base → \"{ex['token_base']}\"  chunked → \"{ex['token_chunked']}\"")

        results[f"chunk_{chunk_size}"] = {
            "total": num_runs,
            "first_token_flip": flips,
            "flip_pct": f"{flips/num_runs*100:.2f}%",
            "flip_examples": flip_examples[:10],
        }
        engine_chunk.shutdown()
        cleanup_engine()

    print(f"\n  SUMMARY:")
    for key, r in results.items():
        print(f"    {key}: first_token_flip = {r['first_token_flip']}/{r['total']} ({r['flip_pct']})")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-runs", type=int, default=1000)
    parser.add_argument("--scenarios", type=str, default="a,b")
    parser.add_argument("--output", type=str, default="sglang_kvcache_results.json")
    args = parser.parse_args()

    scenarios = args.scenarios.lower().split(",")
    all_results = {"engine": "sglang", "num_runs": args.num_runs}

    t0 = time.time()

    if "a" in scenarios:
        all_results["scenario_a_prefix_cache"] = run_scenario_a_sglang(args.num_runs)

    if "b" in scenarios:
        all_results["scenario_b_chunked_prefill"] = run_scenario_b_sglang(args.num_runs)

    elapsed = time.time() - t0
    all_results["elapsed_seconds"] = round(elapsed, 1)

    out_path = os.path.join(os.path.dirname(__file__), args.output)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
