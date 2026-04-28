"""
KV-Cache Layout Determinism Test
================================
All tests use batch_size=1, orthogonal to batch invariance.
Only variable: how KV-cache is physically distributed / laid out.

Scenarios:
  1. Contiguous vs Paged attention (operator level)
  2. Prefix cache hit vs fresh prefill (model level, Llama-3.1-8B)
  3. Chunked prefill with different chunk sizes (model level)

Each scenario runs 1000 iterations across multiple diverse queries.
"""

import torch
import torch.nn.functional as F
import math
import json
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict

# ---------------------------------------------------------------------------
# Scenario 1: Operator-Level — Contiguous vs Paged Attention
# ---------------------------------------------------------------------------

def paged_attention_online_softmax(
    q: torch.Tensor,
    kv_pages: list[torch.Tensor],  # list of (k_page, v_page) tuples
    scale: float,
) -> torch.Tensor:
    """
    Online-softmax paged attention: accumulate partial results page by page.
    q: [num_heads, q_len, head_dim]
    Each kv_pages element: tuple of (k, v) each [num_heads, page_tokens, head_dim]
    """
    num_heads, q_len, head_dim = q.shape
    m = torch.full((num_heads, q_len), float('-inf'), dtype=torch.float32, device=q.device)
    l = torch.zeros(num_heads, q_len, dtype=torch.float32, device=q.device)
    acc = torch.zeros(num_heads, q_len, head_dim, dtype=torch.float32, device=q.device)

    for k_page, v_page in kv_pages:
        scores = torch.bmm(q.float(), k_page.float().transpose(1, 2)) * scale
        page_max = scores.max(dim=-1).values
        new_m = torch.maximum(m, page_max)
        old_scale = torch.exp(m - new_m)
        page_exp = torch.exp(scores - new_m.unsqueeze(-1))
        page_sum = page_exp.sum(dim=-1)
        acc = acc * old_scale.unsqueeze(-1) + torch.bmm(page_exp, v_page.float())
        l = l * old_scale + page_sum
        m = new_m

    out = acc / l.unsqueeze(-1)
    return out.to(q.dtype)


def contiguous_attention(q, k, v, scale):
    """Standard attention on contiguous KV."""
    scores = torch.bmm(q.float(), k.float().transpose(1, 2)) * scale
    attn = torch.softmax(scores, dim=-1)
    out = torch.bmm(attn, v.float())
    return out.to(q.dtype)


def run_scenario1(num_runs=1000, device="cuda"):
    """
    Scenario 1: Same KV content, different physical page layouts.
    - A: contiguous attention
    - B: paged attention, pages in order (page_size=16)
    - C: paged attention, pages in order (page_size=64)
    - D: paged attention, pages in order (page_size=7, odd size => different remainder pattern)
    """
    print("\n" + "="*70)
    print("Scenario 1: Contiguous vs Paged Attention (Operator Level)")
    print("="*70)

    num_heads, head_dim = 32, 128
    scale = 1.0 / math.sqrt(head_dim)
    seq_lens = [128, 256, 512, 1024]
    page_sizes = [16, 64, 7]

    results = {}
    for seq_len in seq_lens:
        diffs = defaultdict(list)
        for run in range(num_runs):
            torch.manual_seed(run * 1000 + seq_len)
            q = torch.randn(num_heads, 1, head_dim, dtype=torch.bfloat16, device=device)
            k = torch.randn(num_heads, seq_len, head_dim, dtype=torch.bfloat16, device=device)
            v = torch.randn(num_heads, seq_len, head_dim, dtype=torch.bfloat16, device=device)

            # A: contiguous
            out_contig = contiguous_attention(q, k, v, scale)

            for ps in page_sizes:
                # Split KV into pages
                kv_pages = []
                for start in range(0, seq_len, ps):
                    end = min(start + ps, seq_len)
                    kv_pages.append((k[:, start:end, :], v[:, start:end, :]))

                out_paged = paged_attention_online_softmax(q, kv_pages, scale)
                diff = (out_contig.float() - out_paged.float()).abs().max().item()
                diffs[f"page_size={ps}"].append(diff)

        results[seq_len] = {}
        for key, diff_list in diffs.items():
            nonzero = sum(1 for d in diff_list if d > 0)
            max_d = max(diff_list)
            mean_d = sum(diff_list) / len(diff_list)
            results[seq_len][key] = {
                "nonzero_count": nonzero,
                "nonzero_pct": f"{nonzero/num_runs*100:.1f}%",
                "max_diff": f"{max_d:.6e}",
                "mean_diff": f"{mean_d:.6e}",
            }
            print(f"  seq_len={seq_len}, {key}: "
                  f"{nonzero}/{num_runs} differ ({nonzero/num_runs*100:.1f}%), "
                  f"max_diff={max_d:.6e}")

    return results


# ---------------------------------------------------------------------------
# Scenario 2: Prefix Cache Hit vs Fresh Prefill (Model Level)
# ---------------------------------------------------------------------------

def get_model_and_tokenizer(model_path):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa",
    )
    model.eval()
    return model, tokenizer


# Diverse query templates — each run generates a unique prompt to get
# truly independent samples (batch_size=1, deterministic per input).
PREFIXES = [
    # Short system prompts
    "You are a helpful AI assistant.\n\nQuestion: ",
    "You are a coding assistant.\n\nTask: ",
    "Answer the following question concisely.\n\nQ: ",
    # Medium system prompts
    "You are an expert financial analyst with deep knowledge of global markets and investment strategies. Provide detailed analysis.\n\nQuestion: ",
    "You are a medical knowledge assistant. Provide accurate health information based on current evidence.\n\nQuestion: ",
    # Long context (RAG-style)
    "Context: The transformer architecture was introduced in the paper 'Attention is All You Need' by Vaswani et al. in 2017. It replaced recurrent neural networks with self-attention mechanisms, enabling much more efficient parallelization during training. The key innovation was the multi-head attention mechanism, which allows the model to jointly attend to information from different representation subspaces. The architecture consists of an encoder and decoder, each composed of layers of multi-head attention and feed-forward networks. Layer normalization and residual connections are used throughout.\n\nBased on the context above, answer: ",
    "Context: Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data. Supervised learning uses labeled data to train models, while unsupervised learning finds patterns in unlabeled data. Reinforcement learning trains agents through reward signals. Deep learning uses neural networks with many layers to learn hierarchical representations. Common architectures include CNNs for images, RNNs for sequences, and transformers for various tasks.\n\nQuestion: ",
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


def _build_test_cases(tokenizer, num_runs, device):
    """
    Build num_runs unique (prefix_ids, suffix_ids, full_ids) triples.

    CRITICAL: tokenize the FULL prompt first, then split by character-level
    prefix boundary to avoid BPE boundary mismatch.
    """
    import itertools
    combos = list(itertools.product(PREFIXES, SUFFIXES))
    # cycle through combos to get num_runs unique cases
    # (8 prefixes × 30 suffixes = 240 unique combos; cycle for more)
    cases = []
    for i in range(num_runs):
        prefix_str, suffix_str = combos[i % len(combos)]
        full_str = prefix_str + suffix_str

        # Tokenize FULL prompt, then find the split point
        full_ids = tokenizer.encode(full_str, return_tensors="pt").to(device)

        # To find correct token boundary: encode prefix alone, then verify
        # each token boundary until we find the one that decodes back to prefix
        prefix_ids_raw = tokenizer.encode(prefix_str, add_special_tokens=False)
        # The full_ids includes BOS; prefix in full_ids starts at index 1 (after BOS)
        # We need to find how many tokens in full_ids correspond to the prefix.
        # Strategy: decode incrementally from full_ids until we recover the prefix string.
        # More efficient: just encode prefix_str with the same BOS and find the split.

        # Robust approach: find split point by decoding prefixes of full_ids
        # until we match the prefix string exactly.
        prefix_char_len = len(prefix_str)
        split_pos = None
        for t in range(1, full_ids.shape[1] + 1):
            decoded = tokenizer.decode(full_ids[0, :t], skip_special_tokens=True)
            if len(decoded) >= prefix_char_len:
                # Check if decoded starts with prefix_str
                if decoded[:prefix_char_len] == prefix_str:
                    split_pos = t
                    break

        if split_pos is None or split_pos >= full_ids.shape[1]:
            continue  # skip if can't find clean boundary

        prefix_ids = full_ids[:, :split_pos]
        suffix_ids = full_ids[:, split_pos:]

        # Sanity: verify the split is correct
        assert suffix_ids.shape[1] > 0, f"Empty suffix for case {i}"

        cases.append({
            "prefix_str": prefix_str,
            "suffix_str": suffix_str,
            "full_ids": full_ids,
            "prefix_ids": prefix_ids,
            "suffix_ids": suffix_ids,
            "prefix_tokens": prefix_ids.shape[1],
            "suffix_tokens": suffix_ids.shape[1],
            "total_tokens": full_ids.shape[1],
        })

    return cases


@torch.no_grad()
def run_scenario2(model_path, num_runs=1000, device="cuda"):
    """
    Scenario 2: Prefix cache hit vs fresh prefill.

    A (fresh): Full prompt prefill in one shot → decode 1 token
    B (cached): Prefill prefix alone → save KV → then prefill suffix
                attending to cached prefix KV → decode 1 token

    Compare decode logits between A and B.

    Each of the num_runs iterations uses a DIFFERENT prompt (unique combo
    of prefix × suffix) so every run is an independent sample.
    Tokenization is done on the full prompt first, then split, to avoid
    BPE boundary mismatch.
    """
    print("\n" + "="*70)
    print("Scenario 2: Prefix Cache Hit vs Fresh Prefill (Model Level)")
    print("="*70)

    model, tokenizer = get_model_and_tokenizer(model_path)

    print("  Building test cases (tokenize + verify boundaries)...")
    cases = _build_test_cases(tokenizer, num_runs, device)
    print(f"  {len(cases)} valid test cases (each is a unique prompt)")

    # Verification: print a few examples
    for i in [0, 1, len(cases)//2]:
        c = cases[i]
        prefix_decoded = tokenizer.decode(c["prefix_ids"][0], skip_special_tokens=True)
        suffix_decoded = tokenizer.decode(c["suffix_ids"][0], skip_special_tokens=False)
        full_decoded = tokenizer.decode(c["full_ids"][0], skip_special_tokens=True)
        match = (full_decoded == prefix_decoded + suffix_decoded) or \
                (full_decoded == prefix_decoded.rstrip() + " " + suffix_decoded.lstrip())
        print(f"    Case {i}: prefix_tok={c['prefix_tokens']}, suffix_tok={c['suffix_tokens']}, "
              f"total={c['total_tokens']}")
        # Verify concatenated KV matches full
        cat_ids = torch.cat([c["prefix_ids"], c["suffix_ids"]], dim=1)
        assert torch.equal(cat_ids, c["full_ids"]), f"Token concat mismatch at case {i}!"

    total_tests = 0
    total_differ = 0
    total_argmax_flip = 0
    max_diff_seen = 0.0
    all_diffs = []
    flip_examples = []  # collect examples where first token flips
    per_prefix_stats = defaultdict(lambda: {"count": 0, "differ": 0, "flip": 0, "max_diff": 0.0})

    for case_idx, case in enumerate(cases):
        full_ids = case["full_ids"]
        prefix_ids = case["prefix_ids"]
        suffix_ids = case["suffix_ids"]

        # --- Path A: Fresh full prefill ---
        outputs_a = model(full_ids, use_cache=True)
        logits_a = outputs_a.logits[:, -1, :]  # prefill last-token logits = first generated token

        # --- Path B: Prefix prefill → cache → suffix prefill ---
        prefix_out = model(prefix_ids, use_cache=True)
        kv_prefix = prefix_out.past_key_values

        suffix_out = model(suffix_ids, past_key_values=kv_prefix, use_cache=True)
        logits_b = suffix_out.logits[:, -1, :]  # same position, via cached prefix

        # --- Compare: first generated token ---
        diff = (logits_a.float() - logits_b.float()).abs().max().item()

        first_token_a = logits_a.argmax(dim=-1).item()
        first_token_b = logits_b.argmax(dim=-1).item()
        flipped = (first_token_a != first_token_b)
        if flipped:
            tok_a = tokenizer.decode([first_token_a])
            tok_b = tokenizer.decode([first_token_b])
            # Also check margin: how close are the top-2 logits?
            topk_a = torch.topk(logits_a.float(), 2, dim=-1)
            margin_a = (topk_a.values[0, 0] - topk_a.values[0, 1]).item()
            flip_examples.append({
                "case_idx": case_idx,
                "suffix": case["suffix_str"][:50],
                "token_fresh": tok_a,
                "token_cached": tok_b,
                "logit_diff": diff,
                "top2_margin": margin_a,
            })

        total_tests += 1
        if diff > 0:
            total_differ += 1
        if flipped:
            total_argmax_flip += 1
        max_diff_seen = max(max_diff_seen, diff)
        all_diffs.append(diff)

        # Track per-prefix-length stats
        pkey = case["prefix_tokens"]
        per_prefix_stats[pkey]["count"] += 1
        if diff > 0:
            per_prefix_stats[pkey]["differ"] += 1
        if flipped:
            per_prefix_stats[pkey]["flip"] += 1
        per_prefix_stats[pkey]["max_diff"] = max(per_prefix_stats[pkey]["max_diff"], diff)

        if (case_idx + 1) % 100 == 0 or case_idx == len(cases) - 1:
            print(f"  [{case_idx+1:4d}/{len(cases)}] "
                  f"running: differ={total_differ}/{total_tests} ({total_differ/total_tests*100:.1f}%) "
                  f"argmax_flip={total_argmax_flip} max_diff={max_diff_seen:.6e}")

    # Summary by prefix length
    print(f"\n  Per prefix-length breakdown:")
    for pkey in sorted(per_prefix_stats.keys()):
        s = per_prefix_stats[pkey]
        print(f"    prefix_tokens={pkey:3d}: "
              f"differ={s['differ']}/{s['count']} ({s['differ']/s['count']*100:.1f}%) "
              f"flip={s['flip']}/{s['count']} ({s['flip']/s['count']*100:.1f}%) "
              f"max_diff={s['max_diff']:.6e}")

    # Diff distribution
    sorted_diffs = sorted(all_diffs)
    p50 = sorted_diffs[len(sorted_diffs)//2]
    p90 = sorted_diffs[int(len(sorted_diffs)*0.9)]
    p99 = sorted_diffs[int(len(sorted_diffs)*0.99)]

    print(f"\n  TOTAL: {total_differ}/{total_tests} differ ({total_differ/total_tests*100:.1f}%), "
          f"first_token_flip={total_argmax_flip}/{total_tests} ({total_argmax_flip/total_tests*100:.2f}%)")
    print(f"  Diff distribution: p50={p50:.6e}, p90={p90:.6e}, p99={p99:.6e}, max={max_diff_seen:.6e}")

    if flip_examples:
        print(f"\n  First-token flip examples ({len(flip_examples)} total):")
        for ex in flip_examples[:20]:  # show up to 20
            print(f"    suffix=\"{ex['suffix']}...\"")
            print(f"      fresh → \"{ex['token_fresh']}\"  |  cached → \"{ex['token_cached']}\"  "
                  f"| logit_diff={ex['logit_diff']:.4f}, top2_margin={ex['top2_margin']:.4f}")

    return {
        "total_tests": total_tests,
        "total_differ": total_differ,
        "differ_pct": f"{total_differ/total_tests*100:.1f}%",
        "total_first_token_flip": total_argmax_flip,
        "first_token_flip_pct": f"{total_argmax_flip/total_tests*100:.2f}%",
        "max_diff": f"{max_diff_seen:.6e}",
        "diff_distribution": {"p50": f"{p50:.6e}", "p90": f"{p90:.6e}", "p99": f"{p99:.6e}"},
        "flip_examples": flip_examples[:20],
        "per_prefix_length": {
            str(k): {
                "count": v["count"],
                "differ": v["differ"],
                "differ_pct": f"{v['differ']/v['count']*100:.1f}%",
                "flip": v["flip"],
                "flip_pct": f"{v['flip']/v['count']*100:.1f}%",
                "max_diff": f"{v['max_diff']:.6e}",
            }
            for k, v in sorted(per_prefix_stats.items())
        },
    }


# ---------------------------------------------------------------------------
# Scenario 3: Chunked Prefill with Different Chunk Sizes
# ---------------------------------------------------------------------------

@torch.no_grad()
def chunked_prefill(model, input_ids, chunk_size):
    """
    Prefill a sequence in chunks of chunk_size tokens.
    Each chunk attends to all previous KV (via past_key_values) + current chunk.
    Returns final logits and full KV cache.
    """
    seq_len = input_ids.shape[1]
    past_kv = None

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        chunk_ids = input_ids[:, start:end]
        outputs = model(chunk_ids, past_key_values=past_kv, use_cache=True)
        past_kv = outputs.past_key_values

    return outputs.logits[:, -1, :], past_kv


# Long context templates for Scenario 3 — we generate many unique prompts
# by combining context blocks + diverse questions.
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
    "memory footprint at the cost of some accuracy. Model parallelism strategies include tensor, pipeline, and "
    "expert parallelism for distributed inference across multiple GPUs.",

    "The human immune system is a complex network of cells, tissues, and organs that work together to defend "
    "the body against harmful pathogens. The innate immune system provides immediate, non-specific defense through "
    "physical barriers like skin, chemical barriers like stomach acid, and cellular defenses like phagocytes. "
    "The adaptive immune system develops targeted responses through T cells and B cells, creating immunological "
    "memory for faster future responses. Vaccines work by training the adaptive immune system to recognize specific "
    "antigens without causing disease. Autoimmune disorders occur when the immune system mistakenly attacks the "
    "body's own cells, leading to conditions like rheumatoid arthritis and type 1 diabetes.",

    "Distributed systems are computer systems whose components are located on different networked computers that "
    "communicate and coordinate their actions by passing messages. Key challenges include handling partial failures, "
    "network partitions, and maintaining consistency across replicas. The CAP theorem states that a distributed "
    "system can only guarantee two of three properties: consistency, availability, and partition tolerance. "
    "Consensus algorithms like Raft and Paxos enable agreement among distributed nodes. Modern architectures "
    "use microservices, message queues like Kafka, and container orchestration with Kubernetes. Database sharding "
    "distributes data across multiple instances for scalability. Load balancers distribute traffic, while circuit "
    "breakers prevent cascading failures.",

    "Climate change refers to long-term shifts in global temperatures and weather patterns. Human activities, "
    "primarily burning fossil fuels, have been the main driver since the 1800s, releasing greenhouse gases like "
    "carbon dioxide and methane. The global average temperature has risen by about 1.1 degrees Celsius since "
    "pre-industrial times. Effects include rising sea levels, more frequent extreme weather events, ocean "
    "acidification, and biodiversity loss. Mitigation strategies include transitioning to renewable energy, "
    "improving energy efficiency, carbon capture technologies, and reforestation. International agreements like "
    "the Paris Agreement aim to limit warming to 1.5 degrees Celsius above pre-industrial levels.",
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


@torch.no_grad()
def run_scenario3(model_path, num_runs=1000, device="cuda"):
    """
    Scenario 3: Same prompt, different chunk sizes for prefill.

    Generates num_runs unique long prompts (context + question combos).
    For each prompt, compares full prefill vs chunked prefill with various
    chunk sizes. Each prompt is tested once (all independent samples).
    """
    print("\n" + "="*70)
    print("Scenario 3: Chunked Prefill — Different Chunk Sizes")
    print("="*70)

    model, tokenizer = get_model_and_tokenizer(model_path)

    import itertools
    combos = list(itertools.product(LONG_CONTEXTS, LONG_QUESTIONS))
    # 5 contexts × 20 questions = 100 unique combos; cycle for more

    chunk_sizes = [128, 64, 32]
    total_tests = 0
    total_differ = defaultdict(int)
    total_argmax_flip = defaultdict(int)
    max_diffs = defaultdict(float)
    all_diffs = defaultdict(list)
    flip_examples = defaultdict(list)

    print(f"  {num_runs} unique prompts, chunk sizes: full (baseline), {chunk_sizes}")

    for test_idx in range(num_runs):
        ctx, question = combos[test_idx % len(combos)]
        # Add variation for cycling: append run index to make prompt unique
        if test_idx >= len(combos):
            variant = f" (variant {test_idx // len(combos)})"
        else:
            variant = ""
        prompt = f"Context: {ctx}{variant}\n\nQuestion: {question}\n\nAnswer:"
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        seq_len = input_ids.shape[1]

        # Baseline: full prefill (only need prefill logits for first-token comparison)
        logits_full, _ = chunked_prefill(model, input_ids, chunk_size=seq_len)

        for cs in chunk_sizes:
            if cs >= seq_len:
                continue
            logits_c, _ = chunked_prefill(model, input_ids, chunk_size=cs)

            diff = (logits_full.float() - logits_c.float()).abs().max().item()
            tok_full = logits_full.argmax(-1).item()
            tok_c = logits_c.argmax(-1).item()
            flipped = (tok_full != tok_c)

            key = f"chunk_{cs}"
            all_diffs[key].append(diff)
            if diff > 0:
                total_differ[key] += 1
            if flipped:
                flip_examples[key].append({
                    "test_idx": test_idx,
                    "question": question[:50],
                    "token_full": tokenizer.decode([tok_full]),
                    "token_chunked": tokenizer.decode([tok_c]),
                    "logit_diff": diff,
                })
                total_argmax_flip[key] += 1
            max_diffs[key] = max(max_diffs[key], diff)

        total_tests += 1

        if (test_idx + 1) % 100 == 0 or test_idx == num_runs - 1:
            print(f"  [{test_idx+1:4d}/{num_runs}] seq_len={seq_len} ", end="")
            for cs in chunk_sizes:
                key = f"chunk_{cs}"
                n = total_differ[key]
                af = total_argmax_flip[key]
                print(f"| cs={cs}: {n}/{total_tests} diff, {af} flip ", end="")
            print()

    print(f"\n  SUMMARY ({total_tests} unique prompts):")
    summary = {}
    for cs in chunk_sizes:
        key = f"chunk_{cs}"
        n = total_differ[key]
        af = total_argmax_flip[key]
        md = max_diffs[key]
        dlist = all_diffs[key]
        sorted_d = sorted(dlist)
        p50 = sorted_d[len(sorted_d)//2] if sorted_d else 0
        p90 = sorted_d[int(len(sorted_d)*0.9)] if sorted_d else 0
        p99 = sorted_d[int(len(sorted_d)*0.99)] if sorted_d else 0
        print(f"    chunk_size={cs}: differ={n}/{total_tests} ({n/total_tests*100:.1f}%) "
              f"first_token_flip={af}/{total_tests} ({af/total_tests*100:.1f}%) "
              f"p50={p50:.6e} p90={p90:.6e} max={md:.6e}")
        summary[key] = {
            "total_differ": n,
            "differ_pct": f"{n/total_tests*100:.1f}%",
            "total_first_token_flip": af,
            "first_token_flip_pct": f"{af/total_tests*100:.1f}%",
            "p50_diff": f"{p50:.6e}",
            "p90_diff": f"{p90:.6e}",
            "p99_diff": f"{p99:.6e}",
            "max_diff": f"{md:.6e}",
            "flip_examples": flip_examples[key][:10],
        }

    # Print flip examples
    for cs in chunk_sizes:
        key = f"chunk_{cs}"
        if flip_examples[key]:
            print(f"\n  First-token flip examples for chunk_size={cs} ({len(flip_examples[key])} total):")
            for ex in flip_examples[key][:10]:
                print(f"    q=\"{ex['question']}...\"")
                print(f"      full → \"{ex['token_full']}\"  |  chunked → \"{ex['token_chunked']}\"  "
                      f"| logit_diff={ex['logit_diff']:.4f}")

    return {"total_tests": total_tests, "summary": summary}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="KV-Cache Layout Determinism Tests")
    parser.add_argument("--model", type=str,
                        default="/home/kec23008/docker-sys/Models/Llama-3.1-8B-Instruct",
                        help="Path to model")
    parser.add_argument("--num-runs", type=int, default=1000,
                        help="Total number of test iterations per scenario")
    parser.add_argument("--scenarios", type=str, default="1,2,3",
                        help="Comma-separated scenario numbers to run")
    parser.add_argument("--output", type=str, default="kvcache_layout_results.json",
                        help="Output JSON file")
    args = parser.parse_args()

    scenarios = [int(s) for s in args.scenarios.split(",")]
    all_results = {"config": {"model": args.model, "num_runs": args.num_runs}}

    t0 = time.time()

    if 1 in scenarios:
        all_results["scenario1"] = run_scenario1(num_runs=args.num_runs)

    if 2 in scenarios:
        all_results["scenario2"] = run_scenario2(args.model, num_runs=args.num_runs)

    if 3 in scenarios:
        all_results["scenario3"] = run_scenario3(args.model, num_runs=args.num_runs)

    elapsed = time.time() - t0
    all_results["elapsed_seconds"] = round(elapsed, 1)
    print(f"\nTotal time: {elapsed:.1f}s")

    out_path = Path(__file__).parent / args.output
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
