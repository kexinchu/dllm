#!/usr/bin/env python3
"""
Agent E: Downstream Impact Experiments
Measures how non-determinism from batch composition affects practical applications:
  1. Reward Signal Variance (RL proxy)
  2. Distillation Signal Corruption
  3. MoE Near-Tie Analysis (synthetic)
"""
import sys, os, time, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy import stats
import numpy as np

MODEL_PATH = "/home/kec23008/docker-sys/Models/Llama-3.1-8B-Instruct"
OUTPUT_JSON = "/home/kec23008/docker-sys/dllm/research/exp_downstream.json"
OUTPUT_MD = "/home/kec23008/docker-sys/dllm/research/exp_downstream.md"

torch.manual_seed(42)

# ============================================================================
# Helpers (from reference: continuous batching sim)
# ============================================================================

def make_equal_length_batch(tok, target_prompt, filler_prompts, target_len, device):
    """
    Tokenize all prompts and truncate/pad to exactly target_len tokens.
    Simulates continuous batching where all sequences are the same length.
    No left-padding, no position shift -- pure batch-size variation.
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


def load_model():
    print("Loading model...")
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16, device_map="auto")
    model.eval()
    print(f"  Model loaded in {time.perf_counter()-t0:.1f}s")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    return model, tok


# 30 diverse prompts
PROMPTS = [
    "What is deterministic inference in large language models?",
    "Explain quantum computing in simple terms for beginners.",
    "Write a short poem about mountains and rivers in spring.",
    "How does photosynthesis work in C3 and C4 plants today?",
    "What is the meaning of life according to philosophy here?",
    "Describe the process of nuclear fusion happening in the sun.",
    "What is machine learning and how does it differ from AI?",
    "How do modern CPUs achieve instruction level parallelism now?",
    "What is the relationship between entropy and information?",
    "Explain the double slit experiment and wave particle duality.",
    "Describe the architecture of a modern transformer network.",
    "How does CRISPR gene editing technology work step by step?",
    "What are the fundamental forces of nature and interactions?",
    "Explain public key cryptography and the RSA algorithm now.",
    "What is the capital of France and why is it important?",
    "Describe the complete lifecycle of a star from nebula.",
    "Explain the principles of thermodynamics with examples.",
    "How does the internet work from cables to application layer?",
    "Explain the central dogma of molecular biology in detail.",
    "Discuss the major breakthroughs in physics in the 20th century.",
    "How does gradient descent optimize neural network parameters?",
    "What is the difference between supervised and unsupervised learning?",
    "Explain how convolutional neural networks process images.",
    "What is the significance of attention in transformer models?",
    "Describe how reinforcement learning trains game-playing agents.",
    "What are the ethical concerns surrounding artificial intelligence?",
    "How does natural language processing enable machine translation?",
    "Explain the concept of overfitting and regularization techniques.",
    "What is transfer learning and why is it useful in practice?",
    "Describe the architecture and training of generative adversarial nets.",
]

# 15 fillers for batching
FILLERS = [
    "Tell me a joke about a programmer and a rubber duck now.",
    "What is the relationship between entropy and information?",
    "Explain the double slit experiment and wave particle duality.",
    "Describe the architecture of a modern transformer network.",
    "How does CRISPR gene editing technology work step by step?",
    "What are the fundamental forces of nature and interactions?",
    "Explain public key cryptography and the RSA algorithm now.",
    "What is the capital of France and why is it important?",
    "Describe the complete lifecycle of a star from nebula.",
    "Explain the principles of thermodynamics with examples.",
    "How does the internet work from cables to application layer?",
    "Explain the central dogma of molecular biology in detail.",
    "Discuss the major breakthroughs in physics in the 20th century.",
    "How does gradient descent optimize neural network parameters?",
    "What is the difference between supervised and unsupervised learning?",
]


# ============================================================================
# Experiment 1: Reward Signal Variance (RL proxy)
# ============================================================================

def experiment_reward_signal_variance(model, tok):
    print("\n" + "="*80)
    print("  EXPERIMENT 1: REWARD SIGNAL VARIANCE (RL PROXY)")
    print("="*80)
    device = next(model.parameters()).device
    results = {}

    for mode_name, flag_val in [("bf16_default", True), ("fp32_accum", False)]:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = flag_val
        mode_label = "BF16 (default)" if flag_val else "FP32 accum"
        print(f"\n  --- {mode_label} ---")

        per_prompt_mean_abs_diff = []
        per_prompt_max_diff = []
        per_prompt_nonzero_frac = []
        all_position_diffs = []

        for p_idx, prompt in enumerate(PROMPTS):
            target_ids = tok.encode(prompt, add_special_tokens=True)
            target_len = len(target_ids)

            # bs=1 reference
            inp_bs1 = make_equal_length_batch(tok, prompt, [], target_len, device)
            with torch.no_grad():
                logits_bs1 = model(**inp_bs1).logits[0].float()  # [seq_len, vocab]

            # bs=8
            inp_bs8 = make_equal_length_batch(tok, prompt, FILLERS[:7], target_len, device)
            with torch.no_grad():
                logits_bs8 = model(**inp_bs8).logits[0].float()  # [seq_len, vocab]

            # Log-prob of top-1 prediction at each position (using bs=1 top-1 as reference token)
            top1_tokens = logits_bs1.argmax(dim=-1)  # [seq_len]

            logprobs_bs1 = F.log_softmax(logits_bs1, dim=-1)
            logprobs_bs8 = F.log_softmax(logits_bs8, dim=-1)

            # Gather log-probs for the top-1 tokens
            lp_bs1 = logprobs_bs1[torch.arange(target_len, device=device), top1_tokens]  # [seq_len]
            lp_bs8 = logprobs_bs8[torch.arange(target_len, device=device), top1_tokens]  # [seq_len]

            abs_diff = (lp_bs1 - lp_bs8).abs()  # [seq_len]
            all_position_diffs.append(abs_diff.cpu())

            mean_ad = abs_diff.mean().item()
            max_ad = abs_diff.max().item()
            nonzero_frac = (abs_diff > 0).float().mean().item()

            per_prompt_mean_abs_diff.append(mean_ad)
            per_prompt_max_diff.append(max_ad)
            per_prompt_nonzero_frac.append(nonzero_frac)

            if (p_idx + 1) % 10 == 0:
                print(f"    [{p_idx+1}/30] mean_abs_diff={mean_ad:.4e} max={max_ad:.4e} nonzero_frac={nonzero_frac:.4f}")

        # Statistical test: paired t-test (H0: mean_abs_diff = 0)
        t_stat, p_value = stats.ttest_1samp(per_prompt_mean_abs_diff, 0.0)

        # Aggregate across all positions
        all_diffs_cat = torch.cat(all_position_diffs)
        global_mean = all_diffs_cat.mean().item()
        global_max = all_diffs_cat.max().item()
        global_nonzero_frac = (all_diffs_cat > 0).float().mean().item()
        global_median = all_diffs_cat.median().item()

        results[mode_name] = {
            "mean_abs_diff_per_prompt": per_prompt_mean_abs_diff,
            "max_diff_per_prompt": per_prompt_max_diff,
            "nonzero_frac_per_prompt": per_prompt_nonzero_frac,
            "global_mean_abs_diff": global_mean,
            "global_max_abs_diff": global_max,
            "global_median_abs_diff": global_median,
            "global_nonzero_frac": global_nonzero_frac,
            "paired_ttest_t_stat": float(t_stat),
            "paired_ttest_p_value": float(p_value),
            "num_prompts": len(PROMPTS),
            "total_positions": int(all_diffs_cat.shape[0]),
        }

        print(f"\n  {mode_label} SUMMARY:")
        print(f"    Global mean |logprob_diff|:  {global_mean:.6e}")
        print(f"    Global max  |logprob_diff|:  {global_max:.6e}")
        print(f"    Global median |logprob_diff|: {global_median:.6e}")
        print(f"    Fraction positions nonzero:   {global_nonzero_frac:.4f}")
        print(f"    Paired t-test: t={t_stat:.4f}, p={p_value:.4e}")

    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    return results


# ============================================================================
# Experiment 2: Distillation Signal Corruption
# ============================================================================

def experiment_distillation_signal(model, tok):
    print("\n" + "="*80)
    print("  EXPERIMENT 2: DISTILLATION SIGNAL CORRUPTION")
    print("="*80)
    device = next(model.parameters()).device
    results = {}
    T = 1.0  # temperature

    for mode_name, flag_val in [("bf16_default", True), ("fp32_accum", False)]:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = flag_val
        mode_label = "BF16 (default)" if flag_val else "FP32 accum"
        print(f"\n  --- {mode_label} ---")

        per_prompt_mean_kl = []
        per_prompt_max_kl = []
        per_prompt_frac_above_1e6 = []
        all_position_kls = []

        for p_idx, prompt in enumerate(PROMPTS):
            target_ids = tok.encode(prompt, add_special_tokens=True)
            target_len = len(target_ids)

            # bs=1 reference (teacher logits in isolation)
            inp_bs1 = make_equal_length_batch(tok, prompt, [], target_len, device)
            with torch.no_grad():
                logits_bs1 = model(**inp_bs1).logits[0].float()  # [seq_len, vocab]

            # bs=8 (teacher logits when batched)
            inp_bs8 = make_equal_length_batch(tok, prompt, FILLERS[:7], target_len, device)
            with torch.no_grad():
                logits_bs8 = model(**inp_bs8).logits[0].float()  # [seq_len, vocab]

            # Soft distributions at temperature T
            p_bs1 = F.softmax(logits_bs1 / T, dim=-1)  # [seq_len, vocab]
            p_bs8 = F.softmax(logits_bs8 / T, dim=-1)  # [seq_len, vocab]

            # Per-position KL(p_bs1 || p_bs8)
            # KL = sum_v p_bs1(v) * log(p_bs1(v) / p_bs8(v))
            # Use kl_div which expects log-space input: kl_div(log_q, p) = sum p*(log_p - log_q)
            log_p_bs8 = torch.log(p_bs8 + 1e-10)
            # kl_div(input=log_q, target=p, reduction='none') = p * (log_p - log_q)
            kl_per_vocab = F.kl_div(log_p_bs8, p_bs1, reduction='none', log_target=False)  # [seq_len, vocab]
            kl_per_position = kl_per_vocab.sum(dim=-1)  # [seq_len]

            all_position_kls.append(kl_per_position.cpu())

            mean_kl = kl_per_position.mean().item()
            max_kl = kl_per_position.max().item()
            frac_above = (kl_per_position > 1e-6).float().mean().item()

            per_prompt_mean_kl.append(mean_kl)
            per_prompt_max_kl.append(max_kl)
            per_prompt_frac_above_1e6.append(frac_above)

            if (p_idx + 1) % 10 == 0:
                print(f"    [{p_idx+1}/30] mean_KL={mean_kl:.4e} max_KL={max_kl:.4e} frac>1e-6={frac_above:.4f}")

        all_kls_cat = torch.cat(all_position_kls)
        global_mean_kl = all_kls_cat.mean().item()
        global_max_kl = all_kls_cat.max().item()
        global_median_kl = all_kls_cat.median().item()
        global_frac_above = (all_kls_cat > 1e-6).float().mean().item()

        results[mode_name] = {
            "mean_kl_per_prompt": per_prompt_mean_kl,
            "max_kl_per_prompt": per_prompt_max_kl,
            "frac_above_1e6_per_prompt": per_prompt_frac_above_1e6,
            "global_mean_kl": global_mean_kl,
            "global_max_kl": global_max_kl,
            "global_median_kl": global_median_kl,
            "global_frac_above_1e6": global_frac_above,
            "temperature": T,
            "num_prompts": len(PROMPTS),
            "total_positions": int(all_kls_cat.shape[0]),
        }

        print(f"\n  {mode_label} SUMMARY:")
        print(f"    Global mean KL:         {global_mean_kl:.6e}")
        print(f"    Global max KL:          {global_max_kl:.6e}")
        print(f"    Global median KL:       {global_median_kl:.6e}")
        print(f"    Frac positions KL>1e-6: {global_frac_above:.4f}")

    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    return results


# ============================================================================
# Experiment 3: MoE Near-Tie Analysis (synthetic)
# ============================================================================

def experiment_moe_near_tie(model=None, tok=None):
    """
    Synthetic MoE gating scenario. Uses a random linear gate to simulate
    expert routing. Measures how batch composition affects expert selection
    due to floating-point non-determinism in matmul.
    """
    print("\n" + "="*80)
    print("  EXPERIMENT 3: MoE NEAR-TIE ANALYSIS (SYNTHETIC)")
    print("="*80)
    device = torch.device("cuda")

    hidden_dim = 2048
    num_experts = 128
    top_k = 8
    total_tokens = 1000
    seq_len = 50  # 1000 tokens / 20 sequences = 50 per sequence
    num_seqs = total_tokens // seq_len  # 20

    torch.manual_seed(42)
    gate_weight = torch.randn(num_experts, hidden_dim, device=device, dtype=torch.bfloat16)

    # Generate random hidden states for the "target" sequence
    target_hidden = torch.randn(1, seq_len, hidden_dim, device=device, dtype=torch.bfloat16)

    batch_sizes_test = [4, 8, 16]
    tau_values = [0.1, 0.01, 0.001]

    results = {}

    for mode_name, flag_val in [("bf16_default", True), ("fp32_accum", False)]:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = flag_val
        mode_label = "BF16 (default)" if flag_val else "FP32 accum"
        print(f"\n  --- {mode_label} ---")

        mode_results = {"batch_sizes": {}, "near_tie_prevalence": {}}

        # Reference: bs=1 router logits
        with torch.no_grad():
            router_logits_ref = F.linear(target_hidden, gate_weight)  # [1, seq_len, num_experts]
            probs_ref = F.softmax(router_logits_ref.float(), dim=-1)
            topk_ref = torch.topk(probs_ref[0], k=top_k, dim=-1)  # values: [seq_len, top_k], indices: [seq_len, top_k]
            experts_ref = topk_ref.indices  # [seq_len, top_k]

        # Near-tie analysis on reference: gap between k-th and (k+1)-th expert
        # Get full sorted probabilities
        sorted_probs_ref, _ = torch.sort(probs_ref[0].float(), dim=-1, descending=True)  # [seq_len, num_experts]
        gap = sorted_probs_ref[:, top_k - 1] - sorted_probs_ref[:, top_k]  # [seq_len]

        for tau in tau_values:
            near_tie_count = (gap < tau).sum().item()
            near_tie_frac = near_tie_count / seq_len
            mode_results["near_tie_prevalence"][str(tau)] = {
                "count": near_tie_count,
                "fraction": near_tie_frac,
                "total_tokens": seq_len,
            }
            print(f"    Near-tie prevalence (gap < {tau}): {near_tie_count}/{seq_len} = {near_tie_frac:.4f}")

        # Test different batch sizes - run multiple trials with different filler data
        num_trials = 20
        for bs in batch_sizes_test:
            total_flips = 0
            total_positions = 0
            per_trial_flips = []

            for trial in range(num_trials):
                torch.manual_seed(42 + trial + 1000)  # Different fillers each trial
                filler_hidden = torch.randn(bs - 1, seq_len, hidden_dim, device=device, dtype=torch.bfloat16)
                batch_hidden = torch.cat([target_hidden, filler_hidden], dim=0)  # [bs, seq_len, hidden_dim]

                with torch.no_grad():
                    router_logits_batch = F.linear(batch_hidden, gate_weight)
                    probs_batch = F.softmax(router_logits_batch.float(), dim=-1)
                    topk_batch = torch.topk(probs_batch[0], k=top_k, dim=-1)
                    experts_batch = topk_batch.indices  # [seq_len, top_k]

                # Compare expert selections: for each position, check if expert sets differ
                flips = 0
                for pos in range(seq_len):
                    ref_set = set(experts_ref[pos].cpu().tolist())
                    batch_set = set(experts_batch[pos].cpu().tolist())
                    if ref_set != batch_set:
                        flips += 1

                total_flips += flips
                total_positions += seq_len
                per_trial_flips.append(flips)

            flip_rate = total_flips / total_positions
            mean_flips = np.mean(per_trial_flips)
            std_flips = np.std(per_trial_flips)

            mode_results["batch_sizes"][str(bs)] = {
                "total_flips": total_flips,
                "total_positions": total_positions,
                "flip_rate": flip_rate,
                "num_trials": num_trials,
                "mean_flips_per_trial": float(mean_flips),
                "std_flips_per_trial": float(std_flips),
                "per_trial_flips": per_trial_flips,
            }

            print(f"    bs={bs:>2}: flip_rate={flip_rate:.4f} ({total_flips}/{total_positions}) "
                  f"mean_per_trial={mean_flips:.1f}+/-{std_flips:.1f}")

        results[mode_name] = mode_results

    # Reset seed
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    return results


# ============================================================================
# Report generation
# ============================================================================

def generate_markdown_report(all_results):
    lines = []
    lines.append("# Experiment E: Downstream Impact of Non-Determinism")
    lines.append("")
    lines.append("Measures how batch-composition non-determinism in BF16 matmul affects")
    lines.append("practical ML applications: RL reward signals, knowledge distillation, and MoE routing.")
    lines.append("")
    lines.append(f"- **Model**: Llama-3.1-8B-Instruct")
    lines.append(f"- **GPU**: NVIDIA RTX A6000")
    lines.append(f"- **Precision**: BF16 weights, comparing BF16 vs FP32 accumulation")
    lines.append(f"- **Batch comparison**: bs=1 (reference) vs bs=8 (continuous batching sim)")
    lines.append("")

    # Experiment 1
    r1 = all_results["experiment_1_reward_signal"]
    lines.append("---")
    lines.append("## Experiment 1: Reward Signal Variance (RL Proxy)")
    lines.append("")
    lines.append("**Motivation**: In RLHF/GRPO, the reward for a generated sequence depends on model logits.")
    lines.append("If logits change with batch composition, the reward signal has artificial noise.")
    lines.append("")
    lines.append("**Method**: For 30 prompts, compute log-probability of each position's top-1 prediction")
    lines.append("at bs=1 vs bs=8. Measure |logprob_bs1 - logprob_bs8| per position.")
    lines.append("")
    lines.append("| Metric | BF16 (default) | FP32 accum |")
    lines.append("|--------|----------------|------------|")

    bf = r1["bf16_default"]
    fp = r1["fp32_accum"]
    lines.append(f"| Mean \\|logprob diff\\| | {bf['global_mean_abs_diff']:.6e} | {fp['global_mean_abs_diff']:.6e} |")
    lines.append(f"| Median \\|logprob diff\\| | {bf['global_median_abs_diff']:.6e} | {fp['global_median_abs_diff']:.6e} |")
    lines.append(f"| Max \\|logprob diff\\| | {bf['global_max_abs_diff']:.6e} | {fp['global_max_abs_diff']:.6e} |")
    lines.append(f"| Frac positions nonzero | {bf['global_nonzero_frac']:.4f} | {fp['global_nonzero_frac']:.4f} |")
    lines.append(f"| Paired t-test (t-stat) | {bf['paired_ttest_t_stat']:.4f} | {fp['paired_ttest_t_stat']:.4f} |")
    lines.append(f"| Paired t-test (p-value) | {bf['paired_ttest_p_value']:.4e} | {fp['paired_ttest_p_value']:.4e} |")
    lines.append(f"| Total positions | {bf['total_positions']} | {fp['total_positions']} |")
    lines.append("")

    # Interpretation
    if bf['paired_ttest_p_value'] < 0.05:
        lines.append("**Finding**: The reward signal difference is statistically significant (p < 0.05) under BF16 defaults.")
        lines.append("This means batch composition injects systematic noise into RL reward signals.")
    else:
        lines.append("**Finding**: The reward signal difference is not statistically significant under BF16 defaults.")
    lines.append("")

    # Experiment 2
    r2 = all_results["experiment_2_distillation"]
    lines.append("---")
    lines.append("## Experiment 2: Distillation Signal Corruption")
    lines.append("")
    lines.append("**Motivation**: In knowledge distillation, the teacher's soft logits are the training signal.")
    lines.append("If these change with batch composition, the student learns inconsistent targets.")
    lines.append("")
    lines.append("**Method**: For 30 prompts, compute token-level KL(softmax(logits_bs1/T) || softmax(logits_bs8/T))")
    lines.append("with T=1.")
    lines.append("")
    lines.append("| Metric | BF16 (default) | FP32 accum |")
    lines.append("|--------|----------------|------------|")

    bf2 = r2["bf16_default"]
    fp2 = r2["fp32_accum"]
    lines.append(f"| Mean KL per position | {bf2['global_mean_kl']:.6e} | {fp2['global_mean_kl']:.6e} |")
    lines.append(f"| Median KL per position | {bf2['global_median_kl']:.6e} | {fp2['global_median_kl']:.6e} |")
    lines.append(f"| Max KL per position | {bf2['global_max_kl']:.6e} | {fp2['global_max_kl']:.6e} |")
    lines.append(f"| Frac positions KL > 1e-6 | {bf2['global_frac_above_1e6']:.4f} | {fp2['global_frac_above_1e6']:.4f} |")
    lines.append(f"| Total positions | {bf2['total_positions']} | {fp2['total_positions']} |")
    lines.append("")

    # Compute reduction
    if bf2['global_mean_kl'] > 0 and fp2['global_mean_kl'] > 0:
        reduction = (1.0 - fp2['global_mean_kl'] / bf2['global_mean_kl']) * 100
        lines.append(f"**Finding**: FP32 accumulation reduces mean KL divergence by {reduction:.1f}% compared to BF16 defaults.")
    lines.append("")

    # Experiment 3
    r3 = all_results["experiment_3_moe_near_tie"]
    lines.append("---")
    lines.append("## Experiment 3: MoE Near-Tie Analysis (Synthetic)")
    lines.append("")
    lines.append("**Motivation**: MoE models select experts via top-k on router logits. Near-tie logits")
    lines.append("(small gap between k-th and (k+1)-th) are vulnerable to expert selection flips.")
    lines.append("")
    lines.append("**Method**: Synthetic MoE with hidden_dim=2048, 128 experts, top-8 routing.")
    lines.append("Compare expert selections at bs=1 vs bs={4,8,16} over 20 trials x 50 tokens each.")
    lines.append("")

    lines.append("### Near-Tie Prevalence (bs=1 reference)")
    lines.append("")
    lines.append("| Threshold (tau) | BF16 fraction | FP32 fraction |")
    lines.append("|-----------------|---------------|---------------|")
    for tau in ["0.1", "0.01", "0.001"]:
        bf3 = r3["bf16_default"]["near_tie_prevalence"][tau]
        fp3 = r3["fp32_accum"]["near_tie_prevalence"][tau]
        lines.append(f"| gap < {tau} | {bf3['fraction']:.4f} ({bf3['count']}/{bf3['total_tokens']}) | {fp3['fraction']:.4f} ({fp3['count']}/{fp3['total_tokens']}) |")
    lines.append("")

    lines.append("### Expert Selection Flip Rate")
    lines.append("")
    lines.append("| Batch Size | BF16 flip rate | FP32 flip rate |")
    lines.append("|------------|----------------|----------------|")
    for bs in ["4", "8", "16"]:
        bf_bs = r3["bf16_default"]["batch_sizes"][bs]
        fp_bs = r3["fp32_accum"]["batch_sizes"][bs]
        lines.append(f"| bs={bs} | {bf_bs['flip_rate']:.4f} ({bf_bs['total_flips']}/{bf_bs['total_positions']}) | {fp_bs['flip_rate']:.4f} ({fp_bs['total_flips']}/{fp_bs['total_positions']}) |")
    lines.append("")

    # Overall summary
    lines.append("---")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Application | BF16 Impact | FP32 Accum Impact | Practical Concern |")
    lines.append("|-------------|------------|-------------------|-------------------|")

    # RL row
    rl_concern = "High" if bf['global_nonzero_frac'] > 0.5 else ("Medium" if bf['global_nonzero_frac'] > 0.1 else "Low")
    rl_concern_fp32 = "High" if fp['global_nonzero_frac'] > 0.5 else ("Medium" if fp['global_nonzero_frac'] > 0.1 else "Low")
    lines.append(f"| RL Reward Signal | {bf['global_nonzero_frac']*100:.1f}% positions affected | {fp['global_nonzero_frac']*100:.1f}% positions affected | {rl_concern} / {rl_concern_fp32} |")

    # Distillation row
    dist_concern = "High" if bf2['global_frac_above_1e6'] > 0.5 else ("Medium" if bf2['global_frac_above_1e6'] > 0.1 else "Low")
    dist_concern_fp32 = "High" if fp2['global_frac_above_1e6'] > 0.5 else ("Medium" if fp2['global_frac_above_1e6'] > 0.1 else "Low")
    lines.append(f"| KD Soft Targets | {bf2['global_frac_above_1e6']*100:.1f}% positions KL>1e-6 | {fp2['global_frac_above_1e6']*100:.1f}% positions KL>1e-6 | {dist_concern} / {dist_concern_fp32} |")

    # MoE row
    bf_moe_8 = r3["bf16_default"]["batch_sizes"]["8"]
    fp_moe_8 = r3["fp32_accum"]["batch_sizes"]["8"]
    moe_concern = "High" if bf_moe_8['flip_rate'] > 0.05 else ("Medium" if bf_moe_8['flip_rate'] > 0.01 else "Low")
    moe_concern_fp32 = "High" if fp_moe_8['flip_rate'] > 0.05 else ("Medium" if fp_moe_8['flip_rate'] > 0.01 else "Low")
    lines.append(f"| MoE Expert Routing | {bf_moe_8['flip_rate']*100:.1f}% expert flips (bs=8) | {fp_moe_8['flip_rate']*100:.1f}% expert flips (bs=8) | {moe_concern} / {moe_concern_fp32} |")

    lines.append("")

    return "\n".join(lines)


# ============================================================================
# Main
# ============================================================================

def main():
    print("="*80)
    print("  AGENT E: DOWNSTREAM IMPACT EXPERIMENTS")
    print("="*80)

    t_start = time.perf_counter()

    model, tok = load_model()
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False  # Reset after load

    all_results = {}

    # Experiment 1
    t0 = time.perf_counter()
    all_results["experiment_1_reward_signal"] = experiment_reward_signal_variance(model, tok)
    t1 = time.perf_counter()
    print(f"\n  Experiment 1 completed in {t1-t0:.1f}s")

    torch.cuda.empty_cache()

    # Experiment 2
    t0 = time.perf_counter()
    all_results["experiment_2_distillation"] = experiment_distillation_signal(model, tok)
    t1 = time.perf_counter()
    print(f"\n  Experiment 2 completed in {t1-t0:.1f}s")

    torch.cuda.empty_cache()

    # Experiment 3 (no model needed, synthetic)
    t0 = time.perf_counter()
    all_results["experiment_3_moe_near_tie"] = experiment_moe_near_tie()
    t1 = time.perf_counter()
    print(f"\n  Experiment 3 completed in {t1-t0:.1f}s")

    # Save results
    total_time = time.perf_counter() - t_start
    all_results["metadata"] = {
        "model": MODEL_PATH,
        "gpu": torch.cuda.get_device_name(0),
        "total_runtime_seconds": total_time,
    }

    # Save JSON
    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to {OUTPUT_JSON}")

    # Save Markdown
    md = generate_markdown_report(all_results)
    with open(OUTPUT_MD, "w") as f:
        f.write(md)
    print(f"  Report saved to {OUTPUT_MD}")

    print(f"\n  TOTAL RUNTIME: {total_time:.1f}s")


if __name__ == "__main__":
    main()
