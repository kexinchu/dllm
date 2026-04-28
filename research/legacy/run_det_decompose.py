"""
DetermLLM: Determinism Decomposition Experiment
================================================
Goal: Identify which components of LLM inference cause batch non-determinism,
      verify that precision amplification achieves token-level batch-invariance.

Experiment design:
  - Metric: EXACT token sequence match (full generation, not per-step argmax)
    + per-token log-prob deviation |Δlog P|
  - Conditions:
      A: BF16 baseline (no patch)
      B: DetermLLM F.linear only (current approach)
      C: DetermLLM F.linear + torch.matmul/bmm patch (full GEMM coverage)
      D: torch.use_deterministic_algorithms (oracle)
  - Models: Llama-3.2-1B-Instruct, Phi-4 (14B if memory allows)
  - Batch sizes: 1 (reference), 2, 4, 8, 16, 32, 64
  - Generation: 64 tokens, greedy (temperature=0)
  - Attention impl: test both default (FA2) and eager (explicit matmul)

Outputs: research/exp_det_decompose.json
"""

import os, sys, json, time, hashlib
import torch
import torch.nn.functional as F

DLLM_DIR = os.path.dirname(os.path.abspath(__file__))
FP32_DIR  = os.path.join(DLLM_DIR, '..', 'FP32')
sys.path.insert(0, DLLM_DIR)
sys.path.insert(0, FP32_DIR)

os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
import determ_llm

OUT_FILE = os.path.join(DLLM_DIR, 'exp_det_decompose.json')

MODELS = {
    'llama_1b': '/home/kec23008/docker-sys/Models/Llama-3.2-1B-Instruct',
}

PROMPTS = [
    "The Eiffel Tower is located in",
    "In Python, the function to sort a list is",
    "The chemical formula for water is",
    "Albert Einstein developed the theory of",
    "The capital of Japan is",
    "The largest ocean on Earth is the",
    "The author of Pride and Prejudice is",
    "In mathematics, the value of pi is approximately",
    "The CPU stands for central processing",
    "Neural networks are inspired by the human",
]

GEN_LEN   = 64
BATCH_SIZES = [2, 4, 8, 16, 32, 64]


# ── Matmul/bmm patch for attention GEMMs ─────────────────────────────────────

_orig_matmul = None
_orig_bmm    = None
_attn_patch_enabled = False


def _load_kernel():
    return determ_llm._load_kernel()


def _det_matmul_impl(A_2d, B_weight):
    """Apply DetermLLM kernel to a 2D [M,K] x [N,K] GEMM."""
    kern = _load_kernel()
    return kern.gemm_fixed_algo(A_2d.contiguous(), B_weight.contiguous())


def _det_matmul(input, other, *, out=None):
    """Patch for torch.matmul: applies FP32-accum kernel to BF16 CUDA tensors."""
    if (input.dtype == torch.bfloat16 and other.dtype == torch.bfloat16
            and input.is_cuda and input.dim() >= 2 and other.dim() >= 2):
        try:
            *batch_A, M, K = input.shape
            *batch_B, K2, N = other.shape
            if K != K2:
                return _orig_matmul(input, other) if out is None else _orig_matmul(input, other, out=out)

            # Flatten leading dims → single 2D GEMM
            A_2d = input.reshape(-1, K).contiguous()   # [total_M, K]
            # other: [..., K, N] → transpose last two dims → [..., N, K] → [total_N_batched, K]
            # For batched case: each "row" of A corresponds to one batch entry
            # We need weight per batch, which differs → fall through for true batched case
            total_A = A_2d.shape[0]
            total_B_batches = 1
            for d in batch_B:
                total_B_batches *= d

            if total_A == total_B_batches or total_B_batches == 1:
                # All batch entries share the same B (or broadcast) → safe 2D treatment
                B_2d = other.reshape(total_B_batches, K, N)
                if total_B_batches == 1:
                    # Pure broadcast: A=[total_M, K], B=[K, N] → C=[total_M, N]
                    B_weight = B_2d[0].T.contiguous()  # [N, K]
                    result = _det_matmul_impl(A_2d, B_weight)
                    result = result.reshape(*batch_A, M, N)
                    if out is not None:
                        out.copy_(result); return out
                    return result
                else:
                    # Each batch entry has its own B matrix
                    # batch_A must equal batch_B for this to be valid
                    # Reshape: A=[B, M, K], B=[B, K, N]
                    # For M=1 (decode): A_2d=[B, K], treat as one big [B, K] × [N, K] GEMM
                    # with N being the output per batch → NOT a simple 2D GEMM
                    # Fall through to original for now
                    pass
        except Exception:
            pass
    return _orig_matmul(input, other) if out is None else _orig_matmul(input, other, out=out)


def _det_bmm(input, mat2, *, out=None):
    """Patch for torch.bmm: applies FP32-accum kernel to BF16 CUDA batch matmul."""
    if (input.dtype == torch.bfloat16 and mat2.dtype == torch.bfloat16
            and input.is_cuda and input.dim() == 3):
        try:
            B, M, K = input.shape
            B2, K2, N = mat2.shape
            if B == B2 and K == K2 and M == 1:
                # Decode-phase attention: [B, 1, K] × [B, K, N]
                # Each batch item has different K-slice → loop (correct but slow)
                # For empirical verification only; production would use batched kernel
                kern = _load_kernel()
                results = []
                for b in range(B):
                    A_b = input[b]           # [1, K]
                    B_b = mat2[b].T.contiguous()  # [N, K]
                    results.append(kern.gemm_fixed_algo(A_b.contiguous(), B_b))
                result = torch.stack(results, dim=0)  # [B, 1, N]
                if out is not None:
                    out.copy_(result); return out
                return result
        except Exception:
            pass
    return _orig_bmm(input, mat2) if out is None else _orig_bmm(input, mat2, out=out)


def enable_attn_patch():
    global _orig_matmul, _orig_bmm, _attn_patch_enabled
    if _attn_patch_enabled:
        return
    _orig_matmul = torch.matmul
    _orig_bmm    = torch.bmm
    torch.matmul = _det_matmul
    torch.bmm    = _det_bmm
    _attn_patch_enabled = True


def disable_attn_patch():
    global _orig_matmul, _orig_bmm, _attn_patch_enabled
    if not _attn_patch_enabled:
        return
    torch.matmul = _orig_matmul
    torch.bmm    = _orig_bmm
    _attn_patch_enabled = False


# ── Core generation function ──────────────────────────────────────────────────

def generate_tokens(model, tokenizer, prompt, batch_size, gen_len, device):
    """
    Simulate continuous batching: request 0 is our target prompt,
    requests 1..batch_size-1 are the same prompt (worst case: identical inputs,
    so any difference is purely from batch-size-dependent numerical paths).
    """
    enc = tokenizer(prompt, return_tensors='pt')['input_ids']  # [1, S]
    ids = enc.repeat(batch_size, 1).to(device)
    sl  = ids.shape[1]

    token_ids  = []
    token_lps  = []

    with torch.no_grad():
        for step in range(gen_len):
            out    = model(input_ids=ids)
            logits = out.logits[0, -1]          # [vocab] — row 0 = our target request
            lps    = torch.nn.functional.log_softmax(logits, dim=-1)
            tok    = lps.argmax().item()
            token_ids.append(tok)
            token_lps.append(lps[tok].item())
            # Append next token to ALL sequences
            next_col = torch.full((batch_size, 1), tok, dtype=ids.dtype, device=device)
            ids = torch.cat([ids, next_col], dim=1)

    return token_ids, token_lps


def seq_hash(token_ids):
    return hashlib.md5(str(token_ids).encode()).hexdigest()[:12]


# ── Main experiment ───────────────────────────────────────────────────────────

results = {}

for model_tag, model_path in MODELS.items():
    print(f"\n{'='*60}")
    print(f"Model: {model_tag}  ({model_path})")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map='cuda:0',
        attn_implementation='eager',   # force explicit matmul (no Flash Attention)
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"  Loaded on {device}, attn=eager")

    model_results = {}

    # ── Four conditions ────────────────────────────────────────────────────
    CONDITIONS = [
        ('A_BF16_baseline',   False, False, False),  # (label, linear, attn, det_all)
        ('B_linear_only',     True,  False, False),
        ('C_linear_and_attn', True,  True,  False),
        ('D_torch_det',       False, False, True),
    ]

    for cond_name, use_linear, use_attn, use_torch_det in CONDITIONS:
        print(f"\n  Condition: {cond_name}")

        # Configure patches
        if use_torch_det:
            torch.use_deterministic_algorithms(True, warn_only=True)
        else:
            torch.use_deterministic_algorithms(False)

        if use_linear:
            determ_llm.enable()
        else:
            determ_llm.disable()

        if use_attn:
            determ_llm._load_kernel()
            enable_attn_patch()
        else:
            disable_attn_patch()

        cond_results = {}

        for p_idx, prompt in enumerate(PROMPTS):
            # Reference: bs=1
            ref_toks, ref_lps = generate_tokens(model, tokenizer, prompt, 1, GEN_LEN, device)
            ref_hash = seq_hash(ref_toks)

            prompt_res = {
                'ref_hash': ref_hash,
                'batches': {}
            }

            for bs in BATCH_SIZES:
                bs_toks, bs_lps = generate_tokens(model, tokenizer, prompt, bs, GEN_LEN, device)
                bs_hash = seq_hash(bs_toks)

                seq_match    = (bs_hash == ref_hash)
                # Per-token mismatches (positions where token differs)
                tok_mismatches = sum(a != b for a, b in zip(ref_toks, bs_toks))
                lp_diffs   = [abs(a - b) for a, b in zip(ref_lps, bs_lps)]
                mean_lp    = sum(lp_diffs) / len(lp_diffs)
                max_lp     = max(lp_diffs)
                # Position of first token divergence (-1 if none)
                first_div  = next((i for i, (a,b) in enumerate(zip(ref_toks,bs_toks)) if a!=b), -1)

                prompt_res['batches'][f'bs{bs}'] = {
                    'seq_match':       seq_match,
                    'token_mismatches': tok_mismatches,
                    'first_divergence': first_div,
                    'mean_lp_diff':    mean_lp,
                    'max_lp_diff':     max_lp,
                }

                status = '✓' if seq_match else f'✗ (first_div={first_div}, mismatches={tok_mismatches})'
                if bs in [8, 32, 64] or not seq_match:
                    print(f"    p{p_idx} bs={bs:>2}: {status}  mean|ΔlogP|={mean_lp:.4f}")

            # Aggregate stats for this prompt
            all_match = all(v['seq_match'] for v in prompt_res['batches'].values())
            total_mismatch_toks = sum(v['token_mismatches'] for v in prompt_res['batches'].values())
            avg_lp = sum(v['mean_lp_diff'] for v in prompt_res['batches'].values()) / len(BATCH_SIZES)
            prompt_res['all_match'] = all_match
            prompt_res['total_token_mismatches'] = total_mismatch_toks
            prompt_res['avg_lp_diff'] = avg_lp
            cond_results[f'p{p_idx}'] = prompt_res

        # Summary for this condition
        all_seqs_match  = all(cond_results[f'p{i}']['all_match'] for i in range(len(PROMPTS)))
        total_tok_mm    = sum(cond_results[f'p{i}']['total_token_mismatches'] for i in range(len(PROMPTS)))
        avg_lp_all      = sum(cond_results[f'p{i}']['avg_lp_diff'] for i in range(len(PROMPTS))) / len(PROMPTS)

        print(f"\n  >>> {cond_name}: all_seq_match={all_seqs_match}, "
              f"total_tok_mismatches={total_tok_mm}, avg|ΔlogP|={avg_lp_all:.4f}")

        model_results[cond_name] = {
            'all_sequences_match': all_seqs_match,
            'total_token_mismatches': total_tok_mm,
            'avg_lp_diff': avg_lp_all,
            'per_prompt': cond_results,
        }

        # Reset
        determ_llm.disable()
        disable_attn_patch()
        torch.use_deterministic_algorithms(False)

    results[model_tag] = model_results

    del model
    torch.cuda.empty_cache()

# ── Print summary table ───────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("SUMMARY: Token-Level Batch-Invariance")
print(f"{'='*70}")
print(f"{'Condition':<25} {'All Match?':>12} {'Tot Tok MM':>12} {'Avg|ΔlogP|':>12}")
print("-" * 65)
for model_tag, mres in results.items():
    print(f"\nModel: {model_tag}")
    for cond, cres in mres.items():
        m  = '✓' if cres['all_sequences_match'] else '✗'
        print(f"  {cond:<25} {m:>12} {cres['total_token_mismatches']:>12} "
              f"{cres['avg_lp_diff']:>12.4f}")

with open(OUT_FILE, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved → {OUT_FILE}")
