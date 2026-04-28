"""
Overhead benchmark: 5-way comparison for deterministic LLM inference.

Approaches:
  A) BF16 baseline          - vanilla, non-deterministic
  B) FP32 flag only         - allow_bf16_reduced_precision_reduction=False
  C) DetermLLM (ours)       - fixed split-K + FP32 accumulation [THE CONTRIBUTION]
  D) torch.use_deterministic_algorithms(True)  - approximates batch_invariant_ops
  E) Full FP32 (LayerCast)  - all weights cast to FP32

For each approach, measures:
  1. Is output deterministic across batch sizes 1..32? (hash comparison)
  2. Latency overhead vs BF16 baseline

Output: research/exp_overhead_bench.json
"""

import os, sys, json, time, hashlib, warnings
import torch
import torch.nn.functional as F

warnings.filterwarnings('ignore')

# ── paths ──────────────────────────────────────────────────────────────────
DLLM_DIR   = os.path.dirname(os.path.abspath(__file__))
FP32_DIR   = os.path.join(DLLM_DIR, '..', 'FP32')
MODEL_PATH = '/home/kec23008/docker-sys/Models/Phi-4'
OUT_FILE   = os.path.join(DLLM_DIR, 'exp_overhead_triton.json')

sys.path.insert(0, DLLM_DIR)
sys.path.insert(0, FP32_DIR)

# ── model loading ───────────────────────────────────────────────────────────
print("Loading model (BF16)...")
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map='cuda:0'
)
model.eval()
device = next(model.parameters()).device
print(f"  Model on {device}, dtype={next(model.parameters()).dtype}")

# ── test prompts ────────────────────────────────────────────────────────────
PROMPTS = [
    "What is deterministic inference in large language models?",
    "Explain the concept of batch processing in neural networks.",
    "How does attention mechanism work in transformers?",
    "Write a short poem about computation.",
    "Summarize the key ideas behind the transformer architecture.",
]
GEN_LEN   = 32
TEST_BS   = [1, 2, 4, 8, 16, 32]
WARMUP    = 3
BENCH_RUNS = 10

# ── LayerCast patcher for approach E ───────────────────────────────────────
_lc_originals = {}

def layercast_enable(model):
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            _lc_originals[name] = mod.forward
            def _make(m):
                def fwd(x):
                    return F.linear(
                        x.float(), m.weight.float(),
                        m.bias.float() if m.bias is not None else None
                    ).to(x.dtype)
                return fwd
            mod.forward = _make(mod)

def layercast_disable(model):
    for name, mod in model.named_modules():
        if name in _lc_originals:
            mod.forward = _lc_originals[name]
    _lc_originals.clear()

# ── core measurement functions ──────────────────────────────────────────────

def gen_tokens(prompt, bs, gen_len):
    """Generate gen_len tokens for a prompt at batch size bs. Returns list of token ids."""
    input_ids = tokenizer(prompt, return_tensors='pt')['input_ids']
    sl = input_ids.shape[1]
    ids = input_ids.repeat(bs, 1).to(device)
    pos = torch.arange(sl, device=device).unsqueeze(0).repeat(bs, 1)
    tokens = []
    with torch.no_grad():
        for step in range(gen_len):
            out = model(input_ids=ids, position_ids=pos).logits
            tok = out[0, -1].argmax().item()
            tokens.append(tok)
            ids = torch.tensor([[tok]], device=device).repeat(bs, 1)
            pos = torch.tensor([[sl + step]], device=device).repeat(bs, 1)
    return tokens


def token_hash(tokens):
    return hashlib.md5(str(tokens).encode()).hexdigest()[:12]


def measure_determinism(prompt, test_bs, gen_len):
    """
    Check if output is consistent across batch sizes.
    Returns: (is_deterministic, n_mismatches, hashes_dict)
    """
    ref = gen_tokens(prompt, 1, gen_len)
    ref_h = token_hash(ref)
    hashes = {1: ref_h}
    mismatches = 0
    for bs in test_bs:
        if bs == 1:
            continue
        h = token_hash(gen_tokens(prompt, bs, gen_len))
        hashes[bs] = h
        if h != ref_h:
            mismatches += 1
    is_det = (mismatches == 0)
    return is_det, mismatches, hashes


def measure_latency(prompt, bs, gen_len, warmup, n_runs):
    """Returns mean ± std latency in ms for generating gen_len tokens."""
    # warmup
    for _ in range(warmup):
        gen_tokens(prompt, bs, gen_len)
    torch.cuda.synchronize()
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        gen_tokens(prompt, bs, gen_len)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    import statistics
    return statistics.mean(times), statistics.stdev(times) if len(times) > 1 else 0.0


# ── approach definitions ────────────────────────────────────────────────────
import determ_llm

approaches = [
    ("A_BF16_baseline",     "BF16 baseline (non-det)"),
    ("B_FP32_flag",         "FP32 flag only"),
    ("C_DetermLLM_cublaslt","DetermLLM cuBLASLt"),
    ("F_DetermLLM_triton",  "DetermLLM Triton (ours)"),
    ("D_TorchDet",          "torch.use_deterministic_algorithms"),
]

results = {}

for approach_id, approach_name in approaches:
    print(f"\n{'='*60}")
    print(f"Approach {approach_id}: {approach_name}")
    print(f"{'='*60}")

    # ── setup ────────────────────────────────────────────────────────────
    if approach_id == "A_BF16_baseline":
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        determ_llm.disable()
        torch.use_deterministic_algorithms(False, warn_only=True)

    elif approach_id == "B_FP32_flag":
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        determ_llm.disable()
        torch.use_deterministic_algorithms(False, warn_only=True)

    elif approach_id == "C_DetermLLM_cublaslt":
        determ_llm.enable(backend='cublaslt')
        torch.use_deterministic_algorithms(False, warn_only=True)

    elif approach_id == "F_DetermLLM_triton":
        determ_llm.enable(backend='triton')
        torch.use_deterministic_algorithms(False, warn_only=True)

    elif approach_id == "D_TorchDet":
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        determ_llm.disable()
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.use_deterministic_algorithms(True, warn_only=True)

    elif approach_id == "E_FullFP32":
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        determ_llm.disable()
        torch.use_deterministic_algorithms(False, warn_only=True)
        layercast_enable(model)

    # ── determinism test (all prompts, bs 1..32) ────────────────────────
    det_results = {}
    total_mismatches = 0
    for i, prompt in enumerate(PROMPTS):
        is_det, n_mis, hashes = measure_determinism(prompt, TEST_BS, GEN_LEN)
        det_results[f"p{i}"] = {
            "is_deterministic": is_det,
            "n_mismatches": n_mis,
            "hashes": hashes,
        }
        total_mismatches += n_mis
        status = "✓ DET" if is_det else f"✗ {n_mis}/{len(TEST_BS)-1} mismatch"
        print(f"  P{i+1}: {status}  hashes={list(hashes.values())}")

    overall_det = (total_mismatches == 0)
    print(f"  → Overall: {'DETERMINISTIC' if overall_det else 'NON-DETERMINISTIC'} "
          f"({total_mismatches} mismatches across {len(PROMPTS)} prompts)")

    # ── latency benchmark (prompt 0, bs 1, 8, 32) ──────────────────────
    latency = {}
    for bs in [1, 8, 32]:
        mean_ms, std_ms = measure_latency(PROMPTS[0], bs, GEN_LEN, WARMUP, BENCH_RUNS)
        latency[bs] = {"mean_ms": mean_ms, "std_ms": std_ms}
        print(f"  bs={bs:>2}: {mean_ms:.1f} ± {std_ms:.1f} ms")

    results[approach_id] = {
        "name": approach_name,
        "deterministic": overall_det,
        "total_mismatches": total_mismatches,
        "det_per_prompt": det_results,
        "latency": latency,
    }

    # ── teardown ──────────────────────────────────────────────────────────
    if approach_id in ("C_DetermLLM_cublaslt", "F_DetermLLM_triton"):
        determ_llm.disable()
    elif approach_id == "D_TorchDet":
        torch.use_deterministic_algorithms(False, warn_only=True)
        if 'CUBLAS_WORKSPACE_CONFIG' in os.environ:
            del os.environ['CUBLAS_WORKSPACE_CONFIG']
    elif approach_id == "E_FullFP32":
        layercast_disable(model)

# ── compute overheads vs baseline ──────────────────────────────────────────
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
baseline_lat = {bs: results["A_BF16_baseline"]["latency"][bs]["mean_ms"]
                for bs in [1, 8, 32]}

print(f"{'Approach':<35} {'Det?':<6} {'bs=1':>8} {'bs=8':>8} {'bs=32':>8}")
print("-" * 70)
for aid, aname in approaches:
    r = results[aid]
    det = "YES" if r["deterministic"] else "NO"
    overheads = []
    for bs in [1, 8, 32]:
        oh = (r["latency"][bs]["mean_ms"] / baseline_lat[bs] - 1) * 100
        overheads.append(f"{oh:+.1f}%")
    print(f"{aid:<35} {det:<6} {overheads[0]:>8} {overheads[1]:>8} {overheads[2]:>8}")

    # add overhead to results
    results[aid]["overhead_vs_baseline"] = {
        bs: (results[aid]["latency"][bs]["mean_ms"] / baseline_lat[bs] - 1) * 100
        for bs in [1, 8, 32]
    }

# ── save ────────────────────────────────────────────────────────────────────
with open(OUT_FILE, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {OUT_FILE}")
