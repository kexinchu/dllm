#!/usr/bin/env python3
"""
E7 — Overhead profile (plan.md §3 / Priority P0).

Single synthetic prompt (~100 input tokens), 256 output tokens, batch sizes
{1, 8, 32}. For every (model × scheme × bs) triple measure:

  TTFT (ms)            — prefill only, via generate(max_new_tokens=1)
  TPOT (ms/token)      — (total_gen_time − TTFT) / (max_new − 1)
  Throughput (tok/s)   — bs × max_new / total_gen_time
  Peak memory (GB)     — torch.cuda.max_memory_allocated after a full run

Schemes (plan.md §0): BF16 / FP32flag / LayerCast / DetermLLM / DetermLLM+attn.
FP32full reloads weights as float32; skipped if OOM.

Warmup: 2 runs / measurement: 3 runs (median reported).
"""
import sys, os, time, json, gc, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings; warnings.filterwarnings("ignore")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from research import determ_llm
from motivation.test_layercast_latency import apply_layercast, remove_layercast

MODELS = {
    "Llama-3.1-8B":        "/home/kec23008/docker-sys/Models/Llama-3.1-8B-Instruct",
    "DeepSeek-R1-Qwen-7B": "/home/kec23008/docker-sys/Models/DeepSeek-R1-Distill-Qwen-7B",
}

PROMPT = (
    "Explain the concept of deterministic inference in large language models. "
    "Discuss why non-determinism arises in batched inference, what hardware "
    "and software factors contribute to it, and how this affects reproducibility "
    "in production serving systems. Please provide a detailed technical analysis "
    "covering GPU kernels, numerical precision, and batching strategies."
)

MAX_NEW   = 256
BS_LIST   = [1, 8, 32]
N_WARMUP  = 2
N_MEASURE = 3


# ── scheme hooks ──────────────────────────────────────────────────────────────
def make_schemes(model):
    """Return list of (name, enter_fn, exit_fn). All run on current BF16 weights."""
    state = {}

    def bf16_enter():
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    def bf16_exit():
        pass

    def fp32flag_enter():
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    def fp32flag_exit():
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True

    def lc_enter():
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        state["lc_orig"] = apply_layercast(model)
    def lc_exit():
        remove_layercast(model, state.pop("lc_orig"))

    def det_enter():
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        determ_llm.enable(backend="triton")
    def det_exit():
        determ_llm.disable()

    def det_attn_enter():
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        determ_llm.enable(backend="triton", attn=True)
    def det_attn_exit():
        determ_llm.disable()

    return [
        ("BF16",          bf16_enter,     bf16_exit),
        ("FP32flag",      fp32flag_enter, fp32flag_exit),
        ("LayerCast",     lc_enter,       lc_exit),
        ("DetermLLM",     det_enter,      det_exit),
        ("DetermLLM+attn",det_attn_enter, det_attn_exit),
    ]


# ── measurement primitive ─────────────────────────────────────────────────────
def measure_one(model, tok, bs, device):
    """Run one (bs, max_new=256) generation suite. Returns dict of metrics."""
    enc = tok(PROMPT, return_tensors="pt")["input_ids"].to(device)
    ids = enc.repeat(bs, 1).contiguous()
    attn_mask = torch.ones_like(ids)

    # ── warmup ──
    for _ in range(N_WARMUP):
        with torch.no_grad():
            model.generate(input_ids=ids, attention_mask=attn_mask,
                           max_new_tokens=MAX_NEW, do_sample=False,
                           pad_token_id=tok.pad_token_id)
    torch.cuda.synchronize()

    # ── TTFT = generate(max_new=1) ──
    torch.cuda.reset_peak_memory_stats()
    ttft_list = []
    for _ in range(N_MEASURE):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model.generate(input_ids=ids, attention_mask=attn_mask,
                           max_new_tokens=1, do_sample=False,
                           pad_token_id=tok.pad_token_id)
        torch.cuda.synchronize()
        ttft_list.append((time.perf_counter() - t0) * 1000.0)

    # ── Full gen: total_time → derive TPOT, throughput ──
    wall_list = []
    for _ in range(N_MEASURE):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model.generate(input_ids=ids, attention_mask=attn_mask,
                           max_new_tokens=MAX_NEW, do_sample=False,
                           pad_token_id=tok.pad_token_id)
        torch.cuda.synchronize()
        wall_list.append((time.perf_counter() - t0) * 1000.0)

    peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    ttft = sorted(ttft_list)[len(ttft_list) // 2]          # median
    wall = sorted(wall_list)[len(wall_list) // 2]
    tpot = (wall - ttft) / (MAX_NEW - 1)                   # ms / token
    throughput = bs * MAX_NEW / (wall / 1000.0)            # tokens / sec

    return {
        "bs":            bs,
        "input_tokens":  int(enc.shape[1]),
        "output_tokens": MAX_NEW,
        "ttft_ms":       ttft,
        "tpot_ms":       tpot,
        "wall_ms":       wall,
        "throughput_tps": throughput,
        "peak_mem_gb":   peak_mem_gb,
        "ttft_list":     ttft_list,
        "wall_list":     wall_list,
    }


# ── per-model runner ──────────────────────────────────────────────────────────
def run_model(model_name, model_path, out_path, include_fp32full):
    print(f"\n{'=' * 84}\n  MODEL: {model_name}\n{'=' * 84}", flush=True)

    tok = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ── BF16 load: used by all except FP32full ──
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation="sdpa",
    )
    model.eval()
    device = next(model.parameters()).device

    schemes = make_schemes(model)
    per_model = {"model": model_name, "model_path": model_path, "results": []}

    for scheme_name, ein, eout in schemes:
        print(f"\n-- scheme: {scheme_name} --", flush=True)
        ein()
        try:
            for bs in BS_LIST:
                try:
                    m = measure_one(model, tok, bs, device)
                    m["scheme"] = scheme_name
                    per_model["results"].append(m)
                    print(f"   bs={bs:>2}: TTFT={m['ttft_ms']:7.1f}ms  "
                          f"TPOT={m['tpot_ms']:6.2f}ms  "
                          f"thr={m['throughput_tps']:7.1f} tok/s  "
                          f"peak={m['peak_mem_gb']:.2f} GB", flush=True)
                except torch.cuda.OutOfMemoryError:
                    print(f"   bs={bs:>2}: OOM — skipped", flush=True)
                    torch.cuda.empty_cache(); gc.collect()
        finally:
            eout()
            torch.cuda.empty_cache(); gc.collect()

        # incremental save
        with open(out_path, "w") as f:
            json.dump(per_model, f, indent=2)

    # ── FP32full — reload model in fp32 ──
    if include_fp32full:
        print(f"\n-- scheme: FP32full (reloading weights as fp32) --", flush=True)
        del model; torch.cuda.empty_cache(); gc.collect()
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, dtype=torch.float32, device_map={"": 0},
                attn_implementation="sdpa",
            )
            model.eval()
            device = next(model.parameters()).device
            for bs in BS_LIST:
                try:
                    m = measure_one(model, tok, bs, device)
                    m["scheme"] = "FP32full"
                    per_model["results"].append(m)
                    print(f"   bs={bs:>2}: TTFT={m['ttft_ms']:7.1f}ms  "
                          f"TPOT={m['tpot_ms']:6.2f}ms  "
                          f"thr={m['throughput_tps']:7.1f} tok/s  "
                          f"peak={m['peak_mem_gb']:.2f} GB", flush=True)
                except torch.cuda.OutOfMemoryError:
                    print(f"   bs={bs:>2}: OOM — skipped", flush=True)
                    torch.cuda.empty_cache(); gc.collect()
        except torch.cuda.OutOfMemoryError:
            print("   FP32full model load OOM — skipped", flush=True)
        finally:
            try: del model
            except Exception: pass
            torch.cuda.empty_cache(); gc.collect()

        with open(out_path, "w") as f:
            json.dump(per_model, f, indent=2)

    return per_model


# ── formatted summary ─────────────────────────────────────────────────────────
def print_summary(all_results):
    print("\n" + "=" * 100)
    print("  E7 OVERHEAD SUMMARY  (median of 3 runs)")
    print("=" * 100)
    for m in all_results:
        print(f"\n  [{m['model']}]")
        print(f"  {'Scheme':<18} {'bs':>3} {'TTFT ms':>10} {'TPOT ms':>10} "
              f"{'Thr tok/s':>11} {'Peak GB':>9} {'TTFT×':>8} {'TPOT×':>8}")
        # pick BF16 bs=1 as baseline row per model
        bf16_rows = {(r["scheme"], r["bs"]): r for r in m["results"] if r["scheme"] == "BF16"}
        for r in m["results"]:
            bf = bf16_rows.get(("BF16", r["bs"]))
            ttft_x = r["ttft_ms"] / bf["ttft_ms"] if bf else 1.0
            tpot_x = r["tpot_ms"] / bf["tpot_ms"] if bf else 1.0
            print(f"  {r['scheme']:<18} {r['bs']:>3} {r['ttft_ms']:>10.1f} "
                  f"{r['tpot_ms']:>10.2f} {r['throughput_tps']:>11.1f} "
                  f"{r['peak_mem_gb']:>9.2f} {ttft_x:>7.2f}x {tpot_x:>7.2f}x")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS.keys()) + ["all"], default="all")
    ap.add_argument("--out-dir", default="/home/kec23008/docker-sys/dllm/research/exp_E7")
    ap.add_argument("--no-fp32full", action="store_true",
                    help="skip FP32full (saves ~32GB GPU mem + 10min)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"GPU: {torch.cuda.get_device_name(0)}  "
          f"free={torch.cuda.mem_get_info()[0] / 1e9:.1f} GB", flush=True)

    targets = list(MODELS.items()) if args.model == "all" else [(args.model, MODELS[args.model])]
    all_results = []
    for name, path in targets:
        out = os.path.join(args.out_dir, f"E7_{name.replace('/', '_')}.json")
        r = run_model(name, path, out, include_fp32full=not args.no_fp32full)
        all_results.append(r)

    combined = os.path.join(args.out_dir, "E7_all.json")
    with open(combined, "w") as f:
        json.dump(all_results, f, indent=2)
    print_summary(all_results)
    print(f"\n  Saved: {combined}")


if __name__ == "__main__":
    main()
