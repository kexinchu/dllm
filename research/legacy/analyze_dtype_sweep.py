"""Analyse exp_matrix_dtype/ — Avg_Std@top1_prob and first-divergence-index
across 7 batch sizes per (model, method).

Metrics
-------
Avg_Std@top1_prob : for each (problem, token-position j) compute the
                    population std of the top-1 logprob across bs in the
                    set (e.g. {1,4,8,16,32,64,128}), then average over j
                    and over problems.

first_div_idx     : for each problem, index of the earliest decoded token
                    that differs across any two bs values. If all bs
                    produce identical sequences, reported as the common
                    output length.
"""
import json
import os
import statistics
from collections import defaultdict

OUT = "/home/kec23008/docker-sys/dllm/research/exp_matrix_dtype"

BS_LIST = [1, 4, 8, 16, 32, 64, 128]

MODEL_SHORT = {
    "DeepSeek-R1-Distill-Qwen-7B": "DeepSeek-7B",
    "Llama-3.1-8B-Instruct":       "Llama-8B",
    "Phi-4":                       "Phi-4",
}
METHOD_LABEL = {
    ("bf16_baseline", "bfloat16"): "BF16",
    ("bf16_baseline", "float32"):  "FP32-full",
    ("dllm_hybrid",   "bfloat16"): "DetermLLM",
}


def load_all():
    """runs[(model, method_label)] -> dict[bs]->run"""
    runs = defaultdict(dict)
    for f in sorted(os.listdir(OUT)):
        if not f.endswith(".json") or f.startswith("summary"):
            continue
        path = os.path.join(OUT, f)
        try:
            with open(path) as fh:
                d = json.load(fh)
        except json.JSONDecodeError:
            continue
        m = d["meta"]
        dtype = m.get("dtype", "bfloat16")
        key = (m["model_short"], METHOD_LABEL[(m["method"], dtype)])
        runs[key][m["batch_size"]] = d
    return runs


def avg_std_top1(runs_by_bs, bs_set):
    """Across bs in bs_set, compute per-position std of top1 logprob; mean."""
    have = [bs for bs in bs_set if bs in runs_by_bs]
    if len(have) < 2:
        return None, 0
    all_lps = [runs_by_bs[bs]["per_problem"] for bs in have]
    n = min(len(pp) for pp in all_lps)
    problem_stds = []
    for i in range(n):
        lps_per_bs = [pp[i].get("top1_logprobs", []) for pp in all_lps]
        L = min(len(x) for x in lps_per_bs)
        if L == 0:
            continue
        pos_stds = [statistics.pstdev([lps_per_bs[k][j] for k in range(len(have))])
                    for j in range(L)]
        problem_stds.append(statistics.mean(pos_stds))
    return (statistics.mean(problem_stds) if problem_stds else 0.0), n


def first_div_idx(runs_by_bs, bs_set):
    """For each problem, find earliest token position where any two bs differ."""
    have = [bs for bs in bs_set if bs in runs_by_bs]
    if len(have) < 2:
        return None
    all_pp = [runs_by_bs[bs]["per_problem"] for bs in have]
    n = min(len(pp) for pp in all_pp)
    idxs = []
    for i in range(n):
        toks_per_bs = [pp[i].get("token_ids") or pp[i].get("first_100_tokens", [])
                       for pp in all_pp]
        L = min(len(t) for t in toks_per_bs)
        div = L  # default: never diverge within window
        for j in range(L):
            if len(set(t[j] for t in toks_per_bs)) > 1:
                div = j; break
        idxs.append(div)
    return idxs


def main():
    runs = load_all()
    print(f"Loaded {sum(len(v) for v in runs.values())} configs, "
          f"{len(runs)} (model, method) groups\n")

    print("=== Avg_Std@top1_prob across bs ∈ {" + ",".join(str(b) for b in BS_LIST) + "} ===")
    fmt = "{:<14} {:<10} {:>8} {:>14} {:>14} {:>14}"
    print(fmt.format("model", "method", "bs_have", "avg_std", "div_idx_med", "div_idx_mean"))
    print("-" * 90)
    rows = []
    for (model, meth), runs_by_bs in sorted(runs.items()):
        avgstd, n = avg_std_top1(runs_by_bs, BS_LIST)
        divs = first_div_idx(runs_by_bs, BS_LIST)
        div_med = statistics.median(divs) if divs else None
        div_mean = statistics.mean(divs) if divs else None
        rows.append({
            "model": model, "method": meth,
            "bs_have": sorted(runs_by_bs.keys()),
            "avg_std_top1": avgstd,
            "div_idx_median": div_med,
            "div_idx_mean": div_mean,
            "div_idx_values": divs,
        })
        print(fmt.format(
            model[:14], meth, len(runs_by_bs),
            f"{avgstd:.4e}" if avgstd is not None else "-",
            f"{div_med:.0f}" if div_med is not None else "-",
            f"{div_mean:.1f}" if div_mean is not None else "-",
        ))

    # Per-bs accuracy + runtime summary
    print("\n=== Per-config accuracy & runtime ===")
    fmt2 = "{:<14} {:<10} {:>4} {:>6} {:>7} {:>8}"
    print(fmt2.format("model", "method", "bs", "acc", "len", "rt"))
    print("-" * 55)
    for (model, meth), runs_by_bs in sorted(runs.items()):
        for bs in sorted(runs_by_bs):
            r = runs_by_bs[bs]["aggregate"]
            print(fmt2.format(model[:14], meth, bs,
                              f"{r['accuracy']*100:.0f}%",
                              f"{r['avg_output_length']:.0f}",
                              f"{r['total_runtime_seconds']:.0f}s"))

    out_path = os.path.join(OUT, "summary_dtype.json")
    with open(out_path, "w") as f:
        json.dump({"rows": rows, "bs_list": BS_LIST}, f, indent=2, default=str)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
