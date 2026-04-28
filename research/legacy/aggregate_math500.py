"""Aggregate overnight MATH500 eval into Std@Acc / Div_Index / length variance tables.

Run in the morning once run_overnight.sh finishes. Produces:
  - exp_math500_overnight/aggregate.json  (machine readable)
  - exp_math500_overnight/aggregate.md    (paper-ready tables)
"""
import hashlib
import json
import os
import statistics
from collections import defaultdict

RES_DIR = "/home/kec23008/docker-sys/dllm/research/exp_math500_overnight"


def load_all():
    """Load every *.json in the results directory, keyed by (method, bs, seed)."""
    runs = {}
    for f in sorted(os.listdir(RES_DIR)):
        if not f.endswith(".json") or f == "aggregate.json":
            continue
        path = os.path.join(RES_DIR, f)
        with open(path) as fh:
            data = json.load(fh)
        m = data["meta"]
        key = (m["method"], m["batch_size"], m["seed"])
        runs[key] = data
    return runs


def compute_divergence_index(runs_same_method):
    """For each problem, find first token where runs diverge.

    runs_same_method: list of run-dicts, each containing per_problem responses.
    Returns median and mean divergence index across problems.
    """
    if len(runs_same_method) < 2:
        return None

    # Each run has per_problem[i]["first_20_tokens"]; we use token_hash for quick
    # equality check and first_20_tokens for early-divergence analysis.
    n_probs = len(runs_same_method[0]["per_problem"])
    div_indices = []
    for i in range(n_probs):
        hashes = [r["per_problem"][i]["token_hash"] for r in runs_same_method]
        if len(set(hashes)) == 1:
            # all equal: divergence index = output length (never diverged)
            length = runs_same_method[0]["per_problem"][i]["output_len"]
            div_indices.append(length)
        else:
            # Find first token index where they differ (using first_20 tokens)
            toks_lists = [r["per_problem"][i]["first_20_tokens"] for r in runs_same_method]
            div_at = None
            max_check = min(len(t) for t in toks_lists)
            for j in range(max_check):
                vals = [t[j] for t in toks_lists]
                if len(set(vals)) > 1:
                    div_at = j
                    break
            if div_at is None:
                div_at = max_check
            div_indices.append(div_at)
    return {
        "median": statistics.median(div_indices),
        "mean": statistics.mean(div_indices),
        "min": min(div_indices),
        "max": max(div_indices),
    }


def main():
    runs = load_all()
    print(f"Loaded {len(runs)} runs from {RES_DIR}")

    if not runs:
        print("No results found. Has the overnight pipeline finished?")
        return

    # Group runs: for each method, we have (bs, seed) combinations
    per_method_bs = defaultdict(list)   # (method, bs) -> list of runs
    per_method = defaultdict(list)      # method -> list of runs across bs,seed
    for (method, bs, seed), run in runs.items():
        per_method_bs[(method, bs)].append(run)
        per_method[method].append(run)

    # Std@Acc: std of accuracy across all (bs, seed) configs for each method
    print("\n=== Std@Acc (across batch sizes and seeds) ===")
    std_acc = {}
    for method, rlist in per_method.items():
        accs = [r["aggregate"]["accuracy"] for r in rlist]
        std_acc[method] = {
            "mean": statistics.mean(accs) if accs else 0.0,
            "std": statistics.stdev(accs) if len(accs) > 1 else 0.0,
            "min": min(accs) if accs else 0.0,
            "max": max(accs) if accs else 0.0,
            "n_configs": len(accs),
        }
        print(f"  {method:<20}  mean={std_acc[method]['mean']:.4f} "
              f"std={std_acc[method]['std']:.4f} "
              f"range=[{std_acc[method]['min']:.4f}, {std_acc[method]['max']:.4f}] "
              f"n={std_acc[method]['n_configs']}")

    # Avg_Std@Output_Length: per-problem std of output length across configs,
    # averaged over problems
    print("\n=== Avg_Std@Output_Length (per-problem std, averaged) ===")
    length_var = {}
    for method, rlist in per_method.items():
        if len(rlist) < 2:
            length_var[method] = None
            continue
        n_probs = len(rlist[0]["per_problem"])
        per_prob_stds = []
        for i in range(n_probs):
            lens = [r["per_problem"][i]["output_len"] for r in rlist]
            if len(lens) > 1:
                per_prob_stds.append(statistics.stdev(lens))
        length_var[method] = {
            "mean_std": statistics.mean(per_prob_stds),
            "median_std": statistics.median(per_prob_stds),
        }
        print(f"  {method:<20}  mean_std={length_var[method]['mean_std']:.2f} "
              f"median_std={length_var[method]['median_std']:.2f}")

    # Div_Index: median token position at first divergence across configs
    print("\n=== Div_Index (first diverging token, per method) ===")
    div_stats = {}
    for method, rlist in per_method.items():
        d = compute_divergence_index(rlist)
        div_stats[method] = d
        if d:
            print(f"  {method:<20}  median={d['median']:.1f} mean={d['mean']:.1f} "
                  f"range=[{d['min']}, {d['max']}]")

    # Avg runtime per method
    print("\n=== Average runtime per eval config ===")
    runtime_stats = {}
    for method, rlist in per_method.items():
        rts = [r["aggregate"]["total_runtime_seconds"] for r in rlist]
        runtime_stats[method] = {
            "mean_s": statistics.mean(rts),
            "std_s": statistics.stdev(rts) if len(rts) > 1 else 0.0,
        }
        print(f"  {method:<20}  {runtime_stats[method]['mean_s']:.1f}s "
              f"(±{runtime_stats[method]['std_s']:.1f}s)")

    # Write machine-readable aggregate
    out_json = os.path.join(RES_DIR, "aggregate.json")
    agg = {
        "std_acc": std_acc,
        "length_var": length_var,
        "div_index": div_stats,
        "runtime": runtime_stats,
        "n_runs": len(runs),
    }
    with open(out_json, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"\nSaved -> {out_json}")

    # Write paper-ready markdown
    out_md = os.path.join(RES_DIR, "aggregate.md")
    with open(out_md, "w") as f:
        f.write("# MATH500 overnight results (DeepSeek-R1-Distill-Qwen-7B)\n\n")
        f.write("## Std@Acc across configurations\n\n")
        f.write("| Method | Mean Acc | Std@Acc | Min | Max | n |\n")
        f.write("|---|---|---|---|---|---|\n")
        for method in ["bf16_baseline", "layercast", "dllm_cublaslt", "dllm_triton"]:
            if method in std_acc:
                s = std_acc[method]
                f.write(f"| {method} | {s['mean']:.2%} | {s['std']:.4f} | "
                        f"{s['min']:.2%} | {s['max']:.2%} | {s['n_configs']} |\n")

        f.write("\n## Avg_Std@Output_Length\n\n")
        f.write("| Method | Mean per-problem std |\n|---|---|\n")
        for method in ["bf16_baseline", "layercast", "dllm_cublaslt", "dllm_triton"]:
            if method in length_var and length_var[method]:
                f.write(f"| {method} | {length_var[method]['mean_std']:.1f} |\n")

        f.write("\n## Div_Index (first diverging token)\n\n")
        f.write("| Method | Median | Mean |\n|---|---|---|\n")
        for method in ["bf16_baseline", "layercast", "dllm_cublaslt", "dllm_triton"]:
            if method in div_stats and div_stats[method]:
                d = div_stats[method]
                f.write(f"| {method} | {d['median']:.1f} | {d['mean']:.1f} |\n")

        f.write("\n## Runtime per eval config (seconds)\n\n")
        f.write("| Method | Mean | Std |\n|---|---|---|\n")
        for method in ["bf16_baseline", "layercast", "dllm_cublaslt", "dllm_triton"]:
            if method in runtime_stats:
                r = runtime_stats[method]
                f.write(f"| {method} | {r['mean_s']:.1f} | {r['std_s']:.1f} |\n")

    print(f"Saved -> {out_md}")


if __name__ == "__main__":
    main()
