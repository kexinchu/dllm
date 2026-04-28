"""Analyze the full matrix: models × datasets × batch sizes × methods.

For each (model, dataset, method), we have 3 configs at bs ∈ {1, 8, 32}.
Report per-config: accuracy, avg output length, runtime.
For each (model, dataset, method) aggregate:
  - mean accuracy across bs
  - std of accuracy across bs  (= Std@Acc over batch-size perturbation)
  - non-det rate: fraction of problems with different token sequences cross-bs
  - first-divergence token position (mean / median)

Produces exp_matrix/summary.{json,md}.
"""
import json
import os
import statistics
from collections import defaultdict

OUT = "/home/kec23008/docker-sys/dllm/research/exp_matrix"


def load_all():
    """Load every *.json in OUT, keyed by (model_short, dataset, method, bs)."""
    runs = {}
    for f in sorted(os.listdir(OUT)):
        if not f.endswith(".json") or f in ("summary.json",):
            continue
        path = os.path.join(OUT, f)
        with open(path) as fh:
            d = json.load(fh)
        m = d["meta"]
        key = (m["model_short"], m["dataset"], m["method"], m["batch_size"])
        runs[key] = d
    return runs


def cross_bs_stats(runs_for_same_mdm, bs_a, bs_b):
    """Compare two configs (bs_a vs bs_b) for the same (model, dataset, method).

    Returns: n_problems, same_toks, same_answer, first_div distribution.
    """
    ra = runs_for_same_mdm.get(bs_a)
    rb = runs_for_same_mdm.get(bs_b)
    if ra is None or rb is None:
        return None

    pa_list = ra["per_problem"]
    pb_list = rb["per_problem"]
    n = min(len(pa_list), len(pb_list))

    same_toks = 0
    same_answer = 0
    divs = []
    for i in range(n):
        pa = pa_list[i]; pb = pb_list[i]
        if pa["token_hash"] == pb["token_hash"]:
            same_toks += 1
            divs.append(-1)   # -1 signals "no divergence"
        else:
            ta = pa.get("first_100_tokens") or pa.get("first_20_tokens", [])
            tb = pb.get("first_100_tokens") or pb.get("first_20_tokens", [])
            div = next((j for j, (x, y) in enumerate(zip(ta, tb)) if x != y),
                       min(len(ta), len(tb)))
            divs.append(div)
        if pa["pred"] == pb["pred"]:
            same_answer += 1

    actual_divs = [d for d in divs if d >= 0]
    return {
        "n": n,
        "same_toks": same_toks,
        "same_answer": same_answer,
        "div_mean": statistics.mean(actual_divs) if actual_divs else None,
        "div_median": statistics.median(actual_divs) if actual_divs else None,
        "div_min": min(actual_divs) if actual_divs else None,
        "div_max": max(actual_divs) if actual_divs else None,
    }


def main():
    runs = load_all()
    print(f"Loaded {len(runs)} configs from {OUT}")
    if not runs:
        return

    # Group by (model, dataset, method) → dict of {bs: run}
    grouped = defaultdict(dict)
    for (model, dataset, method, bs), run in runs.items():
        grouped[(model, dataset, method)][bs] = run

    # Per-config accuracy / runtime table
    print("\n=== Per-config accuracy and runtime ===")
    fmt = "{:<32} {:<8} {:<22} {:>3} {:>6} {:>7} {:>5}"
    print(fmt.format("model", "dataset", "method", "bs", "acc", "len", "rt"))
    print("-" * 95)
    for (model, dataset, method), runs_by_bs in sorted(grouped.items()):
        for bs in sorted(runs_by_bs):
            r = runs_by_bs[bs]["aggregate"]
            print(fmt.format(
                model, dataset, method, bs,
                f"{r['accuracy']*100:.0f}%",
                f"{r['avg_output_length']:.0f}",
                f"{r['total_runtime_seconds']:.0f}s"
            ))

    # Aggregate: Std@Acc across batch sizes for each (model, dataset, method)
    print("\n=== Std@Acc across batch sizes (per model × dataset × method) ===")
    fmt = "{:<32} {:<8} {:<22} {:>3} {:>8} {:>8} {:>8} {:>8}"
    print(fmt.format("model", "dataset", "method", "n_bs", "acc@1", "acc@8", "acc@32", "std"))
    print("-" * 105)
    summary_rows = []
    for (model, dataset, method), runs_by_bs in sorted(grouped.items()):
        accs = {bs: runs_by_bs[bs]["aggregate"]["accuracy"] for bs in runs_by_bs}
        if len(accs) < 2: continue
        acc_vals = list(accs.values())
        std = statistics.stdev(acc_vals) if len(acc_vals) > 1 else 0.0
        row = {
            "model": model, "dataset": dataset, "method": method,
            "accs": accs, "std": std,
            "n_bs": len(accs),
        }
        summary_rows.append(row)
        a1 = f"{accs.get(1, 0)*100:.0f}%" if 1 in accs else "-"
        a8 = f"{accs.get(8, 0)*100:.0f}%" if 8 in accs else "-"
        a32 = f"{accs.get(32, 0)*100:.0f}%" if 32 in accs else "-"
        print(fmt.format(model, dataset, method, len(accs), a1, a8, a32, f"{std:.4f}"))

    # Cross-bs token-level: same_toks / same_answer / first-div-pos
    print("\n=== Cross-BS non-determinism (bs=1 vs bs=8) ===")
    fmt = "{:<32} {:<8} {:<22} {:>9} {:>10} {:>10} {:>10}"
    print(fmt.format("model", "dataset", "method", "same_toks", "same_ans", "div_med", "div_mean"))
    print("-" * 110)
    cross_results = []
    for (model, dataset, method), runs_by_bs in sorted(grouped.items()):
        s = cross_bs_stats(runs_by_bs, 1, 8)
        if s is None: continue
        cross_results.append({
            "model": model, "dataset": dataset, "method": method,
            "pair": "1v8", **s,
        })
        print(fmt.format(
            model, dataset, method,
            f"{s['same_toks']}/{s['n']}",
            f"{s['same_answer']}/{s['n']}",
            f"{s['div_median']:.1f}" if s['div_median'] is not None else "-",
            f"{s['div_mean']:.1f}" if s['div_mean'] is not None else "-",
        ))

    print("\n=== Cross-BS non-determinism (bs=8 vs bs=32) ===")
    print(fmt.format("model", "dataset", "method", "same_toks", "same_ans", "div_med", "div_mean"))
    print("-" * 110)
    for (model, dataset, method), runs_by_bs in sorted(grouped.items()):
        s = cross_bs_stats(runs_by_bs, 8, 32)
        if s is None: continue
        cross_results.append({
            "model": model, "dataset": dataset, "method": method,
            "pair": "8v32", **s,
        })
        print(fmt.format(
            model, dataset, method,
            f"{s['same_toks']}/{s['n']}",
            f"{s['same_answer']}/{s['n']}",
            f"{s['div_median']:.1f}" if s['div_median'] is not None else "-",
            f"{s['div_mean']:.1f}" if s['div_mean'] is not None else "-",
        ))

    # Save machine-readable summary
    out = {
        "n_configs": len(runs),
        "n_groups": len(grouped),
        "std_acc_per_mdm": summary_rows,
        "cross_bs": cross_results,
    }
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved -> {OUT}/summary.json")


if __name__ == "__main__":
    main()
