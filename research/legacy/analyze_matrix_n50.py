"""Phase-2 analysis: n=50 matrix with full token_ids + top1_logprobs.

Computes Yuan et al. 2025-style metrics:
  - Std@Acc           : std of per-config accuracy across batch sizes.
  - Div_Index         : first-divergence token position (no 100-cap).
  - Avg_Std@top1_prob : for each common-prefix position j, std over bs of
                        the top-1 logprob; averaged over j then over problems.
  - Avg_Std@Out_Len   : std of output length across bs, averaged over problems.

Writes exp_matrix_n50/summary_n50.{json,md}.
"""
import json
import math
import os
import statistics
from collections import defaultdict

OUT = "/home/kec23008/docker-sys/dllm/research/exp_matrix_n50"


# ── Loaders ─────────────────────────────────────────────────────────────────
def load_all():
    runs = {}
    for f in sorted(os.listdir(OUT)):
        if not f.endswith(".json") or f.startswith("summary"):
            continue
        path = os.path.join(OUT, f)
        try:
            with open(path) as fh:
                d = json.load(fh)
        except json.JSONDecodeError:
            print(f"  skip malformed {f}")
            continue
        m = d["meta"]
        key = (m["model_short"], m["dataset"], m["method"], m["batch_size"])
        runs[key] = d
    return runs


# ── Pairwise metrics ───────────────────────────────────────────────────────
def first_divergence(pa, pb):
    """Compare two per_problem entries; return first-divergence token index
    (relative to decoded position 0). If sequences fully agree, returns the
    length of the shorter sequence."""
    if pa["token_hash"] == pb["token_hash"]:
        return min(pa["output_len"], pb["output_len"])
    ta = pa.get("token_ids") or pa.get("first_100_tokens", [])
    tb = pb.get("token_ids") or pb.get("first_100_tokens", [])
    for j, (x, y) in enumerate(zip(ta, tb)):
        if x != y:
            return j
    return min(len(ta), len(tb))


def std_at_top1(pa, pb, pc):
    """Across 3 bs variants of the same problem, compute mean-over-prefix
    std of top1_logprob. Returns (avg_std, prefix_len)."""
    lps = [p.get("top1_logprobs", []) for p in (pa, pb, pc)]
    if not all(lps):
        return None, 0
    n = min(len(x) for x in lps)
    if n == 0:
        return 0.0, 0
    stds = []
    for j in range(n):
        vals = [lps[k][j] for k in range(3)]
        stds.append(statistics.pstdev(vals))
    return statistics.mean(stds), n


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    runs = load_all()
    print(f"Loaded {len(runs)} configs from {OUT}")
    if not runs:
        return

    grouped = defaultdict(dict)
    for (model, dataset, method, bs), run in runs.items():
        grouped[(model, dataset, method)][bs] = run

    # Per-config table
    print("\n=== Per-config (accuracy, avg output length, runtime) ===")
    fmt = "{:<30} {:<8} {:<16} {:>4} {:>6} {:>7} {:>8}"
    print(fmt.format("model", "dataset", "method", "bs", "acc", "len", "rt"))
    print("-" * 95)
    for (model, dataset, method), runs_by_bs in sorted(grouped.items()):
        for bs in sorted(runs_by_bs):
            r = runs_by_bs[bs]["aggregate"]
            print(fmt.format(model, dataset, method, bs,
                             f"{r['accuracy']*100:.0f}%",
                             f"{r['avg_output_length']:.0f}",
                             f"{r['total_runtime_seconds']:.0f}s"))

    # Per-(model,dataset,method) aggregate metrics
    rows = []
    print("\n=== Per (model,dataset,method): Std@Acc / Div_Index / Avg_Std@top1_prob ===")
    fmt2 = "{:<30} {:<8} {:<16} {:>8} {:>10} {:>12} {:>12} {:>14}"
    print(fmt2.format("model", "dataset", "method", "Std@Acc",
                      "nondet%", "DivIdx_med", "DivIdx_mean", "AvgStd@top1"))
    print("-" * 125)
    for (model, dataset, method), runs_by_bs in sorted(grouped.items()):
        if set(runs_by_bs.keys()) != {1, 8, 32}:
            continue
        r1, r8, r32 = runs_by_bs[1], runs_by_bs[8], runs_by_bs[32]
        accs = [r1["aggregate"]["accuracy"], r8["aggregate"]["accuracy"], r32["aggregate"]["accuracy"]]
        std_acc = statistics.pstdev(accs)

        n = min(len(r1["per_problem"]), len(r8["per_problem"]), len(r32["per_problem"]))
        divs18, divs832 = [], []
        nondet = 0
        std_top1s, out_lens = [], []
        for i in range(n):
            p1, p8, p32 = r1["per_problem"][i], r8["per_problem"][i], r32["per_problem"][i]
            d18 = first_divergence(p1, p8); divs18.append(d18)
            d832 = first_divergence(p8, p32); divs832.append(d832)
            if p1["token_hash"] != p8["token_hash"] or p8["token_hash"] != p32["token_hash"]:
                nondet += 1
            s, _ = std_at_top1(p1, p8, p32)
            if s is not None:
                std_top1s.append(s)
            out_lens.append(statistics.pstdev([p1["output_len"], p8["output_len"], p32["output_len"]]))

        divs_all = divs18 + divs832
        row = {
            "model": model, "dataset": dataset, "method": method,
            "n_problems": n,
            "acc_bs1": accs[0], "acc_bs8": accs[1], "acc_bs32": accs[2],
            "std_acc": std_acc,
            "nondet_rate": nondet / n,
            "div_median": statistics.median(divs_all),
            "div_mean": statistics.mean(divs_all),
            "div_max": max(divs_all),
            "avg_std_top1": statistics.mean(std_top1s) if std_top1s else None,
            "avg_std_outlen": statistics.mean(out_lens),
        }
        rows.append(row)
        print(fmt2.format(model, dataset, method,
                          f"{std_acc:.4f}",
                          f"{row['nondet_rate']*100:.0f}%",
                          f"{row['div_median']:.0f}",
                          f"{row['div_mean']:.1f}",
                          f"{row['avg_std_top1']:.4f}" if row["avg_std_top1"] is not None else "-"))

    # Write machine-readable summary
    out_path = os.path.join(OUT, "summary_n50.json")
    with open(out_path, "w") as f:
        json.dump({"per_mdm": rows, "n_configs": len(runs)}, f, indent=2, default=str)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
