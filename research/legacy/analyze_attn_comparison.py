"""Analyze cross-bs determinism for attn-patched vs LayerCast vs no-attn.

For each method at seed=0, compare bs=8 vs bs=16:
  - same_toks: problems with identical full token sequences
  - same_answer: problems where final extracted answer matches
  - div_early: problems diverging in first 20 tokens
  - div_later: problems diverging after 20 tokens (but not bit-exact)

Use either the _small.json files (n=10) or the full _seed0.json files.
"""
import json
import os
import statistics

RES = "/home/kec23008/docker-sys/dllm/research/exp_math500_overnight"

def load(method, bs, suffix=""):
    """Load JSON for (method, bs, seed=0, optional _small suffix)."""
    path = os.path.join(RES, f"{method}_bs{bs}_seed0{suffix}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def analyze(method, suffix=""):
    b8 = load(method, 8, suffix)
    b16 = load(method, 16, suffix)
    if b8 is None or b16 is None:
        return None

    n = min(len(b8["per_problem"]), len(b16["per_problem"]))
    same_toks = 0
    same_answer = 0
    early_divs = []   # div position if div<20
    for i in range(n):
        pa = b8["per_problem"][i]
        pb = b16["per_problem"][i]
        if pa["token_hash"] == pb["token_hash"]:
            same_toks += 1
            early_divs.append(-1)   # signals "no divergence"
        else:
            ta = pa["first_20_tokens"]
            tb = pb["first_20_tokens"]
            div = next((j for j, (x, y) in enumerate(zip(ta, tb)) if x != y), 20)
            early_divs.append(div)
        if pa["pred"] == pb["pred"]:
            same_answer += 1

    not_div_in_first_20 = sum(1 for d in early_divs if d == 20)
    div_in_first_20 = sum(1 for d in early_divs if 0 <= d < 20)

    return {
        "method": method,
        "n_problems": n,
        "same_toks": same_toks,
        "same_answer": same_answer,
        "div_in_first_20": div_in_first_20,
        "div_after_20_but_not_exact": not_div_in_first_20 - same_toks,
        "not_exact_total": n - same_toks,
        "acc_bs8": b8["aggregate"]["accuracy"],
        "acc_bs16": b16["aggregate"]["accuracy"],
        "runtime_bs8_s": b8["aggregate"]["total_runtime_seconds"],
        "runtime_bs16_s": b16["aggregate"]["total_runtime_seconds"],
    }


def main():
    methods = ["bf16_baseline", "layercast", "dllm_cublaslt", "dllm_triton",
               "dllm_hybrid", "dllm_hybrid_attn"]

    # Try both full and small versions
    for suffix in ["", "_small"]:
        size_label = "full (n=50)" if suffix == "" else "small (n=10, gen=1024)"
        print(f"\n{'='*80}\n{size_label}\n{'='*80}")
        print(f"{'method':<25} {'n':>3} {'same_toks':>10} {'same_ans':>9} "
              f"{'div<20':>7} {'div>=20':>8} {'acc@8':>7} {'acc@16':>7}")
        for m in methods:
            r = analyze(m, suffix)
            if r is None:
                continue
            print(f"{m:<25} {r['n_problems']:>3} "
                  f"{r['same_toks']:>5}/{r['n_problems']:<4} "
                  f"{r['same_answer']:>5}/{r['n_problems']:<3} "
                  f"{r['div_in_first_20']:>7} "
                  f"{r['div_after_20_but_not_exact']:>8} "
                  f"{r['acc_bs8']:>6.0%} {r['acc_bs16']:>6.0%}")


if __name__ == "__main__":
    main()
