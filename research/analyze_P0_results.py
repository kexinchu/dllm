#!/usr/bin/env python3
"""
Aggregate and analyze P0 experiment outputs.

Reads:
  research/exp_E7/E7_all.json            (overhead profile)
  research/exp_E3/E3_*.json              (prob std)
  research/exp_E4/E4_{model}_{dataset}.json   (div_index + accuracy)
  research/exp_E2/E2.json                (prob gap)

Produces:
  paper-ready markdown tables for each experiment
  per-subject/difficulty breakdown for E5
  answer-flip examples for E1 motivation
  json summary under research/P0_summary/summary.json
"""
import os, sys, json, glob, argparse, statistics
from collections import defaultdict

SUMMARY_DIR = "/home/kec23008/docker-sys/dllm/research/P0_summary"
os.makedirs(SUMMARY_DIR, exist_ok=True)


# ── E7: Overhead table ────────────────────────────────────────────────────────
def analyze_E7():
    paths = sorted(glob.glob("/home/kec23008/docker-sys/dllm/research/exp_E7/E7_*.json"))
    paths = [p for p in paths if not p.endswith("/E7_all.json")]
    if not paths: return None

    out = ["## E7 — Overhead Profile", ""]
    for p in paths:
        d = json.load(open(p))
        out.append(f"### Model: {d['model']}")
        out.append("")
        out.append("| Scheme | bs | TTFT (ms) | TPOT (ms) | Throughput (tok/s) | Peak mem (GB) | TTFT× | TPOT× |")
        out.append("|---|---|---|---|---|---|---|---|")
        bf = {(r["scheme"], r["bs"]): r for r in d["results"] if r["scheme"] == "BF16"}
        for r in d["results"]:
            b = bf.get(("BF16", r["bs"]))
            ttx = r["ttft_ms"] / b["ttft_ms"] if b else 1.0
            tpx = r["tpot_ms"] / b["tpot_ms"] if b else 1.0
            out.append(f"| {r['scheme']} | {r['bs']} | {r['ttft_ms']:.1f} | "
                       f"{r['tpot_ms']:.2f} | {r['throughput_tps']:.1f} | "
                       f"{r['peak_mem_gb']:.2f} | {ttx:.2f}× | {tpx:.2f}× |")
        out.append("")
    return "\n".join(out)


# ── E3: Prob std table ────────────────────────────────────────────────────────
def analyze_E3():
    paths = sorted(glob.glob("/home/kec23008/docker-sys/dllm/research/exp_E3/E3_*.json"))
    if not paths: return None

    out = ["## E3 — Avg_Std@top1_prob", ""]
    for p in paths:
        model = os.path.basename(p).replace("E3_", "").replace(".json", "")
        d = json.load(open(p))
        out.append(f"### Model: {model}  (N_problems={d[0]['n_problems']}, bs={d[0]['bs_list']})")
        out.append("")
        out.append("| Scheme | Avg_Std@top1 | Median | Avg max-diff | vs BF16 |")
        out.append("|---|---|---|---|---|")
        bf = next((r["avg_std_top1_prob"] for r in d if r["scheme"] == "BF16"), None)
        for r in d:
            ratio = r["avg_std_top1_prob"] / bf if bf and bf > 0 else 0.0
            out.append(f"| {r['scheme']} | {r['avg_std_top1_prob']:.3e} | "
                       f"{r['median_std']:.3e} | {r['avg_max_diff']:.3e} | "
                       f"{ratio:.3f}× |")
        out.append("")
    return "\n".join(out)


# ── E4: Div_Index + bit_exact + accuracy ──────────────────────────────────────
def analyze_E4():
    paths = sorted(glob.glob("/home/kec23008/docker-sys/dllm/research/exp_E4/E4_*.json"))
    if not paths: return None

    out = ["## E4 — Div_Index + Bit-Exact + Accuracy", ""]
    for p in paths:
        d = json.load(open(p))
        model, dataset = d["model"], d.get("dataset", "?")
        N = d["n_problems"]; gl = d["gen_len"]; bsl = d["bs_list"]
        out.append(f"### {model} / {dataset}  (N={N}, gen_len={gl}, bs={bsl})")
        out.append("")
        out.append("| Scheme | bit-exact % | %problems diverge | Div_Index median | p25 | p75 | "
                   f"Acc(bs={bsl[0]}) | Acc(bs={bsl[-1]}) | runtime s |")
        out.append("|---|---|---|---|---|---|---|---|---|")
        for s in d["schemes"]:
            agg = s["aggregate"]
            acc_ref  = agg["accuracy_by_bs"][str(bsl[0])] if str(bsl[0]) in agg["accuracy_by_bs"] else agg["accuracy_by_bs"][bsl[0]]
            acc_max  = agg["accuracy_by_bs"][str(bsl[-1])] if str(bsl[-1]) in agg["accuracy_by_bs"] else agg["accuracy_by_bs"][bsl[-1]]
            out.append(f"| {s['scheme']} | {agg['bit_exact_rate']*100:.1f}% | "
                       f"{agg['problem_diverge_rate']*100:.1f}% | "
                       f"{agg['div_index_median']} | {agg['div_index_p25']} | {agg['div_index_p75']} | "
                       f"{acc_ref*100:.1f}% | {acc_max*100:.1f}% | "
                       f"{agg['runtime_s']:.0f} |")
        out.append("")
    return "\n".join(out)


# ── E5: Task-difficulty breakdown (derived from E4) ───────────────────────────
def analyze_E5():
    paths = sorted(glob.glob("/home/kec23008/docker-sys/dllm/research/exp_E4/E4_*.json"))
    if not paths: return None

    out = ["## E5 — Divergence rate by task difficulty (derived from E4)", ""]
    for p in paths:
        d = json.load(open(p))
        bsl = d["bs_list"]; gl = d["gen_len"]
        out.append(f"### {d['model']} / {d.get('dataset','?')}  (N={d['n_problems']}, bs={bsl})")
        out.append("")
        # group per problem by subject
        out.append("| Scheme | Subject | N | %diverge | median Div_Index |")
        out.append("|---|---|---|---|---|")
        for s in d["schemes"]:
            by_subj = defaultdict(list)
            for q in s["per_problem"]:
                by_subj[q.get("subject", "?")].append(q)
            for subj, qs in sorted(by_subj.items()):
                div_idxs = []
                n_div = 0
                for q in qs:
                    bsbs = q["per_bs"]
                    for bs in bsl[1:]:
                        bs_key = bs if bs in bsbs else str(bs)
                        div_idxs.append(bsbs[bs_key]["div_idx"])
                    if any(not bsbs[bs if bs in bsbs else str(bs)]["bit_exact"] for bs in bsl[1:]):
                        n_div += 1
                pct = 100 * n_div / len(qs)
                med = statistics.median(div_idxs)
                out.append(f"| {s['scheme']} | {subj} | {len(qs)} | {pct:.1f}% | {med} |")
        out.append("")
    return "\n".join(out)


# ── E1 motivation: extract answer-flip examples (derived from E4) ──────────────
def analyze_E1():
    paths = sorted(glob.glob("/home/kec23008/docker-sys/dllm/research/exp_E4/E4_*.json"))
    if not paths: return None

    out = ["## E1 — Motivation: answer-flip examples (derived from E4)", ""]
    for p in paths:
        d = json.load(open(p))
        bsl = d["bs_list"]
        out.append(f"### {d['model']} / {d.get('dataset','?')}")
        out.append("")
        # for BF16 scheme: list problems where pred differs across bs
        bf = next((s for s in d["schemes"] if s["scheme"] == "BF16"), None)
        if not bf: continue
        flips = []
        for q in bf["per_problem"]:
            preds = [q["per_bs"][bs if bs in q["per_bs"] else str(bs)]["pred"] for bs in bsl]
            corrs = [q["per_bs"][bs if bs in q["per_bs"] else str(bs)]["correct"] for bs in bsl]
            if len(set(str(p) for p in preds)) > 1:
                flips.append({"idx": q["idx"], "gold": q["gold"], "preds": preds, "correct": corrs,
                              "subject": q.get("subject", "")})
        out.append(f"BF16: **{len(flips)}/{d['n_problems']}** problems had answer change across bs={bsl}")
        out.append("")
        if flips:
            out.append("| idx | subject | gold | " + " | ".join(f"pred@bs={b}" for b in bsl) + " | "
                       + " | ".join(f"✓@bs={b}" for b in bsl) + " |")
            out.append("|" + "---|" * (2 + 2 * len(bsl) + 1))
            for f in flips[:15]:
                row = f"| {f['idx']} | {f['subject']} | {f['gold']} | " \
                      + " | ".join(str(p)[:20] for p in f["preds"]) + " | " \
                      + " | ".join("✓" if c else "✗" for c in f["correct"]) + " |"
                out.append(row)
        out.append("")

        # accuracy flip table
        out.append("Accuracy per bs under BF16:")
        out.append("")
        acc_per_bs = bf["aggregate"]["accuracy_by_bs"]
        out.append("| bs | accuracy |")
        out.append("|---|---|")
        for bs in bsl:
            k = bs if bs in acc_per_bs else str(bs)
            out.append(f"| {bs} | {acc_per_bs[k]*100:.1f}% |")
        out.append("")
    return "\n".join(out)


# ── E2: Prob gap summary ──────────────────────────────────────────────────────
def analyze_E2():
    p = "/home/kec23008/docker-sys/dllm/research/exp_E2/E2.json"
    if not os.path.exists(p): return None

    d = json.load(open(p))
    out = ["## E2 — Top-1 / Top-2 probability gap at divergence", ""]
    out.append(f"Model: {d['model']}  bs_ref={d['bs_ref']}  bs_perturb={d['bs_perturb']}")
    out.append("")
    for sc_name, sc_data in d["schemes"].items():
        divs = [e for e in sc_data["per_problem"] if e.get("div_idx") is not None]
        out.append(f"### {sc_name}:  #diverged = {len(divs)}/{sc_data['n_total']}")
        if divs:
            gaps = sorted(d["ref_p1_minus_p2"] for d in divs)
            out.append(f"p1−p2 gap @ divergence:  "
                       f"median={gaps[len(gaps)//2]:.4f}  "
                       f"p25={gaps[len(gaps)//4]:.4f}  "
                       f"p75={gaps[3*len(gaps)//4]:.4f}  "
                       f"min={gaps[0]:.4f}  max={gaps[-1]:.4f}")
            out.append("")
            out.append("Examples (first 5):")
            out.append("| idx | ref_top1 | ref_p1 | ref_p2 | gap | prt_top1 |")
            out.append("|---|---|---|---|---|---|")
            for e in divs[:5]:
                out.append(f"| {e['idx']} | {e['ref_top1_text']!r} | "
                           f"{e['ref_top5_probs'][0]:.3f} | "
                           f"{e['ref_top5_probs'][1]:.3f} | "
                           f"{e['ref_p1_minus_p2']:.4f} | "
                           f"{e['prt_top1_text']!r} |")
        out.append("")
    return "\n".join(out)


def main():
    sections = []
    for fn in (analyze_E7, analyze_E3, analyze_E4, analyze_E5, analyze_E1, analyze_E2):
        s = fn()
        if s: sections.append(s)

    md = "# P0 Experiment Summary\n\n" + "\n\n".join(sections)
    out_md = os.path.join(SUMMARY_DIR, "P0_summary.md")
    with open(out_md, "w") as f:
        f.write(md)
    print(f"Saved: {out_md}")
    print()
    print(md)


if __name__ == "__main__":
    main()
