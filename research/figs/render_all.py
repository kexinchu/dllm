#!/usr/bin/env python3
"""Render every figure (1-7). Each script is independent; this is a convenience
runner used by analyze_P0_results / nightly job.
"""
import os, sys, traceback
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

scripts = [
    "fig1_motivation",
    "fig2_prob_gap",
    "fig3_avg_std",
    "fig4_div_index_cdf",
    "fig5_bit_exact",
    "fig6_difficulty",
    "fig7_overhead",
]

for name in scripts:
    print(f"\n=== {name} ===")
    try:
        m = __import__(name)
        m.main()
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        traceback.print_exc()
