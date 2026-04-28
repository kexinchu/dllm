#!/usr/bin/env python3
"""Add `level` field (1..5) to research/math500_cached.json by re-fetching the
HuggingFaceH4/MATH-500 dataset and matching on problem text. Idempotent.

Output: rewrites cache with extra 'level' key; backups the old cache as
research/math500_cached.json.bak (only if backup doesn't already exist).
"""
import json, os, sys

CACHE = "/home/kec23008/docker-sys/dllm/research/math500_cached.json"
BAK   = CACHE + ".bak"

def main():
    cache = json.load(open(CACHE))
    if all("level" in p for p in cache):
        print("Cache already enriched, no action."); return

    if not os.path.exists(BAK):
        with open(BAK, "w") as f:
            json.dump(cache, f)
        print(f"Backed up cache → {BAK}")

    print("Loading HuggingFaceH4/MATH-500 from HF Hub ...")
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    by_problem = {ex["problem"]: ex for ex in ds}
    print(f"  HF dataset has {len(by_problem)} problems")

    enriched, missing = [], 0
    for p in cache:
        ref = by_problem.get(p["problem"])
        if ref is None:
            missing += 1
            p2 = dict(p, level=None)
        else:
            p2 = dict(p, level=ref.get("level"))
        enriched.append(p2)

    with open(CACHE, "w") as f:
        json.dump(enriched, f)
    print(f"Wrote enriched cache: {len(enriched)} problems, "
          f"{missing} unmatched (level=None)")
    # spot-check
    levels = [p["level"] for p in enriched if p["level"] is not None]
    if levels:
        from collections import Counter
        print("level distribution:", dict(sorted(Counter(levels).items())))


if __name__ == "__main__":
    main()
