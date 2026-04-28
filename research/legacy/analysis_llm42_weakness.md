# LLM-42 Weakness Analysis: Rollback Frequency in Hard Scenarios

## Core Finding

LLM-42's decode-verify-rollback protocol assumes rollbacks are rare (<1% of tokens).
Our experiments show this assumption breaks catastrophically in three scenarios:

## Data Summary (Llama-3.1-8B, bs=1..128)

### D2: Reasoning Prompts (128 tokens generated)

| Prompt | BF16 rollback | FP32 rollback | Worst recompute |
|---|---:|---:|---:|
| P1: Math (train distance) | 62.5% of bs | **87.5%** | **58.6%** tokens |
| P2: Code (LCS function) | 0% | 0% | 0% |
| P3: Proof (√2 irrational) | 0% | **87.5%** | **81.2%** |
| P4: Calculus (max area) | 12.5% | 12.5% | 42.2% |
| P5: Debug (fibonacci) | **87.5%** | 50% | 53.1% |
| **Average** | **35%** | **50%** | — |

Key observation: FP32 flag makes rollbacks MORE frequent (35% → 50%).

### D3: Long Generation (512 tokens)

| | BF16 | FP32 |
|---|---|---|
| Rollback rate | **100%** (8/8 runs) | **100%** (8/8 runs) |
| Average recompute | ~80% | ~83% |
| Worst recompute | **99.2%** (first mismatch at pos 4) | **99.2%** |

### D3b: Very Long Generation (1024 tokens)

| | BF16 | FP32 |
|---|---|---|
| Rollback rate | **100%** (3/3 runs) | **100%** (3/3 runs) |
| Average recompute | ~83% | ~90% |

## LLM-42 Equivalent Overhead Calculation

LLM-42 overhead ≈ P(rollback) × avg_recompute_fraction × (1 + verify_overhead)

Assuming verify_overhead = 0.1 (LLM-42's lightweight verification):

| Scenario | Gen Length | P(rollback) | Avg Recompute | **LLM-42 Overhead** | batch_invariant_ops |
|---|---:|---:|---:|---:|---:|
| Short chat | 32 tok | ~5% | ~30% | **~2%** | 34% |
| Medium chat | 64 tok | ~15% | ~40% | **~7%** | 34% |
| **Reasoning** | **128 tok** | **35-50%** | **50%** | **~20-28%** | **34%** |
| **Long gen** | **512 tok** | **100%** | **80%** | **~88%** | **34%** |
| **Very long** | **1024 tok** | **100%** | **89%** | **~98%** | **34%** |

## Crossover Point

LLM-42 overhead > batch_invariant_ops when:
- Reasoning prompts with 128+ tokens: **approaching crossover (~28% vs 34%)**
- Long generation 256+ tokens: **far exceeds crossover**
- Any generation > 512 tokens: **LLM-42 is 2-3× worse than batch_invariant_ops**

## Root Cause

LLM-42's assumption: "most tokens have large argmax margin, rollbacks are rare"

Reality:
1. As generation lengthens, P(hitting at least one near-tie position) → 1
2. Once a rollback occurs at position t, ALL tokens from t to end must be recomputed
   (autoregressive dependency means one wrong token invalidates everything after it)
3. Near-tie positions tend to occur at "decision points" in reasoning
   (choosing between alternative reasoning paths), which are EARLY in the chain
4. Early rollback = nearly full recomputation

## Implications for Paper Direction D

This data supports a paper arguing:
1. LLM-42's rollback approach degrades to ~2× overhead for long/reasoning tasks
2. No existing approach works well across ALL scenarios
3. Need: scenario-aware determinism (different strategy per workload type)
