# P0 Experiment Summary

## E7 — Overhead Profile

### Model: Llama-3.1-8B

| Scheme | bs | TTFT (ms) | TPOT (ms) | Throughput (tok/s) | Peak mem (GB) | TTFT× | TPOT× |
|---|---|---|---|---|---|---|---|
| BF16 | 1 | 37.4 | 29.89 | 33.4 | 15.01 | 1.00× | 1.00× |
| BF16 | 8 | 99.7 | 31.40 | 252.6 | 15.29 | 1.00× | 1.00× |
| BF16 | 32 | 350.0 | 36.24 | 854.0 | 16.25 | 1.00× | 1.00× |
| FP32flag | 1 | 39.2 | 29.64 | 33.7 | 15.01 | 1.05× | 0.99× |
| FP32flag | 8 | 102.9 | 31.75 | 249.7 | 15.29 | 1.03× | 1.01× |
| FP32flag | 32 | 355.0 | 36.32 | 852.0 | 16.25 | 1.01× | 1.00× |
| LayerCast | 1 | 153.1 | 132.03 | 7.6 | 16.96 | 4.09× | 4.42× |
| LayerCast | 8 | 532.9 | 139.79 | 56.6 | 17.24 | 5.35× | 4.45× |
| LayerCast | 32 | 1851.8 | 146.27 | 209.2 | 18.21 | 5.29× | 4.04× |
| DetermLLM | 1 | 45.6 | 43.08 | 23.2 | 15.01 | 1.22× | 1.44× |
| DetermLLM | 8 | 102.9 | 42.82 | 185.8 | 15.29 | 1.03× | 1.36× |
| DetermLLM | 32 | 384.2 | 43.01 | 721.7 | 16.25 | 1.10× | 1.19× |
| DetermLLM+attn | 1 | 45.0 | 42.83 | 23.3 | 15.01 | 1.20× | 1.43× |
| DetermLLM+attn | 8 | 103.2 | 42.51 | 187.1 | 15.29 | 1.04× | 1.35× |
| DetermLLM+attn | 32 | 381.5 | 42.87 | 724.0 | 16.25 | 1.09× | 1.18× |


## E3 — Avg_Std@top1_prob

### Model: llama8b  (N_problems=50, bs=[1, 4, 8])

| Scheme | Avg_Std@top1 | Median | Avg max-diff | vs BF16 |
|---|---|---|---|---|
| BF16 | 6.479e-03 | 6.445e-03 | 6.087e-02 | 1.000× |
| FP32flag | 5.535e-03 | 5.539e-03 | 5.496e-02 | 0.854× |
| LayerCast | 6.108e-03 | 5.873e-03 | 5.898e-02 | 0.943× |
| DetermLLM | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000× |
| DetermLLM+attn | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000× |


## E4 — Div_Index + Bit-Exact + Accuracy

### llama8b / math500  (N=50, gen_len=512, bs=[1, 8, 16])

| Scheme | bit-exact % | %problems diverge | Div_Index median | p25 | p75 | Acc(bs=1) | Acc(bs=16) | runtime s |
|---|---|---|---|---|---|---|---|---|
| BF16 | 28.0% | 80.0% | 134 | 45 | 512 | 32.0% | 40.0% | 1825 |


## E5 — Divergence rate by task difficulty (derived from E4)

### llama8b / math500  (N=50, bs=[1, 8, 16])

| Scheme | Subject | N | %diverge | median Div_Index |
|---|---|---|---|---|
| BF16 | Algebra | 12 | 75.0% | 85.0 |
| BF16 | Counting & Probability | 4 | 100.0% | 118.0 |
| BF16 | Geometry | 5 | 60.0% | 329.5 |
| BF16 | Intermediate Algebra | 8 | 75.0% | 124.5 |
| BF16 | Number Theory | 7 | 71.4% | 158.0 |
| BF16 | Prealgebra | 7 | 85.7% | 214.0 |
| BF16 | Precalculus | 7 | 100.0% | 83.5 |


## E1 — Motivation: answer-flip examples (derived from E4)

### llama8b / math500

BF16: **17/50** problems had answer change across bs=[1, 8, 16]

| idx | subject | gold | pred@bs=1 | pred@bs=8 | pred@bs=16 | ✓@bs=1 | ✓@bs=8 | ✓@bs=16 |
|---|---|---|---|---|---|---|---|---|
| 0 | Precalculus | \left( 3, \frac{\pi}{2} \right) | (3, \frac{\pi}{2}) | (3, \frac{\pi}{2}) | (3,\frac{\pi}{2}) | ✗ | ✗ | ✗ |
| 4 | Algebra | \text{Evelyn} | Angela | None | Angela | ✗ | ✗ | ✗ |
| 6 | Number Theory | 27 | 19683 | 19683 | 729 | ✗ | ✗ | ✗ |
| 10 | Number Theory | 2220 | None | 2000 | 2000 | ✗ | ✗ | ✗ |
| 14 | Precalculus | \sqrt{51} | 10 | \sqrt{51} | \sqrt{51} | ✗ | ✓ | ✓ |
| 17 | Precalculus | \pi | None | \pi | \pi | ✗ | ✓ | ✓ |
| 19 | Intermediate Algebra | 3 | \sqrt[3]{3} | \sqrt[3]{3} | 3 | ✗ | ✗ | ✓ |
| 22 | Algebra | 5 | 5 | None | None | ✓ | ✗ | ✗ |
| 24 | Algebra | 10 | 10 | 49.3 | 10 | ✓ | ✗ | ✓ |
| 26 | Counting & Probability | 144 | None | None | 576 | ✗ | ✗ | ✗ |
| 28 | Precalculus | -2 + 7i | 7i - 2 | -2 + 7i | -2 + 7i | ✗ | ✓ | ✓ |
| 29 | Counting & Probability | 225 | None | 1 | None | ✗ | ✗ | ✗ |
| 30 | Number Theory | 52_8 | None | 52_8 | 52_8 | ✗ | ✓ | ✓ |
| 32 | Counting & Probability | 720 | 720 | 720 | 240 | ✓ | ✓ | ✗ |
| 33 | Algebra | \frac{243}{625} | 25/9 | None | None | ✗ | ✗ | ✗ |

Accuracy per bs under BF16:

| bs | accuracy |
|---|---|
| 1 | 32.0% |
| 8 | 40.0% |
| 16 | 40.0% |
