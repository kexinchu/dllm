# Avg_Std@top1_prob validation — DeepSeek-R1-Distill-Qwen-7B / MATH500 (5 problems)

Configs: bs ∈ [1, 2, 4, 8, 16]
Decode: greedy 256 tokens

## Per scheme

| Scheme | Avg_Std@top1_prob | min pre_div | median pre_div | max pre_div |
|---|---|---|---|---|
| BF16 | 3.615e-03 | 6 | 56 | 256 |
| SRP-FP32 | 0.000e+00 | 256 | 256 | 256 |
| FP32-all | 3.590e-07 | 256 | 256 | 256 |

## Per problem (per scheme)

### BF16

| problem | level | input_len | pre_div | avg_std_problem | max_std_pre_div |
|---|---|---|---|---|---|
| 0 | 2 | 69 | 256 | 3.016e-03 | 2.913e-02 |
| 1 | 5 | 132 | 14 | 5.235e-03 | 1.544e-02 |
| 2 | 3 | 68 | 124 | 3.659e-03 | 3.045e-02 |
| 3 | 3 | 35 | 56 | 3.377e-03 | 3.007e-02 |
| 4 | 2 | 360 | 6 | 2.790e-03 | 1.121e-02 |

### SRP-FP32

| problem | level | input_len | pre_div | avg_std_problem | max_std_pre_div |
|---|---|---|---|---|---|
| 0 | 2 | 69 | 256 | 0.000e+00 | 0.000e+00 |
| 1 | 5 | 132 | 256 | 0.000e+00 | 0.000e+00 |
| 2 | 3 | 68 | 256 | 0.000e+00 | 0.000e+00 |
| 3 | 3 | 35 | 256 | 0.000e+00 | 0.000e+00 |
| 4 | 2 | 360 | 256 | 0.000e+00 | 0.000e+00 |

### FP32-all

| problem | level | input_len | pre_div | avg_std_problem | max_std_pre_div |
|---|---|---|---|---|---|
| 0 | 2 | 69 | 256 | 2.839e-07 | 2.485e-06 |
| 1 | 5 | 132 | 256 | 3.959e-07 | 2.852e-06 |
| 2 | 3 | 68 | 256 | 4.175e-07 | 3.917e-06 |
| 3 | 3 | 35 | 256 | 3.110e-07 | 4.801e-06 |
| 4 | 2 | 360 | 256 | 3.868e-07 | 3.839e-06 |
