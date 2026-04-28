# MATH500 overnight results (DeepSeek-R1-Distill-Qwen-7B)

## Std@Acc across configurations

| Method | Mean Acc | Std@Acc | Min | Max | n |
|---|---|---|---|---|---|
| bf16_baseline | 43.00% | 0.0115 | 42.00% | 44.00% | 4 |
| layercast | 41.00% | 0.0141 | 40.00% | 42.00% | 2 |
| dllm_cublaslt | 40.00% | 0.0231 | 38.00% | 42.00% | 4 |
| dllm_triton | 41.00% | 0.0115 | 40.00% | 42.00% | 4 |

## Avg_Std@Output_Length

| Method | Mean per-problem std |
|---|---|
| bf16_baseline | 60.9 |
| layercast | 87.9 |
| dllm_cublaslt | 79.0 |
| dllm_triton | 92.0 |

## Div_Index (first diverging token)

| Method | Median | Mean |
|---|---|---|
| bf16_baseline | 20.0 | 17.8 |
| layercast | 20.0 | 18.7 |
| dllm_cublaslt | 20.0 | 18.8 |
| dllm_triton | 20.0 | 18.2 |

## Runtime per eval config (seconds)

| Method | Mean | Std |
|---|---|---|
| bf16_baseline | 3937.2 | 657.7 |
| layercast | 11618.5 | 869.1 |
| dllm_cublaslt | 3990.7 | 658.8 |
| dllm_triton | 4518.2 | 547.3 |
