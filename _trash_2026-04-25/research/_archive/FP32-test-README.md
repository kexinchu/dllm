# FP32 reduction-only tests

本目录为 **PyTorch 层面的数值/行为模拟**，用于比较「仅规约用 FP32 再压回 BF16」与 BF16 规约的 latency 与 deterministic。  
**真实部署**该策略需要修改 kernel（见下方「部署需改 kernel」）。

Tests live under **`FP32/`**:

- **`FP32/reduction_ops.py`**: Reduction variants  
  - (1) BF16 + atomicAdd style  
  - (2) FP32 + atomicAdd -> BF16  
  - (3) Deterministic sequential (bit-by-bit)  

- **`FP32/test_reduction_fp32_vs_bf16.py`**: Benchmark script (latency + determinism + batch_invariance).

Run from repo root:

```bash
cd /workspace/dllm && python -m FP32.test_reduction_fp32_vs_bf16
```

Or:

```bash
cd /workspace/dllm/FP32 && python test_reduction_fp32_vs_bf16.py
```

## Metrics

- **Latency**: (1) BF16 vs (2) FP32-then-BF16 reduction (vs LayerCast which casts everything to FP32).
- **Determinism**: (2) vs (3); variance of (1) across runs with different chunk orders.
- **Batch invariance**: Same first row at M=1 vs M=8; first-row output should match for deterministic methods (see `docs/batch_invariance.md`).

## 方案 A：GEMM kernel（已实现）

`FP32/gemm_fp32_accum.py` 实现了方案 A，**优先使用推理引擎内融合的 FP32 累加 GEMM**（cuBLASLt），而不是在 Python 里逐 Linear 替换：

- **cuBLASLt 扩展**（推荐）：`FP32/csrc/gemm_fp32_accum_cuda.cu` 使用 cuBLASLt 做 BF16×BF16、FP32 累加、BF16 输出的融合 GEMM。编译后 `matmul_fp32_accum` 会优先走该 kernel。
- **Triton kernel**：无扩展时，有 Triton 且 CUDA 则用 Triton 的 FP32 累加 matmul。
- **PyTorch fallback**：无 CUDA 或上述都不可用时用 `(A.float() @ B.float()).to(bf16)`，数值等价。

### 编译 cuBLASLt 扩展

在 **FP32/** 目录下执行（需 CUDA + PyTorch CUDA 环境）：

```bash
cd /workspace/dllm/FP32 && python setup.py build_ext --inplace
```

编译成功后，`FP32/` 下会生成 `_gemm_fp32_accum_cuda*.so`，`run_tests` 会显示 Backend: cuBLASLt (fused)。也可从仓库根目录用 `setup_fp32_ext.py` 构建：`python setup_fp32_ext.py build_ext --inplace`。

**用法：**

- `matmul_fp32_accum(a, b)`：等价于 `a @ b`，但规约在 FP32 中完成。
- `linear_fp32_accum(input, weight, bias=None)`：等价于 `F.linear(input, weight, bias)`。
- `LinearFP32Accum(in_features, out_features, bias=True)`：可替换 `nn.Linear`，forward 时用 FP32 累加再压回 BF16。

替换模型中的 Linear：`module.linear = LinearFP32Accum(in_f, out_f); module.linear.load_state_dict(orig.state_dict())`。

**运行**：`python -m FP32.run_tests` 在 CUDA 下会多跑 GEMM 正确性 + latency 对比（gemm_fp32_accum vs torch linear bf16）。

## 方案 B（未实现）

GEMM 输出 partial sums（BF16），再用单独的 FP32 reduction kernel 做 sum 并 cast 回 BF16。

---

## 全套 FP32 Accumulation（新增）

除 GEMM 外，所有带 reduction 的 op 都实现了 FP32 accumulator 版本。**统一范式**：

```
BF16 input -> load -> FP32 accumulator 做 reduction -> cast to BF16 -> store
```

### 新增模块

| 模块 | 文件 | Reduction | 实现 |
|---|---|---|---|
| **RMSNorm** | `FP32/rmsnorm_fp32_accum.py` | mean(x^2) over hidden_dim | Triton kernel + PyTorch fallback |
| **Softmax** | `FP32/softmax_fp32_accum.py` | max(x) + sum(exp(x-max)) | Triton kernel + PyTorch fallback |
| **Attention** | `FP32/attention_fp32_accum.py` | QK^T + online softmax + attn@V | Triton Flash-style kernel (O(1) memory, no split-KV) + PyTorch fallback |
| **Model Patcher** | `FP32/model_patcher.py` | - | 一键 patch HF 模型所有 reduction op |

### 使用方法

```python
from FP32.model_patcher import fp32_accum_mode

model = AutoModelForCausalLM.from_pretrained(..., torch_dtype=torch.bfloat16)

# 一键开启全套 FP32 accumulation
with fp32_accum_mode(model):
    outputs = model.generate(...)

# 可选择性开启
with fp32_accum_mode(model, patch_linear=True, patch_rmsnorm=True,
                     patch_attention=False, patch_softmax=True):
    outputs = model.generate(...)
```

### 运行测试

```bash
# 单元测试：correctness / determinism / batch invariance
python -m FP32.test_ops_fp32_accum

# 端到端测试：模型级别 batch invariance + latency
python motivation/test_fp32_accum_e2e.py --model-path /path/to/model
```

### Attention kernel 设计要点

- **Triton Flash-style**：tiling + online softmax，O(1) 额外内存
- **所有 accumulator 为 FP32**：m_i (running max), l_i (running sum), acc (output)
- **无 split-KV**：每个 (batch, head, q_block) 独立遍历完整 KV 序列
  - reduction 顺序仅取决于 seq_len 和 BLOCK_N，与 batch size 无关
  - 天然 batch-invariant
- **QK^T 和 attn@V**：`tl.dot(..., out_dtype=tl.float32)` FP32 accumulation
- Decode 性能损失 ~20-40%（无 split-KV 并行），prefill 接近原始

### 支持模型

通过 `model_patcher` 自动识别并 patch：
- **Llama** 系列：LlamaRMSNorm, LlamaSdpaAttention
- **Qwen3-MoE** 系列：Qwen2RMSNorm, Qwen3MoeSdpaAttention
- 其他使用标准 HF attention/norm 类的模型
