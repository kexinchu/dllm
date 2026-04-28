# KV-Cache Layout 对 Deterministic Inference 的影响

## 问题定义

在 batch_size=1（与 batch invariance 正交）条件下，KV-cache 的**物理分布/page layout** 是否影响推理结果的 determinism？

具体场景：
1. **Prefix cache hit vs miss**: 同一 prompt，prefix 部分从 cache 复用 vs 全部 fresh prefill
2. **Chunked prefill**: 同一 prompt，不同 chunk size 做 prefill

## 影响机制（理论分析）

KV-cache layout 影响 determinism 的路径：

### 路径 1: Prefix cache 改变 attention 的计算路径

```
一次性 prefill (cache miss):
  所有 token 在每层的 attention 中同时互相 attend (causal mask)
  Q_all @ K_all^T → softmax → V_all

分步 prefill (cache hit):
  step 1: prefix tokens 互相 attend → 产生 prefix KV cache
  step 2: suffix tokens attend to prefix_KV(from cache) + suffix_KV(fresh)
  Q_suffix @ cat(K_prefix_cached, K_suffix)^T → softmax → cat(V_prefix_cached, V_suffix)
```

即使最终 KV 值相同，SDPA kernel 处理不同形状的 Q 矩阵时，可能选择不同的 tiling/split 策略 → 不同的浮点 reduction 顺序 → 数值差异。

### 路径 2: Chunked prefill 改变中间层的 attention 输出

chunked prefill 时，每个 chunk 的 attention 只能 attend to 已有的 KV（之前的 chunks）+ 当前 chunk。不同 chunk 边界意味着每层 attention 的计算范围不同：

```
chunk_size=全部 (无 chunking):  layer L 的 attention 一次性处理全部 token
chunk_size=128:                layer L 先处理 [0:128]，再处理 [128:256] attend to [0:256]
```

每层 attention 输出的微小差异 → 传播到下一层 → 逐层累积放大。

### 路径 3: Online softmax 的分页累积

Paged Attention 使用 online softmax 逐页累积 partial attention：
```
for each page:
  partial_score = Q @ K_page^T
  merge with running max/sum/weighted_value
```
vs 标准 attention 一次性计算全部 softmax。浮点非结合律导致：
`merge(page0, page1, page2) ≠ softmax_full(K_all)`

## 实验设计

### 测试环境
- **Model**: Llama-3.1-8B-Instruct (BF16)
- **GPU**: NVIDIA RTX A6000
- **所有测试 batch_size=1**，与 batch invariance 正交
- **1000 个独立 prompt**（8 prefix × 30 suffix = 240 unique combos，循环扩展）
- **Greedy decoding (temperature=0)**，比较 first generated token

### Scenario 1: 算子级 — Contiguous vs Paged Attention

纯 PyTorch 实现，比较标准 attention（连续 KV）vs online-softmax paged attention（按 page 逐块累积），验证浮点非结合律的基础效应。

### Scenario 2 (A): Prefix Cache Hit vs Fresh Prefill

```
Path A (cache miss): 完整 prompt 一次性 prefill → first token
Path B (cache hit):  先 warm cache (prefix) → 再发完整 prompt (prefix hits cache) → first token
比较: first generated token 是否相同
```

### Scenario 3 (B): Chunked Prefill

```
Baseline: 完整 prompt 一次性 prefill (chunk_size=全长)
Variant:  chunk_size = 128 / 64 / 32
比较: first generated token 是否相同
```

## 实验结果

### Scenario 1: 算子级 (Contiguous vs Paged Attention)

> 1000 runs per config, random Q/K/V tensors, BF16

| seq_len | page_size=16 | page_size=64 | page_size=7 |
|---------|-------------|-------------|------------|
| 128 | 52.5% differ, max 2e-3 | 57.3% | 54.6% |
| 256 | 70.6% differ | 71.7% | 71.7% |
| 512 | 79.1% differ | 78.8% | 78.8% |
| 1024 | 72.1% differ | 70.7% | 74.7% |

**结论**: Online softmax 逐页累积 vs 一次性 softmax，50-80% 的 case 产生数值差异（BF16 量级 ~1e-3）。page_size 的具体值影响不大——只要分页了就有差异。

### Scenario A: Prefix Cache Hit vs Fresh Prefill

> 1000 unique prompts, Llama-3.1-8B-Instruct, greedy decoding

| Engine | Version | first_token_flip | flip rate |
|--------|---------|-----------------|-----------|
| HuggingFace + SDPA | transformers | 19/1000 | **1.90%** |
| vLLM (PagedAttention) | 0.8.5 | 17/1000 | **1.70%** |
| SGLang (RadixAttention, Triton) | 0.5.10 | 10/1000 | **1.00%** |

**vLLM flip 示例**:
| Query | no_cache → | cached → |
|-------|-----------|----------|
| "How does encryption work?" | " What" | "\n\n" |
| "What are the differences between Python and Java?" | " Python" | " (" |
| "What is machine learning?" | " A" | "\n\n" |

**SGLang flip 示例**:
| Query | no_cache → | cached → |
|-------|-----------|----------|
| "What is the water cycle?" | "\n\n" | " The" |
| "What is the Turing test?" | "\n\n" | " The" |
| "How does GPS navigation work?" | " GPS" | "\n\n" |

**HF flip 示例 (logit diff + top2 margin)**:
| Query | fresh → | cached → | logit_diff | top2_margin |
|-------|---------|----------|-----------|------------|
| "How does GPS navigation work?" | "\n" | " GPS" | 0.188 | 0.000 |
| "What is DNA replication?" | " DNA" | "\n\n" | 0.136 | 0.063 |
| "How does a compiler work?" | " A" | "\n\n" | 0.219 | 0.063 |

- HF logit diff 分布: **p50=0.156, p90=0.203, p99=0.250, max=0.281**
- 所有 flip 都发生在 **top-2 margin ≈ 0** (near-tie) 的场景

### Scenario B: Chunked Prefill

> 1000 unique prompts, Llama-3.1-8B-Instruct, greedy decoding

| chunk_size | HF + SDPA | vLLM 0.8.5 | SGLang 0.5.10 |
|-----------|-----------|------------|---------------|
| 128 | 14/1000 (**1.40%**) | 0/1000 (**0.00%**) | 1/1000 (**0.10%**) |
| 64 | 15/1000 (**1.50%**) | 2/1000 (**0.20%**) | 1/1000 (**0.10%**) |
| 32 | 17/1000 (**1.70%**) | 2/1000 (**0.20%**) | 2/1000 (**0.20%**) |

**Flip 示例 (vLLM, chunk=64)**:
| Prompt | full → | chunked → |
|--------|--------|-----------|
| "Summarize the main points..." | " The" | "\n" |
| "What are the limitations..." | " The" | " While" |

**Flip 示例 (SGLang, chunk=32)**:
| Prompt | full → | chunked → |
|--------|--------|-----------|
| "What is the most important..." | " The" | " This" |
| 另一个 | " Climate" | " The" |

## 关键发现

### 1. KV-cache 分布确实影响 determinism（即使 batch_size=1）

三个独立引擎（HF/vLLM/SGLang）均验证：prefix cache hit 和 chunked prefill 会改变 first generated token。

### 2. Prefix caching 是更显著的 non-determinism 源

| 场景 | flip rate 范围 |
|------|--------------|
| Prefix cache hit vs miss | **1.0% – 1.9%** |
| Chunked prefill | **0.0% – 0.2%** (真实引擎) |

真实引擎的 chunked prefill 几乎做到 deterministic（0–0.2%），但 prefix caching 仍有 1–1.7% 的 flip 率。这印证了 SGLang 文档中 "需要 `--disable-radix-cache` 才能保证 deterministic" 的设计。

### 3. 真实引擎的 flip 率显著低于 HF 模拟

HF + SDPA 的 chunked prefill flip 率 (1.4–1.7%) 远高于 vLLM/SGLang (0–0.2%)。原因：vLLM/SGLang 在 attention kernel 中做了优化（fixed split size、统一 KV 逻辑布局），使 chunk 边界对 reduction 顺序的影响降到很低。

### 4. 所有 flip 都发生在 near-tie 场景

logit diff 仅 0.1–0.3（BF16 精度），top-2 margin ≈ 0。KV-cache layout 差异本身很小，但在 near-tie 时足以翻转 argmax 选择。典型的 flip 模式："The" ↔ "\n\n"、"The" ↔ "This"、"A" ↔ "\n\n" 等高频 token 之间的微小偏好翻转。

### 5. 差异通过 transformer 层逐层放大

| 层级 | 差异量级 |
|-----|---------|
| 算子级 (单次 paged attention) | ~1e-3 |
| 模型级 logit diff (32 层) | ~0.15–0.28 |

单次 attention 的 ~1e-3 差异经过 32 层 transformer 传播后放大到 ~0.2 logit 空间。

## 根因分析：为什么 Prefix Cache Hit 会导致不同结果

### vLLM 的实现 (0.8.5)

vLLM 对 prefix cache hit 和 cache miss 走的是**完全不同的 attention 计算路径**：

```
# flash_attn.py (vllm/attention/backends/flash_attn.py)

# Path A: Cache miss — 标准 prefill
flash_attn_varlen_func(q=query, k=key, v=value, ...)

# Path B: Cache hit — paged attention over KV cache
flash_attn_varlen_func(q=query, k=key_cache, v=value_cache,
                      block_table=prefill_meta.block_tables, ...)
```

- **Cache miss**: Q shape=[全部 token], K/V 是连续内存中的 dense tensor
- **Cache hit**: Q shape=[suffix token only], K/V 通过 block_table 从 paged KV cache 读取

两条路径导致：
1. Q 矩阵形状不同 → kernel tiling 策略不同 → reduction 顺序不同
2. K/V 的内存布局不同（dense vs block-strided）→ 读取模式不同

对于 MLA backend (DeepSeek 等)，vLLM 更显式地拆分计算：
```
# 1. 计算 prefix 部分 (from cache)
context_output, context_lse = _compute_prefill_context(q, kv_cache, ...)
# 2. 计算 suffix 部分 (fresh, causal)
suffix_output, suffix_lse = flash_attn_varlen_func(q, k, v, causal=True, ...)
# 3. 通过 LSE merge 合并
merge_attn_states(output, prefix_output, prefix_lse, suffix_output, suffix_lse)
```

这个 prefix/suffix 分开计算再 merge 的架构，**在数学上等价但在浮点计算上不等价**于一次性 full attention。

### SGLang 的实现 (0.5.10)

SGLang 默认也走两阶段（prefix + extend），但在 deterministic mode 下提供了**统一 KV 布局**方案：

```python
# triton_ops/extend_attention.py

def build_unified_kv_indices(prefix_kv_indptr, prefix_kv_indices,
                              extend_start_loc, extend_seq_lens,
                              extend_kv_indices, bs):
    """将 cached prefix KV 和 fresh extend KV 合并成统一的 index 布局"""
    # 输出: unified_kv_indptr, unified_kv_indices
    # 让 attention kernel 看到的是一个统一的 KV 序列

# 统一 1-stage kernel
@triton.jit
def _fwd_kernel_unified(...):
    """prefix 和 extend KV 通过 unified_kv_indices 以相同路径访问"""
    for start_n in range(0, cur_seq_kv_len, BLOCK_N):
        # 不区分 prefix/extend，统一遍历
        kv_idx = tl.load(unified_kv_indices + ...)
```

## 解决方案分析

### 方案 1: 统一 KV 逻辑布局 (SGLang 已实现)

**核心思想**: 在进 attention kernel 之前，将 cached prefix KV 和 fresh extend KV 统一到一个逻辑地址空间。

```
Before (2-stage, non-deterministic):
  stage 1: Q_suffix @ K_prefix_cached → partial_output_1, lse_1
  stage 2: Q_suffix @ K_suffix_fresh  → partial_output_2, lse_2
  merge(partial_output_1, lse_1, partial_output_2, lse_2)

After (unified, deterministic):
  unified_kv = build_unified_index(prefix_kv_indices, extend_kv_indices)
  Q_suffix @ K_unified → full_output  (单次 reduction，与 full prefill 等价)
```

**关键点**:
- 物理内存仍然是 paged/block pool（不需要 memcpy）
- 只是让 kernel 通过统一的 index 序列遍历所有 KV，保证 reduction 顺序一致
- 需要配合 **fixed split tile size** (SGLang 默认 256)，使得不同 KV 长度下分块方式一致

**SGLang 代码路径**:
```python
# triton_backend.py
if self.enable_deterministic:
    self.split_tile_size = 256  # fixed, not adaptive
    return self._forward_extend_unified(...)  # 统一 1-stage kernel
```

**局限**: 统一布局只解决了 "prefix/extend 分开计算" 的问题。如果 prefix cache hit 时 KV 的**值本身就不同**（因为之前缓存时的 batch shape 不同），还需要 batch-invariant ops。

### 方案 2: Batch-Invariant Ops (SGLang + vLLM 均部分实现)

替换 PyTorch 默认的 matmul/reduction kernel，确保不同输入 shape 下走相同的 reduction 路径：

```python
# sglang/srt/batch_invariant_ops/batch_invariant_ops.py

def enable_batch_invariant_mode():
    """替换 PyTorch ops 为 batch-invariant 版本"""
    torch.library.impl(aten.mm, "CUDA")(mm_batch_invariant)
    torch.library.impl(aten.addmm, "CUDA")(addmm_batch_invariant)
    torch.library.impl(aten._log_softmax, "CUDA")(log_softmax_batch_invariant)
    torch.library.impl(aten.mean.dim, "CUDA")(mean_batch_invariant)
```

- **mm/addmm**: 使用 persistent block kernel，固定 tile 顺序（不随 M 维度变化而改变 split-K 策略）
- **softmax/mean**: 固定 per-element reduction 路径，不因向量长度不同而改变分块
- 代价: 20-35% 性能损失

### 方案 3: Fixed Split-KV Size (Attention Decode)

Decode 阶段的 split-KV attention 需要固定 split 大小：

```python
# SGLang: triton_backend.py
self.split_tile_size = 256  # 固定，不根据 KV 长度自适应

# FlashAttention backend: flashattention_backend.py
self.num_splits = 1  # 强制不 split (牺牲并行度)
```

- 确保 KV 长度变化不影响 split 边界 → reduction 顺序一致
- FlashAttention 3 在 SGLang 中只能设 num_splits=1（性能损失较大）

### 方案对比

| 方案 | 解决的问题 | 性能代价 | 实现复杂度 |
|------|-----------|---------|-----------|
| 统一 KV 布局 | prefix/extend 分开计算的差异 | 低（只改 index） | 中 |
| Batch-invariant ops | 不同 batch shape 下的 kernel 选择差异 | 高 (20-35%) | 高 |
| Fixed split-KV | decode 阶段 split 数变化 | 中等 | 低 |
| Disable radix cache | 彻底避免 prefix 复用 | 高（失去 cache 收益） | 无 |

### 完整解决方案 (SGLang 的方法)

```bash
# Level 1: 基础 deterministic (解决 ~90% 问题)
--enable-deterministic-inference
# → 启用 batch-invariant ops + unified KV layout + fixed split

# Level 2: 完全 deterministic (解决剩余 prefix cache 问题)
--enable-deterministic-inference --disable-radix-cache
# → 彻底关闭 prefix 复用，所有请求都 fresh prefill
# → 代价: 失去 prefix caching 的吞吐收益
```

### 开放问题

1. **能否在保留 prefix caching 的同时实现完全 deterministic?**
   - 理论上可以：如果 batch-invariant ops 保证 KV 值不随 batch 变化，且 attention kernel 使用统一布局 + fixed split
   - SGLang 的 unified kernel 已经做到了大部分，但仍有边缘 case（所以还是推荐 disable-radix-cache）
   - 关键难点: radix cache 中存储的 KV 可能来自不同的 "batch 上下文"（不同并发度），除非所有 op 都是 batch-invariant

2. **Chunked prefill 的解决方案更成熟**
   - vLLM chunk=128 已达 0% flip，SGLang 也接近 0%
   - 因为 chunked prefill 不涉及 "不同来源的 KV"，只是分段执行，引擎已优化好

3. **性能与 determinism 的 trade-off**
   - 当前 SGLang: 34.35% 性能损失 (batch-invariant mode)
   - 主要来源: 禁用 split-K、禁用动态 tile、固定 reduction 策略

## 测试代码

| 文件 | 说明 |
|------|------|
| `paged_attention_sim.py` | 手工 paged attention 实现（PyTorch） |
| `test_kvcache_layout_determinism.py` | HF + SDPA 测试（Scenario 1/A/B） |
| `test_vllm_kvcache.py` | vLLM 测试（Scenario A/B） |
| `test_sglang_kvcache.py` | SGLang 测试（Scenario A/B） |

### 运行方式

```bash
# HF tests
python3 test_kvcache_layout_determinism.py --scenarios 1,2,3 --num-runs 1000

# SGLang tests (需要 CUDA 12.9 nvcc)
PATH=/usr/local/cuda-12.9/bin:$PATH python3 test_sglang_kvcache.py --scenarios a,b --num-runs 1000

# vLLM tests (需要 conda env 隔离)
PYTHONNOUSERSITE=1 PYTHONPATH="" /home/kec23008/miniconda3/envs/vllm_test/bin/python3 test_vllm_kvcache.py --scenarios a,b --num-runs 1000
```
