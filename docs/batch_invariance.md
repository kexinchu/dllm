## Batch-Invariance

### Blog from Thinking Machine Lab
- [Link](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
- 浮点数计算的非结合性(具有不同的"小数位数"时)
```py
import random

vals = [1e-10, 1e-5, 1e-2, 1]
vals = vals + [-v for v in vals]

results = []
random.seed(42)
for _ in range(10000):
    random.shuffle(vals)
    results.append(sum(vals))

results = sorted(set(results))
print(f"There are {len(results)} unique results")

# Output:
# There are 102 unique results
```
- GPU上的原子操作(类似"原子加法")时"非确定性的"
    - 涉及多个SM核心参与同一个向量的规约，执行顺序完全取决于哪个核心先完成计算
    - 但是问题源头并不是来自于“GPU本身并行就不确定”；而是kernel选择
        - 为了得到最佳效率，对于 matmul / attention 这类带 reduction 的 kernel，会根据 batch 大小、形状动态选择不同的分块 / split-K / tensor core 指令。
        - 不同的划分 → 不同的加法顺序 → 浮点非结合性 → logit 有微小差异 → greedy 采样在某个 token 开始走到另一条路径。
        - 影响：即使temperature = 0， 也会返回不同的结果：(作者在vLLM下执行推理，相同prompt采样1000次得到了 80 个不同的结果)
    - 问题定义“batch invariance”: 对同一个样本，无论它是在 batch_size=1 跑，还是和其他请求一起 batch_size=32 跑，甚至 batch 内位置变化，该样本的每一步数值轨迹都要一模一样
    - Horace的方案是：重写RMSNorm/matual/attention这三个关键op，保证其reduction 方式对 batch尺度 / chunking / prefix caching 不敏感，包括：
        - RMSNorm：
        ```py
        # x: [batch_size, hidden_dim]
        def rms_norm(x, weight):
            return x * torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True)) * weight
        ```
            - 修改前：
                - 大 batch 时：data-parallel，每个 batch element（一行）交给一个 core/SM，整条向量的 mormalization 在该 core 内完成。
                - 小 batch 时：为了“把所有 SM 喂饱”，会启用 split reduction：一个向量被拆成多个 chunk，分给多个 core 做局部 sum，再在最后合并。 => 可能会导致顺序改变
            - 修改后：
                - 坚持使用data parallel, 每个element的normalization始终由一个core持有并完成；batch增大时，就多开启一些thread block；小 batch 时，牺牲并行度。
        <img src="./pictures/RMSNorm.jpg" width=400>
        
        - Matmul：
            - 修改前：
                - “正常情况”：data parallel，将输出matrix切分成2D tiles，每个tile分给一个core，在这个core内完成所有dot-prodect的规约计算
                - 当在小batch下，如果K比较大，为了饱和GPU，会使用Split-K/Stream-K策略，在K维进行拆分，使多个core计算同一个输出tile的不同K区间，再合并 => 导致顺序改变
                - 另外，tensor core选择也有影响：大tile在batch size大时效率高，小batch时，会切换小tile。也会导致内部的累加顺序改变
            - 修改后：
                - 对 matmul 只保留一套内核配置(tile size + tensor core指令 + parallel划分方案)，只按照输出2D tiles做data parallel，不再拆分K；同时禁用 split-K/stream-K、动态tensor core 的策略
                - 相比于cuBLAS，损失了~20% 的性能，小batch时更明显
        - Attention：
            - 比较复杂，在feature dim 和 sequence dim上分别进行规约
            - 修改前（以FlashAttention2为例）：
                - 并行策略：data-parallel in Q，reduce over K/V
                - 在如vllm的系统中，会结合
                    - KV-Cache单独处理：
                        - Paged KV-Cache / Chunked prefill
                        - KV-Cache不连续 => 计算时对于每一个KV-Cache block分别规约(P_cache)，再对当前新token的KV block规约(P_current); 最后再合并
                        - example
                        ```shell
                        想象 KV cache 有 80 个 token，新来 48 个 token：
                        实现 A：把 80 个 cache 元素分成 3 个 block（两满一残），48 个新元素分成 2 个 block ⇒ 总共 5 个 block 的规约。
                        实现 B（比如一次性处理 128 个元素，没有 cache 概念）：用 4 个 block 规约。
                        ```
                    - Split-KV/FlashDecoding
                        - decode 阶段 query length 很小，几乎无法沿 Q 并行，所以会沿 KV 长度方向拆分，多个 core 处理不同 KV 区间，再合并 => 块间顺序改变
                        - 给定需要的并行度，算出“需要 𝑆 个 split”，然后把 KV 长度 𝐿 均匀分成 𝑆 段，每段长度约 𝐿/𝑆。 => 分块改变会导致顺序改变 (a|bc vs. ab|c)
            - 修改之后：
                - 统一KV布局，消除"KV-Cache 和 当前token"的分开规约 => 连续起来
                    - update the KV cache and page table before the attention kernel itself, ensuring that our keys and values are always consistently laid out regardless of how many tokens are being processed.
                    - kernel 内部不再区分“cache 部分”和“当前 token 部分”，只是在一个统一的 K 维度上规约。
                - 选择一个固定的 split 大小 𝐶（例如 256），对任意 KV 长度 𝐿，按 [0:𝐶),[𝐶:2𝐶),… 切分，最后一段可能是残长 ≤𝐶。
                    - 合并阶段也固定为“按这些固定 index 区间的顺序做规约”
    - 影响： 初步实现之后，能够实现批次不变性(1000次采样得到相同的结果；但是延迟增加了61.5%)

### vLLM中的适配
- link: https://docs.vllm.ai/en/latest/features/batch_invariance/
- 提供开关： “VLLM_BATCH_INVARIANT=1”；
- 开发中，目前支持部分硬件 + 部分模型；开启后会禁用某些可能引入不确定性的优化(例如张量并行模式下的自定义 all-reduce 操作)6
- 适配PR(https://github.com/vllm-project/vllm/pull/24583)
    - 在 FlexAttention backend 里固定 BLOCK_M/BLOCK_N/IS_DIVISIBLE 等 kernel 配置，让 attention kernel 的 tile / mask 行为固定下来；(https://arxiv.org/pdf/2412.05496)
    - 再配合 batch_invariant_ops 里对 torch.mm/addmm/mean/log_softmax 的 override，替换 PyTorch 默认的 matmul/reduction kernels。
    - 固定 tile / 固定 split 规则 + 统一 KV logical layout
    - 对 KV-cache, 内存仍是 paged/block pool;会通过更新page table的方式将cache + current tokens 统一成一个逻辑序列。
    - 即 不用砍掉 PagedAttention 的内存管理，只是要换一套 batch-invariant attention backend 来吃这套布局

### sglang中的适配
- 这个问题SGlang中已经做了集成[link](https://lmsys.org/blog/2025-09-22-sglang-deterministic/); 通过"--enable-deterministic-inference + --disable-radix-cache" 开启
- 适配PR: https://github.com/sgl-project/sglang/issues/10278
- 目标：
    - Attention Backend（FlashInfer / Triton / FA3）
    - NCCL 通信（TP 下 deterministic all-reduce）
    - Radix Cache Support：“Making Prefill with Radix Cache has the same output as Prefill without Radix cache”
- 目前的问题：
    - 性能损失: Batch-Invariant 要求GPU在不同 batch 下走完全相同的计算路径 → 把 GPU 里所有的动态调优、动态切分、动态并行、动态缓存复用通通锁死了；(sglang当前版本仍造成34.35%的性能损失)

### KV-Cache管理
- 并不要求物理上联系的KV-Cache管理，依旧可以复用paged/block KV-Cache管理
- 在进attention kernel之前，通过更新page table / metadata，将同一个sequence的KV-Cache 和 current KV以一致的逻辑顺序呈现。
- 固定 tile / 固定 split 规则 + 统一 KV logical layout

### 从系统角度

| Component / 模块 | 为什么会影响 deterministic / batch-invariant inference？| vLLM 现状  | SGLang 现状 |
| ----- | ------- | --------- | --------- |
| **RMSNorm / LayerNorm / mean / log_softmax 等 reduction op** | -  | 集成了 TML 的 `batch_invariant_ops`，整体：**dense+单卡：✅；MoE/TP：未完全覆盖**。  | 集成了 TML 的 `batch_invariant_ops`，整体：**dense+单卡：✅；MoE/TP：受限于后面通信问题**。 |
| **Matmul / Linear GEMM**  | -  | batch-invariant 模式下使用 `mm_batch_invariant` / `matmul_persistent` 等，固定 tile 配置和 split-K 策略；结合 FlexAttention 的路径，保证 dense 模型在单卡上 matmul 是 batch-invariant 的。**单卡 dense：✅；多卡/部分量化 / MoE fused GEMM：未完全覆盖**。   | deterministic 模式下同样通过集成 `batch_invariant_ops` 替换 matmul，实现固定的 reduction 逻辑；在 Qwen3-8B TP=1 上通过 single/mixed/prefix 测试。对 MoE/TP 的 GEMM 还没有系统性 batch-invariant 保证。**dense+单卡：✅；MoE/TP：未完全覆盖**。|
| **Self-Attention 核心（QKᵀ, softmax, AV）**  | - | Batch-invariant 模式强制使用 FlexAttention backend，并配合 batch-invariant kernel 实现固定 split-KV、统一 prefill/decode 路径。对 **dense, TP=1** 的 text-only 模型基本解决；对 **VLM / 多模态 attention** 官方有 issue 在补，尚未完全 batch-invariant。**文本 dense：✅；VLM/特殊 backend：未完全覆盖**。 | 针对 **FlashInfer / FlashAttention-3 / Triton** 三类 backend，分别实现 deterministic 版本：固定 split-KV size，prefill & decode 使用相同 kernel 配置，并让 chunked prefill 与 split-KV 对齐。Qwen3-8B TP=1 已验证通过。对 VLM / 部分 backend + 大 TP 仍有问题。**文本 dense：✅；多模态/大 TP：未完全覆盖**。 |
| **Chunked Prefill（长 context 切分）**   | 长序列通常被切成多个 chunk 逐步送入 KV cache。若 chunk 的切分点根据当前 batch 负载动态调整，同一长序列在不同并发度下会被切成不同的段，每段参与 attention/reduction 的方式就不同，造成数值路径差异。  | **未优化**  | 重写了 chunked prefill 算法，使 **chunk 边界与 split-KV size 对齐**，并为每条序列固定切分 pattern，使引入更多 request 时不会改变该序列自身的 chunk 序列。Qwen3-8B TP=1 场景验证通过。**单卡 dense：✅；和 radix cache/大 TP 联动仍在演进：未完全覆盖**。 |
| **KV Cache 布局 / Paged / Radix Cache**  | KV cache 在内存中的布局会决定 attention kernel 如何读取和分块。如果不同 batch 场景下 page/segment 合并、prefix 共享/重排方式不同，同一 token 在 kernel 中所参与的分块结构会不同，从而改变 reduction 顺序。 | vLLM 的 paged KV cache 是核心特性；batch-invariant 模式通过 FlexAttention + 固定 layout 的思路减少布局变化对数值的影响，但是： prefix-sharing 策略、compression、offload 等高级玩法在 batch invariance 模式下支持有限。**基础 paged KV + 单卡 dense OK；更激进的缓存/压缩：未完全覆盖**。| SGLang 当前 deterministic 模式主要和普通 KV cache + chunked prefill 集成，radix cache 在 FlashInfer/Triton 下是**暂时关闭**。**普通 KV：✅；radix cache：目前暂不支持（为 determinism 关闭）**。|
| **Scheduler：动态 batching / request 排布**  | 只影响batch的管理，只要消除batch variance, 就可以消除影响  | -  | - |
| **分布式 Tensor Parallel / DP 下的 all-reduce / all-gather**     | TP/DP 需要 all-reduce / all-gather；浮点 all-reduce 的加法顺序依赖 rank 排布、chunk 切分、通信算法（ring/tree/nvls）。  | vLLM 暂时 **只对 TP=1 有强保证**；**TP>1：暂不支持**。  | SGLang 已经在 Qwen3-8B 上测试了 TP=2/4；结果显示：TP=2 可以通过 deterministic test；TP=4 在 Blackwell 上 prefix 模式仍然出现不一致。主要原因是 NVLS 自定义 all-reduce 目前不 deterministic。**TP=1：✅；TP=2：部分 ✅；TP>=4：暂不支持**。    |
| **MoE: gating（softmax+top-k） & expert dispatch**    | gating 是对每个 token 的 expert logits 做 softmax+top-k，本质上又是一个 reduction+排序过程，对数值微小扰动高度敏感。随后 expert dispatch 会引入 all-to-all 通信、不同 expert 的 GEMM。当 batch 改变或通信/reduction 路径不稳定时，很容易导致选择不同 expert，从而完全改变后续轨迹。 | **MoE：暂不支持**。    | TP>1 + MoE 的测试已观察到明显不 determinism。**MoE：暂不支持**。  |
| **量化 kernel（FP8/INT8、nvFP4 等）**  | 量化/反量化通常包含 scale 估计、clamp、整数 GEMM + 反缩放，其中也会有 reduction、max/min、统计计算。当 kernel 和调度对 batch/形状敏感时，同一输入在不同 batch 下可能得到不同的 scale/clip 行为，引起微小数值差异。  | **量化 + batch invariance：暂不支持**。  | **量化 deterministic：暂不支持**。  |
| **VLM / Vision encoder + cross-attention**   | VLM 在文本 attention 之外，还包含 image patch embedding / ViT / resampler、以及 text–image cross-attention。这些模块有自己的 big matmul/reduction，与文本部分 share 或不 share kernel。若这些 kernel 不 batch-invariant，那么多模态场景下 deterministic 更难保证。 | 官方有 [issue](https://github.com/vllm-project/vllm/issues/27059)，还在doing：**text-only dense 相对成熟，VLM 仍在 TODO**。  | SGLang 的 deterministic 工作主要在 text-only LLM 上**多模态 deterministic：暂不支持**。  |


### 其他问题
- Posttraining 前向和反向之间share 中间结果吗？
    - Rollout阶段（推理）
        - SGLang 接收 prompt → 前向生成 output tokens、logprobs、value predictions（如果 actor-critic 模型）
        - SGLang 不保留（或只轻微保留）用于推理的中间 activations；主要保存生成的结果数据（比如 sequence, logprob, value）
        - 这些生成数据被送入训练数据池
    - 训练阶段
        - Megatron 从数据池读入生成数据 +原始状态
        - Megatron 加载当前模型权重（可能已被从推理端同步）
        - Megatron 对输入（相同 prompt）执行 自己的前向（产生 activations）→ 计算 loss（如 policy loss, value loss, critic loss）→反向更新权重
        - 更新后的权重再同步回推理端（SGLang）用于下一轮rollout

- NCCL中的deterministic
    - NCCL 负责的是多卡之间的 all-reduce / all-gather / reduce-scatter 等通信
    - NCCL 的 ring / tree / hierarchical 算法会按拓扑做分段加法，顺序可能随进程数量、分组策略变化而变化
    - 新版 NCCL 提供了一些 环境变量, 确保在 同一个并行配置(拓扑/rank等)下，尽量保证运算路径固定，使多次运行结果稳定 =>  **需要在上层保证每次all-reduce的参与rank，tensor切分方式，调用顺序一致**
