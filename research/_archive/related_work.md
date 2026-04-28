# Related Work (Complete)

## 6.1 Numerical Precision in Deep Learning

Mixed-precision training with FP16 weights and FP32 master copies was established by Micikevicius et al. (2018). Kalamkar et al. (2019) introduced BF16 for training, exploiting its wider dynamic range at the cost of reduced mantissa precision. Blanchard et al. (2020) analyzed subspace-descent methods specifically for BF16 computations. The theoretical foundation for our error analysis derives from Higham (2002), whose recursive summation error bounds (Theorem 4.4) we extend to the batch-invariance setting. Crucially, our work addresses a different concern than these: not the *accuracy* of reduced-precision inference (whether BF16 outputs are close to FP32 outputs), but its *determinism* (whether BF16 outputs are identical across batch configurations).

## 6.2 Deterministic GPU Computation

PyTorch provides `torch.use_deterministic_algorithms(True)` and the `CUBLAS_WORKSPACE_CONFIG` environment variable to enforce deterministic behavior in CUDA operations (Paszke et al., 2019). NVIDIA's cuDNN offers deterministic algorithm selection flags. However, these mechanisms address *run-to-run* non-determinism caused by atomic operations in backward passes and non-deterministic reduction kernels. They do *not* address *batch-composition* non-determinism, which arises from cuBLAS kernel selection heuristics in the forward pass. Our work is orthogonal: we identify and resolve a distinct class of non-determinism that persists even when all existing determinism flags are enabled.

## 6.3 Deterministic LLM Serving Systems

The problem of batch-invariant LLM inference has recently attracted attention from serving system developers.

**Thinking Machine Lab (2025)** published "Defeating Nondeterminism in LLM Inference," identifying kernel selection (not atomic operations) as the root cause of batch-dependent non-determinism. They proposed custom GEMM, attention, and RMSNorm kernels that fix reduction orders regardless of batch size, achieving determinism at 61.5% latency overhead in their initial implementation. Our work formalizes their empirical insight with Theorems 1--3, and demonstrates that for HuggingFace inference, a single cuBLAS flag achieves the same GEMM determinism at zero overhead.

**SGLang** (Zheng et al., 2025) implemented a deterministic inference mode activated via `--enable-deterministic-inference --disable-radix-cache`. Their approach includes fixed split-KV attention (using FlashInfer and FA3 backends), deterministic NCCL all-reduce for tensor parallelism, and batch-invariant matmul/reduction ops. The reported performance overhead is 34.35%. Our Theorem 3 provides the theoretical justification for their fixed split-KV design, and our cuBLAS flag finding is complementary --- it should be applied as a zero-cost first step before more expensive kernel interventions.

**vLLM** (Kwon et al., 2023) is developing a batch-invariant mode (`VLLM_BATCH_INVARIANT=1`) using FlexAttention with fixed tile configurations (`BLOCK_M`, `BLOCK_N`, `IS_DIVISIBLE`), combined with `batch_invariant_ops` that override PyTorch's default matmul and reduction kernels (PR #24583). The estimated overhead is 20--35%.

Our contribution relative to these systems: (1) we provide the *theoretical framework* (Theorems 1--3) explaining why serving engines need heavy interventions while HuggingFace does not; (2) we demonstrate that a single cuBLAS flag eliminates GEMM batch variance at zero cost; and (3) we identify MoE routing as an open problem that none of these systems fully address.

## 6.4 FlashAttention and Efficient Attention

FlashAttention (Dao et al., 2022) introduced tiled attention with online softmax, achieving IO-aware exact attention computation. FlashAttention-2 (Dao, 2023) improved parallelism and work partitioning. FlashDecoding (Tri Dao et al., 2023) introduced split-KV parallelism for the decode phase, splitting the KV sequence across SMs to improve GPU utilization when the query length is small. Our Theorem 2 formalizes why this split-KV approach introduces batch-dependent non-determinism: different batch sizes lead to different split counts, producing structurally different computation graphs with cascaded multiplicative rescaling.

FlashDecoding++ (Hong et al., 2023) proposed using a unified maximum value for softmax to simplify the cross-split reduction, which partially addresses the structural mismatch but does not eliminate it. FlashInfer (Ye et al., 2024) provides configurable attention kernels with explicit control over split strategies, enabling the fixed-split design prescribed by our Theorem 3.

## 6.5 Mixture-of-Experts Routing

The Mixture-of-Experts architecture was introduced by Shazeer et al. (2017) and popularized for large-scale models by Switch Transformers (Fedus et al., 2022) and ST-MoE (Zoph et al., 2022). Recent production models including DeepSeek-V2 (DeepSeek-AI, 2024) and Qwen3-MoE (Qwen Team, 2025) employ MoE for efficient scaling. Prior work on MoE routing has focused on load balancing, routing collapse, and expert specialization during *training*. Our work identifies a new failure mode: *batch-composition-dependent expert selection during inference*, where sub-ULP GEMM perturbations propagate through the gate-softmax-topk pipeline to flip expert choices. This is a numerical precision problem fundamentally different from training dynamics.

## 6.6 Reproducibility in Machine Learning

Henderson et al. (2018) demonstrated that deep RL results are highly sensitive to implementation details, with reward variance often exceeding the effect of algorithmic improvements. Pineau et al. (2021) proposed reproducibility checklists for ML research. Our work connects *numerical determinism* in inference to these practical reproducibility concerns, quantifying that batch-dependent non-determinism injects 2.3% mean reward signal variance and 6.3e-4 mean KL divergence in distillation targets --- noise that is systematic, not random, and proportional to batch composition differences.
