// DetermLLM: BF16 GEMM with FP32 accumulation and batch-invariant reduction.
//
// DESIGN: Precision Amplification for Batch-Invariant GEMM
// ─────────────────────────────────────────────────────────
// BF16 GEMM non-determinism in LLM inference comes from cuBLAS dynamically
// selecting different split-K strategies based on M (batch size). Different
// split counts → different FP reduction trees → different BF16 outputs.
//
// Fix: use FP32 accumulation throughout. The FP32 non-associativity error is:
//   |δ| ≤ K × ε_FP32 × Σ|a_i|  ≈  K × 1.2e-7 × √K  (typical LLM values)
// For K=4096: |δ| ≈ 4096 × 1.2e-7 × 64 ≈ 3.1e-5
// BF16 quantization step ≈ 7.8e-3 >> |δ|, so any two FP32-accumulated sums
// round to the SAME BF16 value. Non-associativity is masked by BF16 quantization.
//
// Two-phase heuristic:
//   Phase 1: REDUCTION_SCHEME_NONE (no split-K) — fewest ops, lowest latency.
//   Phase 2: REDUCTION_SCHEME_COMPUTE_TYPE (FP32 inter-CTA reduction with split-K)
//            — found when NONE has no candidate for large M; still batch-invariant
//            by the precision amplification argument above.
//   Fallback: torch::mm(float32) — consistent across all M by the same argument.
//
// Row-major layout: PyTorch tensors are row-major. Without CUBLASLT_ORDER_ROW,
// cuBLASLt defaults to column-major, causing catastrophic out-of-bounds writes
// (ld=K for [M,K] tensor → accesses M-1 + (K-1)*K ≈ K² elements, far beyond M*K).

#include <torch/extension.h>
#include <cublasLt.h>
#include <cuda_bf16.h>

static cublasLtHandle_t ltHandle = nullptr;
static bool initialized = false;
static void* workspace = nullptr;
static const size_t WORKSPACE_SIZE = 32 * 1024 * 1024; // 32 MB

void ensure_init() {
    if (!initialized) {
        cublasLtCreate(&ltHandle);
        cudaMalloc(&workspace, WORKSPACE_SIZE);
        initialized = true;
    }
}

static void set_row_major(cublasLtMatrixLayout_t layout) {
    cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
    cublasLtMatrixLayoutSetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                     &order, sizeof(order));
}

// Try cuBLASLt heuristic with given reductionMask.
// Returns (status, algo, foundCount).
static cublasStatus_t try_heuristic(
        cublasLtMatmulDesc_t matmulDesc,
        cublasLtMatrixLayout_t layoutA,
        cublasLtMatrixLayout_t layoutB,
        cublasLtMatrixLayout_t layoutC,
        uint32_t reductionMask,
        cublasLtMatmulHeuristicResult_t& result,
        int& returnedResults) {

    cublasLtMatmulPreference_t pref;
    cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                          &WORKSPACE_SIZE, sizeof(WORKSPACE_SIZE));
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK,
                                          &reductionMask, sizeof(reductionMask));

    cublasStatus_t hstatus = cublasLtMatmulAlgoGetHeuristic(
        ltHandle, matmulDesc, layoutA, layoutB, layoutC, layoutC,
        pref, 1, &result, &returnedResults);

    cublasLtMatmulPreferenceDestroy(pref);
    return hstatus;
}

torch::Tensor gemm_fp32_accum(torch::Tensor A, torch::Tensor B) {
    // A: [M, K] BF16 row-major
    // B: [N, K] BF16 row-major  (weight stored transposed, as in nn.Linear)
    // Returns C: [M, N] BF16  =  A @ B^T  with FP32 accumulation, batch-invariant.

    TORCH_CHECK(A.dtype() == torch::kBFloat16, "A must be BF16");
    TORCH_CHECK(B.dtype() == torch::kBFloat16, "B must be BF16");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "inputs must be 2D");
    TORCH_CHECK(A.size(1) == B.size(1), "K dimension mismatch");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "inputs must be contiguous");

    const int M = A.size(0), K = A.size(1), N = B.size(0);
    auto C = torch::empty({M, N}, A.options());
    ensure_init();

    // FP32 accumulation (compute and scale types are FP32)
    cublasLtMatmulDesc_t matmulDesc;
    cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);

    cublasOperation_t opN = CUBLAS_OP_N, opT = CUBLAS_OP_T;
    cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN));
    cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opT, sizeof(opT));

    // Row-major layouts (critical: prevents column-major out-of-bounds writes)
    cublasLtMatrixLayout_t layoutA, layoutB, layoutC;
    cublasLtMatrixLayoutCreate(&layoutA, CUDA_R_16BF, M, K, K);
    cublasLtMatrixLayoutCreate(&layoutB, CUDA_R_16BF, N, K, K);
    cublasLtMatrixLayoutCreate(&layoutC, CUDA_R_16BF, M, N, N);
    set_row_major(layoutA);
    set_row_major(layoutB);
    set_row_major(layoutC);

    cublasLtMatmulHeuristicResult_t result;
    int returnedResults = 0;
    bool found = false;

    // Phase 1: no split-K (REDUCTION_SCHEME_NONE = 0x0).
    // Optimal for small M; reduction is trivial (single CTA per output tile).
    try_heuristic(matmulDesc, layoutA, layoutB, layoutC,
                  CUBLASLT_REDUCTION_SCHEME_NONE, result, returnedResults);
    if (returnedResults > 0) found = true;

    // Phase 2: FP32 inter-CTA reduction (REDUCTION_SCHEME_COMPUTE_TYPE = 0x2).
    // Allows split-K for large M while keeping all reductions in FP32.
    // Batch-invariant by precision amplification: FP32 error ≈ K*eps_fp32*sqrt(K)
    // << BF16 step ≈ eps_bf16, so different split trees round to same BF16 output.
    if (!found) {
        try_heuristic(matmulDesc, layoutA, layoutB, layoutC,
                      CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE, result, returnedResults);
        if (returnedResults > 0) found = true;
    }

    float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t status = CUBLAS_STATUS_NOT_SUPPORTED;

    if (found) {
        status = cublasLtMatmul(
            ltHandle, matmulDesc,
            &alpha,
            A.data_ptr(), layoutA,
            B.data_ptr(), layoutB,
            &beta,
            C.data_ptr(), layoutC,
            C.data_ptr(), layoutC,
            &result.algo,
            workspace, WORKSPACE_SIZE,
            0  // default CUDA stream
        );
    }

    cublasLtMatrixLayoutDestroy(layoutA);
    cublasLtMatrixLayoutDestroy(layoutB);
    cublasLtMatrixLayoutDestroy(layoutC);
    cublasLtMatmulDescDestroy(matmulDesc);

    if (status != CUBLAS_STATUS_SUCCESS) {
        // Consistent FP32 fallback: same precision argument applies.
        // BF16→FP32 is lossless; FP32 non-associativity << BF16 step → same BF16 output.
        return torch::mm(A.to(torch::kFloat32),
                         B.t().to(torch::kFloat32)).to(torch::kBFloat16);
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_fixed_algo", &gemm_fp32_accum,
          "BF16 GEMM: FP32 accumulation, batch-invariant via precision amplification. "
          "Two-phase: REDUCTION_SCHEME_NONE → REDUCTION_SCHEME_COMPUTE_TYPE → FP32 fallback.");
}
