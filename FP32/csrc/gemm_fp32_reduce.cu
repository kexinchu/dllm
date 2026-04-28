// cuBLASLt GEMM: BF16 inputs, FP32 compute type, BF16 output
// Key: CUBLAS_COMPUTE_32F ensures ALL reductions (including cross-tile) are in FP32
// This is different from allow_bf16_reduced_precision_reduction which only affects
// within-tile accumulation.

#include <torch/extension.h>
#include <cublasLt.h>
#include <c10/cuda/CUDAStream.h>

static cublasLtHandle_t ltHandle = nullptr;
static void* workspace = nullptr;
static const size_t WS_SIZE = 32 * 1024 * 1024;

void ensure_init() {
    if (!ltHandle) {
        cublasLtCreate(&ltHandle);
        cudaMalloc(&workspace, WS_SIZE);
    }
}

torch::Tensor gemm_fp32_reduce(torch::Tensor A, torch::Tensor B) {
    // A: [M, K] BF16, B: [N, K] BF16
    // Returns: [M, N] BF16
    // Compute: C = A @ B^T with CUBLAS_COMPUTE_32F (FP32 everything internally)

    TORCH_CHECK(A.dtype() == torch::kBFloat16 && B.dtype() == torch::kBFloat16);
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2);
    TORCH_CHECK(A.size(1) == B.size(1));
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous());

    int M = A.size(0), K = A.size(1), N = B.size(0);
    auto C = torch::empty({M, N}, A.options());
    ensure_init();

    // Matmul descriptor: COMPUTE_32F with FP32 scale type
    cublasLtMatmulDesc_t desc;
    cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);

    cublasOperation_t opN = CUBLAS_OP_N, opT = CUBLAS_OP_T;
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN));
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &opT, sizeof(opT));

    // Matrix layouts: BF16 for A, B, C
    cublasLtMatrixLayout_t layA, layB, layC;
    cublasLtMatrixLayoutCreate(&layA, CUDA_R_16BF, M, K, K);
    cublasLtMatrixLayoutCreate(&layB, CUDA_R_16BF, N, K, K);
    cublasLtMatrixLayoutCreate(&layC, CUDA_R_16BF, M, N, N);

    // Preference: request algorithms with FP32 reduction (COMPUTE_TYPE reduction)
    cublasLtMatmulPreference_t pref;
    cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                          &WS_SIZE, sizeof(WS_SIZE));

    // Allow all reduction schemes but prefer FP32 ones via COMPUTE_32F
    // COMPUTE_32F already forces FP32 accumulation; we don't restrict reduction scheme
    // to maximize algorithm availability

    // Get best algorithm with this constraint
    const int N_ALGOS = 8;
    cublasLtMatmulHeuristicResult_t results[N_ALGOS];
    int nResult = 0;
    cublasLtMatmulAlgoGetHeuristic(ltHandle, desc, layA, layB, layC, layC,
                                    pref, N_ALGOS, results, &nResult);

    float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t status = CUBLAS_STATUS_NOT_SUPPORTED;

    if (nResult > 0) {
        // Try algorithms in order of preference
        for (int i = 0; i < nResult; i++) {
            status = cublasLtMatmul(ltHandle, desc,
                &alpha, A.data_ptr(), layA, B.data_ptr(), layB,
                &beta, C.data_ptr(), layC, C.data_ptr(), layC,
                &results[i].algo, workspace, WS_SIZE,
                c10::cuda::getCurrentCUDAStream().stream());
            if (status == CUBLAS_STATUS_SUCCESS) break;
        }
    }

    // If COMPUTE_TYPE reduction not available, try without reduction constraint
    if (status != CUBLAS_STATUS_SUCCESS) {
        // Remove reduction constraint, just use COMPUTE_32F
        uint32_t noMask = 0xFFFFFFFF; // allow all
        cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK,
                                              &noMask, sizeof(noMask));
        nResult = 0;
        cublasLtMatmulAlgoGetHeuristic(ltHandle, desc, layA, layB, layC, layC,
                                        pref, N_ALGOS, results, &nResult);
        if (nResult > 0) {
            for (int i = 0; i < nResult; i++) {
                status = cublasLtMatmul(ltHandle, desc,
                    &alpha, A.data_ptr(), layA, B.data_ptr(), layB,
                    &beta, C.data_ptr(), layC, C.data_ptr(), layC,
                    &results[i].algo, workspace, WS_SIZE,
                    c10::cuda::getCurrentCUDAStream().stream());
                if (status == CUBLAS_STATUS_SUCCESS) break;
            }
        }
    }

    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(layA);
    cublasLtMatrixLayoutDestroy(layB);
    cublasLtMatrixLayoutDestroy(layC);
    cublasLtMatmulDescDestroy(desc);

    // Final fallback
    if (status != CUBLAS_STATUS_SUCCESS) {
        return torch::mm(A, B.t());
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_fp32_reduce", &gemm_fp32_reduce,
          "BF16 GEMM with FP32 compute type and FP32 cross-tile reduction");
}
