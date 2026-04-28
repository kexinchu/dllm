/*
 * GEMM with BF16 inputs, FP32 accumulation, BF16 output.
 * Uses cuBLASLt (fused kernel in inference engine; no Python-level Linear replacement).
 * Direct row-major GEMM: C = A @ B without transpose tricks.
 */
#include <c10/cuda/CUDAStream.h>
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

#define CUBLASLT_CHECK(expr)                                                                    \
  do {                                                                                          \
    cublasStatus_t __err = (expr);                                                              \
    TORCH_CHECK(__err == CUBLAS_STATUS_SUCCESS, "cuBLASLt error ", (int)__err, ": ", #expr);   \
  } while (0)

// C = A @ B, A (M,K), B (K,N), C (M,N). BF16 in/out, FP32 accum.
// Use cuBLAS column-major with transpose to handle row-major PyTorch tensors.
torch::Tensor gemm_fp32_accum_cuda_impl(torch::Tensor a, torch::Tensor b) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda(), "inputs must be CUDA tensors");
  TORCH_CHECK(a.dtype() == torch::kBFloat16 && b.dtype() == torch::kBFloat16,
              "inputs must be bfloat16");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "inputs must be 2D");
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "inputs must be contiguous");
  
  const int64_t M = a.size(0), K = a.size(1), N = b.size(1);
  TORCH_CHECK(b.size(0) == K, "A (M,K) @ B (K,N): B must have K rows");

  auto c = torch::empty({M, N}, a.options().dtype(torch::kBFloat16));

  if (M == 0 || N == 0 || K == 0)
    return c;

  cublasLtHandle_t ltHandle = nullptr;
  CUBLASLT_CHECK(cublasLtCreate(&ltHandle));

  cublasLtMatmulDesc_t opDesc = nullptr;
  cublasLtMatrixLayout_t aDesc = nullptr, bDesc = nullptr, cDesc = nullptr;
  cublasLtMatmulPreference_t pref = nullptr;

  // Create matmul descriptor with FP32 compute type
  // For BF16 inputs with FP32 accumulation: computeType=CUBLAS_COMPUTE_32F, scaleType=CUDA_R_32F
  CUBLASLT_CHECK(cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  
  // Row-major C = A @ B can be computed as column-major: C^T = B^T @ A^T
  // So we compute: op(B) @ op(A) where op = transpose
  // This gives us C^T, then we interpret the result as row-major C
  cublasOperation_t transA = CUBLAS_OP_T;  // Transpose A
  cublasOperation_t transB = CUBLAS_OP_T;  // Transpose B
  CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transB, sizeof(transB)));
  CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transA, sizeof(transA)));

  // Matrix layouts in column-major convention:
  // For row-major A (M,K): when viewed as column-major, it's (K,M) with ld=K
  // For row-major B (K,N): when viewed as column-major, it's (N,K) with ld=N
  // For row-major C (M,N): when viewed as column-major, it's (N,M) with ld=N
  
  // A: row-major (M,K) = column-major (K,M), ld=K
  CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&aDesc, CUDA_R_16BF, K, M, K));
  // B: row-major (K,N) = column-major (N,K), ld=N
  CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&bDesc, CUDA_R_16BF, N, K, N));
  // C: row-major (M,N) = column-major (N,M), ld=N
  CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&cDesc, CUDA_R_16BF, N, M, N));

  CUBLASLT_CHECK(cublasLtMatmulPreferenceCreate(&pref));
  size_t workspaceSize = 1 << 23;  // 8 MB workspace
  CUBLASLT_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

  // Get heuristic algorithm
  cublasLtMatmulHeuristicResult_t heuristicResult[32];
  int returnedResults = 0;
  cublasStatus_t heurStatus = cublasLtMatmulAlgoGetHeuristic(
      ltHandle, opDesc, 
      bDesc,  // First operand: B^T (N,K) -> after transpose (K,N)
      aDesc,  // Second operand: A^T (K,M) -> after transpose (M,K)
      cDesc,  // C descriptor
      cDesc,  // D descriptor (same as C for C = alpha*A*B + beta*C)
      pref, 
      32,  // Request up to 32 algorithms
      heuristicResult, 
      &returnedResults);

  if (heurStatus != CUBLAS_STATUS_SUCCESS || returnedResults == 0) {
    // Cleanup on failure
    if (pref) cublasLtMatmulPreferenceDestroy(pref);
    if (cDesc) cublasLtMatrixLayoutDestroy(cDesc);
    if (bDesc) cublasLtMatrixLayoutDestroy(bDesc);
    if (aDesc) cublasLtMatrixLayoutDestroy(aDesc);
    if (opDesc) cublasLtMatmulDescDestroy(opDesc);
    cublasLtDestroy(ltHandle);
    TORCH_CHECK(false, "cuBLASLt: no algorithm found (status=", (int)heurStatus, ", returned=", returnedResults, ")");
  }

  float alpha = 1.0f, beta = 0.0f;
  auto workspace = torch::empty({static_cast<int64_t>(workspaceSize)}, a.options().dtype(torch::kByte));
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  // Get pointers using void* to avoid template instantiation issues
  const void* a_ptr = a.const_data_ptr();
  const void* b_ptr = b.const_data_ptr();
  void* c_ptr = c.mutable_data_ptr();
  void* ws_ptr = workspace.mutable_data_ptr();

  // Perform matmul: C = alpha * op(B) * op(A) + beta * C
  // Since we're using column-major with transposes, this computes row-major C = A @ B
  CUBLASLT_CHECK(cublasLtMatmul(
      ltHandle,
      opDesc,
      &alpha,
      b_ptr,  // First matrix: B
      bDesc,
      a_ptr,  // Second matrix: A
      aDesc,
      &beta,
      c_ptr,  // C matrix (output)
      cDesc,
      c_ptr,  // D matrix (same as C)
      cDesc,
      &heuristicResult[0].algo,
      ws_ptr,
      workspaceSize,
      stream));

  // Cleanup
  if (pref) cublasLtMatmulPreferenceDestroy(pref);
  if (cDesc) cublasLtMatrixLayoutDestroy(cDesc);
  if (bDesc) cublasLtMatrixLayoutDestroy(bDesc);
  if (aDesc) cublasLtMatrixLayoutDestroy(aDesc);
  if (opDesc) cublasLtMatmulDescDestroy(opDesc);
  cublasLtDestroy(ltHandle);

  return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemm_fp32_accum_cuda", &gemm_fp32_accum_cuda_impl, "BF16 GEMM with FP32 accumulation (cuBLASLt)");
}
