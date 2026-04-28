"""
Build cuBLASLt GEMM extension (BF16 x BF16, FP32 accum, BF16 out).
Run from repo root: pip install -e .  (if using this as setup.py) or:
  python setup_fp32_ext.py build_ext --inplace
Then the extension is available as FP32._gemm_fp32_accum_cuda.
"""
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="dllm_fp32_gemm",
    version="0.1.0",
    ext_modules=[
        CUDAExtension(
            name="FP32._gemm_fp32_accum_cuda",
            sources=["FP32/csrc/gemm_fp32_accum_cuda.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math"],
            },
            libraries=["cublasLt"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
