"""
Build cuBLASLt GEMM extension: BF16 x BF16 with FP32 accumulation, BF16 output.
Run from FP32/: python setup.py build_ext --inplace
Then the .so is in FP32/ and importable as _gemm_fp32_accum_cuda when FP32 is on path.
"""
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="gemm_fp32_accum_cuda",
    ext_modules=[
        CUDAExtension(
            name="_gemm_fp32_accum_cuda",
            sources=["csrc/gemm_fp32_accum_cuda.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math"],
            },
            libraries=["cublasLt"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
