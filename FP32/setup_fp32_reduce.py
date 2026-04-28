from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='gemm_fp32_reduce',
    ext_modules=[
        CUDAExtension(
            name='_gemm_fp32_reduce',
            sources=['csrc/gemm_fp32_reduce.cu'],
            libraries=['cublasLt'],
            extra_compile_args={'nvcc': ['-O3', '--use_fast_math']},
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
