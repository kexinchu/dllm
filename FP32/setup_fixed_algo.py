from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='gemm_fixed_algo',
    ext_modules=[
        CUDAExtension(
            name='_gemm_fixed_algo',
            sources=['csrc/gemm_fixed_algo.cu'],
            libraries=['cublasLt'],
            extra_compile_args={'nvcc': ['-O3', '--use_fast_math']},
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
