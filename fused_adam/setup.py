from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

nvcc_args = ['-maxrregcount=16']

setup(
    name='fused_adam',
    ext_modules=[
        CUDAExtension('fused_adam', [
            'pybind_adam.cpp',
            'fused_adam_kernel.cu',
        ], extra_compile_args={'nvcc': nvcc_args})
    ],
    cmdclass={
        'build_ext': BuildExtension
    })