from setuptools import setup, Extension
from torch.utils import cpp_extension
import os

# Check if CUDA is available
try:
    from torch.utils.cpp_extension import CUDAExtension
    USE_CUDA = True
except ImportError:
    USE_CUDA = False

ext_modules = []

if USE_CUDA:
    # CUDA extension for GPU acceleration
    cuda_extension = CUDAExtension(
        name='nsct_torch.nsct_cuda',
        sources=[
            'csrc/nsct_cuda.cpp',
            'csrc/atrousc_kernel.cu',
            'csrc/zconv2_kernel.cu',
        ],
        extra_compile_args={
            'cxx': ['-O3'],
            'nvcc': ['-O3', '--use_fast_math']
        }
    )
    ext_modules.append(cuda_extension)

setup(
    name='nsct_torch',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='PyTorch GPU-accelerated Nonsubsampled Contourlet Transform',
    long_description=open('README.md').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    packages=['nsct_torch'],
    ext_modules=ext_modules,
    cmdclass={'build_ext': cpp_extension.BuildExtension},
    install_requires=[
        'torch>=2.0.0',
        'numpy>=1.20.0',
        'scipy>=1.7.0',
    ],
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: C++',
        'Programming Language :: CUDA',
        'Topic :: Scientific/Engineering :: Image Processing',
    ],
)
