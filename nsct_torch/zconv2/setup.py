"""
setup.py for building the zconv2 CUDA extension module

This script compiles the CUDA implementation of 2D convolution with 
upsampled filter and creates a PyTorch C++ extension module.

Requirements:
    - NVIDIA CUDA Toolkit (compatible with your PyTorch version)
    - PyTorch with CUDA support
    - C++ compiler (MSVC on Windows, GCC on Linux)

Build instructions:
    Build:      python setup.py build_ext --inplace
    Install:    pip install .
    Clean:      python setup.py clean --all

Note:
    You may see warnings about CUDA version mismatches. As long as the major 
    version matches (e.g., 11.x with 11.y), the build should work fine.
"""

import os
import sys
import platform
import warnings
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

# Suppress specific warnings that are not critical
warnings.filterwarnings('ignore', message='.*ninja.*')
warnings.filterwarnings('ignore', message='.*compiler version.*')

def get_cuda_version():
    """Get CUDA version from PyTorch"""
    if torch.cuda.is_available():
        # torch.version.cuda exists at runtime but Pylance doesn't recognize it
        return torch.version.cuda  # type: ignore[attr-defined]
    return None

def get_compute_capability():
    """Get compute capability of the first GPU"""
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability(0)
        return capability[0], capability[1]
    return None, None

def get_cuda_version_tuple():
    """Get CUDA version as tuple (major, minor)"""
    # Try to get actual installed CUDA version from nvcc
    import subprocess
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            # Parse output like: "Cuda compilation tools, release 11.3, V11.3.109"
            for line in result.stdout.split('\n'):
                if 'release' in line.lower():
                    # Extract version like "11.3"
                    import re
                    match = re.search(r'release\s+(\d+)\.(\d+)', line, re.IGNORECASE)
                    if match:
                        return int(match.group(1)), int(match.group(2))
    except Exception:
        pass
    
    # Fallback to PyTorch's CUDA version
    if torch.cuda.is_available():
        # torch.version.cuda exists at runtime but Pylance doesn't recognize it
        cuda_version_str = torch.version.cuda  # type: ignore[attr-defined]
        if cuda_version_str:
            parts = cuda_version_str.split('.')
            return int(parts[0]), int(parts[1])
    return None, None

# Check CUDA availability
if not torch.cuda.is_available():
    print("WARNING: CUDA is not available. This extension requires CUDA.")
    print("Please install a CUDA-enabled version of PyTorch.")
    sys.exit(1)

cuda_version = get_cuda_version()
cuda_major, cuda_minor = get_cuda_version_tuple()
compute_major, compute_minor = get_compute_capability()

print(f"\n{'='*70}")
print(f"Building zconv2 extension")
print(f"{'='*70}")
print(f"PyTorch version: {torch.__version__}")
print(f"PyTorch CUDA version: {cuda_version}")
if cuda_major and cuda_minor:
    print(f"Detected CUDA Toolkit: {cuda_major}.{cuda_minor}")
    # Check for version mismatch
    pytorch_cuda = cuda_version.split('.') if cuda_version else []
    if len(pytorch_cuda) >= 2:
        pt_major, pt_minor = int(pytorch_cuda[0]), int(pytorch_cuda[1])
        if pt_major != cuda_major:
            # Only warn if major version differs
            print(f"\n⚠️  WARNING: PyTorch CUDA {pt_major}.{pt_minor} vs Installed CUDA {cuda_major}.{cuda_minor}")
            print(f"   Major version mismatch! This may cause runtime errors.")
            print(f"   Please install matching versions.\n")
        elif pt_minor != cuda_minor:
            # Minor version mismatch is usually OK
            print(f"\nℹ️  Info: CUDA minor version mismatch ({pt_major}.{pt_minor} vs {cuda_major}.{cuda_minor})")
            print(f"   This is typically fine as long as major versions match.\n")
if compute_major and compute_minor:
    print(f"GPU compute capability: sm_{compute_major}{compute_minor}")

# Compiler flags for C++ code
if platform.system() == 'Windows':
    # MSVC uses different flags
    cxx_flags = ['/O2', '/std:c++17']
else:
    # GCC/Clang
    cxx_flags = ['-O3', '-std=c++17']

# Compiler flags for CUDA code
nvcc_flags = [
    '--use_fast_math',  # Enable fast math optimizations
    '--expt-relaxed-constexpr',  # Allow constexpr in device code
]

# Add optimization flags for CUDA (nvcc understands these)
if platform.system() == 'Windows':
    nvcc_flags.extend(['-O3'])
else:
    nvcc_flags.extend(['-O3'])

# Add GPU architecture flags if detected
if compute_major and compute_minor:
    target_compute = compute_major * 10 + compute_minor
    
    # Map CUDA version to maximum supported compute capability
    # CUDA 11.0-11.3: up to sm_86
    # CUDA 11.4+: up to sm_87
    # CUDA 11.8+: up to sm_89
    # CUDA 12.0+: up to sm_90
    max_supported_compute = 86  # Default for CUDA 11.0-11.3
    
    if cuda_major and cuda_minor:
        if cuda_major >= 12:
            max_supported_compute = 90
        elif cuda_major == 11:
            if cuda_minor >= 8:
                max_supported_compute = 89
            elif cuda_minor >= 4:
                max_supported_compute = 87
            else:
                max_supported_compute = 86
    
    # Use the lower of GPU capability and CUDA support
    actual_compute = min(target_compute, max_supported_compute)
    
    if actual_compute < target_compute:
        print(f"\nℹ️  Compiling for: sm_{actual_compute} (GPU supports sm_{target_compute})")
        print(f"   Note: CUDA {cuda_major}.{cuda_minor} limits to sm_{max_supported_compute}")
        print(f"   Upgrade CUDA Toolkit to utilize newer GPU features.\n")
    else:
        print(f"Compiling for GPU architecture: sm_{actual_compute}")
    
    nvcc_flags.append(f'-gencode=arch=compute_{actual_compute},code=sm_{actual_compute}')
else:
    # Default to common architectures if not detected
    # Only use architectures supported by most CUDA versions
    print("\nℹ️  Using default GPU architectures (auto-detect failed)")
    print("   Targeting: sm_75, sm_80, sm_86 (covers most modern GPUs)\n")
    nvcc_flags.extend([
        '-gencode=arch=compute_75,code=sm_75',  # Turing (RTX 20xx)
        '-gencode=arch=compute_80,code=sm_80',  # Ampere (RTX 30xx, A100)
        '-gencode=arch=compute_86,code=sm_86',  # Ampere (RTX 30xx)
    ])

print(f"{'='*70}")
print("Starting compilation... (this may take a few minutes)")
print(f"{'='*70}\n")

# Platform-specific flags
if platform.system() == 'Windows':
    cxx_flags.extend(['/wd4819'])  # Disable warning C4819 (code page encoding)
elif platform.system() == 'Linux':
    cxx_flags.extend(['-Wno-sign-compare'])

# Define the extension module
ext_modules = [
    CUDAExtension(
        name='zconv2',
        sources=[
            'zconv2.cpp',
            'zconv2_launcher.cu',
            'zconv2_kernel.cu',
        ],
        extra_compile_args={
            'cxx': cxx_flags,
            'nvcc': nvcc_flags,
        }
    ),
]

# Setup configuration
setup(
    name='zconv2',
    version='1.0.0',
    author='NSCT Toolbox Contributors',
    description='CUDA-accelerated 2D convolution with upsampled filter for NSCT',
    long_description=open('README.md', encoding='utf-8').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension
    },
    python_requires='>=3.8',
    install_requires=[
        'torch>=2.0.0',
        'numpy>=1.20.0',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Image Processing',
        'Programming Language :: Python :: 3',
        'Programming Language :: C++',
        'Programming Language :: CUDA',
    ],
)
