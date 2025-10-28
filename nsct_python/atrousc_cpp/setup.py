"""
setup.py for building the atrousc_cpp C++ extension module

This script compiles the high-performance C++ implementation of 
à trous convolution and creates a Python extension module.

Build instructions:
    Build:      python setup.py build_ext --inplace
    Install:    pip install .
"""

import os
import sys
import platform
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11

class get_pybind_include(object):
    """Helper class to determine the pybind11 include path"""
    def __str__(self):
        return pybind11.get_include()


# Determine compiler flags based on platform
extra_compile_args = []
extra_link_args = []

if platform.system() == 'Windows':
    # MSVC compiler flags
    extra_compile_args = [
        '/O2',           # Optimize for speed
        '/fp:fast',      # Fast floating-point model
        '/std:c++17',    # C++17 standard
        '/openmp',       # Enable OpenMP multi-threading
        '/arch:AVX2',    # Enable AVX2 vectorization
        '/wd4819',       # Disable warning C4819 (code page encoding)
    ]
            
elif platform.system() == 'Linux' or platform.system() == 'Darwin':
    # GCC/Clang compiler flags
    extra_compile_args = [
        '-O3',                  # Maximum optimization
        '-march=native',        # Optimize for current CPU (includes AVX2 if available)
        '-ffast-math',          # Fast math operations
        '-fopenmp',             # Enable OpenMP multi-threading
        '-std=c++17',           # C++17 standard
    ]
    extra_link_args = ['-fopenmp']  # Link OpenMP runtime


# Define the extension module
ext_modules = [
    Extension(
        'atrousc_cpp',
        sources=['atrousc.cpp'],
        include_dirs=[
            get_pybind_include(),
            pybind11.get_include(),
        ],
        language='c++',
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]


# Custom build_ext to ensure proper C++17 support
class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    
    def build_extensions(self):
        # Get the compiler type
        ct = self.compiler.compiler_type
        
        opts = []
        if ct == 'msvc':
            opts.append('/std:c++17')
        else:
            opts.append('-std=c++17')
            
        for ext in self.extensions:
            ext.extra_compile_args = opts + ext.extra_compile_args
            
        build_ext.build_extensions(self)


setup(
    name='atrousc_cpp',
    version='1.0.0',
    author='NSCT Toolbox Contributors',
    description='High-performance C++ implementation of à trous convolution',
    long_description=open('README.md', 'r', encoding='utf-8').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    ext_modules=ext_modules,
    install_requires=['pybind11>=2.6.0', 'numpy>=1.19.0'],
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
    python_requires='>=3.7',
)
