"""
Build script for the batched NSCT CUDA extensions.

This file compiles the CUDA kernels for zconv2 and atrousc under the names
`nsct.zconv2.zconv2` and `nsct.atrousc.atrousc` respectively.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).parent.resolve()


def _extension_sources(*relative: str) -> List[str]:
    return [str(ROOT.joinpath(path)) for path in relative]


extra_compile_args: Dict[str, List[str]] = {
    "cxx": ["-O3", "-std=c++17"],
    "nvcc": ["-O3", "--use_fast_math"],
}

ext_modules = [
    CUDAExtension(
        name="nsct.zconv2.zconv2",
        sources=_extension_sources(
            "nsct/zconv2/zconv2.cpp",
            "nsct/zconv2/zconv2_launcher.cu",
            "nsct/zconv2/zconv2_kernel.cu",
        ),
        extra_compile_args=extra_compile_args,
    ),
    CUDAExtension(
        name="nsct.atrousc.atrousc",
        sources=_extension_sources(
            "nsct/atrousc/atrousc.cpp",
            "nsct/atrousc/atrousc_launcher.cu",
            "nsct/atrousc/atrousc_kernel.cu",
        ),
        extra_compile_args=extra_compile_args,
    ),
]

setup(
    name="nsct",
    version="0.1.0",
    packages=find_packages(exclude=("tests", "results")),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
