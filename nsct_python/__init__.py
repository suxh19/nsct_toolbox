"""
NSCT (Nonsubsampled Contourlet Transform) Toolbox - Python Implementation

This package provides a Python translation of the MATLAB NSCT toolbox.
"""

import os
import sys

# 在 Windows 上,需要添加 PyTorch DLL 路径到 DLL 搜索路径
# 这样 CUDA 扩展模块才能正确加载
if sys.platform == 'win32' and sys.version_info >= (3, 8):
    try:
        import torch
        torch_lib_path = os.path.join(os.path.dirname(torch.__file__), 'lib')
        if os.path.exists(torch_lib_path):
            os.add_dll_directory(torch_lib_path)
    except Exception:
        pass

from nsct_python.core import nssfbdec, nssfbrec, nsfbdec
from nsct_python.filters import (
    efilter2, dmaxflat, ldfilter, ld2quin, mctrans, 
    dfilters, atrousfilters, parafilters
)
from nsct_python.utils import (
    extend2, upsample2df, modulate2, resampz, qupz, symext
)

__all__ = [
    # Core functions
    'nssfbdec', 'nssfbrec', 'nsfbdec',
    # Filter functions
    'efilter2', 'dmaxflat', 'ldfilter', 'ld2quin', 'mctrans',
    'dfilters', 'atrousfilters', 'parafilters',
    # Utility functions
    'extend2', 'upsample2df', 'modulate2', 'resampz', 'qupz', 'symext',
]

__version__ = '0.1.0'