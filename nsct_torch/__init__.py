"""
NSCT Torch - PyTorch Implementation of Nonsubsampled Contourlet Transform

This package provides PyTorch-based implementations with CUDA acceleration.

Data Type Support:
------------------
All main functions support optional dtype parameter for computation precision:
- None (default): Preserves input tensor dtype
- torch.float32: 32-bit floating point, faster computation
- torch.float64: 64-bit floating point, higher precision

Functions supporting dtype parameter:
- nsctdec, nsctrec: Main NSCT decomposition/reconstruction functions
- atrousfilters, dfilters: Filter generation functions
- efilter2, dmaxflat, ldfilter: Filter utility functions

Example:
    >>> import torch
    >>> from nsct_torch import nsctdec, nsctrec
    >>> 
    >>> # Dtype automatically matches input (float32)
    >>> img = torch.rand(256, 256)  
    >>> coeffs = nsctdec(img, [2, 3, 4])  # Uses float32
    >>> 
    >>> # Explicitly use float32 for faster computation
    >>> coeffs_32 = nsctdec(img, [2, 3, 4], dtype=torch.float32)
    >>> 
    >>> # Use float64 for higher precision
    >>> coeffs_64 = nsctdec(img, [2, 3, 4], dtype=torch.float64)
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

from nsct_torch.core_torch import nssfbdec, nssfbrec, nsfbdec, nsfbrec, nsdfbdec, nsdfbrec, nsctdec, nsctrec
from nsct_torch.filters_torch import (
    efilter2, dmaxflat, ldfilter, ld2quin, mctrans,
    dfilters, atrousfilters, parafilters
)
from nsct_torch.utils_torch import (
    extend2, upsample2df, modulate2, resampz, qupz, symext
)

__all__ = [
    # Core functions
    'nssfbdec', 'nssfbrec', 'nsfbdec', 'nsfbrec', 'nsdfbdec', 'nsdfbrec', 'nsctdec', 'nsctrec',
    # Filter functions
    'efilter2', 'dmaxflat', 'ldfilter', 'ld2quin', 'mctrans',
    'dfilters', 'atrousfilters', 'parafilters',
    # Utility functions
    'extend2', 'upsample2df', 'modulate2', 'resampz', 'qupz', 'symext',
]

__version__ = '0.1.0'
