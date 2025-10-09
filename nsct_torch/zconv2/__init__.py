"""
zconv2 - CUDA-accelerated 2D convolution with upsampled filter

This module provides a GPU-accelerated implementation of the zconv2 function
using CUDA and PyTorch C++ extensions.
"""

import os
import sys

# 在 Windows 上,需要添加 PyTorch DLL 路径到 DLL 搜索路径
# 这样才能正确加载 CUDA 扩展模块的依赖项
if sys.platform == 'win32' and sys.version_info >= (3, 8):
    try:
        import torch
        torch_lib_path = os.path.join(os.path.dirname(torch.__file__), 'lib')
        if os.path.exists(torch_lib_path):
            # 添加 PyTorch lib 目录到 DLL 搜索路径
            os.add_dll_directory(torch_lib_path)
    except Exception:
        # 如果添加失败,继续尝试导入,可能会失败但给出更清晰的错误信息
        pass

try:
    from .zconv2 import zconv2
    CUDA_AVAILABLE = True
except ImportError as e:
    CUDA_AVAILABLE = False
    zconv2 = None
    _import_error = str(e)

def is_available():
    """Check if CUDA extension is available"""
    return CUDA_AVAILABLE

def get_import_error():
    """Get the import error message if CUDA is not available"""
    if CUDA_AVAILABLE:
        return None
    return _import_error

__all__ = ['zconv2', 'is_available', 'get_import_error', 'CUDA_AVAILABLE']
