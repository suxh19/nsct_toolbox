"""
atrousc - CUDA-accelerated à trous convolution

This module provides a GPU-accelerated implementation of the atrousc function
using CUDA and PyTorch C++ extensions.
"""

import os
import sys

# Initialize error tracking
_import_error = None
CUDA_AVAILABLE = False
atrousc = None

# 在 Windows 上,需要添加 PyTorch DLL 路径到 DLL 搜索路径
# 这样才能正确加载 CUDA 扩展模块的依赖项
if sys.platform == 'win32' and sys.version_info >= (3, 8):
    try:
        import torch
        torch_lib_path = os.path.join(os.path.dirname(torch.__file__), 'lib')
        if os.path.exists(torch_lib_path):
            # 添加 PyTorch lib 目录到 DLL 搜索路径
            os.add_dll_directory(torch_lib_path)
            
        # 同时添加 CUDA bin 目录
        cuda_path = os.environ.get('CUDA_PATH')
        if cuda_path:
            cuda_bin = os.path.join(cuda_path, 'bin')
            if os.path.exists(cuda_bin):
                os.add_dll_directory(cuda_bin)
    except Exception as e:
        _import_error = f"Failed to add DLL directories: {e}"

try:
    from .atrousc import atrousc
    CUDA_AVAILABLE = True
    _import_error = None
except ImportError as e:
    CUDA_AVAILABLE = False
    atrousc = None
    _import_error = str(e)
except Exception as e:
    CUDA_AVAILABLE = False
    atrousc = None
    _import_error = f"Unexpected error: {str(e)}"

def is_available():
    """Check if CUDA extension is available"""
    return CUDA_AVAILABLE

def get_import_error():
    """Get the import error message if CUDA is not available"""
    if CUDA_AVAILABLE:
        return None
    return _import_error

__all__ = ['atrousc', 'is_available', 'get_import_error', 'CUDA_AVAILABLE']
