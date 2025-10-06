"""
zconv2_cpp - High-performance C++ implementation of 2D convolution with upsampled filter

This module provides optimized C++ implementations of the zconv2 operation
used in the Nonsubsampled Contourlet Transform (NSCT).
"""

import numpy as np
import sys
import os

# Try to import the C++ extension
CPP_AVAILABLE = False
_cpp_import_error = None
_zconv2_cpp = None

# Add current directory to path to allow importing the compiled module
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    # Import the compiled C++ extension module
    # The module is in the same directory as this __init__.py
    import importlib.util
    module_name = 'zconv2_cpp'
    
    # Find the .pyd or .so file
    pyd_files = [f for f in os.listdir(current_dir) if f.startswith('zconv2_cpp') and (f.endswith('.pyd') or f.endswith('.so'))]
    
    if pyd_files:
        spec = importlib.util.spec_from_file_location(module_name, os.path.join(current_dir, pyd_files[0]))
        if spec and spec.loader:
            _cpp_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(_cpp_module)
            _zconv2_cpp = _cpp_module.zconv2
            CPP_AVAILABLE = True
    else:
        _cpp_import_error = Exception("Compiled module file (.pyd or .so) not found")
        
except Exception as e:
    _cpp_import_error = e


def zconv2(x: np.ndarray, h: np.ndarray, mup: np.ndarray) -> np.ndarray:
    """
    2D convolution with upsampled filter using periodic boundary (C++ implementation).
    
    This computes convolution as if the filter had been upsampled by matrix mup,
    but without actually upsampling the filter (efficient stepping through zeros).
    
    Args:
        x: Input signal (2D array)
        h: Filter (2D array)
        mup: Upsampling matrix (2x2 array) [[M0, M1], [M2, M3]]
    
    Returns:
        y: Convolution output (same size as x)
    
    Raises:
        RuntimeError: If C++ extension is not available.
    """
    if not CPP_AVAILABLE:
        raise RuntimeError(
            "C++ extension is not available. Please compile it with:\n"
            "  cd nsct_python/zconv2_cpp\n"
            "  python setup.py build_ext --inplace\n"
            f"Import error: {_cpp_import_error}"
        )
    
    # Ensure input is contiguous and of correct dtype
    x = np.ascontiguousarray(x, dtype=np.float64)
    h = np.ascontiguousarray(h, dtype=np.float64)
    
    # Ensure mup is 2x2 matrix
    mup = np.array(mup, dtype=np.float64)
    if mup.shape != (2, 2):
        raise ValueError("mup must be a 2x2 matrix")
    
    mup = np.ascontiguousarray(mup, dtype=np.float64)
    
    return _zconv2_cpp(x, h, mup)




def is_cpp_available() -> bool:
    """Check if C++ extension is available."""
    return CPP_AVAILABLE


def get_backend_info() -> dict:
    """Get information about the current backend."""
    return {
        'cpp_available': CPP_AVAILABLE,
        'backend': 'C++' if CPP_AVAILABLE else 'Pure Python',
        'import_error': str(_cpp_import_error) if not CPP_AVAILABLE and _cpp_import_error else None
    }


# Export main function
__all__ = ['zconv2', 'is_cpp_available', 'get_backend_info']
