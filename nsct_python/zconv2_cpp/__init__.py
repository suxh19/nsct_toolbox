"""
zconv2_cpp - High-performance C++ implementation of 2D convolution with upsampled filter

This module provides optimized C++ implementations of the zconv2 operation
used in the Nonsubsampled Contourlet Transform (NSCT).
"""

import numpy as np

# Try to import the C++ extension
CPP_AVAILABLE = False
_cpp_import_error = None

try:
    import zconv2_cpp as _cpp_module
    _zconv2_cpp = _cpp_module.zconv2
    CPP_AVAILABLE = True
except ImportError as e:
    _cpp_import_error = e
    try:
        from zconv2_cpp import zconv2 as _zconv2_cpp
        CPP_AVAILABLE = True
        _cpp_import_error = None
    except ImportError as e2:
        _cpp_import_error = e2


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
