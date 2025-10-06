"""
atrousc_cpp - High-performance C++ implementation of à trous convolution

This module provides optimized C++ implementations of the à trous convolution
operation used in the Nonsubsampled Contourlet Transform (NSCT).
"""

import numpy as np

# Try to import the C++ extension
CPP_AVAILABLE = False
_cpp_import_error = None

try:
    import atrousc_cpp as _cpp_module
    _atrousc_cpp = _cpp_module.atrousc
    CPP_AVAILABLE = True
except ImportError as e:
    _cpp_import_error = e
    try:
        from atrousc_cpp import atrousc as _atrousc_cpp
        CPP_AVAILABLE = True
        _cpp_import_error = None
    except ImportError as e2:
        _cpp_import_error = e2


def atrousc(x: np.ndarray, h: np.ndarray, M: np.ndarray) -> np.ndarray:
    """
    Performs à trous convolution with an upsampled filter using C++ implementation.
    
    Args:
        x (np.ndarray): Extended input signal (2D).
        h (np.ndarray): Original filter (not upsampled).
        M (np.ndarray): Upsampling matrix (2x2 or scalar).
    
    Returns:
        np.ndarray: Result of convolution, 'valid' mode.
    
    Raises:
        RuntimeError: If C++ extension is not available.
    """
    if not CPP_AVAILABLE:
        raise RuntimeError(
            "C++ extension is not available. Please compile it with:\n"
            "  cd nsct_python/atrousc_cpp\n"
            "  python setup.py build_ext --inplace\n"
            f"Import error: {_cpp_import_error}"
        )
    
    # Ensure input is contiguous and of correct dtype
    x = np.ascontiguousarray(x, dtype=np.float64)
    h = np.ascontiguousarray(h, dtype=np.float64)
    
    # Handle scalar M
    if np.isscalar(M):
        M = np.array([[M, 0], [0, M]], dtype=np.float64)
    elif M.shape == (2, 2):
        M = np.ascontiguousarray(M, dtype=np.float64)
    else:
        raise ValueError("M must be a scalar or 2x2 matrix")
    
    return _atrousc_cpp(x, h, M)




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
__all__ = ['atrousc', 'is_cpp_available', 'get_backend_info']
