"""
atrousc_cpp - High-performance C++ implementation of à trous convolution

This module provides optimized C++ implementations of the à trous convolution
operation used in the Nonsubsampled Contourlet Transform (NSCT).
"""

import numpy as np
import sys
import os

# Try to import the C++ extension
CPP_AVAILABLE = False
_cpp_import_error = None
_atrousc_cpp = None

# Add current directory to path to allow importing the compiled module
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    # Import the compiled C++ extension module
    # The module is in the same directory as this __init__.py
    import importlib.util
    import platform
    module_name = 'atrousc_cpp'
    
    # Find the .pyd or .so file, prioritizing the correct platform
    pyd_files = [f for f in os.listdir(current_dir) if f.startswith('atrousc_cpp') and (f.endswith('.pyd') or f.endswith('.so'))]
    
    # Filter by platform
    if platform.system() == 'Windows':
        pyd_files = [f for f in pyd_files if f.endswith('.pyd')]
    else:
        pyd_files = [f for f in pyd_files if f.endswith('.so')]
    
    if pyd_files:
        spec = importlib.util.spec_from_file_location(module_name, os.path.join(current_dir, pyd_files[0]))
        if spec and spec.loader:
            _cpp_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(_cpp_module)
            _atrousc_cpp = _cpp_module.atrousc
            CPP_AVAILABLE = True
    else:
        _cpp_import_error = Exception("Compiled module file (.pyd or .so) not found for this platform")
        
except Exception as e:
    _cpp_import_error = e


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
