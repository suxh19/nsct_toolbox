"""
Type stubs for zconv2 module
"""
import torch
from typing import Union

CUDA_AVAILABLE: bool

def zconv2(
    x: torch.Tensor,
    h: torch.Tensor,
    mup: torch.Tensor,
) -> torch.Tensor:
    """
    2D convolution with upsampled filter (periodic boundary).
    
    This is a CUDA-accelerated implementation that performs 2D convolution
    with upsampled filters, using periodic boundary conditions.
    
    Args:
        x: Input signal tensor (2D, on CUDA device)
        h: Filter tensor (2D, on CUDA device)
        mup: Upsampling matrix (2x2 tensor) [[M0, M1], [M2, M3]]
    
    Returns:
        Convolved tensor with same shape as input
    
    Raises:
        RuntimeError: If input is not on CUDA device or if CUDA operation fails
    """
    ...

def is_available() -> bool:
    """Check if CUDA extension is available"""
    ...

def get_import_error() -> Union[str, None]:
    """Get the import error message if CUDA is not available"""
    ...

__all__ = ['zconv2', 'is_available', 'get_import_error', 'CUDA_AVAILABLE']
