"""
Type stubs for atrousc module
"""
import torch
from typing import Union

CUDA_AVAILABLE: bool

def atrousc(
    x: torch.Tensor,
    h: torch.Tensor,
    M: torch.Tensor,
) -> torch.Tensor:
    """
    Atrous convolution with symmetric extension.
    
    This is a CUDA-accelerated implementation that performs atrous (à trous)
    convolution with symmetric boundary extension.
    
    Args:
        x: Extended input signal tensor (2D, on CUDA device)
        h: Filter tensor (2D, on CUDA device)
        M: Upsampling matrix (2x2 tensor or scalar)
    
    Returns:
        Convolved output tensor
    
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

__all__ = ['atrousc', 'is_available', 'get_import_error', 'CUDA_AVAILABLE']
