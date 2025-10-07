"""
Filter bank operations for NSCT
包含非下采样滤波器组的分解和重构函数
"""

import torch
import torch.nn.functional as F
from typing import Union, Tuple
from nsct_torch.filters import efilter2
from nsct_torch.utils import symext, upsample2df
from .convolution import _convolve_upsampled, _atrousc_torch


def nssfbdec(x: torch.Tensor, f1: torch.Tensor, f2: torch.Tensor, 
             mup: Union[torch.Tensor, None] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Two-channel nonsubsampled filter bank decomposition.
    
    Args:
        x: Input tensor
        f1: First filter
        f2: Second filter
        mup: Upsampling matrix (optional)
    
    Returns:
        Tuple of (y1, y2) filtered outputs
    """
    if mup is None:
        y1 = efilter2(x, f1)
        y2 = efilter2(x, f2)
    else:
        y1 = _convolve_upsampled(x, f1, mup, is_rec=False)
        y2 = _convolve_upsampled(x, f2, mup, is_rec=False)
    return y1, y2


def nssfbrec(x1: torch.Tensor, x2: torch.Tensor, 
             f1: torch.Tensor, f2: torch.Tensor,
             mup: Union[torch.Tensor, None] = None) -> torch.Tensor:
    """
    Two-channel nonsubsampled filter bank reconstruction.
    
    Args:
        x1: First input tensor
        x2: Second input tensor
        f1: First filter
        f2: Second filter
        mup: Upsampling matrix (optional)
    
    Returns:
        Reconstructed tensor
    """
    if x1.shape != x2.shape:
        raise ValueError("Input sizes for the two branches must be the same")
    
    if mup is None:
        y1 = efilter2(x1, f1)
        y2 = efilter2(x2, f2)
    else:
        y1 = _convolve_upsampled(x1, f1, mup, is_rec=True)
        y2 = _convolve_upsampled(x2, f2, mup, is_rec=True)
    
    return y1 + y2


def nsfbdec(x: torch.Tensor, h0: torch.Tensor, h1: torch.Tensor, 
           lev: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Nonsubsampled filter bank decomposition at a given level.
    PyTorch translation of nsfbdec.m.
    
    Args:
        x: Input image at finer scale
        h0: Lowpass à trous filter
        h1: Highpass à trous filter
        lev: Decomposition level (0 for first level, >0 for subsequent levels)
    
    Returns:
        Tuple of (y0, y1):
            y0: Image at coarser scale (lowpass output)
            y1: Wavelet highpass output (bandpass output)
    """
    if lev != 0:
        # For levels > 0, use upsampled filters
        I2 = torch.eye(2, dtype=torch.int64, device=x.device)
        shift = [-2**(lev-1), -2**(lev-1)]
        shift = [s + 2 for s in shift]
        L = 2**lev
        
        # Upsample filters
        h0_up = upsample2df(h0, lev)
        h1_up = upsample2df(h1, lev)
        
        # Symmetric extension
        x_ext_h0 = symext(x, h0_up, shift)
        x_ext_h1 = symext(x, h1_up, shift)
        
        # Atrous convolution
        mup = I2 * L
        y0 = _atrousc_torch(x_ext_h0, h0, mup)
        y1 = _atrousc_torch(x_ext_h1, h1, mup)
    else:
        # First level (lev == 0)
        shift = [1, 1]
        
        # Symmetric extension
        x_ext_h0 = symext(x, h0, shift)
        x_ext_h1 = symext(x, h1, shift)
        
        # Regular convolution with 'valid' mode
        # Use PyTorch conv2d
        x_ext_h0_4d = x_ext_h0.unsqueeze(0).unsqueeze(0)
        x_ext_h1_4d = x_ext_h1.unsqueeze(0).unsqueeze(0)
        
        # Important: PyTorch's conv2d performs true convolution (flips kernel),
        # while NumPy's convolve2d performs correlation (no flip).
        # To match NumPy behavior, we need to flip the filters.
        h0_flipped = torch.flip(h0, [0, 1])
        h1_flipped = torch.flip(h1, [0, 1])
        h0_4d = h0_flipped.unsqueeze(0).unsqueeze(0)
        h1_4d = h1_flipped.unsqueeze(0).unsqueeze(0)
        
        y0 = F.conv2d(x_ext_h0_4d, h0_4d, padding=0).squeeze()
        y1 = F.conv2d(x_ext_h1_4d, h1_4d, padding=0).squeeze()
    
    return y0, y1


def nsfbrec(y0: torch.Tensor, y1: torch.Tensor, 
           g0: torch.Tensor, g1: torch.Tensor, 
           lev: int) -> torch.Tensor:
    """
    Nonsubsampled filter bank reconstruction at a given level.
    PyTorch translation of nsfbrec.m.
    
    Args:
        y0: Lowpass image (coarse scale)
        y1: Highpass image (wavelet details)
        g0: Lowpass synthesis filter
        g1: Highpass synthesis filter
        lev: Reconstruction level
    
    Returns:
        Reconstructed image at finer scale
    """
    I2 = torch.eye(2, dtype=torch.int64, device=y0.device)
    
    if lev != 0:
        # Higher levels: use upsampled filters with atrous convolution
        shift = -2**(lev-1) * torch.tensor([1, 1], device=y0.device) + 2
        L = 2**lev
        
        # Upsample filters
        g0_up = upsample2df(g0, lev)
        g1_up = upsample2df(g1, lev)
        
        # Extend inputs
        y0_ext = symext(y0, g0_up, shift.tolist())
        y1_ext = symext(y1, g1_up, shift.tolist())
        
        # Apply atrous convolution and sum
        x = _atrousc_torch(y0_ext, g0, L * I2) + \
            _atrousc_torch(y1_ext, g1, L * I2)
    else:
        # Level 0: use regular convolution
        shift = [1, 1]
        
        # Extend inputs
        y0_ext = symext(y0, g0, shift)
        y1_ext = symext(y1, g1, shift)
        
        # Convolve and sum
        y0_ext_4d = y0_ext.unsqueeze(0).unsqueeze(0)
        y1_ext_4d = y1_ext.unsqueeze(0).unsqueeze(0)
        
        # Important: PyTorch's conv2d performs true convolution (flips kernel),
        # while NumPy's convolve2d performs correlation (no flip).
        # To match NumPy behavior, we need to flip the filters.
        g0_flipped = torch.flip(g0, [0, 1])
        g1_flipped = torch.flip(g1, [0, 1])
        g0_4d = g0_flipped.unsqueeze(0).unsqueeze(0)
        g1_4d = g1_flipped.unsqueeze(0).unsqueeze(0)
        
        x = F.conv2d(y0_ext_4d, g0_4d, padding=0).squeeze() + \
            F.conv2d(y1_ext_4d, g1_4d, padding=0).squeeze()
    
    return x
