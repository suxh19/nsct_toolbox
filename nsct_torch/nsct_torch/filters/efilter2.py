"""
2D Filtering with edge handling (via extension).
PyTorch translation of efilter2.m
"""

import torch
import torch.nn.functional as F
from typing import Optional, List
from nsct_torch.utils import extend2


def efilter2(x: torch.Tensor, f: torch.Tensor, extmod: str = 'per', 
             shift: Optional[List[int]] = None) -> torch.Tensor:
    """
    2D Filtering with edge handling (via extension).
    
    Args:
        x: Input image tensor.
        f: 2D filter tensor.
        extmod: Extension mode (default is 'per'). See extend2 for details.
        shift: Specify the window over which the convolution occurs. 
               Defaults to [0, 0].
    
    Returns:
        Filtered image tensor of the same size as the input.
    """
    if shift is None:
        shift = [0, 0]
    
    x_float = x.to(dtype=torch.float64)
    
    # The origin of filter f is assumed to be floor(size(f)/2) + 1.
    # Amount of shift should be no more than floor((size(f)-1)/2).
    sf = (torch.tensor(f.shape, dtype=torch.float64) - 1) / 2
    
    # Extend the image
    xext = extend2(
        x_float,
        int(torch.floor(sf[0]).item() + shift[0]),
        int(torch.ceil(sf[0]).item() - shift[0]),
        int(torch.floor(sf[1]).item() + shift[1]),
        int(torch.ceil(sf[1]).item() - shift[1]),
        extmod
    )
    
    # Use PyTorch conv2d for filtering
    # conv2d expects (N, C, H, W) format
    xext_4d = xext.unsqueeze(0).unsqueeze(0)
    
    # Important: PyTorch's conv2d performs true convolution (flips kernel),
    # while NumPy's convolve2d performs correlation (no flip).
    # To match NumPy behavior, we need to flip the filter.
    f_flipped = torch.flip(f, [0, 1])
    f_4d = f_flipped.unsqueeze(0).unsqueeze(0)
    
    # Valid convolution (no padding)
    y = F.conv2d(xext_4d, f_4d, padding=0).squeeze()
    
    return y
