"""
Construct quincunx filters from a ladder network structure allpass filter.
PyTorch translation of ld2quin.m
"""

import torch
from typing import Tuple
from nsct_torch.utils import qupz


def ld2quin(beta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Construct quincunx filters from a ladder network structure allpass filter.
    
    Args:
        beta: 1D allpass filter tensor.
    
    Returns:
        tuple: (h0, h1) quincunx filters.
    """
    if beta.ndim != 1:
        raise ValueError('The input must be a 1-D filter')
    
    lf = beta.shape[0]
    n = lf // 2
    
    if n * 2 != lf:
        raise ValueError('The input allpass filter must be even length')
    
    # beta(z1) * beta(z2) -> outer product
    sp = torch.outer(beta, beta)
    
    # beta(z1*z2^{-1}) * beta(z1*z2)
    # Obtained by quincunx upsampling type 1 (with zero padded)
    h = qupz(sp, 1)
    
    # Lowpass quincunx filter
    h0 = h.clone()
    center_idx = lf - 1
    h0[center_idx, center_idx] += 1
    h0 = h0 / 2.0
    
    # Highpass quincunx filter
    # Use torch.nn.functional.conv2d for 2D convolution
    # PyTorch conv2d expects (N, C, H, W) format
    h_4d = h.unsqueeze(0).unsqueeze(0)
    h0_rot_4d = torch.rot90(h0, 2).unsqueeze(0).unsqueeze(0)
    
    # Full convolution
    h1 = -torch.nn.functional.conv2d(
        h_4d, 
        h0_rot_4d, 
        padding=(h0_rot_4d.shape[2] - 1, h0_rot_4d.shape[3] - 1)
    ).squeeze()
    
    center_idx_h1 = 4 * n - 2
    h1[center_idx_h1, center_idx_h1] += 1
    
    return h0, h1
