"""
Main NSCT (Nonsubsampled Contourlet Transform) functions
包含NSCT的主要分解和重构函数
"""

import torch
from typing import List, Any
from nsct_torch.filters import dfilters, parafilters, atrousfilters
from nsct_torch.utils import modulate2
from .filterbank import nsfbdec, nsfbrec
from .directional import nsdfbdec, nsdfbrec


def nsctdec(x: torch.Tensor, levels: List[int], dfilt: str = 'dmaxflat7',
           pfilt: str = 'maxflat', device: str = 'cpu') -> List:
    """
    Nonsubsampled Contourlet Transform Decomposition.
    PyTorch translation of nsctdec.m.
    
    Args:
        x: Input image (2D tensor)
        levels: List of directional decomposition levels at each pyramidal level
        dfilt: Filter name for directional decomposition
        pfilt: Filter name for pyramidal decomposition
        device: Device to use
    
    Returns:
        List where y[0] is lowpass and y[1:] are bandpass directional subbands
    """
    # Input validation
    levels_array = torch.tensor(levels, dtype=torch.int64)
    if not torch.all(levels_array == torch.round(levels_array.to(torch.float32))):
        raise ValueError('The decomposition levels shall be integers')
    
    if torch.any(levels_array < 0):
        raise ValueError('The decomposition levels shall be non-negative integers')
    
    # Get filters
    h1, h2 = dfilters(dfilt, 'd', device=device)
    h1 = h1 / torch.sqrt(torch.tensor(2.0, device=device))
    h2 = h2 / torch.sqrt(torch.tensor(2.0, device=device))
    
    k1 = modulate2(h1, 'c')
    k2 = modulate2(h2, 'c')
    
    f1, f2 = parafilters(h1, h2)
    
    filters = {
        'k1': k1,
        'k2': k2,
        'f1': f1,
        'f2': f2
    }
    
    # Get pyramid filters
    h1_pyr, h2_pyr, g1_pyr, g2_pyr = atrousfilters(pfilt, device=device)
    
    # Number of levels
    clevels = len(levels)
    nIndex = clevels
    
    # Initialize output
    y: List[Any] = [None] * (clevels + 1)
    
    # Nonsubsampled pyramid decomposition
    for i in range(clevels):
        xlo, xhi = nsfbdec(x, h1_pyr, h2_pyr, i)
        
        if levels[nIndex - 1] > 0:
            xhi_dir = nsdfbdec(xhi, filters, levels[nIndex - 1], device=device)
            y[nIndex] = xhi_dir
        else:
            y[nIndex] = xhi
        
        nIndex = nIndex - 1
        x = xlo
    
    # The lowpass output
    y[0] = x
    
    return y


def nsctrec(y: List, dfilt: str = 'dmaxflat7', pfilt: str = 'maxflat',
           device: str = 'cpu') -> torch.Tensor:
    """
    Nonsubsampled Contourlet Reconstruction.
    PyTorch translation of nsctrec.m.
    
    Args:
        y: List of subbands from nsctdec
        dfilt: Filter name for directional reconstruction
        pfilt: Filter name for pyramidal reconstruction
        device: Device to use
    
    Returns:
        Reconstructed image
    """
    # Get filters (synthesis filters 'r')
    h1_dir, h2_dir = dfilters(dfilt, 'r', device=device)
    h1_dir = h1_dir / torch.sqrt(torch.tensor(2.0, device=device))
    h2_dir = h2_dir / torch.sqrt(torch.tensor(2.0, device=device))
    
    k1 = modulate2(h1_dir, 'c')
    k2 = modulate2(h2_dir, 'c')
    f1, f2 = parafilters(h1_dir, h2_dir)
    
    filters = {
        'k1': k1,
        'k2': k2,
        'f1': f1,
        'f2': f2
    }
    
    # Get pyramid filters
    h1_pyr, h2_pyr, g1_pyr, g2_pyr = atrousfilters(pfilt, device=device)
    
    # Number of pyramid levels
    n = len(y) - 1
    
    # Special case: no pyramid levels
    if n == 0:
        return y[0]
    
    # Start with lowpass
    xlo = y[0]
    nIndex = n - 1
    
    # Reconstruct from coarsest to finest level
    for i in range(n):
        if isinstance(y[i + 1], list):
            xhi = nsdfbrec(y[i + 1], filters, device=device)
        else:
            xhi = y[i + 1]
        
        x = nsfbrec(xlo, xhi, g1_pyr, g2_pyr, nIndex)
        xlo = x
        nIndex = nIndex - 1
    
    return x
