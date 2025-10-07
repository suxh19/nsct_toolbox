"""
Directional filter bank operations for NSCT
包含非下采样方向滤波器组的分解和重构函数
"""

import torch
from typing import Union, List, Dict
from nsct_torch.filters import dfilters, parafilters
from nsct_torch.utils import modulate2
from .filterbank import nssfbdec, nssfbrec


def nsdfbdec(x: torch.Tensor, dfilter: Union[str, Dict], 
            clevels: int = 0, device: str = 'cpu') -> List[torch.Tensor]:
    """
    Nonsubsampled Directional Filter Bank (NSDFB) decomposition.
    PyTorch translation of nsdfbdec.m.
    
    Args:
        x: Input image (2D tensor)
        dfilter: Either str (filter name) or dict with filters
        clevels: Number of decomposition levels
        device: Device to use for tensor creation
    
    Returns:
        List of output subbands (length 2^clevels)
    """
    # Input validation
    if clevels != round(clevels) or clevels < 0:
        raise ValueError('Number of decomposition levels must be a non-negative integer')
    
    # No decomposition case
    if clevels == 0:
        return [x]
    
    # Get filters
    if isinstance(dfilter, str):
        h1, h2 = dfilters(dfilter, 'd', device=device)
        h1 = h1 / torch.sqrt(torch.tensor(2.0, device=device))
        h2 = h2 / torch.sqrt(torch.tensor(2.0, device=device))
        
        k1 = modulate2(h1, 'c')
        k2 = modulate2(h2, 'c')
        
        f1, f2 = parafilters(h1, h2)
    elif isinstance(dfilter, dict):
        if not all(key in dfilter for key in ['k1', 'k2', 'f1', 'f2']):
            raise ValueError("Filter dict must contain keys: 'k1', 'k2', 'f1', 'f2'")
        k1 = dfilter['k1']
        k2 = dfilter['k2']
        f1 = dfilter['f1']
        f2 = dfilter['f2']
    else:
        raise TypeError('dfilter must be a string or dict')
    
    # Quincunx sampling matrix
    q1 = torch.tensor([[1, -1], [1, 1]], dtype=torch.int64, device=device)
    
    # First-level decomposition
    if clevels == 1:
        y1, y2 = nssfbdec(x, k1, k2)
        return [y1, y2]
    
    # Multi-level decomposition (clevels >= 2)
    x1, x2 = nssfbdec(x, k1, k2)
    
    y: List[torch.Tensor] = [torch.empty(0)] * 4
    y[0], y[1] = nssfbdec(x1, k1, k2, q1)
    y[2], y[3] = nssfbdec(x2, k1, k2, q1)
    
    # Third and higher levels decomposition
    for l in range(3, clevels + 1):
        y_old = y
        y = [torch.empty(0)] * (2 ** l)
        
        # The first half channels
        for k in range(1, 2 ** (l - 2) + 1):
            slk = 2 * ((k - 1) // 2) - 2 ** (l - 3) + 1
            mkl = 2 * torch.tensor([[2 ** (l - 3), 0], [0, 1]], 
                                  dtype=torch.int64, device=device) @ \
                  torch.tensor([[1, 0], [-slk, 1]], dtype=torch.int64, device=device)
            
            i = ((k - 1) % 2)
            y[2 * k - 2], y[2 * k - 1] = nssfbdec(y_old[k - 1], f1[i], f2[i], mkl)
        
        # The second half channels
        for k in range(2 ** (l - 2) + 1, 2 ** (l - 1) + 1):
            slk = 2 * ((k - 2 ** (l - 2) - 1) // 2) - 2 ** (l - 3) + 1
            mkl = 2 * torch.tensor([[1, 0], [0, 2 ** (l - 3)]], 
                                  dtype=torch.int64, device=device) @ \
                  torch.tensor([[1, -slk], [0, 1]], dtype=torch.int64, device=device)
            
            i = ((k - 1) % 2) + 2
            y[2 * k - 2], y[2 * k - 1] = nssfbdec(y_old[k - 1], f1[i], f2[i], mkl)
    
    return y


def nsdfbrec(y: List[torch.Tensor], dfilter: Union[str, Dict],
            device: str = 'cpu') -> torch.Tensor:
    """
    Nonsubsampled directional filter bank reconstruction.
    PyTorch translation of nsdfbrec.m.
    
    Args:
        y: List of directional subbands (2^clevels subbands)
        dfilter: Filter specification (str or dict)
        device: Device to use
    
    Returns:
        Reconstructed image
    """
    # Check input
    if len(y) == 0:
        raise ValueError('Number of subbands must be a power of 2')
    
    # Determine clevels
    clevels = int(torch.log2(torch.tensor(len(y))).item())
    if 2**clevels != len(y):
        raise ValueError('Number of subbands must be a power of 2')
    
    # No reconstruction case
    if clevels == 0:
        return y[0]
    
    # Get filters (synthesis filters 'r')
    if isinstance(dfilter, str):
        h1, h2 = dfilters(dfilter, 'r', device=device)
        h1 = h1 / torch.sqrt(torch.tensor(2.0, device=device))
        h2 = h2 / torch.sqrt(torch.tensor(2.0, device=device))
        
        k1 = modulate2(h1, 'c')
        k2 = modulate2(h2, 'c')
        
        f1, f2 = parafilters(h1, h2)
    elif isinstance(dfilter, dict):
        if not all(key in dfilter for key in ['k1', 'k2', 'f1', 'f2']):
            raise ValueError("Filter dict must contain keys: 'k1', 'k2', 'f1', 'f2'")
        k1 = dfilter['k1']
        k2 = dfilter['k2']
        f1 = dfilter['f1']
        f2 = dfilter['f2']
    else:
        raise TypeError('dfilter must be a string or dict')
    
    # Quincunx sampling matrix
    q1 = torch.tensor([[1, -1], [1, 1]], dtype=torch.int64, device=device)
    
    # First-level reconstruction
    if clevels == 1:
        return nssfbrec(y[0], y[1], k1, k2)
    
    # Multi-level reconstruction
    x = y.copy()
    
    # Third and higher levels reconstructions (from highest to lowest)
    for l in range(clevels, 2, -1):
        # The first half channels
        for k in range(1, 2 ** (l - 2) + 1):
            slk = 2 * ((k - 1) // 2) - 2 ** (l - 3) + 1
            mkl = 2 * torch.tensor([[2 ** (l - 3), 0], [0, 1]], 
                                  dtype=torch.int64, device=device) @ \
                  torch.tensor([[1, 0], [-slk, 1]], dtype=torch.int64, device=device)
            
            i = ((k - 1) % 2)
            x[k - 1] = nssfbrec(x[2 * k - 2], x[2 * k - 1], f1[i], f2[i], mkl)
        
        # The second half channels
        for k in range(2 ** (l - 2) + 1, 2 ** (l - 1) + 1):
            slk = 2 * ((k - 2 ** (l - 2) - 1) // 2) - 2 ** (l - 3) + 1
            mkl = 2 * torch.tensor([[1, 0], [0, 2 ** (l - 3)]], 
                                  dtype=torch.int64, device=device) @ \
                  torch.tensor([[1, -slk], [0, 1]], dtype=torch.int64, device=device)
            
            i = ((k - 1) % 2) + 2
            x[k - 1] = nssfbrec(x[2 * k - 2], x[2 * k - 1], f1[i], f2[i], mkl)
    
    # Second-level reconstruction
    x[0] = nssfbrec(x[0], x[1], k1, k2, q1)
    x[1] = nssfbrec(x[2], x[3], k1, k2, q1)
    
    # First-level reconstruction
    result = nssfbrec(x[0], x[1], k1, k2)
    
    return result
