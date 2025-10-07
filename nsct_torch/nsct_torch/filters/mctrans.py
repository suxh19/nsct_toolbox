"""
McClellan transformation.
PyTorch translation of mctrans.m
"""

import torch
import torch.nn.functional as F


def mctrans(b: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    McClellan transformation.
    Produces the 2-D FIR filter H that corresponds to the 1-D FIR filter B
    using the transform T.
    
    Args:
        b: 1-D FIR filter (row vector) tensor.
        t: 2-D transformation filter tensor.
    
    Returns:
        The resulting 2-D FIR filter tensor.
    """
    n = (b.shape[0] - 1) // 2
    b = torch.fft.ifftshift(b)
    a = torch.cat([b[0:1], 2 * b[1:n + 1]])
    
    inset = torch.floor((torch.tensor(t.shape, dtype=torch.float64) - 1) / 2).to(torch.int64)
    
    # Use Chebyshev polynomials to compute h
    P0 = torch.tensor(1.0, dtype=t.dtype, device=t.device)
    P1 = t
    h = a[1] * P1
    
    # Add a[0]*P0 to the center of h
    r_h, c_h = h.shape
    h[r_h // 2, c_h // 2] += a[0]
    
    for i in range(2, n + 1):
        # Full convolution using conv2d
        t_4d = t.unsqueeze(0).unsqueeze(0)
        P1_4d = P1.unsqueeze(0).unsqueeze(0)
        
        P2 = 2 * F.conv2d(
            P1_4d, 
            t_4d, 
            padding=(t.shape[0] - 1, t.shape[1] - 1)
        ).squeeze()
        
        # Subtract P0 from the center of P2
        r_p2, c_p2 = P2.shape
        if P0.ndim == 0:  # scalar
            r_p0, c_p0 = 1, 1
            P2[r_p2 // 2, c_p2 // 2] -= P0
        else:
            r_p0, c_p0 = P0.shape
            start_r = (r_p2 - r_p0) // 2
            start_c = (c_p2 - c_p0) // 2
            P2[start_r: start_r + r_p0, start_c: start_c + c_p0] -= P0
        
        # Add the previous h to the center of the new h
        hh = h
        h = a[i] * P2
        r_h, c_h = h.shape
        r_hh, c_hh = hh.shape
        start_r = (r_h - r_hh) // 2
        start_c = (c_h - c_hh) // 2
        h[start_r: start_r + r_hh, start_c: start_c + c_hh] += hh
        
        P0 = P1
        P1 = P2
    
    # Rotate for use with filter2 (correlation)
    return torch.rot90(h, 2)
