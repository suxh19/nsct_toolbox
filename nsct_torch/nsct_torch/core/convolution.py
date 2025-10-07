"""
Convolution operations for NSCT
包含底层卷积实现函数
"""

import torch
from typing import Union


def _zconv2_torch(x: torch.Tensor, h: torch.Tensor, mup: torch.Tensor) -> torch.Tensor:
    """
    2D convolution with upsampled filter using periodic boundary.
    PyTorch implementation (CPU/GPU compatible).
    
    Args:
        x: Input signal (2D tensor)
        h: Filter (2D tensor)
        mup: Upsampling matrix (2x2 tensor) [[M0, M1], [M2, M3]]
    
    Returns:
        y: Convolution output (same size as x)
    """
    s_rows, s_cols = x.shape
    f_rows, f_cols = h.shape
    
    # Extract upsampling matrix elements
    M0, M1 = int(mup[0, 0].item()), int(mup[0, 1].item())
    M2, M3 = int(mup[1, 0].item()), int(mup[1, 1].item())
    
    # Calculate upsampled filter dimensions
    new_f_rows = (M0 - 1) * (f_rows - 1) + M2 * (f_cols - 1) + f_rows - 1
    new_f_cols = (M3 - 1) * (f_cols - 1) + M1 * (f_rows - 1) + f_cols - 1
    
    # Create output tensor
    result = torch.zeros_like(x)
    
    # Initialize
    start1 = new_f_rows // 2
    start2 = new_f_cols // 2
    mn1_init = start1 % s_rows
    mn2_save = start2 % s_cols
    
    # Main convolution loop
    for n1 in range(s_rows):
        mn1 = (mn1_init + n1) % s_rows
        mn2 = mn2_save
        
        for n2 in range(s_cols):
            out_index_x = mn1
            out_index_y = mn2
            total = 0.0
            
            # Loop over filter rows
            for l1 in range(f_rows):
                index_x = out_index_x
                index_y = out_index_y
                
                # Loop over filter columns
                for l2 in range(f_cols):
                    total += x[index_x, index_y].item() * h[l1, l2].item()
                    
                    # Step through input with M2, M3 (periodic boundary)
                    index_x = (index_x - M2) % s_rows
                    index_y = (index_y - M3) % s_cols
                
                # Step for outer filter loop with M0, M1
                out_index_x = (out_index_x - M0) % s_rows
                out_index_y = (out_index_y - M1) % s_cols
            
            result[n1, n2] = total
            
            # Update mn2 for next column
            mn2 = (mn2 + 1) % s_cols
    
    return result


def _atrousc_torch(x: torch.Tensor, h: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """
    À trous convolution with upsampled filter.
    PyTorch implementation (CPU/GPU compatible).
    
    Args:
        x: Extended input signal (2D tensor)
        h: Original filter, not upsampled (2D tensor)
        M: Upsampling matrix (2x2 tensor or scalar)
    
    Returns:
        Result of convolution in 'valid' mode (2D tensor)
    """
    S_rows, S_cols = x.shape
    F_rows, F_cols = h.shape
    
    # Extract upsampling factors
    if M.ndim == 0:
        M0 = M3 = int(M.item())
    else:
        M0 = int(M[0, 0].item())
        M3 = int(M[1, 1].item())
    
    # Calculate output dimensions
    O_rows = S_rows - M0 * F_rows + 1
    O_cols = S_cols - M3 * F_cols + 1
    
    if O_rows <= 0 or O_cols <= 0:
        return torch.zeros(max(0, O_rows), max(0, O_cols), 
                          dtype=x.dtype, device=x.device)
    
    # Create output tensor
    result = torch.zeros(O_rows, O_cols, dtype=x.dtype, device=x.device)
    
    # Flip the filter for convolution
    h_flipped = torch.flip(torch.flip(h, [0]), [1])
    
    # Main convolution loop
    for n1 in range(O_cols):
        for n2 in range(O_rows):
            total = 0.0
            kk1 = n1 + M0 - 1
            
            # Loop over filter columns
            for k1 in range(F_cols):
                kk2 = n2 + M3 - 1
                
                # Loop over filter rows
                for k2 in range(F_rows):
                    # Flipped indices
                    f1 = F_cols - 1 - k1
                    f2 = F_rows - 1 - k2
                    
                    # Accumulate
                    total += h_flipped[f2, f1].item() * x[kk2, kk1].item()
                    kk2 += M3
                
                kk1 += M0
            
            result[n2, n1] = total
    
    return result


def _convolve_upsampled(x: torch.Tensor, f: torch.Tensor, 
                       mup: Union[int, float, torch.Tensor], 
                       is_rec: bool = False) -> torch.Tensor:
    """
    Helper for convolution with an upsampled filter, handling reconstruction.
    
    Args:
        x: Input tensor
        f: Filter tensor
        mup: Upsampling factor or matrix
        is_rec: Whether this is reconstruction (uses time-reversed filter)
    
    Returns:
        Convolution result
    """
    # If the filter is all zeros, return zeros
    if not torch.any(f):
        return torch.zeros_like(x)
    
    # Convert mup to matrix form
    if isinstance(mup, (int, float)):
        mup_mat = torch.tensor([[mup, 0], [0, mup]], 
                              dtype=torch.int64, device=x.device)
    else:
        mup_mat = mup.to(torch.int64)
    
    # For reconstruction, use time-reversed filter
    f_to_use = torch.rot90(f, 2) if is_rec else f
    
    # Use zconv2 for periodic convolution with upsampled filter
    return _zconv2_torch(x, f_to_use, mup_mat)
