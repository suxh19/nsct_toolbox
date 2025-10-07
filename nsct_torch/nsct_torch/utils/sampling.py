"""
Sampling operations for NSCT
包含上采样、重采样和quincunx采样函数
"""

import torch
from typing import Union


def upsample2df(h: torch.Tensor, power: int = 1) -> torch.Tensor:
    """
    Upsample a 2D filter by 2^power by inserting zeros.
    PyTorch translation of upsample2df.m.

    Args:
        h: Input filter tensor.
        power: Power of 2 for upsampling factor.

    Returns:
        Upsampled filter tensor.
    """
    factor = 2 ** power
    m, n = h.shape
    ho = torch.zeros((factor * m, factor * n), dtype=h.dtype, device=h.device)
    ho[::factor, ::factor] = h
    return ho


def resampz(x: torch.Tensor, type: int, shift: int = 1) -> torch.Tensor:
    """
    Resampling of a matrix (shearing). PyTorch translation of resampz.m.

    Args:
        x: Input matrix tensor.
        type: One of {1, 2, 3, 4}.
        shift: Amount of shift.

    Returns:
        Resampled matrix tensor.
    """
    sx = x.shape

    if type in [1, 2]:  # Vertical shearing
        y = torch.zeros((sx[0] + abs(shift * (sx[1] - 1)), sx[1]), dtype=x.dtype, device=x.device)

        if type == 1:
            shift1 = torch.arange(sx[1], device=x.device) * (-shift)
        else:  # type == 2
            shift1 = torch.arange(sx[1], device=x.device) * shift

        if torch.any(shift1 < 0):
            shift1 = shift1 - shift1.min()

        for n in range(sx[1]):
            y[shift1[n]: shift1[n] + sx[0], n] = x[:, n]

        # Trim zero rows
        row_norms = torch.norm(y, dim=1)
        non_zero_rows = torch.where(row_norms > 0)[0]
        if len(non_zero_rows) == 0:
            return torch.zeros((0, sx[1]), dtype=x.dtype, device=x.device)
        return y[non_zero_rows.min():non_zero_rows.max() + 1, :]

    elif type in [3, 4]:  # Horizontal shearing
        y = torch.zeros((sx[0], sx[1] + abs(shift * (sx[0] - 1))), dtype=x.dtype, device=x.device)

        if type == 3:
            shift2 = torch.arange(sx[0], device=x.device) * (-shift)
        else:  # type == 4
            shift2 = torch.arange(sx[0], device=x.device) * shift

        if torch.any(shift2 < 0):
            shift2 = shift2 - shift2.min()

        for m in range(sx[0]):
            y[m, shift2[m]: shift2[m] + sx[1]] = x[m, :]

        # Trim zero columns
        col_norms = torch.norm(y, dim=0)
        non_zero_cols = torch.where(col_norms > 0)[0]
        if len(non_zero_cols) == 0:
            return torch.zeros((sx[0], 0), dtype=x.dtype, device=x.device)
        return y[:, non_zero_cols.min():non_zero_cols.max() + 1]

    else:
        raise ValueError("Type must be one of {1, 2, 3, 4}")


def qupz(x: torch.Tensor, type: int = 1) -> torch.Tensor:
    """
    Quincunx Upsampling, based on the transform's mathematical definition.
    PyTorch translation.
    
    Note: To match MATLAB behavior, this function trims all-zero rows and columns
    from the output.
    
    Args:
        x: Input tensor.
        type: 1 or 2 for different quincunx upsampling types.
        
    Returns:
        Upsampled tensor.
    """
    r, c = x.shape

    # The output grid size is (r+c-1) x (r+c-1)
    out_size = (r + c - 1, r + c - 1)
    y = torch.zeros(out_size, dtype=x.dtype, device=x.device)

    if type == 1:  # Upsampling by Q1 = [[1, -1], [1, 1]]
        offset_r = c - 1
        for r_idx in range(r):
            for c_idx in range(c):
                n1 = r_idx - c_idx
                n2 = r_idx + c_idx
                y[n1 + offset_r, n2] = x[r_idx, c_idx]

    elif type == 2:  # Upsampling by Q2 = [[1, 1], [-1, 1]]
        offset_c = r - 1
        for r_idx in range(r):
            for c_idx in range(c):
                n1 = r_idx + c_idx
                n2 = -r_idx + c_idx
                y[n1, n2 + offset_c] = x[r_idx, c_idx]
    else:
        raise ValueError("type must be 1 or 2")

    # Trim all-zero rows and columns
    row_norms = torch.norm(y, dim=1)
    col_norms = torch.norm(y, dim=0)
    
    non_zero_rows = torch.where(row_norms > 0)[0]
    non_zero_cols = torch.where(col_norms > 0)[0]
    
    if len(non_zero_rows) == 0 or len(non_zero_cols) == 0:
        return torch.zeros((0, 0), dtype=x.dtype, device=x.device)
    
    y = y[non_zero_rows.min():non_zero_rows.max() + 1, 
          non_zero_cols.min():non_zero_cols.max() + 1]
    
    return y
