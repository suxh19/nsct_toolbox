"""
Utility functions for NSCT Torch - PyTorch implementation
Migrated from nsct_python/utils.py
"""

import torch
import torch.nn.functional as F
from typing import Tuple, List, Optional, Union


def extend2(x: torch.Tensor, ru: int, rd: int, cl: int, cr: int, extmod: str = 'per') -> torch.Tensor:
    """
    2D extension of an image. PyTorch translation of extend2.m.

    Args:
        x: Input image tensor (H x W).
        ru: Amount of extension up for rows.
        rd: Amount of extension down for rows.
        cl: Amount of extension left for columns.
        cr: Amount of extension right for columns.
        extmod: Extension mode. Valid modes are:
            'per': Periodized extension (default).
            'qper_row': Quincunx periodized extension in row.
            'qper_col': Quincunx periodized extension in column.

    Returns:
        Extended image tensor.
    """
    if extmod == 'per':
        # Periodic extension using torch pad with circular mode
        # PyTorch pad requires at least 3D for circular mode, so add batch dim
        # PyTorch pad order: (left, right, top, bottom)
        x_3d = x.unsqueeze(0)
        padded = F.pad(x_3d, (cl, cr, ru, rd), mode='circular')
        return padded.squeeze(0)

    rx, cx = x.shape

    if extmod == 'qper_row':
        # Quincunx periodized extension in row
        # Step 1: Extend left and right with vertical circular shift
        left_part = torch.roll(x[:, -cl:], rx // 2, dims=0)
        right_part = torch.roll(x[:, :cr], -rx // 2, dims=0)
        y = torch.cat([left_part, x, right_part], dim=1)

        # Step 2: Extend top and bottom periodically (add batch dim for circular mode)
        y_3d = y.unsqueeze(0)
        y_padded = F.pad(y_3d, (0, 0, ru, rd), mode='circular')
        return y_padded.squeeze(0)

    if extmod == 'qper_col':
        # Quincunx periodized extension in column
        # Step 1: Extend top and bottom with horizontal circular shift
        top_part = torch.roll(x[-ru:, :], cx // 2, dims=1)
        bottom_part = torch.roll(x[:rd, :], -cx // 2, dims=1)
        y = torch.cat([top_part, x, bottom_part], dim=0)

        # Step 2: Extend left and right periodically (add batch dim for circular mode)
        y_3d = y.unsqueeze(0)
        y_padded = F.pad(y_3d, (cl, cr, 0, 0), mode='circular')
        return y_padded.squeeze(0)

    raise ValueError(f"Invalid extension mode: {extmod}")


def symext(x: torch.Tensor, h: torch.Tensor, shift: Union[List[int], Tuple[int, int]]) -> torch.Tensor:
    """
    Symmetric extension for image x with filter h.
    PyTorch translation of symext.m.
    
    Performs symmetric extension (H/V symmetry) for image x and filter h.
    The filter h is assumed to have odd dimensions.
    If the filter has horizontal and vertical symmetry, then 
    the nonsymmetric part of conv2(h,x) has the same size as x.
    
    Args:
        x: Input image tensor (m×n).
        h: 2D filter coefficients tensor.
        shift: Shift values [s1, s2].
    
    Returns:
        Symmetrically extended image with size (m+p-1, n+q-1),
        where p and q are the filter dimensions.
    """
    m, n = x.shape
    p, q = h.shape
    
    p2 = int(torch.floor(torch.tensor(p / 2)).item())
    q2 = int(torch.floor(torch.tensor(q / 2)).item())
    s1 = shift[0]
    s2 = shift[1]
    
    # Calculate extension amounts
    ss = p2 - s1 + 1
    rr = q2 - s2 + 1
    
    # Horizontal extension (left and right)
    if ss > 0:
        left_ext = torch.flip(x[:, :ss], dims=[1])
    else:
        left_ext = torch.empty((m, 0), dtype=x.dtype, device=x.device)
    
    # Right extension
    right_start = n - 1
    right_end = n - p - s1
    
    if right_end <= right_start:
        right_ext = torch.flip(x[:, right_end:right_start + 1], dims=[1])
    else:
        right_ext = torch.empty((m, 0), dtype=x.dtype, device=x.device)
    
    yT = torch.cat([left_ext, x, right_ext], dim=1)
    
    # Vertical extension (top and bottom)
    if rr > 0:
        top_ext = torch.flip(yT[:rr, :], dims=[0])
    else:
        top_ext = torch.empty((0, yT.shape[1]), dtype=x.dtype, device=x.device)
    
    # Bottom extension
    bottom_start = m - 1
    bottom_end = m - q - s2
    
    if bottom_end <= bottom_start:
        bottom_ext = torch.flip(yT[bottom_end:bottom_start + 1, :], dims=[0])
    else:
        bottom_ext = torch.empty((0, yT.shape[1]), dtype=x.dtype, device=x.device)
    
    yT = torch.cat([top_ext, yT, bottom_ext], dim=0)
    
    # Crop to final size
    yT = yT[:m + p - 1, :n + q - 1]
    
    return yT


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


def modulate2(x: torch.Tensor, mode: str = 'b', center: Optional[List[int]] = None) -> torch.Tensor:
    """
    2D modulation. PyTorch translation of modulate2.m.

    Args:
        x: Input matrix tensor.
        mode: 'r' for row, 'c' for column, 'b' for both.
        center: Modulation center offset. Defaults to None.

    Returns:
        Modulated matrix tensor.
    """
    s = x.shape
    if center is None:
        center = [0, 0]

    o = torch.floor(torch.tensor(s, dtype=torch.float64) / 2) + 1 + torch.tensor(center, dtype=torch.float64)

    n1 = torch.arange(1, s[0] + 1, dtype=torch.float64, device=x.device) - o[0]
    n2 = torch.arange(1, s[1] + 1, dtype=torch.float64, device=x.device) - o[1]

    y = x.to(dtype=torch.float64).clone()
    
    if mode in ['r', 'b']:
        m1 = (-1) ** n1
        y *= m1.unsqueeze(1)

    if mode in ['c', 'b']:
        m2 = (-1) ** n2
        y *= m2.unsqueeze(0)

    return y


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
