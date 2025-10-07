"""
Extension functions for image processing
包含图像扩展和对称扩展函数
"""

import torch
import torch.nn.functional as F
from typing import Union, List, Tuple


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
