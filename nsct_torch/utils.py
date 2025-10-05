import torch
import torch.nn.functional as F
from typing import Tuple

def extend2(x: torch.Tensor, ru: int, rd: int, cl: int, cr: int, extmod: str = 'per') -> torch.Tensor:
    """
    2D extension of a tensor. PyTorch translation of extend2.m.

    Args:
        x (torch.Tensor): Input tensor (H, W).
        ru (int): Amount of extension up for rows.
        rd (int): Amount of extension down for rows.
        cl (int): Amount of extension left for columns.
        cr (int): Amount of extension right for columns.
        extmod (str): Extension mode. Valid modes are:
            'per': Periodized extension (default).
            'qper_row': Quincunx periodized extension in row.
            'qper_col': Quincunx periodized extension in column.

    Returns:
        torch.Tensor: Extended tensor.
    """
    if extmod == 'per':
        # PyTorch's 'circular' padding is equivalent to numpy's 'wrap'.
        # The padding order is (pad_left, pad_right, pad_top, pad_bottom).
        return F.pad(x.unsqueeze(0).unsqueeze(0), (cl, cr, ru, rd), 'circular').squeeze(0).squeeze(0)

    rx, cx = x.shape

    if extmod == 'qper_row':
        # Quincunx periodized extension in row
        # Step 1: Extend left and right with vertical circular shift
        y = torch.cat([
            torch.roll(x[:, -cl:], shifts=rx // 2, dims=0),
            x,
            torch.roll(x[:, :cr], shifts=-rx // 2, dims=0)
        ], dim=1)

        # Step 2: Extend top and bottom periodically
        y = F.pad(y.unsqueeze(0).unsqueeze(0), (0, 0, ru, rd), 'circular').squeeze(0).squeeze(0)
        return y

    if extmod == 'qper_col':
        # Quincunx periodized extension in column
        # Step 1: Extend top and bottom with horizontal circular shift
        y = torch.cat([
            torch.roll(x[-ru:, :], shifts=cx // 2, dims=1),
            x,
            torch.roll(x[:rd, :], shifts=-cx // 2, dims=1)
        ], dim=0)

        # Step 2: Extend left and right periodically
        y = F.pad(y.unsqueeze(0).unsqueeze(0), (cl, cr, 0, 0), 'circular').squeeze(0).squeeze(0)
        return y

    raise ValueError(f"Invalid extension mode: {extmod}")


def upsample2df(h: torch.Tensor, power: int = 1) -> torch.Tensor:
    """
    Upsample a 2D filter by 2^power by inserting zeros.
    Translation of upsample2df.m.

    Args:
        h (torch.Tensor): Input filter.
        power (int): Power of 2 for upsampling factor.

    Returns:
        torch.Tensor: Upsampled filter.
    """
    factor = 2**power
    m, n = h.shape
    ho = torch.zeros(factor * m, factor * n, dtype=h.dtype, device=h.device)
    ho[::factor, ::factor] = h
    return ho

def modulate2(x: torch.Tensor, mode: str = 'b', center: Tuple[int, int] = None) -> torch.Tensor:
    """
    2D modulation. Translation of modulate2.m.

    Args:
        x (torch.Tensor): Input matrix.
        mode (str): 'r' for row, 'c' for column, 'b' for both.
        center (tuple, optional): Modulation center offset. Defaults to None.

    Returns:
        torch.Tensor: Modulated matrix.
    """
    s = x.shape
    if center is None:
        center = (0, 0)

    # Note: MATLAB is 1-based, so floor(s/2)+1 is the center.
    # PyTorch/Numpy are 0-based, so floor((s-1)/2) is the center.
    # The original MATLAB code's logic is preserved here.
    o = (torch.tensor(s, dtype=torch.float64) / 2).floor() + 1 + torch.tensor(center, dtype=torch.float64)

    n1 = torch.arange(1, s[0] + 1, device=x.device) - o[0]
    n2 = torch.arange(1, s[1] + 1, device=x.device) - o[1]

    y = x.clone().to(torch.float64)  # Cast to float to handle multiplication by -1
    if mode in ['r', 'b']:
        m1 = (-1.0) ** n1
        y *= m1.unsqueeze(1)

    if mode in ['c', 'b']:
        m2 = (-1.0) ** n2
        y *= m2.unsqueeze(0)

    return y


def resampz(x: torch.Tensor, type: int, shift: int = 1) -> torch.Tensor:
    """
    Resampling of a matrix (shearing). Translation of resampz.m.

    Args:
        x (torch.Tensor): Input matrix.
        type (int): One of {1, 2, 3, 4}.
        shift (int): Amount of shift.

    Returns:
        torch.Tensor: Resampled matrix.
    """
    sx = x.shape

    if type in [1, 2]: # Vertical shearing
        y = torch.zeros(sx[0] + abs(shift * (sx[1] - 1)), sx[1], dtype=x.dtype, device=x.device)

        if type == 1:
            shift1 = torch.arange(sx[1], device=x.device) * (-shift)
        else: # type == 2
            shift1 = torch.arange(sx[1], device=x.device) * shift

        if (shift1 < 0).any():
            # Normalize to be non-negative for indexing
            shift1 = shift1 - shift1.min()

        for n in range(sx[1]):
            y[shift1[n] : shift1[n] + sx[0], n] = x[:, n]

        # Trim zero rows
        row_norms = torch.linalg.norm(y.float(), axis=1)
        non_zero_rows = torch.where(row_norms > 1e-6)[0]
        if len(non_zero_rows) == 0:
            return torch.zeros((0, sx[1]), dtype=x.dtype, device=x.device)
        return y[non_zero_rows.min():non_zero_rows.max()+1, :]

    elif type in [3, 4]: # Horizontal shearing
        y = torch.zeros(sx[0], sx[1] + abs(shift * (sx[0] - 1)), dtype=x.dtype, device=x.device)

        if type == 3:
            shift2 = torch.arange(sx[0], device=x.device) * (-shift)
        else: # type == 4
            shift2 = torch.arange(sx[0], device=x.device) * shift

        if (shift2 < 0).any():
            # Normalize to be non-negative for indexing
            shift2 = shift2 - shift2.min()

        for m in range(sx[0]):
            y[m, shift2[m] : shift2[m] + sx[1]] = x[m, :]

        # Trim zero columns
        col_norms = torch.linalg.norm(y.float(), axis=0)
        non_zero_cols = torch.where(col_norms > 1e-6)[0]
        if len(non_zero_cols) == 0:
            return torch.zeros((sx[0], 0), dtype=x.dtype, device=x.device)
        return y[:, non_zero_cols.min():non_zero_cols.max()+1]

    else:
        raise ValueError("Type must be one of {1, 2, 3, 4}")

def qupz(x: torch.Tensor, type: int = 1) -> torch.Tensor:
    """
    Quincunx Upsampling, based on the transform's mathematical definition.
    This replaces the complex resampz-based implementation.

    Note: To match MATLAB behavior, this function trims all-zero rows and columns
    from the output, similar to how MATLAB's resampz works.
    """
    r, c = x.shape
    device = x.device
    dtype = x.dtype

    # The output grid size is (r+c-1) x (r+c-1)
    out_size = (r + c - 1, r + c - 1)
    y = torch.zeros(out_size, dtype=dtype, device=device)

    if type == 1: # Upsampling by Q1 = [[1, -1], [1, 1]]
        offset_r = c - 1
        for r_idx in range(r):
            for c_idx in range(c):
                n1 = r_idx - c_idx
                n2 = r_idx + c_idx
                y[n1 + offset_r, n2] = x[r_idx, c_idx]

    elif type == 2: # Upsampling by Q2 = [[1, 1], [-1, 1]]
        offset_c = r - 1
        for r_idx in range(r):
            for c_idx in range(c):
                n1 = r_idx + c_idx
                n2 = -r_idx + c_idx
                y[n1, n2 + offset_c] = x[r_idx, c_idx]
    else:
        raise ValueError("type must be 1 or 2")

    # Trim all-zero rows and columns
    row_norms = torch.linalg.norm(y.float(), axis=1)
    col_norms = torch.linalg.norm(y.float(), axis=0)

    non_zero_rows = torch.where(row_norms > 1e-6)[0]
    non_zero_cols = torch.where(col_norms > 1e-6)[0]

    if len(non_zero_rows) == 0 or len(non_zero_cols) == 0:
        return torch.zeros((0, 0), dtype=dtype, device=device)

    y = y[non_zero_rows.min():non_zero_rows.max()+1,
          non_zero_cols.min():non_zero_cols.max()+1]

    return y