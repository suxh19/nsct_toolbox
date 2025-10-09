import torch
import torch.nn.functional as F
from typing import Optional


def extend2(x, ru, rd, cl, cr, extmod='per'):
    """
    2D extension of an image. PyTorch translation of extend2.m.

    Args:
        x (torch.Tensor): Input image (2D tensor).
        ru (int): Amount of extension up for rows.
        rd (int): Amount of extension down for rows.
        cl (int): Amount of extension left for columns.
        cr (int): Amount of extension right for columns.
        extmod (str): Extension mode. Valid modes are:
            'per': Periodized extension (default).
            'qper_row': Quincunx periodized extension in row.
            'qper_col': Quincunx periodized extension in column.

    Returns:
        torch.Tensor: Extended image.
    """
    if extmod == 'per':
        # Periodic extension using circular padding
        # For 2D tensors, need to add batch and channel dimensions
        # torch.nn.functional.pad uses (left, right, top, bottom) order
        x_4d = x.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        padded = F.pad(x_4d, (cl, cr, ru, rd), mode='circular')
        return padded.squeeze(0).squeeze(0)  # Remove added dims

    rx, cx = x.shape

    if extmod == 'qper_row':
        # Quincunx periodized extension in row
        # Step 1: Extend left and right with vertical circular shift
        left_ext = torch.roll(x[:, -cl:], rx // 2, dims=0)
        right_ext = torch.roll(x[:, :cr], -rx // 2, dims=0)
        y = torch.cat([left_ext, x, right_ext], dim=1)

        # Step 2: Extend top and bottom periodically
        y_4d = y.unsqueeze(0).unsqueeze(0)
        padded = F.pad(y_4d, (0, 0, ru, rd), mode='circular')
        return padded.squeeze(0).squeeze(0)

    if extmod == 'qper_col':
        # Quincunx periodized extension in column
        # Step 1: Extend top and bottom with horizontal circular shift
        top_ext = torch.roll(x[-ru:, :], cx // 2, dims=1)
        bottom_ext = torch.roll(x[:rd, :], -cx // 2, dims=1)
        y = torch.cat([top_ext, x, bottom_ext], dim=0)

        # Step 2: Extend left and right periodically
        y_4d = y.unsqueeze(0).unsqueeze(0)
        padded = F.pad(y_4d, (cl, cr, 0, 0), mode='circular')
        return padded.squeeze(0).squeeze(0)

    raise ValueError(f"Invalid extension mode: {extmod}")


def _reflect_indices(
    length: int,
    pad: int,
    side: str,
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Generate reflection indices for symmetric padding. Handles arbitrarily large
    padding values by repeatedly mirroring the borders, matching MATLAB's
    symmetric extension semantics.
    """
    dtype = torch.float32 if dtype is None else dtype

    if pad <= 0:
        return torch.empty(0, dtype=dtype, device=device)

    if length <= 0:
        raise ValueError("Input length must be positive for symmetric extension")

    if side == "left":
        idx = torch.arange(-pad, 0, device=device, dtype=dtype)
    elif side == "right":
        idx = torch.arange(length, length + pad, device=device, dtype=dtype)
    else:
        raise ValueError("side must be 'left' or 'right'")

    if length == 1:
        return torch.zeros_like(idx)

    period = 2 * length
    idx_mod = torch.remainder(idx, period)
    reflected = torch.where(idx_mod < length, idx_mod, period - idx_mod - 1)
    return reflected.to(dtype)


def symext(x, h, shift, *, index_dtype: Optional[torch.dtype] = None):
    """
    Symmetric extension for image x with filter h (PyTorch translation).

    This implementation mirrors the MATLAB behaviour even when the requested
    extension exceeds the signal length by repeatedly reflecting the borders.

    Args:
        x (torch.Tensor): Input signal.
        h (torch.Tensor): Filter kernel.
        shift (iterable): Extension shift offsets.
        index_dtype (torch.dtype, optional): dtype used when constructing index vectors.
            Defaults to torch.float32.
    """
    if x.dim() != 2:
        raise ValueError("symext expects a 2D tensor")

    m, n = x.shape
    p, q = h.shape

    p2 = int(p // 2)
    q2 = int(q // 2)
    s1 = int(shift[0])
    s2 = int(shift[1])

    pad_left = p2 - s1 + 1
    pad_right = p + s1
    pad_top = q2 - s2 + 1
    pad_bottom = q + s2

    device = x.device
    index_dtype = torch.float32 if index_dtype is None else index_dtype

    center_cols = torch.arange(n, device=device, dtype=index_dtype)
    left_idx = _reflect_indices(n, pad_left, "left", device, dtype=index_dtype)
    right_idx = _reflect_indices(n, pad_right, "right", device, dtype=index_dtype)
    col_idx = torch.cat([left_idx, center_cols, right_idx], dim=0).to(torch.long)
    extended = x.index_select(1, col_idx)

    center_rows = torch.arange(m, device=device, dtype=index_dtype)
    top_idx = _reflect_indices(m, pad_top, "left", device, dtype=index_dtype)
    bottom_idx = _reflect_indices(m, pad_bottom, "right", device, dtype=index_dtype)
    row_idx = torch.cat([top_idx, center_rows, bottom_idx], dim=0).to(torch.long)
    extended = extended.index_select(0, row_idx)

    return extended[: m + p - 1, : n + q - 1]


def upsample2df(h, power=1):
    """
    Upsample a 2D filter by 2^power by inserting zeros.
    PyTorch translation of upsample2df.m.

    Args:
        h (torch.Tensor): Input filter.
        power (int): Power of 2 for upsampling factor.

    Returns:
        torch.Tensor: Upsampled filter.
    """
    factor = 2 ** power
    m, n = h.shape
    ho = torch.zeros((factor * m, factor * n), dtype=h.dtype, device=h.device)
    ho[::factor, ::factor] = h
    return ho


def modulate2(x, mode='b', center=None):
    """
    2D modulation. PyTorch translation of modulate2.m.

    Args:
        x (torch.Tensor): Input matrix.
        mode (str): 'r' for row, 'c' for column, 'b' for both.
        center (list or tuple, optional): Modulation center offset. Defaults to None.

    Returns:
        torch.Tensor: Modulated matrix (preserves input dtype).
    """
    s = x.shape
    device = x.device
    dtype = x.dtype
    if center is None:
        center = [0, 0]

    shape_tensor = torch.tensor(s, dtype=torch.float32, device=device)
    center_tensor = torch.tensor(center, dtype=torch.float32, device=device)
    o = torch.floor(shape_tensor / 2) + 1.0 + center_tensor

    n1 = torch.arange(1, s[0] + 1, dtype=torch.float32, device=device) - o[0]
    n2 = torch.arange(1, s[1] + 1, dtype=torch.float32, device=device) - o[1]

    y = x.clone()
    
    if mode in ['r', 'b']:
        m1 = (-1) ** n1
        y = y * m1.unsqueeze(1)

    if mode in ['c', 'b']:
        m2 = (-1) ** n2
        y = y * m2.unsqueeze(0)

    return y


def resampz(x, type, shift=1):
    """
    Resampling of a matrix (shearing). PyTorch translation of resampz.m.

    Args:
        x (torch.Tensor): Input matrix.
        type (int): One of {1, 2, 3, 4}.
        shift (int): Amount of shift.

    Returns:
        torch.Tensor: Resampled matrix.
    """
    sx = x.shape

    if type in [1, 2]:  # Vertical shearing
        y = torch.zeros((sx[0] + abs(shift * (sx[1] - 1)), sx[1]), 
                       dtype=x.dtype, device=x.device)

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
        return y[non_zero_rows.min().item():non_zero_rows.max().item() + 1, :]

    elif type in [3, 4]:  # Horizontal shearing
        y = torch.zeros((sx[0], sx[1] + abs(shift * (sx[0] - 1))), 
                       dtype=x.dtype, device=x.device)

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
        return y[:, non_zero_cols.min().item():non_zero_cols.max().item() + 1]

    else:
        raise ValueError("Type must be one of {1, 2, 3, 4}")


def qupz(x, type=1):
    """
    Quincunx Upsampling, based on the transform's mathematical definition.
    PyTorch translation of qupz.
    
    Note: To match MATLAB behavior, this function trims all-zero rows and columns
    from the output, similar to how MATLAB's resampz works.
    
    Args:
        x (torch.Tensor): Input matrix.
        type (int): Type of quincunx upsampling (1 or 2).
    
    Returns:
        torch.Tensor: Upsampled matrix.
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
    
    # Handle edge case: completely zero matrix
    if len(non_zero_rows) == 0 or len(non_zero_cols) == 0:
        return torch.zeros((0, 0), dtype=x.dtype, device=x.device)
    
    # Trim to bounding box of non-zero elements
    y = y[non_zero_rows.min().item():non_zero_rows.max().item() + 1, 
          non_zero_cols.min().item():non_zero_cols.max().item() + 1]
    
    return y


if __name__ == '__main__':
    # --- Tests for extend2 ---
    print("--- Running tests for extend2 ---")
    img = torch.arange(16, dtype=torch.float32).reshape((4, 4))
    ext_per = extend2(img, 1, 1, 1, 1, 'per')
    assert ext_per.shape == (6, 6)
    print("extend2 tests passed!")

    # --- Tests for upsample2df ---
    print("\n--- Running tests for upsample2df ---")
    h = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    h_up = upsample2df(h, power=1)
    expected_h_up = torch.tensor([[1, 0, 2, 0], [0, 0, 0, 0], [3, 0, 4, 0], [0, 0, 0, 0]], 
                                  dtype=torch.float32)
    print("Original filter:\n", h)
    print("Upsampled filter (power=1):\n", h_up)
    assert torch.equal(h_up, expected_h_up)
    assert h_up.shape == (4, 4)
    print("upsample2df tests passed!")

    # --- Tests for modulate2 ---
    print("\n--- Running tests for modulate2 ---")
    m = torch.ones((3, 4), dtype=torch.float32)
    m_mod_b = modulate2(m, 'b')
    print("Original matrix:\n", m)
    print("Modulated 'both':\n", m_mod_b)
    assert m_mod_b[0, 0] == -1.0
    assert m_mod_b[1, 0] == 1.0
    assert m_mod_b[0, 1] == 1.0
    print("modulate2 tests passed!")

    # --- Tests for resampz ---
    print("\n--- Running tests for resampz ---")
    r_in = torch.arange(1, 7, dtype=torch.float32).reshape(2, 3)
    r_out1 = resampz(r_in, 1, shift=1)
    expected_r1 = torch.tensor([[0, 0, 3], [0, 2, 6], [1, 5, 0], [4, 0, 0]], 
                                dtype=torch.float32)
    print("Original for resampz:\n", r_in)
    print("Resampled (type 1, shift 1):\n", r_out1)
    assert torch.equal(r_out1, expected_r1)
    
    r_out3 = resampz(r_in, 3, shift=1)
    expected_r3 = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 0]], dtype=torch.float32)
    print("Resampled (type 3, shift 1):\n", r_out3)
    assert torch.equal(r_out3, expected_r3)
    print("resampz tests passed!")

    # --- Tests for qupz ---
    print("\n--- Running tests for qupz ---")
    q_in = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    q_out = qupz(q_in, 1)
    expected_q_out = torch.tensor([[0, 2, 0],
                                    [1, 0, 4],
                                    [0, 3, 0]], dtype=torch.float32)
    print("qupz(type=1) output:\n", q_out)
    assert torch.equal(q_out, expected_q_out)
    print("qupz tests passed!")

    print("\n=== All utils.py tests passed! ===")
