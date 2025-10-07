import torch
import torch.nn.functional as F


def extend2(x, ru, rd, cl, cr, extmod='per'):
    """
    2D extension of an image. PyTorch translation of extend2.m.

    Args:
        x (torch.Tensor): Input image.
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
        # Periodic extension using torch.nn.functional.pad with circular mode
        # F.pad circular mode requires 3D+ tensors, so add batch dimension temporarily
        x_3d = x.unsqueeze(0)  # Add batch dimension
        # F.pad expects (left, right, top, bottom) for 2D
        padded = F.pad(x_3d, (cl, cr, ru, rd), mode='circular')
        return padded.squeeze(0)  # Remove batch dimension

    rx, cx = x.shape

    if extmod == 'qper_row':
        # Quincunx periodized extension in row
        # Step 1: Extend left and right with vertical circular shift
        left_part = torch.roll(x[:, -cl:], shifts=rx // 2, dims=0)
        right_part = torch.roll(x[:, :cr], shifts=-rx // 2, dims=0)
        y = torch.cat([left_part, x, right_part], dim=1)

        # Step 2: Extend top and bottom periodically
        y_3d = y.unsqueeze(0)
        y_padded = F.pad(y_3d, (0, 0, ru, rd), mode='circular')
        return y_padded.squeeze(0)

    if extmod == 'qper_col':
        # Quincunx periodized extension in column
        # Step 1: Extend top and bottom with horizontal circular shift
        top_part = torch.roll(x[-ru:, :], shifts=cx // 2, dims=1)
        bottom_part = torch.roll(x[:rd, :], shifts=-cx // 2, dims=1)
        y = torch.cat([top_part, x, bottom_part], dim=0)

        # Step 2: Extend left and right periodically
        y_3d = y.unsqueeze(0)
        y_padded = F.pad(y_3d, (cl, cr, 0, 0), mode='circular')
        return y_padded.squeeze(0)

    raise ValueError(f"Invalid extension mode: {extmod}")


def symext(x, h, shift):
    """
    Symmetric extension for image x with filter h.
    PyTorch translation of symext.m.
    
    Performs symmetric extension (H/V symmetry) for image x and filter h.
    The filter h is assumed to have odd dimensions.
    If the filter has horizontal and vertical symmetry, then 
    the nonsymmetric part of conv2(h,x) has the same size as x.
    
    Args:
        x (torch.Tensor): Input image (m×n).
        h (torch.Tensor): 2D filter coefficients.
        shift (list or tuple): Shift values [s1, s2].
    
    Returns:
        torch.Tensor: Symmetrically extended image with size (m+p-1, n+q-1),
                     where p and q are the filter dimensions.
    
    Notes:
        - Created by A. Cunha, Fall 2003
        - Modified 12/2005 by A. Cunha (fixed bug on swapped indices)
        - PyTorch translation maintains exact MATLAB behavior
    
    Example:
        >>> x = torch.arange(16).reshape(4, 4).float()
        >>> h = torch.ones((3, 3))
        >>> shift = [1, 1]
        >>> y = symext(x, h, shift)
        >>> y.shape
        torch.Size([6, 6])
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
        left_ext = torch.fliplr(x[:, :ss])
    else:
        left_ext = torch.empty((m, 0), dtype=x.dtype, device=x.device)
    
    # Right extension
    right_start = n - 1
    right_end = n - p - s1
    
    if right_end <= right_start:
        right_ext = torch.fliplr(x[:, right_end:right_start + 1])
    else:
        right_ext = torch.empty((m, 0), dtype=x.dtype, device=x.device)
    
    yT = torch.cat([left_ext, x, right_ext], dim=1)
    
    # Vertical extension (top and bottom)
    if rr > 0:
        top_ext = torch.flipud(yT[:rr, :])
    else:
        top_ext = torch.empty((0, yT.shape[1]), dtype=x.dtype, device=x.device)
    
    # Bottom extension
    bottom_start = m - 1
    bottom_end = m - q - s2
    
    if bottom_end <= bottom_start:
        bottom_ext = torch.flipud(yT[bottom_end:bottom_start + 1, :])
    else:
        bottom_ext = torch.empty((0, yT.shape[1]), dtype=x.dtype, device=x.device)
    
    yT = torch.cat([top_ext, yT, bottom_ext], dim=0)
    
    # Crop to final size
    yT = yT[:m + p - 1, :n + q - 1]
    
    return yT


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
    factor = 2**power
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
        torch.Tensor: Modulated matrix.
    """
    s = x.shape
    if center is None:
        center = [0, 0]

    o = torch.floor(torch.tensor(s, dtype=torch.float32) / 2) + 1 + torch.tensor(center, dtype=torch.float32)

    n1 = torch.arange(1, s[0] + 1, device=x.device) - o[0]
    n2 = torch.arange(1, s[1] + 1, device=x.device) - o[1]

    y = x.to(torch.float64).clone()  # Cast to float to handle multiplication by -1
    
    if mode in ['r', 'b']:
        m1 = (-1) ** n1
        y *= m1[:, None]

    if mode in ['c', 'b']:
        m2 = (-1) ** n2
        y *= m2[None, :]

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
        y = torch.zeros((sx[0] + abs(shift * (sx[1] - 1)), sx[1]), dtype=x.dtype, device=x.device)

        if type == 1:
            shift1 = torch.arange(sx[1], device=x.device) * (-shift)
        else:  # type == 2
            shift1 = torch.arange(sx[1], device=x.device) * shift

        if (shift1 < 0).any():
            # Normalize to be non-negative for indexing
            shift1 = shift1 - shift1.min()

        for n in range(sx[1]):
            y[shift1[n] : shift1[n] + sx[0], n] = x[:, n]

        # Trim zero rows
        row_norms = torch.linalg.norm(y, dim=1)
        non_zero_rows = torch.where(row_norms > 0)[0]
        if len(non_zero_rows) == 0:
            return torch.zeros((0, sx[1]), dtype=x.dtype, device=x.device)
        return y[non_zero_rows.min():non_zero_rows.max()+1, :]

    elif type in [3, 4]:  # Horizontal shearing
        y = torch.zeros((sx[0], sx[1] + abs(shift * (sx[0] - 1))), dtype=x.dtype, device=x.device)

        if type == 3:
            shift2 = torch.arange(sx[0], device=x.device) * (-shift)
        else:  # type == 4
            shift2 = torch.arange(sx[0], device=x.device) * shift

        if (shift2 < 0).any():
            # Normalize to be non-negative for indexing
            shift2 = shift2 - shift2.min()

        for m in range(sx[0]):
            y[m, shift2[m] : shift2[m] + sx[1]] = x[m, :]

        # Trim zero columns
        col_norms = torch.linalg.norm(y, dim=0)
        non_zero_cols = torch.where(col_norms > 0)[0]
        if len(non_zero_cols) == 0:
            return torch.zeros((sx[0], 0), dtype=x.dtype, device=x.device)
        return y[:, non_zero_cols.min():non_zero_cols.max()+1]

    else:
        raise ValueError("Type must be one of {1, 2, 3, 4}")


def qupz(x, type=1):
    """
    Quincunx Upsampling, based on the transform's mathematical definition.
    PyTorch translation.
    
    Note: To match MATLAB behavior, this function trims all-zero rows and columns
    from the output, similar to how MATLAB's resampz works.
    """
    r, c = x.shape

    # The output grid size is (r+c-1) x (r+c-1)
    out_size = (r + c - 1, r + c - 1)
    y = torch.zeros(out_size, dtype=x.dtype, device=x.device)

    if type == 1:  # Upsampling by Q1 = [[1, -1], [1, 1]]
        # To handle negative indices from n1 = r_idx - c_idx
        offset_r = c - 1
        for r_idx in range(r):
            for c_idx in range(c):
                n1 = r_idx - c_idx
                n2 = r_idx + c_idx
                y[n1 + offset_r, n2] = x[r_idx, c_idx]

    elif type == 2:  # Upsampling by Q2 = [[1, 1], [-1, 1]]
        # To handle negative indices from n2 = -r_idx + c_idx
        offset_c = r - 1
        for r_idx in range(r):
            for c_idx in range(c):
                n1 = r_idx + c_idx
                n2 = -r_idx + c_idx
                y[n1, n2 + offset_c] = x[r_idx, c_idx]
    else:
        raise ValueError("type must be 1 or 2")

    # Trim all-zero rows and columns (to match MATLAB's resampz behavior)
    row_norms = torch.linalg.norm(y, dim=1)
    col_norms = torch.linalg.norm(y, dim=0)
    
    non_zero_rows = torch.where(row_norms > 0)[0]
    non_zero_cols = torch.where(col_norms > 0)[0]
    
    # Handle edge case: completely zero matrix
    if len(non_zero_rows) == 0 or len(non_zero_cols) == 0:
        return torch.zeros((0, 0), dtype=x.dtype, device=x.device)
    
    # Trim to bounding box of non-zero elements
    y = y[non_zero_rows.min():non_zero_rows.max()+1, 
          non_zero_cols.min():non_zero_cols.max()+1]
    
    return y


if __name__ == '__main__':
    # --- Tests for extend2 ---
    print("--- Running tests for extend2 ---")
    img = torch.arange(16).reshape((4, 4)).float()
    ext_per = extend2(img, 1, 1, 1, 1, 'per')
    assert ext_per.shape == (6, 6)
    print("extend2 tests passed!")

    # --- Tests for upsample2df ---
    print("\n--- Running tests for upsample2df ---")
    h = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    h_up = upsample2df(h, power=1)
    expected_h_up = torch.tensor([[1, 0, 2, 0], [0, 0, 0, 0], [3, 0, 4, 0], [0, 0, 0, 0]], dtype=torch.float32)
    print("Original filter:\n", h)
    print("Upsampled filter (power=1):\n", h_up)
    assert torch.equal(h_up, expected_h_up)
    assert h_up.shape == (4, 4)
    print("upsample2df tests passed!")

    # --- Tests for modulate2 ---
    print("\n--- Running tests for modulate2 ---")
    m = torch.ones((3, 4))
    m_mod_b = modulate2(m, 'b')
    print("Original matrix:\n", m)
    print("Modulated 'both':\n", m_mod_b)
    assert m_mod_b[0, 0] == -1.0
    assert m_mod_b[1, 0] == 1.0
    assert m_mod_b[0, 1] == 1.0
    print("modulate2 tests passed!")

    # --- Tests for resampz ---
    print("\n--- Running tests for resampz ---")
    r_in = torch.arange(1, 7).reshape(2, 3).float()
    # Type 1: R1 = [1, 1; 0, 1] -> shifts columns down
    r_out1 = resampz(r_in, 1, shift=1)
    expected_r1 = torch.tensor([[0, 0, 3], [0, 2, 6], [1, 5, 0], [4, 0, 0]], dtype=torch.float32)
    print("Original for resampz:\n", r_in)
    print("Resampled (type 1, shift 1):\n", r_out1)
    assert torch.equal(r_out1, expected_r1)
    # Type 3: R3 = [1, 0; 1, 1] -> shifts rows right
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

    print("\n=== All tests passed! ===")
