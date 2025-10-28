import numpy as np

def extend2(x, ru, rd, cl, cr, extmod='per'):
    """
    2D extension of an image. Python translation of extend2.m.

    Args:
        x (np.ndarray): Input image.
        ru (int): Amount of extension up for rows.
        rd (int): Amount of extension down for rows.
        cl (int): Amount of extension left for columns.
        cr (int): Amount of extension right for columns.
        extmod (str): Extension mode. Valid modes are:
            'per': Periodized extension (default).
            'qper_row': Quincunx periodized extension in row.
            'qper_col': Quincunx periodized extension in column.

    Returns:
        np.ndarray: Extended image.
    """
    if extmod == 'per':
        # Periodic extension using numpy.pad
        return np.pad(x, ((ru, rd), (cl, cr)), 'wrap')

    rx, cx = x.shape

    if extmod == 'qper_row':
        # Quincunx periodized extension in row
        # Step 1: Extend left and right with vertical circular shift
        y = np.concatenate([
            np.roll(x[:, -cl:], rx // 2, axis=0),
            x,
            np.roll(x[:, :cr], -rx // 2, axis=0)
        ], axis=1)

        # Step 2: Extend top and bottom periodically
        y = np.pad(y, ((ru, rd), (0, 0)), 'wrap')
        return y

    if extmod == 'qper_col':
        # Quincunx periodized extension in column
        # Step 1: Extend top and bottom with horizontal circular shift
        y = np.concatenate([
            np.roll(x[-ru:, :], cx // 2, axis=1),
            x,
            np.roll(x[:rd, :], -cx // 2, axis=1)
        ], axis=0)

        # Step 2: Extend left and right periodically
        y = np.pad(y, ((0, 0), (cl, cr)), 'wrap')
        return y

    raise ValueError(f"Invalid extension mode: {extmod}")


def symext(x, h, shift):
    """
    Symmetric extension for image x with filter h.
    Translation of symext.m.
    
    Performs symmetric extension (H/V symmetry) for image x and filter h.
    The filter h is assumed to have odd dimensions.
    If the filter has horizontal and vertical symmetry, then 
    the nonsymmetric part of conv2(h,x) has the same size as x.
    
    Args:
        x (np.ndarray): Input image (m×n).
        h (np.ndarray): 2D filter coefficients.
        shift (list or tuple): Shift values [s1, s2].
    
    Returns:
        np.ndarray: Symmetrically extended image with size (m+p-1, n+q-1),
                    where p and q are the filter dimensions.
    
    Notes:
        - Created by A. Cunha, Fall 2003
        - Modified 12/2005 by A. Cunha (fixed bug on swapped indices)
        - Python translation maintains exact MATLAB behavior
    
    Example:
        >>> x = np.arange(16).reshape(4, 4)
        >>> h = np.ones((3, 3))
        >>> shift = [1, 1]
        >>> y = symext(x, h, shift)
        >>> y.shape
        (6, 6)
    """
    m, n = x.shape
    p, q = h.shape
    
    # MATLAB: parp = 1 - mod(p, 2); parq = 1 - mod(q, 2)
    # These variables are computed but not used in MATLAB code
    # parp = 1 - (p % 2)
    # parq = 1 - (q % 2)
    
    p2 = int(np.floor(p / 2))
    q2 = int(np.floor(q / 2))
    s1 = shift[0]
    s2 = shift[1]
    
    # Calculate extension amounts
    ss = p2 - s1 + 1
    rr = q2 - s2 + 1
    
    # MATLAB: yT = [fliplr(x(:,1:ss)) x  x(:,n  :-1: n-p-s1+1)];
    # Horizontal extension (left and right)
    if ss > 0:
        left_ext = np.fliplr(x[:, :ss])
    else:
        left_ext = np.empty((m, 0))
    
    # Right extension: x(:, n:-1:n-p-s1+1)
    # MATLAB indices: n (last column, index n-1 in Python) down to n-p-s1+1
    # In Python: n-1 down to n-p-s1 (inclusive)
    right_start = n - 1
    right_end = n - p - s1  # This will be used as: right_end : right_start + 1
    
    if right_end <= right_start:
        right_ext = np.fliplr(x[:, right_end:right_start + 1])
    else:
        right_ext = np.empty((m, 0))
    
    yT = np.concatenate([left_ext, x, right_ext], axis=1)
    
    # MATLAB: yT = [flipud(yT(1:rr,:)); yT ;  yT(m  :-1: m-q-s2+1,:)];
    # Vertical extension (top and bottom)
    if rr > 0:
        top_ext = np.flipud(yT[:rr, :])
    else:
        top_ext = np.empty((0, yT.shape[1]))
    
    # Bottom extension: yT(m:-1:m-q-s2+1, :)
    # MATLAB indices: m (last row, index m-1 in Python) down to m-q-s2+1
    # In Python: m-1 down to m-q-s2 (inclusive)
    bottom_start = m - 1
    bottom_end = m - q - s2
    
    if bottom_end <= bottom_start:
        bottom_ext = np.flipud(yT[bottom_end:bottom_start + 1, :])
    else:
        bottom_ext = np.empty((0, yT.shape[1]))
    
    yT = np.concatenate([top_ext, yT, bottom_ext], axis=0)
    
    # MATLAB: yT = yT(1:m+p-1, 1:n+q-1);
    # Crop to final size (this is already the expected size, so just verify)
    yT = yT[:m + p - 1, :n + q - 1]
    
    return yT


def upsample2df(h, power=1):
    """
    Upsample a 2D filter by 2^power by inserting zeros.
    Translation of upsample2df.m.

    Args:
        h (np.ndarray): Input filter.
        power (int): Power of 2 for upsampling factor.

    Returns:
        np.ndarray: Upsampled filter.
    """
    factor = 2**power
    m, n = h.shape
    ho = np.zeros((factor * m, factor * n), dtype=h.dtype)
    ho[::factor, ::factor] = h
    return ho

def modulate2(x, mode='b', center=None):
    """
    2D modulation. Translation of modulate2.m.

    Args:
        x (np.ndarray): Input matrix.
        mode (str): 'r' for row, 'c' for column, 'b' for both.
        center (list or tuple, optional): Modulation center offset. Defaults to None.

    Returns:
        np.ndarray: Modulated matrix.
    """
    s = x.shape
    if center is None:
        center = [0, 0]

    o = np.floor(np.array(s) / 2) + 1 + np.array(center)

    n1 = np.arange(1, s[0] + 1) - o[0]
    n2 = np.arange(1, s[1] + 1) - o[1]

    y = x.astype(np.float64, copy=True) # Cast to float to handle multiplication by -1
    if mode in ['r', 'b']:
        m1 = (-1) ** n1
        y *= m1[:, np.newaxis]

    if mode in ['c', 'b']:
        m2 = (-1) ** n2
        y *= m2[np.newaxis, :]

    return y


def resampz(x, type, shift=1):
    """
    Resampling of a matrix (shearing). Translation of resampz.m.

    Args:
        x (np.ndarray): Input matrix.
        type (int): One of {1, 2, 3, 4}.
        shift (int): Amount of shift.

    Returns:
        np.ndarray: Resampled matrix.
    """
    sx = x.shape

    if type in [1, 2]: # Vertical shearing
        y = np.zeros((sx[0] + abs(shift * (sx[1] - 1)), sx[1]), dtype=x.dtype)

        if type == 1:
            shift1 = np.arange(sx[1]) * (-shift)
        else: # type == 2
            shift1 = np.arange(sx[1]) * shift

        if np.any(shift1 < 0):
            # Normalize to be non-negative for indexing
            shift1 = shift1 - shift1.min()

        for n in range(sx[1]):
            y[shift1[n] : shift1[n] + sx[0], n] = x[:, n]

        # Trim zero rows
        row_norms = np.linalg.norm(y, axis=1)
        non_zero_rows = np.where(row_norms > 0)[0]
        if len(non_zero_rows) == 0:
            return np.zeros((0, sx[1]), dtype=x.dtype)  # Correct shape for empty result
        return y[non_zero_rows.min():non_zero_rows.max()+1, :]

    elif type in [3, 4]: # Horizontal shearing
        y = np.zeros((sx[0], sx[1] + abs(shift * (sx[0] - 1))), dtype=x.dtype)

        if type == 3:
            shift2 = np.arange(sx[0]) * (-shift)
        else: # type == 4
            shift2 = np.arange(sx[0]) * shift

        if np.any(shift2 < 0):
            # Normalize to be non-negative for indexing
            shift2 = shift2 - shift2.min()

        for m in range(sx[0]):
            y[m, shift2[m] : shift2[m] + sx[1]] = x[m, :]

        # Trim zero columns
        col_norms = np.linalg.norm(y, axis=0)
        non_zero_cols = np.where(col_norms > 0)[0]
        if len(non_zero_cols) == 0:
            return np.zeros((sx[0], 0), dtype=x.dtype)  # Correct shape for empty result
        return y[:, non_zero_cols.min():non_zero_cols.max()+1]

    else:
        raise ValueError("Type must be one of {1, 2, 3, 4}")

def qupz(x, type=1):
    """
    Quincunx Upsampling, based on the transform's mathematical definition.
    This replaces the complex resampz-based implementation.
    
    Note: To match MATLAB behavior, this function trims all-zero rows and columns
    from the output, similar to how MATLAB's resampz works.
    """
    r, c = x.shape

    # The output grid size is (r+c-1) x (r+c-1)
    out_size = (r + c - 1, r + c - 1)
    y = np.zeros(out_size, dtype=x.dtype)

    if type == 1: # Upsampling by Q1 = [[1, -1], [1, 1]]
        # To handle negative indices from n1 = r_idx - c_idx
        offset_r = c - 1
        for r_idx in range(r):
            for c_idx in range(c):
                n1 = r_idx - c_idx
                n2 = r_idx + c_idx
                y[n1 + offset_r, n2] = x[r_idx, c_idx]

    elif type == 2: # Upsampling by Q2 = [[1, 1], [-1, 1]]
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
    # This is necessary because MATLAB's qupz implementation uses resampz
    # which automatically removes zero rows/columns
    row_norms = np.linalg.norm(y, axis=1)
    col_norms = np.linalg.norm(y, axis=0)
    
    non_zero_rows = np.where(row_norms > 0)[0]
    non_zero_cols = np.where(col_norms > 0)[0]
    
    # Handle edge case: completely zero matrix
    if len(non_zero_rows) == 0 or len(non_zero_cols) == 0:
        return np.zeros((0, 0), dtype=x.dtype)
    
    # Trim to bounding box of non-zero elements
    y = y[non_zero_rows.min():non_zero_rows.max()+1, 
          non_zero_cols.min():non_zero_cols.max()+1]
    
    return y

if __name__ == '__main__':
    # --- Tests for extend2 ---
    print("--- Running tests for extend2 ---")
    img = np.arange(16).reshape((4, 4))
    ext_per = extend2(img, 1, 1, 1, 1, 'per')
    assert ext_per.shape == (6, 6)
    print("extend2 tests passed!")

    # --- Tests for upsample2df ---
    print("\n--- Running tests for upsample2df ---")
    h = np.array([[1, 2], [3, 4]])
    h_up = upsample2df(h, power=1)
    expected_h_up = np.array([[1, 0, 2, 0], [0, 0, 0, 0], [3, 0, 4, 0], [0, 0, 0, 0]])
    print("Original filter:\n", h)
    print("Upsampled filter (power=1):\n", h_up)
    assert np.array_equal(h_up, expected_h_up)
    assert h_up.shape == (4, 4)
    print("upsample2df tests passed!")

    # --- Tests for modulate2 ---
    print("\n--- Running tests for modulate2 ---")
    m = np.ones((3, 4))
    m_mod_b = modulate2(m, 'b')
    print("Original matrix:\n", m)
    print("Modulated 'both':\n", m_mod_b)
    # Correct assertions based on MATLAB logic trace
    # o = [2, 3], n1 = [-1, 0, 1], n2 = [-2, -1, 0, 1]
    # m(0,0) = (-1)^-1 * (-1)^-2 = -1 * 1 = -1
    assert m_mod_b[0, 0] == -1.0
    # m(1,0) = (-1)^0 * (-1)^-2 = 1 * 1 = 1
    assert m_mod_b[1, 0] == 1.0
    # m(0,1) = (-1)^-1 * (-1)^-1 = -1 * -1 = 1
    assert m_mod_b[0, 1] == 1.0
    print("modulate2 tests passed!")

    # --- Tests for resampz ---
    print("\n--- Running tests for resampz ---")
    r_in = np.arange(1, 7).reshape(2, 3)
    # Type 1: R1 = [1, 1; 0, 1] -> shifts columns down
    r_out1 = resampz(r_in, 1, shift=1)
    # Expected output based on MATLAB logic trace
    expected_r1 = np.array([[0, 0, 3], [0, 2, 6], [1, 5, 0], [4, 0, 0]])
    print("Original for resampz:\n", r_in)
    print("Resampled (type 1, shift 1):\n", r_out1)
    assert np.array_equal(r_out1, expected_r1)
    # Type 3: R3 = [1, 0; 1, 1] -> shifts rows right
    r_out3 = resampz(r_in, 3, shift=1)
    # Expected output based on MATLAB logic trace
    expected_r3 = np.array([[0, 1, 2, 3], [4, 5, 6, 0]])
    print("Resampled (type 3, shift 1):\n", r_out3)
    assert np.array_equal(r_out3, expected_r3)
    print("resampz tests passed!")

    # --- Tests for qupz ---
    print("\n--- Running tests for qupz ---")
    q_in = np.array([[1, 2], [3, 4]])
    q_out = qupz(q_in, 1)
    # Based on careful re-trace of MATLAB code
    expected_q_out = np.array([[0, 2, 0],
                               [1, 0, 4],
                               [0, 3, 0]])
    print("qupz(type=1) output:\n", q_out)
    assert np.array_equal(q_out, expected_q_out)
    print("qupz tests passed!")

    # --- Tests for qupz ---
    print("\n--- Running tests for qupz ---")
    q_in = np.array([[1, 2], [3, 4]])
    q_out = qupz(q_in, 1)
    # Expected output from the mathematical definition of quincunx upsampling
    expected_q_out = np.array([[0, 2, 0],
                               [1, 0, 4],
                               [0, 3, 0]])
    print("qupz(type=1) output:\n", q_out)
    assert np.array_equal(q_out, expected_q_out)
    print("qupz tests passed!")