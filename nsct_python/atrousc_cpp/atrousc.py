import numpy as np

def _atrousc_equivalent(x: np.ndarray, h: np.ndarray, M: np.ndarray) -> np.ndarray:
    """
    Equivalent of MATLAB's atrousc MEX function using pure Python.
    
    Computes convolution with an upsampled filter (à trous convolution).
    The filter is conceptually upsampled by matrix M, but we compute the
    convolution directly without actually upsampling.
    
    Args:
        x (np.ndarray): Extended input signal (2D).
        h (np.ndarray): Original filter (not upsampled).
        M (np.ndarray): Upsampling matrix (2x2 or scalar that becomes diagonal matrix).
    
    Returns:
        np.ndarray: Result of convolution, 'valid' mode.
    
    Notes:
        This function is optimized for separable (diagonal) upsampling matrices,
        which is the typical case: M = [[L, 0], [0, L]]
    """
    # Convert scalar to diagonal matrix
    if np.isscalar(M):
        M = np.array([[M, 0], [0, M]], dtype=int)
    elif M.shape == (2, 2):
        M = M.astype(int)
    else:
        raise ValueError("M must be a scalar or 2x2 matrix")
    
    # Get dimensions
    S_rows, S_cols = x.shape
    F_rows, F_cols = h.shape
    
    # For diagonal matrix M = [[M0, 0], [0, M3]]
    M0 = int(M[0, 0])
    M3 = int(M[1, 1])
    
    # Output size (matching MATLAB's atrousc 'valid' mode)
    O_rows = S_rows - M0 * F_rows + 1
    O_cols = S_cols - M3 * F_cols + 1
    
    if O_rows <= 0 or O_cols <= 0:
        return np.zeros((max(0, O_rows), max(0, O_cols)))
    
    # Initialize output
    out = np.zeros((O_rows, O_cols), dtype=x.dtype)
    
    # Flip the filter (for convolution)
    h_flipped = np.flipud(np.fliplr(h))
    
    # Convolution loop (optimized for separable upsampling)
    for n1 in range(O_cols):  # Note: MATLAB's column-major order
        for n2 in range(O_rows):
            total = 0.0
            kk1 = n1 + M0 - 1
            for k1 in range(F_cols):
                kk2 = n2 + M3 - 1
                for k2 in range(F_rows):
                    f1 = F_cols - 1 - k1  # Flipped index
                    f2 = F_rows - 1 - k2
                    total += h_flipped[f2, f1] * x[kk2, kk1]
                    kk2 += M3
                kk1 += M0
            out[n2, n1] = total
    
    return out