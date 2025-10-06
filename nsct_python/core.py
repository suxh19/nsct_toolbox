import numpy as np
from scipy.signal import convolve2d
from typing import Union, Tuple, Optional
from nsct_python.filters import efilter2
from nsct_python.utils import extend2, symext, upsample2df

def _upsample_and_find_origin(f: np.ndarray, mup: Union[int, float, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Upsamples a filter and returns the upsampled filter and its new origin.
    """
    if (np.isscalar(mup) and mup == 1) or \
       (isinstance(mup, np.ndarray) and np.array_equal(mup, np.eye(2))):
        return f, (np.array(f.shape) - 1) // 2

    if isinstance(mup, (int, float)):
        mup = np.array([[mup, 0], [0, mup]], dtype=int)
    mup = np.array(mup, dtype=int)

    taps_r, taps_c = np.nonzero(f)
    tap_coords = np.array([taps_r, taps_c])
    upsampled_coords = mup @ tap_coords

    orig_origin = (np.array(f.shape) - 1) // 2
    new_origin_coord = mup @ orig_origin

    min_coords = upsampled_coords.min(axis=1)
    max_coords = upsampled_coords.max(axis=1)

    new_size = max_coords - min_coords + 1
    f_up = np.zeros(new_size, dtype=f.dtype)

    shifted_coords = upsampled_coords - min_coords[:, np.newaxis]
    f_up[shifted_coords[0, :], shifted_coords[1, :]] = f[taps_r, taps_c]

    f_up_origin = new_origin_coord - min_coords

    return f_up, f_up_origin

def _convolve_upsampled(x: np.ndarray, f: np.ndarray, mup: Union[int, float, np.ndarray], is_rec: bool = False) -> np.ndarray:
    """ Helper for convolution with an upsampled filter, handling reconstruction. """
    # If the filter is all zeros, the output is all zeros.
    if not np.any(f):
        return np.zeros_like(x)

    f_up, f_up_origin = _upsample_and_find_origin(f, mup)

    # For reconstruction, we use the time-reversed filter and its origin
    if is_rec:
        f_up = np.rot90(f_up, 2)
        f_up_origin = np.array(f_up.shape) - 1 - f_up_origin

    pad_top = f_up_origin[0]
    pad_bottom = f_up.shape[0] - 1 - f_up_origin[0]
    pad_left = f_up_origin[1]  # type: ignore
    pad_right = f_up.shape[1] - 1 - f_up_origin[1]

    x_ext = extend2(x, pad_top, pad_bottom, pad_left, pad_right)

    # Perform correlation by rotating the kernel 180 degrees for convolve2d
    return convolve2d(x_ext, np.rot90(f_up, 2), 'valid')

def nssfbdec(x, f1, f2, mup=None):
    """
    Two-channel nonsubsampled filter bank decomposition.
    """
    if mup is None:
        y1 = efilter2(x, f1)
        y2 = efilter2(x, f2)
    else:
        y1 = _convolve_upsampled(x, f1, mup, is_rec=False)
        y2 = _convolve_upsampled(x, f2, mup, is_rec=False)
    return y1, y2

def nssfbrec(x1, x2, f1, f2, mup=None):
    """
    Two-channel nonsubsampled filter bank reconstruction.
    """
    if x1.shape != x2.shape:
        raise ValueError("Input sizes for the two branches must be the same")

    if mup is None:
        y1 = efilter2(x1, f1)
        y2 = efilter2(x2, f2)
    else:
        y1 = _convolve_upsampled(x1, f1, mup, is_rec=True)
        y2 = _convolve_upsampled(x2, f2, mup, is_rec=True)

    return y1 + y2


def nsfbdec(x: np.ndarray, h0: np.ndarray, h1: np.ndarray, lev: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Nonsubsampled filter bank decomposition at a given level.
    Translation of nsfbdec.m.
    
    Computes the nonsubsampled pyramid decomposition at level lev with filters h0, h1.
    
    Args:
        x (np.ndarray): Input image at finer scale.
        h0 (np.ndarray): Lowpass à trous filter (obtained from atrousfilters).
        h1 (np.ndarray): Highpass à trous filter (obtained from atrousfilters).
        lev (int): Decomposition level (0 for first level, >0 for subsequent levels).
    
    Returns:
        tuple: (y0, y1) where:
            - y0: Image at coarser scale (lowpass output)
            - y1: Wavelet highpass output (bandpass output)
    
    Notes:
        - For lev=0: Uses regular conv2 with symext
        - For lev>0: Uses upsampled filters with atrous convolution
        - This function replaces the MEX function atrousc with pure Python implementation
    
    History:
        - Adapted from atrousdec by Arthur Cunha
        - Modified on Aug 2004 by A. C.
        - Modified on Oct 2004 by A. C.
        - Python translation: Oct 2025
    
    See also:
        nsfbrec, atrousfilters
    
    Example:
        >>> from nsct_python.filters import atrousfilters
        >>> h0, h1, g0, g1 = atrousfilters('maxflat')
        >>> x = np.random.rand(32, 32)
        >>> y0, y1 = nsfbdec(x, h0, h1, 0)
        >>> y0.shape == x.shape
        True
    """
    if lev != 0:
        # For levels > 0, use upsampled filters
        # MATLAB: I2 = eye(2); % delay compensation
        # MATLAB: shift = -2^(lev-1)*[1,1] + 2; L=2^lev;
        I2 = np.eye(2, dtype=int)
        shift = [-2**(lev-1), -2**(lev-1)]
        shift = [s + 2 for s in shift]  # shift = -2^(lev-1)*[1,1] + 2
        L = 2**lev
        
        # Upsample filters
        h0_up = upsample2df(h0, lev)
        h1_up = upsample2df(h1, lev)
        
        # Symmetric extension
        x_ext_h0 = symext(x, h0_up, shift)
        x_ext_h1 = symext(x, h1_up, shift)
        
        # Atrous convolution (using _convolve_upsampled with diagonal upsampling matrix)
        # MATLAB: y0 = atrousc(symext(x,upsample2df(h0,lev),shift),h0,I2 * L);
        # The upsampling matrix is I2 * L = [[L, 0], [0, L]]
        mup = I2 * L
        
        # We need to convolve the extended image with the original filter (not upsampled)
        # because the extension was done with the upsampled filter
        # But we pass the upsampling matrix to simulate atrous convolution
        y0 = _atrousc_equivalent(x_ext_h0, h0, mup)
        y1 = _atrousc_equivalent(x_ext_h1, h1, mup)
    else:
        # First level (lev == 0)
        # MATLAB: shift = [1, 1]; % delay compensation
        # MATLAB: y0 = conv2(symext(x,h0,shift),h0,'valid');
        # MATLAB: y1 = conv2(symext(x,h1,shift),h1,'valid');
        shift = [1, 1]
        
        # Symmetric extension
        x_ext_h0 = symext(x, h0, shift)
        x_ext_h1 = symext(x, h1, shift)
        
        # Regular convolution with 'valid' mode
        y0 = convolve2d(x_ext_h0, h0, mode='valid')
        y1 = convolve2d(x_ext_h1, h1, mode='valid')
    
    return y0, y1


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


def nsfbrec(y0: np.ndarray, y1: np.ndarray, g0: np.ndarray, g1: np.ndarray, lev: int) -> np.ndarray:
    """
    Nonsubsampled filter bank reconstruction at a given level.
    Translation of nsfbrec.m.
    
    Computes the inverse of 2-D à trous decomposition at level lev.
    
    Args:
        y0 (np.ndarray): Lowpass image (coarse scale).
        y1 (np.ndarray): Highpass image (wavelet details).
        g0 (np.ndarray): Lowpass synthesis filter (obtained from atrousfilters).
        g1 (np.ndarray): Highpass synthesis filter (obtained from atrousfilters).
        lev (int): Reconstruction level (0 for first level, >0 for subsequent levels).
    
    Returns:
        np.ndarray: Reconstructed image at finer scale.
    
    Notes:
        - For lev=0: Uses regular conv2 with symext
        - For lev>0: Uses upsampled filters with atrous convolution
        - This is the inverse operation of nsfbdec
        - Perfect reconstruction property: x ≈ nsfbrec(nsfbdec(x, h0, h1, lev), g0, g1, lev)
    
    History:
        - Created on May 2004 by Arthur Cunha
        - Modified on Aug 2004 by A. C.
        - Modified on Oct 2004 by A. C.
        - Python translation: Oct 2025
    
    See also:
        nsfbdec, atrousfilters
    
    Example:
        >>> from nsct_python.core import nsfbdec, nsfbrec
        >>> from nsct_python.filters import atrousfilters
        >>> import numpy as np
        >>> h0, h1, g0, g1 = atrousfilters('maxflat')
        >>> x = np.random.rand(64, 64)
        >>> y0, y1 = nsfbdec(x, h0, h1, 0)
        >>> x_rec = nsfbrec(y0, y1, g0, g1, 0)
        >>> print(f"Reconstruction error: {np.mean((x - x_rec)**2):.2e}")
    """
    # Identity matrix for upsampling
    I2 = np.eye(2, dtype=int)
    
    if lev != 0:
        # Higher levels: use upsampled filters with atrous convolution
        shift = -2**(lev-1) * np.array([1, 1]) + 2  # Delay correction
        L = 2**lev
        
        # Upsample filters
        g0_up = upsample2df(g0, lev)
        g1_up = upsample2df(g1, lev)
        
        # Extend inputs
        y0_ext = symext(y0, g0_up, shift)
        y1_ext = symext(y1, g1_up, shift)
        
        # Apply atrous convolution and sum
        x = _atrousc_equivalent(y0_ext, g0, L * I2) + \
            _atrousc_equivalent(y1_ext, g1, L * I2)
    else:
        # Level 0: use regular convolution
        shift = np.array([1, 1])
        
        # Extend inputs
        y0_ext = symext(y0, g0, shift)
        y1_ext = symext(y1, g1, shift)
        
        # Convolve and sum (using 'valid' to match MATLAB)
        x = convolve2d(y0_ext, g0, mode='valid') + \
            convolve2d(y1_ext, g1, mode='valid')
    
    return x


if __name__ == '__main__':
    from nsct_python.filters import dfilters

    print("--- Running tests for nssfbdec and nssfbrec (Perfect Reconstruction) ---")

    # Create a sample image (must be even-sized for some filters)
    img = np.random.rand(32, 32)

    # Get a pair of analysis and synthesis filters
    # Using 'pkva' as it's a well-defined filter pair in the project
    h0, h1 = dfilters('pkva', 'd') # Analysis filters
    g0, g1 = dfilters('pkva', 'r') # Synthesis filters

    # Define a quincunx upsampling matrix
    mup = np.array([[1, 1], [-1, 1]])

    # --- Test Decomposition and Reconstruction ---
    print("Testing with upsampling matrix M =", mup.flatten())

    # Decompose
    y1, y2 = nssfbdec(img, h0, h1, mup)

    # Reconstruct
    recon_img = nssfbrec(y1, y2, g0, g1, mup)

    # Check for perfect reconstruction
    print("Original vs. Reconstructed MSE:", np.mean((img - recon_img)**2))
    assert np.allclose(img, recon_img, atol=1e-9)

    print("nssfbdec/rec perfect reconstruction test passed!")