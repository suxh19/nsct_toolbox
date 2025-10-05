import numpy as np
from scipy.signal import convolve2d
from nsct_python.filters import efilter2
from nsct_python.utils import extend2

def _upsample_and_find_origin(f, mup):
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

def _convolve_upsampled(x, f, mup, is_rec=False):
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
    pad_left = f_up_origin[1]
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