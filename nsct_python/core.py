import numpy as np
from scipy.signal import convolve2d
from typing import Union, Tuple, Optional, List, Any
from nsct_python.filters import efilter2, dfilters, modulate2, parafilters, atrousfilters
from nsct_python.utils import extend2, symext, upsample2df

# Import C++ implementation of zconv2
from nsct_python.zconv2_cpp import zconv2 as _zconv2_cpp

# Import C++ implementation of atrousc
from nsct_python.atrousc_cpp import atrousc as _atrousc_cpp


def _zconv2(x: np.ndarray, h: np.ndarray, mup: np.ndarray) -> np.ndarray:
    """
    2D convolution with upsampled filter using periodic boundary.
    
    Uses high-performance C++ implementation.
    
    Args:
        x: Input signal (2D array)
        h: Filter (2D array)
        mup: Upsampling matrix (2x2 array) [[M0, M1], [M2, M3]]
    
    Returns:
        y: Convolution output (same size as x)
    """
    mup_mat = np.array(mup, dtype=np.float64)
    return _zconv2_cpp(x, h, mup_mat)


def _convolve_upsampled(x: np.ndarray, f: np.ndarray, mup: Union[int, float, np.ndarray], is_rec: bool = False) -> np.ndarray:
    """ 
    Helper for convolution with an upsampled filter, handling reconstruction.
    Uses zconv2-style periodic convolution when mup is a 2x2 matrix.
    """
    # If the filter is all zeros, the output is all zeros.
    if not np.any(f):
        return np.zeros_like(x)
    
    # Convert mup to matrix form
    if isinstance(mup, (int, float)):
        mup_mat = np.array([[mup, 0], [0, mup]], dtype=int)
    else:
        mup_mat = np.array(mup, dtype=int)
    
    # For reconstruction, use time-reversed filter
    f_to_use = np.rot90(f, 2) if is_rec else f
    
    # Use zconv2 for periodic convolution with upsampled filter
    return _zconv2(x, f_to_use, mup_mat)

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
        y0 = _atrousc_cpp(x_ext_h0, h0, mup)
        y1 = _atrousc_cpp(x_ext_h1, h1, mup)
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
        x = _atrousc_cpp(y0_ext, g0, L * I2) + \
            _atrousc_cpp(y1_ext, g1, L * I2)
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


def nsdfbdec(x: np.ndarray, dfilter, clevels: int = 0):
    """
    Nonsubsampled Directional Filter Bank (NSDFB) decomposition.
    Translation of nsdfbdec.m.
    
    Decomposes the image X by a nonsubsampled directional filter bank
    with a binary-tree structure. It outputs the final branches, totally 2^clevels.
    There is no subsampling and hence the operation is shift-invariant.
    
    Args:
        x (np.ndarray): Input image (2D array).
        dfilter: Either:
            - str: Directional filter name (e.g., 'pkva', 'dmaxflat7')
            - dict: Dictionary with keys 'k1', 'k2', 'f1', 'f2' containing precomputed filters
        clevels (int): Number of decomposition levels (non-negative integer).
                      clevels=0: No decomposition, return input.
                      clevels=1: 2 subbands.
                      clevels=n: 2^n subbands.
    
    Returns:
        list: List of output subbands (length 2^clevels).
    
    Notes:
        - Uses nssfbdec for two-channel nonsubsampled decomposition
        - First level uses fan filters (k1, k2)
        - Higher levels use parallelogram filters (f1, f2)
        - Upsampling matrices are computed according to Minh N. Do's thesis (eq. 3.18)
    
    History:
        - Created on 08/06/2004 by Jianping Zhou
        - Python translation: Oct 2025
    
    See also:
        dfilters, parafilters, nssfbdec, nsdfbrec
    
    Example:
        >>> from nsct_python.core import nsdfbdec
        >>> import numpy as np
        >>> x = np.random.rand(64, 64)
        >>> y = nsdfbdec(x, 'pkva', 2)  # 2 levels -> 4 subbands
        >>> len(y)
        4
    """
    from nsct_python.filters import dfilters, parafilters
    from nsct_python.utils import modulate2
    
    # Input validation
    if clevels != round(clevels) or clevels < 0:
        raise ValueError('Number of decomposition levels must be a non-negative integer')
    
    # No decomposition case
    if clevels == 0:
        return [x]
    
    # Get filters
    if isinstance(dfilter, str):
        # Get the directional filters for the critically sampled DFB
        h1, h2 = dfilters(dfilter, 'd')
        
        # A scale is required for the nonsubsampled case
        h1 = h1 / np.sqrt(2)
        h2 = h2 / np.sqrt(2)
        
        # Generate the first-level fan filters by modulations
        k1 = modulate2(h1, 'c')
        k2 = modulate2(h2, 'c')
        
        # Obtain the parallelogram filters from the diamond filters
        f1, f2 = parafilters(h1, h2)
        
    elif isinstance(dfilter, dict):
        # Copy filters directly from dict
        if not all(key in dfilter for key in ['k1', 'k2', 'f1', 'f2']):
            raise ValueError("Filter dict must contain keys: 'k1', 'k2', 'f1', 'f2'")
        k1 = dfilter['k1']
        k2 = dfilter['k2']
        f1 = dfilter['f1']
        f2 = dfilter['f2']
    else:
        raise TypeError('dfilter must be a string or dict')
    
    # Quincunx sampling matrix
    q1 = np.array([[1, -1], [1, 1]])
    
    # First-level decomposition
    if clevels == 1:
        # No upsampling for filters at the first level
        y1, y2 = nssfbdec(x, k1, k2)
        return [y1, y2]
    
    # Multi-level decomposition (clevels >= 2)
    
    # Second-level decomposition
    # No upsampling at filters for the first level
    x1, x2 = nssfbdec(x, k1, k2)
    
    # Convolution with upsampled filters
    y: List[Union[np.ndarray, None]] = [None] * 4
    y[0], y[1] = nssfbdec(x1, k1, k2, q1)
    y[2], y[3] = nssfbdec(x2, k1, k2, q1)
    
    # Third and higher levels decomposition
    for l in range(3, clevels + 1):
        # Allocate space for the new subband outputs
        y_old = y
        y: List[Union[np.ndarray, None]] = [None] * (2 ** l)
        
        # The first half channels
        for k in range(1, 2 ** (l - 2) + 1):
            # Compute the upsampling matrix by the formula (3.18) of Minh N. Do's thesis
            # The upsampling matrix for the channel k in a l-levels DFB is M_k^{(l-1)}
            
            # Compute s_{(l-1)}(k):
            slk = 2 * ((k - 1) // 2) - 2 ** (l - 3) + 1
            
            # Compute the sampling matrix:
            mkl = 2 * np.array([[2 ** (l - 3), 0], [0, 1]]) @ np.array([[1, 0], [-slk, 1]])
            
            i = ((k - 1) % 2)  # Index 0 or 1
            
            # Decompose by the two-channel filter bank
            y[2 * k - 2], y[2 * k - 1] = nssfbdec(y_old[k - 1], f1[i], f2[i], mkl)
        
        # The second half channels
        for k in range(2 ** (l - 2) + 1, 2 ** (l - 1) + 1):
            # Compute the upsampling matrix by the extension of the formula (3.18)
            # of Minh N. Do's thesis to the second half channels
            
            # Compute s_{(l-1)}(k):
            slk = 2 * ((k - 2 ** (l - 2) - 1) // 2) - 2 ** (l - 3) + 1
            
            # Compute the sampling matrix:
            mkl = 2 * np.array([[1, 0], [0, 2 ** (l - 3)]]) @ np.array([[1, -slk], [0, 1]])
            
            i = ((k - 1) % 2) + 2  # Index 2 or 3
            
            # Decompose by the two-channel filter bank
            y[2 * k - 2], y[2 * k - 1] = nssfbdec(y_old[k - 1], f1[i], f2[i], mkl)
    
    return y


def nsdfbrec(y: List[np.ndarray], dfilter: Union[str, dict]) -> np.ndarray:
    """
    Nonsubsampled directional filter bank reconstruction.
    
    Reconstructs the image from directional subbands obtained from nsdfbdec.
    Uses a binary-tree structure with no subsampling (shift-invariant).
    Translation of nsdfbrec.m.
    
    Args:
        y (list): List of directional subbands (2^clevels subbands).
        dfilter (str or dict): Directional filter specification.
            - str: Filter name (e.g., 'pkva', 'dmaxflat7')
            - dict: Dict with keys 'k1', 'k2', 'f1', 'f2'
    
    Returns:
        np.ndarray: Reconstructed image.
    
    Notes:
        - This is the inverse operation of nsdfbdec
        - Number of subbands must be a power of 2
        - Uses synthesis filters (type 'r')
        - Perfect reconstruction is achieved with matching analysis/synthesis filters
    
    Examples:
        >>> x = np.random.rand(64, 64)
        >>> y = nsdfbdec(x, 'pkva', 2)  # Decompose into 4 subbands
        >>> x_rec = nsdfbrec(y, 'pkva')  # Reconstruct
        >>> np.allclose(x, x_rec, atol=1e-10)
        True
    """
    from nsct_python.filters import dfilters, parafilters
    from nsct_python.utils import modulate2
    
    # Determine clevels from number of subbands
    clevels = int(np.log2(len(y)))
    if 2**clevels != len(y):
        raise ValueError('Number of subbands must be a power of 2')
    
    # No reconstruction case
    if clevels == 0:
        return y[0]
    
    # Get filters (use synthesis filters 'r')
    if isinstance(dfilter, str):
        # Get the directional filters for the critically sampled DFB
        h1, h2 = dfilters(dfilter, 'r')  # Note: 'r' for reconstruction
        
        # A scale is required for the nonsubsampled case
        h1 = h1 / np.sqrt(2)
        h2 = h2 / np.sqrt(2)
        
        # Generate the first-level fan filters by modulations
        k1 = modulate2(h1, 'c')
        k2 = modulate2(h2, 'c')
        
        # Obtain the parallelogram filters from the diamond filters
        f1, f2 = parafilters(h1, h2)
        
    elif isinstance(dfilter, dict):
        # Copy filters directly from dict
        if not all(key in dfilter for key in ['k1', 'k2', 'f1', 'f2']):
            raise ValueError("Filter dict must contain keys: 'k1', 'k2', 'f1', 'f2'")
        k1 = dfilter['k1']
        k2 = dfilter['k2']
        f1 = dfilter['f1']
        f2 = dfilter['f2']
    else:
        raise TypeError('dfilter must be a string or dict')
    
    # Quincunx sampling matrix
    q1 = np.array([[1, -1], [1, 1]])
    
    # First-level reconstruction
    if clevels == 1:
        # No upsampling for filters at the first level
        return nssfbrec(y[0], y[1], k1, k2)
    
    # Multi-level reconstruction (clevels >= 2)
    # To save memory, we use a copy of the input list to store middle outputs
    x = y.copy()
    
    # Third and higher levels reconstructions (from highest to lowest)
    for l in range(clevels, 2, -1):
        # The first half channels
        for k in range(1, 2 ** (l - 2) + 1):
            # Compute the upsampling matrix by the formula (3.18) of Minh N. Do's thesis
            # The upsampling matrix for the channel k in a l-levels DFB is M_k^{(l-1)}
            
            # Compute s_{(l-1)}(k):
            slk = 2 * ((k - 1) // 2) - 2 ** (l - 3) + 1
            
            # Compute the sampling matrix:
            mkl = 2 * np.array([[2 ** (l - 3), 0], [0, 1]]) @ np.array([[1, 0], [-slk, 1]])
            
            i = ((k - 1) % 2)  # Index 0 or 1
            
            # Reconstruct the two-channel filter bank
            x[k - 1] = nssfbrec(x[2 * k - 2], x[2 * k - 1], f1[i], f2[i], mkl)
        
        # The second half channels
        for k in range(2 ** (l - 2) + 1, 2 ** (l - 1) + 1):
            # Compute the upsampling matrix by the extension of the formula (3.18)
            # of Minh N. Do's thesis to the second half channels
            
            # Compute s_{(l-1)}(k):
            slk = 2 * ((k - 2 ** (l - 2) - 1) // 2) - 2 ** (l - 3) + 1
            
            # Compute the sampling matrix:
            mkl = 2 * np.array([[1, 0], [0, 2 ** (l - 3)]]) @ np.array([[1, -slk], [0, 1]])
            
            i = ((k - 1) % 2) + 2  # Index 2 or 3
            
            # Reconstruct the two-channel filter bank
            x[k - 1] = nssfbrec(x[2 * k - 2], x[2 * k - 1], f1[i], f2[i], mkl)
    
    # Second-level reconstruction
    # Convolution with upsampled filters for the second level
    x[0] = nssfbrec(x[0], x[1], k1, k2, q1)
    x[1] = nssfbrec(x[2], x[3], k1, k2, q1)
    
    # First-level reconstruction
    # No upsampling for filters at the first level
    result = nssfbrec(x[0], x[1], k1, k2)
    
    return result


def nsctdec(x: np.ndarray, levels: Union[List[int], np.ndarray], dfilt: str = 'dmaxflat7', 
            pfilt: str = 'maxflat') -> List:
    """
    Nonsubsampled Contourlet Transform Decomposition.
    
    Decomposes the image using NSCT with pyramidal decomposition and 
    directional filter banks. This is the main function for NSCT analysis.
    Translation of nsctdec.m.
    
    Args:
        x (np.ndarray): Input image (2D array).
        levels (list): Vector of directional decomposition levels at each pyramidal level
                      (from coarse to fine scale). If level is 0, only pyramid decomposition
                      is performed without directional decomposition.
        dfilt (str): Filter name for directional decomposition (default: 'dmaxflat7').
                    See dfilters for available filters.
        pfilt (str): Filter name for pyramidal decomposition (default: 'maxflat').
                    See atrousfilters for available filters.
    
    Returns:
        list: Cell vector of length len(levels) + 1, where:
            - y[0]: Lowpass subband (coarsest scale)
            - y[1] to y[len(levels)]: Bandpass directional subbands at each pyramidal level
              Each y[i] (i > 0) is either:
                - A list of directional subbands if levels[i-1] > 0
                - A single bandpass image if levels[i-1] == 0
    
    Notes:
        - Combines nsfbdec (pyramidal) and nsdfbdec (directional)
        - Number of directional subbands at level i: 2^levels[i]
        - Index convention follows MATLAB implementation
    
    Examples:
        >>> x = np.random.rand(64, 64)
        >>> # 2 pyramid levels with 2 and 3 directional levels
        >>> levels = [2, 3]  
        >>> y = nsctdec(x, levels, 'pkva', 'maxflat')
        >>> len(y)
        3
        >>> # y[0]: lowpass, y[1]: 4 directional subbands, y[2]: 8 directional subbands
    """
    from nsct_python.filters import dfilters, atrousfilters, parafilters
    from nsct_python.utils import modulate2
    
    # Input validation
    if not isinstance(levels, (list, tuple, np.ndarray)):
        raise TypeError('The decomposition levels shall be a list or array of integers')
    
    levels = np.array(levels, dtype=int)
    if not np.all(levels == np.round(levels)):
        raise ValueError('The decomposition levels shall be integers')
    
    if np.any(levels < 0):
        raise ValueError('The decomposition levels shall be non-negative integers')
    
    # Get filters
    # Get the directional filters for the critically sampled DFB
    h1, h2 = dfilters(dfilt, 'd')
    
    # A scale is required for the nonsubsampled case
    h1 = h1 / np.sqrt(2)
    h2 = h2 / np.sqrt(2)
    
    # Generate the first-level fan filters by modulations
    k1 = modulate2(h1, 'c')
    k2 = modulate2(h2, 'c')
    
    # Obtain the parallelogram filters from the diamond filters
    f1, f2 = parafilters(h1, h2)
    
    # Package filters for nsdfbdec
    filters = {
        'k1': k1,
        'k2': k2,
        'f1': f1,
        'f2': f2
    }
    
    # Get pyramid filters
    h1_pyr, h2_pyr, g1_pyr, g2_pyr = atrousfilters(pfilt)
    
    # Number of levels
    clevels = len(levels)
    nIndex = clevels  # Index for output array position (Python 0-based)
    
    # Initialize the output
    y: List[Any] = [None] * (clevels + 1)
    
    # Nonsubsampled pyramid decomposition
    for i in range(clevels):
        # Nonsubsampled pyramid decomposition
        # Note: MATLAB uses i-1 for the level parameter (0-indexed level)
        xlo, xhi = nsfbdec(x, h1_pyr, h2_pyr, i)
        
        # Check directional decomposition level (MATLAB: levels(nIndex-1))
        if levels[nIndex - 1] > 0:
            # Nonsubsampled DFB decomposition on the bandpass image
            xhi_dir = nsdfbdec(xhi, filters, levels[nIndex - 1])
            y[nIndex] = xhi_dir
        else:
            # Copy the result directly (no directional decomposition)
            y[nIndex] = xhi
        
        # Update the index for the Nonsubsampled Pyramids
        nIndex = nIndex - 1
        
        # Prepare for next iteration
        x = xlo
    
    # The lowpass output
    y[0] = x
    
    return y


def nsctrec(y, dfilt='dmaxflat7', pfilt='maxflat'):
    """
    NSCTREC - Nonsubsampled Contourlet Reconstruction
    
    Reconstructs an image from its nonsubsampled contourlet decomposition.
    This is the inverse operation of nsctdec().
    
    Parameters
    ----------
    y : list
        A list of length n+1, where:
        - y[0] is the lowpass subband (2D array)
        - y[1:n+1] are the bandpass directional subbands, where each can be:
          * A list of 2D arrays (directional subbands from DFB) if level > 0
          * A single 2D array if level = 0 (no directional decomposition)
    dfilt : str, optional
        Filter name for the directional reconstruction step.
        Default is 'dmaxflat7'. See dfilters() for available filters.
    pfilt : str, optional
        Filter name for the pyramidal reconstruction step.
        Default is 'maxflat'. See atrousfilters() for available filters.
    
    Returns
    -------
    x : ndarray
        Reconstructed image (2D array)
    
    Notes
    -----
    This function performs perfect reconstruction when used with nsctdec().
    The reconstruction error should be on the order of machine precision (~1e-15).
    
    The reconstruction process:
    1. Get directional filters and pyramid filters
    2. For each pyramid level (from finest to coarsest):
       a. Reconstruct directional subbands using nsdfbrec() if decomposed
       b. Reconstruct pyramid level using nsfbrec()
    3. Return the final reconstructed image
    
    Examples
    --------
    >>> import numpy as np
    >>> x_orig = np.random.rand(64, 64)
    >>> y = nsctdec(x_orig, [2, 3], 'pkva', 'maxflat')
    >>> x_rec = nsctrec(y, 'pkva', 'maxflat')
    >>> error = np.max(np.abs(x_orig - x_rec))
    >>> print(f"Reconstruction error: {error}")  # Should be ~1e-15
    
    See Also
    --------
    nsctdec : Nonsubsampled contourlet decomposition
    nsfbrec : Nonsubsampled filter bank reconstruction
    nsdfbrec : Nonsubsampled directional filter bank reconstruction
    atrousfilters : Atrous pyramid filters
    dfilters : Directional filters
    
    References
    ----------
    .. [1] A. L. da Cunha, J. Zhou, and M. N. Do, "The Nonsubsampled Contourlet
           Transform: Theory, Design, and Applications," IEEE Transactions on
           Image Processing, vol. 15, no. 10, pp. 3089-3101, Oct. 2006.
    """
    # Get the directional filters for the DFB (use 'r' for reconstruction)
    h1_dir, h2_dir = dfilters(dfilt, 'r')
    
    # Scale for the nonsubsampled case (MATLAB: h1 = h1./sqrt(2))
    h1_dir = h1_dir / np.sqrt(2)
    h2_dir = h2_dir / np.sqrt(2)
    
    # Generate filters for DFB reconstruction
    # Create a dictionary with the required filter keys for nsdfbrec
    k1 = modulate2(h1_dir, 'c')
    k2 = modulate2(h2_dir, 'c')
    f1, f2 = parafilters(h1_dir, h2_dir)
    
    filters = {
        'k1': k1,
        'k2': k2,
        'f1': f1,
        'f2': f2
    }
    
    # Get pyramid filters (synthesis filters for reconstruction)
    h1_pyr, h2_pyr, g1_pyr, g2_pyr = atrousfilters(pfilt)
    
    # Number of pyramid levels
    n = len(y) - 1
    
    # Special case: no pyramid levels (just lowpass)
    if n == 0:
        return y[0]
    
    # Start with the lowpass subband
    xlo = y[0]
    
    # Index for pyramid reconstruction (starts from n-1, counts down)
    nIndex = n - 1
    
    # Reconstruct from coarsest to finest level
    for i in range(n):
        # Process the detail subbands
        if isinstance(y[i + 1], list):
            # Nonsubsampled DFB reconstruction (directional subbands)
            xhi = nsdfbrec(y[i + 1], filters)
        else:
            # No DFB decomposition, copy directly (level was 0)
            xhi = y[i + 1]
        
        # Nonsubsampled Pyramid reconstruction
        # Combine lowpass and highpass to get reconstructed image
        x = nsfbrec(xlo, xhi, g1_pyr, g2_pyr, nIndex)
        
        # Prepare for the next level
        xlo = x
        nIndex = nIndex - 1
    
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