import torch
import torch.nn.functional as F
from typing import Union, Tuple, Optional, List, Any

from nsct_torch.filters_torch import efilter2, dfilters, modulate2, parafilters, atrousfilters
from nsct_torch.utils_torch import extend2, symext, upsample2df

# Import CUDA implementation of zconv2 if available
try:
    from nsct_torch.zconv2_cuda import zconv2_cuda as _zconv2_cuda
    ZCONV2_CUDA_AVAILABLE = True
except ImportError:
    ZCONV2_CUDA_AVAILABLE = False
    _zconv2_cuda = None

# Import CUDA implementation of atrousc if available
try:
    from nsct_torch.atrousc_cuda import atrousc_cuda as _atrousc_cuda
    ATROUSC_CUDA_AVAILABLE = True
except ImportError:
    ATROUSC_CUDA_AVAILABLE = False
    _atrousc_cuda = None


def _zconv2_torch(x: torch.Tensor, h: torch.Tensor, mup: torch.Tensor) -> torch.Tensor:
    """
    2D convolution with upsampled filter using periodic boundary (PyTorch implementation).
    
    Vectorized implementation for better performance.
    
    Args:
        x: Input signal (2D tensor)
        h: Filter (2D tensor)
        mup: Upsampling matrix (2x2 tensor) [[M0, M1], [M2, M3]]
    
    Returns:
        y: Convolution output (same size as x)
    """
    s_row_len, s_col_len = x.shape
    f_row_len, f_col_len = h.shape
    
    # Upsampling factors from the matrix
    # mup = [[M0, M1], [M2, M3]]
    mup_int = mup.long()
    M0 = mup_int[0, 0].item()
    M1 = mup_int[0, 1].item()
    M2 = mup_int[1, 0].item()
    M3 = mup_int[1, 1].item()
    
    # Calculate upsampled filter dimensions
    new_f_row_len = (M0 - 1) * (f_row_len - 1) + M2 * (f_col_len - 1) + f_row_len - 1
    new_f_col_len = (M3 - 1) * (f_col_len - 1) + M1 * (f_row_len - 1) + f_col_len - 1
    
    # Starting positions
    start1 = new_f_row_len // 2
    start2 = new_f_col_len // 2
    mn1_init = start1 % s_row_len
    mn2_save = start2 % s_col_len
    
    # Initialize output
    y = torch.zeros_like(x)
    
    # Vectorized approach: compute all indices for each filter tap
    # For each filter element (l1, l2), compute the indices it affects
    for l1 in range(f_row_len):
        for l2 in range(f_col_len):
            if h[l1, l2] == 0:
                continue  # Skip zero filter coefficients
            
            # Precompute index offsets for this filter tap
            # For output position (n1, n2), we need input at specific shifted location
            # Based on C++ logic: start from mn1_init + n1, mn2_save + n2
            # Then apply filter-specific shifts
            
            # Create row indices for all output positions
            n1_range = torch.arange(s_row_len, dtype=torch.long, device=x.device)
            n2_range = torch.arange(s_col_len, dtype=torch.long, device=x.device)
            
            # Compute starting position for each output position
            mn1_arr = (mn1_init + n1_range) % s_row_len
            mn2_arr = (mn2_save + n2_range) % s_col_len
            
            # Apply filter-specific offset
            # offset_x = -M0 * l1
            # offset_y = -M1 * l1 - M3 * l2
            offset_x = (-M0 * l1 - M2 * l2)
            offset_y = (-M1 * l1 - M3 * l2)
            
            # Compute indices for this filter tap
            idx_x = (mn1_arr.unsqueeze(1) + offset_x) % s_row_len
            idx_y = (mn2_arr.unsqueeze(0) + offset_y) % s_col_len
            
            # Accumulate: y += h[l1, l2] * x[idx_x, idx_y]
            y += h[l1, l2] * x[idx_x, idx_y]
    
    return y


def _zconv2(x: torch.Tensor, h: torch.Tensor, mup: torch.Tensor) -> torch.Tensor:
    """
    2D convolution with upsampled filter using periodic boundary.
    
    Uses CUDA implementation if available, otherwise falls back to pure PyTorch.
    
    Args:
        x: Input signal (2D tensor)
        h: Filter (2D tensor)
        mup: Upsampling matrix (2x2 tensor) [[M0, M1], [M2, M3]]
    
    Returns:
        y: Convolution output (same size as x)
    """
    if ZCONV2_CUDA_AVAILABLE and x.is_cuda:
        if _zconv2_cuda is not None:
            return _zconv2_cuda(x, h, mup)
        # Fallback in case the check somehow fails but function is None
        return _zconv2_torch(x, h, mup)
    else:
        return _zconv2_torch(x, h, mup)


def _convolve_upsampled(x: torch.Tensor, f: torch.Tensor, mup: Union[int, float, torch.Tensor], is_rec: bool = False) -> torch.Tensor:
    """ 
    Helper for convolution with an upsampled filter, handling reconstruction.
    Uses zconv2-style periodic convolution when mup is a 2x2 matrix.
    """
    # If the filter is all zeros, the output is all zeros.
    if not torch.any(f):
        return torch.zeros_like(x)
    
    # Convert mup to matrix form
    if isinstance(mup, (int, float)):
        mup_mat = torch.tensor([[mup, 0], [0, mup]], dtype=torch.long, device=x.device)
    else:
        mup_mat = mup.long().to(x.device)
    
    # For reconstruction, use time-reversed filter
    f_to_use = torch.rot90(f, 2) if is_rec else f
    
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


def _atrousc_torch(x: torch.Tensor, f: torch.Tensor, mup: torch.Tensor) -> torch.Tensor:
    """
    Atrous convolution with symmetric extension (PyTorch implementation).
    
    Matches the CUDA kernel implementation logic.
    
    Args:
        x: Input extended signal (2D tensor)
        f: Filter (not upsampled, 2D tensor)
        mup: Upsampling matrix (2x2 tensor or scalar)
    
    Returns:
        y: Convolution output
    """
    # Extract upsampling factors
    if mup.numel() == 1:
        M0 = M3 = mup.item()
    else:
        M0 = mup[1, 1].item()  # Column upsampling factor
        M3 = mup[0, 0].item()  # Row upsampling factor
    
    fm, fn = f.shape
    xm, xn = x.shape
    
    # Calculate output size based on upsampling factors
    m = xm - M3 * fm + 1
    n = xn - M0 * fn + 1
    
    if m <= 0 or n <= 0:
        return torch.zeros(max(0, int(m)), max(0, int(n)), dtype=x.dtype, device=x.device)
    
    # Initialize output
    y = torch.zeros(int(m), int(n), dtype=x.dtype, device=x.device)

    # Rotate filter 180 degrees to match convolution behaviour
    f_rot = torch.rot90(f, 2)
    
    # Perform atrous convolution
    # Following CUDA kernel logic: kk = n + M - 1, then kk += M for each filter element
    # This means for output position (n2, n1), we sample input at:
    # (n2 + M3 - 1 + k2 * M3, n1 + M0 - 1 + k1 * M0) for filter position (k2, k1)
    # Which simplifies to: (n2 + (k2 + 1) * M3 - 1, n1 + (k1 + 1) * M0 - 1)
    # Or: (n2 + k2 * M3 + M3 - 1, n1 + k1 * M0 + M0 - 1)
    
    for k2 in range(fm):  # Filter rows
        for k1 in range(fn):  # Filter columns
            # Calculate starting positions for this filter element
            # kk2 = n2 + M3 - 1 + k2 * M3  (for all n2 from 0 to m-1)
            # kk1 = n1 + M0 - 1 + k1 * M0  (for all n1 from 0 to n-1)
            row_start = M3 - 1 + k2 * M3
            col_start = M0 - 1 + k1 * M0
            
            # Extract the portion and accumulate
            y = y + f_rot[k2, k1] * x[row_start:row_start + m, col_start:col_start + n]
    
    return y


def _atrousc(x: torch.Tensor, f: torch.Tensor, mup: torch.Tensor) -> torch.Tensor:
    """
    Atrous convolution with symmetric extension.
    
    Uses CUDA implementation if available, otherwise falls back to pure PyTorch.
    
    Args:
        x: Input extended signal (2D tensor)
        f: Filter (2D tensor)
        mup: Upsampling matrix (2x2 tensor or scalar)
    
    Returns:
        y: Convolution output
    """
    if ATROUSC_CUDA_AVAILABLE and x.is_cuda:
        if _atrousc_cuda is not None:
            return _atrousc_cuda(x, f, mup)
        # Fallback in case the check somehow fails but function is None
        return _atrousc_torch(x, f, mup)
    else:
        return _atrousc_torch(x, f, mup)


def nsfbdec(x: torch.Tensor, h0: torch.Tensor, h1: torch.Tensor, lev: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Nonsubsampled filter bank decomposition at a given level.
    PyTorch translation of nsfbdec.m.
    
    Computes the nonsubsampled pyramid decomposition at level lev with filters h0, h1.
    
    Args:
        x (torch.Tensor): Input image at finer scale.
        h0 (torch.Tensor): Lowpass à trous filter (obtained from atrousfilters).
        h1 (torch.Tensor): Highpass à trous filter (obtained from atrousfilters).
        lev (int): Decomposition level (0 for first level, >0 for subsequent levels).
    
    Returns:
        tuple: (y0, y1) where:
            - y0: Image at coarser scale (lowpass output)
            - y1: Wavelet highpass output (bandpass output)
    
    Notes:
        - For lev=0: Uses regular conv2 with symext
        - For lev>0: Uses upsampled filters with atrous convolution
    """
    if lev != 0:
        # For levels > 0, use upsampled filters
        shift = [-2**(lev-1), -2**(lev-1)]
        shift = [s + 2 for s in shift]  # shift = -2^(lev-1)*[1,1] + 2
        L = 2**lev
        
        # Upsampling matrix for atrous convolution
        mup = torch.tensor([[L, 0], [0, L]], dtype=torch.long, device=x.device)
        
        # Upsample filters
        h0_up = upsample2df(h0, lev)
        h1_up = upsample2df(h1, lev)
        
        # Symmetric extension with upsampled filter size
        x_ext_h0 = symext(x, h0_up, shift)
        x_ext_h1 = symext(x, h1_up, shift)
        
        # Atrous convolution (use original filter, not upsampled)
        y0 = _atrousc(x_ext_h0, h0, mup)
        y1 = _atrousc(x_ext_h1, h1, mup)
    else:
        # For level 0, use regular convolution
        shift = [1, 1]
        
        # Upsampling matrix for regular convolution
        mup = torch.tensor([[1, 0], [0, 1]], dtype=torch.long, device=x.device)
        
        # Symmetric extension
        x_ext_h0 = symext(x, h0, shift)
        x_ext_h1 = symext(x, h1, shift)
        
        # Regular convolution using _atrousc
        y0 = _atrousc(x_ext_h0, h0, mup)
        y1 = _atrousc(x_ext_h1, h1, mup)
    
    return y0, y1


def nsfbrec(x0: torch.Tensor, x1: torch.Tensor, g0: torch.Tensor, g1: torch.Tensor, lev: int) -> torch.Tensor:
    """
    Nonsubsampled filter bank reconstruction.
    PyTorch translation of nsfbrec.m.
    
    Args:
        x0 (torch.Tensor): Lowpass subband
        x1 (torch.Tensor): Highpass subband
        g0 (torch.Tensor): Lowpass synthesis filter
        g1 (torch.Tensor): Highpass synthesis filter
        lev (int): Reconstruction level
    
    Returns:
        torch.Tensor: Reconstructed image
    """
    if x0.shape != x1.shape:
        raise ValueError("Input sizes for the two branches must be the same")
    
    if lev != 0:
        # For levels > 0, use upsampled filters
        shift = [-2**(lev-1), -2**(lev-1)]
        shift = [s + 2 for s in shift]
        L = 2**lev
        
        # Upsampling matrix for atrous convolution
        mup = torch.tensor([[L, 0], [0, L]], dtype=torch.long, device=x0.device)
        
        # Upsample filters
        g0_up = upsample2df(g0, lev)
        g1_up = upsample2df(g1, lev)
        
        # Symmetric extension with upsampled filter size
        x0_ext = symext(x0, g0_up, shift)
        x1_ext = symext(x1, g1_up, shift)
        
        # Atrous convolution (use original filter, not upsampled)
        y0 = _atrousc(x0_ext, g0, mup)
        y1 = _atrousc(x1_ext, g1, mup)
    else:
        # For level 0, use regular convolution
        shift = [1, 1]
        
        # Upsampling matrix for regular convolution
        mup = torch.tensor([[1, 0], [0, 1]], dtype=torch.long, device=x0.device)
        
        # Symmetric extension
        x0_ext = symext(x0, g0, shift)
        x1_ext = symext(x1, g1, shift)
        
        # Regular convolution
        y0 = _atrousc(x0_ext, g0, mup)
        y1 = _atrousc(x1_ext, g1, mup)
    
    return y0 + y1


def nsdfbdec(x: torch.Tensor, filters: dict, n: int) -> List[torch.Tensor]:
    """
    Nonsubsampled directional filter bank decomposition.
    PyTorch translation of nsdfbdec.m (following NumPy implementation).
    
    Args:
        x (torch.Tensor): Input image
        filters (dict): Dictionary containing filters k1, k2, f1, f2
        n (int): Number of decomposition levels (0 means no decomposition)
    
    Returns:
        list: List of directional subbands (length 2^n)
    """
    if n == 0:
        # No decomposition
        return [x]
    
    # Extract filters
    k1 = filters['k1']
    k2 = filters['k2']
    f1 = filters['f1']  # This is a list of 4 filters
    f2 = filters['f2']  # This is a list of 4 filters
    
    # First level of decomposition (no upsampling matrix)
    if n == 1:
        y1, y2 = nssfbdec(x, k1, k2)
        return [y1, y2]
    
    # Quincunx sampling matrix
    q1 = torch.tensor([[1, -1], [1, 1]], dtype=torch.long, device=x.device)
    
    # Second-level decomposition
    # First level: no upsampling
    x1, x2 = nssfbdec(x, k1, k2)
    
    # Second level: with quincunx upsampling
    y: List[torch.Tensor] = [torch.empty(0)] * 4
    y[0], y[1] = nssfbdec(x1, k1, k2, q1)
    y[2], y[3] = nssfbdec(x2, k1, k2, q1)
    
    if n == 2:
        return y
    
    # Third and higher levels decomposition
    for l in range(3, n + 1):
        y_old = y
        y = [torch.empty(0)] * (2 ** l)
        
        # The first half channels
        for k in range(1, 2 ** (l - 2) + 1):
            # Compute s_{(l-1)}(k):
            slk = 2 * ((k - 1) // 2) - 2 ** (l - 3) + 1
            
            # Compute the sampling matrix:
            mkl_part1 = 2 * torch.tensor([[2 ** (l - 3), 0], [0, 1]], dtype=torch.long, device=x.device)
            mkl_part2 = torch.tensor([[1, 0], [-slk, 1]], dtype=torch.long, device=x.device)
            mkl = torch.mm(mkl_part1.float(), mkl_part2.float()).long()
            
            i = (k - 1) % 2  # Index 0 or 1
            
            # Decompose by the two-channel filter bank
            y[2 * k - 2], y[2 * k - 1] = nssfbdec(y_old[k - 1], f1[i], f2[i], mkl)
        
        # The second half channels
        for k in range(2 ** (l - 2) + 1, 2 ** (l - 1) + 1):
            # Compute s_{(l-1)}(k):
            slk = 2 * ((k - 2 ** (l - 2) - 1) // 2) - 2 ** (l - 3) + 1
            
            # Compute the sampling matrix:
            mkl_part1 = 2 * torch.tensor([[1, 0], [0, 2 ** (l - 3)]], dtype=torch.long, device=x.device)
            mkl_part2 = torch.tensor([[1, -slk], [0, 1]], dtype=torch.long, device=x.device)
            mkl = torch.mm(mkl_part1.float(), mkl_part2.float()).long()
            
            i = ((k - 1) % 2) + 2  # Index 2 or 3
            
            # Decompose by the two-channel filter bank
            y[2 * k - 2], y[2 * k - 1] = nssfbdec(y_old[k - 1], f1[i], f2[i], mkl)
    
    return y


def nsdfbrec(y: List[torch.Tensor], filters: dict) -> torch.Tensor:
    """
    Nonsubsampled directional filter bank reconstruction.
    PyTorch translation of nsdfbrec.m (following NumPy implementation).
    
    Args:
        y (list): List of directional subbands (length 2^n)
        filters (dict): Dictionary containing filters k1, k2, f1, f2
    
    Returns:
        torch.Tensor: Reconstructed image
    """
    n_bands = len(y)
    
    if n_bands == 0:
        raise ValueError('Number of subbands must be a power of 2')
    
    # Calculate number of levels
    n = int(torch.log2(torch.tensor(n_bands, dtype=torch.float32)).item())
    
    if 2**n != n_bands:
        raise ValueError(f"Number of subbands must be a power of 2, got {n_bands}")
    
    # No reconstruction case
    if n == 0:
        return y[0]
    
    # Extract filters
    k1 = filters['k1']
    k2 = filters['k2']
    f1 = filters['f1']  # List of 4 filters
    f2 = filters['f2']  # List of 4 filters
    
    # First-level reconstruction (no upsampling)
    if n == 1:
        return nssfbrec(y[0], y[1], k1, k2)
    
    # Multi-level reconstruction (n >= 2)
    # To save memory, we use a copy of the input list to store middle outputs
    x = y.copy()
    
    # Quincunx sampling matrix
    q1 = torch.tensor([[1, -1], [1, 1]], dtype=torch.long, device=y[0].device)
    
    # Third and higher levels reconstructions (from highest to lowest)
    for l in range(n, 2, -1):
        # The first half channels
        for k in range(1, 2 ** (l - 2) + 1):
            # Compute s_{(l-1)}(k):
            slk = 2 * ((k - 1) // 2) - 2 ** (l - 3) + 1
            
            # Compute the sampling matrix:
            mkl_part1 = 2 * torch.tensor([[2 ** (l - 3), 0], [0, 1]], dtype=torch.long, device=y[0].device)
            mkl_part2 = torch.tensor([[1, 0], [-slk, 1]], dtype=torch.long, device=y[0].device)
            mkl = torch.mm(mkl_part1.float(), mkl_part2.float()).long()
            
            i = (k - 1) % 2  # Index 0 or 1
            
            # Reconstruct the two-channel filter bank
            x[k - 1] = nssfbrec(x[2 * k - 2], x[2 * k - 1], f1[i], f2[i], mkl)
        
        # The second half channels
        for k in range(2 ** (l - 2) + 1, 2 ** (l - 1) + 1):
            # Compute s_{(l-1)}(k):
            slk = 2 * ((k - 2 ** (l - 2) - 1) // 2) - 2 ** (l - 3) + 1
            
            # Compute the sampling matrix:
            mkl_part1 = 2 * torch.tensor([[1, 0], [0, 2 ** (l - 3)]], dtype=torch.long, device=y[0].device)
            mkl_part2 = torch.tensor([[1, -slk], [0, 1]], dtype=torch.long, device=y[0].device)
            mkl = torch.mm(mkl_part1.float(), mkl_part2.float()).long()
            
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


def nsctdec(x: torch.Tensor, nlevs: List[int], dfilt: str = 'dmaxflat7', pfilt: str = 'maxflat') -> List[Any]:
    """
    Nonsubsampled Contourlet Transform (NSCT) decomposition.
    PyTorch translation of nsctdec.m.
    
    Args:
        x (torch.Tensor): Input image (2D tensor)
        nlevs (list): Number of directional decomposition levels at each pyramid level
        dfilt (str): Directional filter name (default: 'dmaxflat7')
        pfilt (str): Pyramid filter name (default: 'maxflat')
    
    Returns:
        list: NSCT coefficients [lowpass, band1, band2, ...]
    """
    # Get directional filters
    h1_dir, h2_dir = dfilters(dfilt, 'd')
    
    # Scale for nonsubsampled case
    h1_dir = h1_dir / torch.sqrt(torch.tensor(2.0))
    h2_dir = h2_dir / torch.sqrt(torch.tensor(2.0))
    
    # Create filter dictionary for DFB
    k1 = modulate2(h1_dir, 'c')
    k2 = modulate2(h2_dir, 'c')
    f1, f2 = parafilters(h1_dir, h2_dir)
    
    filters = {
        'k1': k1,
        'k2': k2,
        'f1': f1,
        'f2': f2
    }
    
    # Get pyramid filters
    h1_pyr, h2_pyr, g1_pyr, g2_pyr = atrousfilters(pfilt)
    
    # Number of pyramid levels
    n = len(nlevs)
    
    # Initialize output list (lowpass + each pyramid level)
    coeffs: List[Any] = [None] * (n + 1)
    
    x_current = x
    for pyr_level in range(n):
        x_low, x_high = nsfbdec(x_current, h1_pyr, h2_pyr, pyr_level)
        
        dir_level = nlevs[n - pyr_level - 1]
        if dir_level > 0:
            coeffs[n - pyr_level] = nsdfbdec(x_high, filters, dir_level)
        else:
            coeffs[n - pyr_level] = x_high
        
        x_current = x_low
    
    coeffs[0] = x_current
    return coeffs


def nsctrec(y: List[Any], dfilt: str = 'dmaxflat7', pfilt: str = 'maxflat') -> torch.Tensor:
    """
    Nonsubsampled Contourlet Transform (NSCT) reconstruction.
    PyTorch translation of nsctrec.m.
    
    Args:
        y (list): NSCT coefficients [lowpass, band1, band2, ...]
        dfilt (str): Directional filter name (default: 'dmaxflat7')
        pfilt (str): Pyramid filter name (default: 'maxflat')
    
    Returns:
        torch.Tensor: Reconstructed image
    """
    # Get directional filters for reconstruction
    h1_dir, h2_dir = dfilters(dfilt, 'r')
    
    # Scale for nonsubsampled case
    h1_dir = h1_dir / torch.sqrt(torch.tensor(2.0))
    h2_dir = h2_dir / torch.sqrt(torch.tensor(2.0))
    
    # Create filter dictionary for DFB
    k1 = modulate2(h1_dir, 'c')
    k2 = modulate2(h2_dir, 'c')
    f1, f2 = parafilters(h1_dir, h2_dir)
    
    filters = {
        'k1': k1,
        'k2': k2,
        'f1': f1,
        'f2': f2
    }
    
    # Get pyramid synthesis filters
    h1_pyr, h2_pyr, g1_pyr, g2_pyr = atrousfilters(pfilt)
    
    # Number of pyramid levels
    n = len(y) - 1
    
    # Special case: no pyramid levels
    if n == 0:
        return y[0]
    
    # Start with lowpass
    xlo = y[0]
    
    # Pyramid reconstruction index
    nIndex = n - 1
    
    # Reconstruct from coarsest to finest
    for i in range(n):
        # Reconstruct directional subbands
        if isinstance(y[i + 1], list):
            xhi = nsdfbrec(y[i + 1], filters)
        else:
            xhi = y[i + 1]
        
        # Pyramid reconstruction
        x = nsfbrec(xlo, xhi, g1_pyr, g2_pyr, nIndex)
        
        # Update for next level
        xlo = x
        nIndex = nIndex - 1
    
    return x


if __name__ == '__main__':
    from nsct_torch.filters_torch import dfilters

    print("--- Running tests for nssfbdec and nssfbrec (Perfect Reconstruction) ---")

    # Create a sample image (must be even-sized for some filters)
    img = torch.rand(32, 32)

    # Get a pair of analysis and synthesis filters
    h0, h1 = dfilters('pkva', 'd')
    g0, g1 = dfilters('pkva', 'r')

    # Define a quincunx upsampling matrix
    mup = torch.tensor([[1, 1], [-1, 1]])

    # Test Decomposition and Reconstruction
    print("Testing with upsampling matrix M =", mup.flatten())

    # Decompose
    y1, y2 = nssfbdec(img, h0, h1, mup)

    # Reconstruct
    recon_img = nssfbrec(y1, y2, g0, g1, mup)

    # Check for perfect reconstruction
    print("Original vs. Reconstructed MSE:", torch.mean((img - recon_img)**2).item())
    assert torch.allclose(img, recon_img, atol=1e-6)

    print("nssfbdec/rec perfect reconstruction test passed!")
