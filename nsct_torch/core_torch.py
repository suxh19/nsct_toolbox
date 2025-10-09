"""
PyTorch implementation of NSCT (Nonsubsampled Contourlet Transform).

CUDA-ONLY IMPLEMENTATION
------------------------
This module requires CUDA and will only work with GPU tensors.
All input tensors must be on CUDA devices.
For CPU support, use nsct_python.core module instead.

Key functions directly use CUDA kernels:
- zconv2_cuda: 2D convolution with upsampled filter (periodic boundary)
- atrousc_cuda: Atrous convolution with symmetric extension

"""
import torch
import torch.nn.functional as F
from typing import Union, Tuple, Optional, List, Any

from nsct_torch.filters_torch import efilter2, dfilters, modulate2, parafilters, atrousfilters
from nsct_torch.utils_torch import extend2, symext, upsample2df
from nsct_torch.zconv2_cuda import zconv2_cuda
from nsct_torch.atrousc_cuda import atrousc_cuda


def _ensure_filter_device_dtype(filter_tensor: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """
    Ensure filters share the same device and dtype as the reference tensor to avoid host/device transfers.
    """
    if filter_tensor.device == reference.device and filter_tensor.dtype == reference.dtype:
        return filter_tensor
    return filter_tensor.to(device=reference.device, dtype=reference.dtype)


def _zconv2(x: torch.Tensor, h: torch.Tensor, mup: torch.Tensor) -> torch.Tensor:
    """
    2D convolution with upsampled filter using periodic boundary.
    
    Args:
        x: Input signal (2D tensor)
        h: Filter (2D tensor)
        mup: Upsampling matrix (2x2 tensor) [[M0, M1], [M2, M3]]
    
    Returns:
        y: Convolution output (same size as x)
    """
    return zconv2_cuda(x, h, mup)


def _convolve_upsampled(x: torch.Tensor, f: torch.Tensor, mup: Union[int, float, torch.Tensor], is_rec: bool = False) -> torch.Tensor:
    """ 
    Helper for convolution with an upsampled filter, handling reconstruction.
    Uses zconv2-style periodic convolution when mup is a 2x2 matrix.
    """
    f = _ensure_filter_device_dtype(f, x)

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
    f1 = _ensure_filter_device_dtype(f1, x)
    f2 = _ensure_filter_device_dtype(f2, x)

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
    
    f1 = _ensure_filter_device_dtype(f1, x1)
    f2 = _ensure_filter_device_dtype(f2, x1)

    if mup is None:
        y1 = efilter2(x1, f1)
        y2 = efilter2(x2, f2)
    else:
        y1 = _convolve_upsampled(x1, f1, mup, is_rec=True)
        y2 = _convolve_upsampled(x2, f2, mup, is_rec=True)

    return y1 + y2


def _atrousc(x: torch.Tensor, f: torch.Tensor, mup: torch.Tensor) -> torch.Tensor:
    """
    Atrous convolution with symmetric extension.
    
    Args:
        x: Input extended signal (2D tensor)
        f: Filter (2D tensor)
        mup: Upsampling matrix (2x2 tensor or scalar)
    
    Returns:
        y: Convolution output
    """
    f = _ensure_filter_device_dtype(f, x)
    return atrousc_cuda(x, f, mup)


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
    h0 = _ensure_filter_device_dtype(h0, x)
    h1 = _ensure_filter_device_dtype(h1, x)

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
    
    g0 = _ensure_filter_device_dtype(g0, x0)
    g1 = _ensure_filter_device_dtype(g1, x0)
    
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
    
    # Extract filters and align them with the input tensor
    k1_raw = filters['k1']
    k1 = _ensure_filter_device_dtype(k1_raw, x)
    if k1 is not k1_raw:
        filters['k1'] = k1

    k2_raw = filters['k2']
    k2 = _ensure_filter_device_dtype(k2_raw, x)
    if k2 is not k2_raw:
        filters['k2'] = k2

    f1_list = filters['f1']  # This is a list of 4 filters
    f1 = []
    for idx, filt in enumerate(f1_list):
        filt_prepared = _ensure_filter_device_dtype(filt, x)
        if filt_prepared is not filt:
            f1_list[idx] = filt_prepared
        f1.append(filt_prepared)

    f2_list = filters['f2']  # This is a list of 4 filters
    f2 = []
    for idx, filt in enumerate(f2_list):
        filt_prepared = _ensure_filter_device_dtype(filt, x)
        if filt_prepared is not filt:
            f2_list[idx] = filt_prepared
        f2.append(filt_prepared)
    
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
    
    # Extract filters and align them with the coefficient tensors
    reference = y[0]

    k1_raw = filters['k1']
    k1 = _ensure_filter_device_dtype(k1_raw, reference)
    if k1 is not k1_raw:
        filters['k1'] = k1

    k2_raw = filters['k2']
    k2 = _ensure_filter_device_dtype(k2_raw, reference)
    if k2 is not k2_raw:
        filters['k2'] = k2

    f1_list = filters['f1']  # List of 4 filters
    f1 = []
    for idx, filt in enumerate(f1_list):
        filt_prepared = _ensure_filter_device_dtype(filt, reference)
        if filt_prepared is not filt:
            f1_list[idx] = filt_prepared
        f1.append(filt_prepared)

    f2_list = filters['f2']  # List of 4 filters
    f2 = []
    for idx, filt in enumerate(f2_list):
        filt_prepared = _ensure_filter_device_dtype(filt, reference)
        if filt_prepared is not filt:
            f2_list[idx] = filt_prepared
        f2.append(filt_prepared)
    
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


def nsctdec(x: torch.Tensor, nlevs: List[int], dfilt: str = 'dmaxflat7', pfilt: str = 'maxflat', dtype: Optional[torch.dtype] = None) -> List[Any]:
    """
    Nonsubsampled Contourlet Transform (NSCT) decomposition.
    PyTorch translation of nsctdec.m.
    
    Args:
        x (torch.Tensor): Input image (2D tensor)
        nlevs (list): Number of directional decomposition levels at each pyramid level
        dfilt (str): Directional filter name (default: 'dmaxflat7')
        pfilt (str): Pyramid filter name (default: 'maxflat')
        dtype (torch.dtype, optional): Data type for computation. If None, uses input dtype.
                                      Default is None. Supports torch.float32 and torch.float64.
    
    Returns:
        list: NSCT coefficients [lowpass, band1, band2, ...]
    """
    # Use input dtype if not specified
    if dtype is None:
        dtype = x.dtype
    
    # Convert input to specified dtype
    x = x.to(dtype)
    
    # Get directional filters with specified dtype
    h1_dir, h2_dir = dfilters(dfilt, 'd', dtype=dtype)
    
    # Move directional filters to the same device as input
    h1_dir = h1_dir.to(device=x.device)
    h2_dir = h2_dir.to(device=x.device)
    
    # Scale for nonsubsampled case
    scale = torch.tensor(2.0, dtype=x.dtype, device=x.device).sqrt()
    h1_dir = h1_dir / scale
    h2_dir = h2_dir / scale
    
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
    
    # Get pyramid filters with specified dtype
    h1_pyr, h2_pyr, g1_pyr, g2_pyr = atrousfilters(pfilt, dtype=dtype)
    
    # Move filters to the same device as input
    h1_pyr = h1_pyr.to(device=x.device)
    h2_pyr = h2_pyr.to(device=x.device)
    g1_pyr = g1_pyr.to(device=x.device)
    g2_pyr = g2_pyr.to(device=x.device)
    
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


def nsctrec(y: List[Any], dfilt: str = 'dmaxflat7', pfilt: str = 'maxflat', dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """
    Nonsubsampled Contourlet Transform (NSCT) reconstruction.
    PyTorch translation of nsctrec.m.
    
    Args:
        y (list): NSCT coefficients [lowpass, band1, band2, ...]
        dfilt (str): Directional filter name (default: 'dmaxflat7')
        pfilt (str): Pyramid filter name (default: 'maxflat')
        dtype (torch.dtype, optional): Data type for computation. If None, uses dtype from input coefficients.
                                       Default is None. Supports torch.float32 and torch.float64.
    
    Returns:
        torch.Tensor: Reconstructed image
    """
    # Determine device and dtype from input if not specified
    device = y[0].device if len(y) > 0 and isinstance(y[0], torch.Tensor) else torch.device('cpu')
    if dtype is None:
        dtype = y[0].dtype if len(y) > 0 and isinstance(y[0], torch.Tensor) else torch.float32
    
    # Get directional filters for reconstruction with specified dtype
    h1_dir, h2_dir = dfilters(dfilt, 'r', dtype=dtype)
    
    # Move directional filters to the same device as input
    h1_dir = h1_dir.to(device=device)
    h2_dir = h2_dir.to(device=device)
    
    # Scale for nonsubsampled case
    scale = torch.tensor(2.0, dtype=dtype, device=device).sqrt()
    h1_dir = h1_dir / scale
    h2_dir = h2_dir / scale
    
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
    
    # Get pyramid synthesis filters with specified dtype
    h1_pyr, h2_pyr, g1_pyr, g2_pyr = atrousfilters(pfilt, dtype=dtype)
    
    # Move filters to the same device as input
    g1_pyr = g1_pyr.to(device=device)
    g2_pyr = g2_pyr.to(device=device)
    h1_pyr = h1_pyr.to(device=device)
    h2_pyr = h2_pyr.to(device=device)
    
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
