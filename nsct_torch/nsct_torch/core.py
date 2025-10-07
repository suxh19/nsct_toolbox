"""
Core NSCT (Nonsubsampled Contourlet Transform) functions
PyTorch translation from nsct_python/core.py
"""

import torch
import torch.nn.functional as F
from typing import Union, Tuple, List, Any, Dict
from nsct_torch.filters import efilter2, dfilters, parafilters, atrousfilters
from nsct_torch.utils import symext, upsample2df, modulate2


def _zconv2_torch(x: torch.Tensor, h: torch.Tensor, mup: torch.Tensor) -> torch.Tensor:
    """
    2D convolution with upsampled filter using periodic boundary.
    PyTorch implementation (CPU/GPU compatible).
    
    Args:
        x: Input signal (2D tensor)
        h: Filter (2D tensor)
        mup: Upsampling matrix (2x2 tensor) [[M0, M1], [M2, M3]]
    
    Returns:
        y: Convolution output (same size as x)
    """
    s_rows, s_cols = x.shape
    f_rows, f_cols = h.shape
    
    # Extract upsampling matrix elements
    M0, M1 = int(mup[0, 0].item()), int(mup[0, 1].item())
    M2, M3 = int(mup[1, 0].item()), int(mup[1, 1].item())
    
    # Calculate upsampled filter dimensions
    new_f_rows = (M0 - 1) * (f_rows - 1) + M2 * (f_cols - 1) + f_rows - 1
    new_f_cols = (M3 - 1) * (f_cols - 1) + M1 * (f_rows - 1) + f_cols - 1
    
    # Create output tensor
    result = torch.zeros_like(x)
    
    # Initialize
    start1 = new_f_rows // 2
    start2 = new_f_cols // 2
    mn1_init = start1 % s_rows
    mn2_save = start2 % s_cols
    
    # Main convolution loop
    for n1 in range(s_rows):
        mn1 = (mn1_init + n1) % s_rows
        mn2 = mn2_save
        
        for n2 in range(s_cols):
            out_index_x = mn1
            out_index_y = mn2
            total = 0.0
            
            # Loop over filter rows
            for l1 in range(f_rows):
                index_x = out_index_x
                index_y = out_index_y
                
                # Loop over filter columns
                for l2 in range(f_cols):
                    total += x[index_x, index_y].item() * h[l1, l2].item()
                    
                    # Step through input with M2, M3 (periodic boundary)
                    index_x = (index_x - M2) % s_rows
                    index_y = (index_y - M3) % s_cols
                
                # Step for outer filter loop with M0, M1
                out_index_x = (out_index_x - M0) % s_rows
                out_index_y = (out_index_y - M1) % s_cols
            
            result[n1, n2] = total
            
            # Update mn2 for next column
            mn2 = (mn2 + 1) % s_cols
    
    return result


def _atrousc_torch(x: torch.Tensor, h: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """
    À trous convolution with upsampled filter.
    PyTorch implementation (CPU/GPU compatible).
    
    Args:
        x: Extended input signal (2D tensor)
        h: Original filter, not upsampled (2D tensor)
        M: Upsampling matrix (2x2 tensor or scalar)
    
    Returns:
        Result of convolution in 'valid' mode (2D tensor)
    """
    S_rows, S_cols = x.shape
    F_rows, F_cols = h.shape
    
    # Extract upsampling factors
    if M.ndim == 0:
        M0 = M3 = int(M.item())
    else:
        M0 = int(M[0, 0].item())
        M3 = int(M[1, 1].item())
    
    # Calculate output dimensions
    O_rows = S_rows - M0 * F_rows + 1
    O_cols = S_cols - M3 * F_cols + 1
    
    if O_rows <= 0 or O_cols <= 0:
        return torch.zeros(max(0, O_rows), max(0, O_cols), 
                          dtype=x.dtype, device=x.device)
    
    # Create output tensor
    result = torch.zeros(O_rows, O_cols, dtype=x.dtype, device=x.device)
    
    # Flip the filter for convolution
    h_flipped = torch.flip(torch.flip(h, [0]), [1])
    
    # Main convolution loop
    for n1 in range(O_cols):
        for n2 in range(O_rows):
            total = 0.0
            kk1 = n1 + M0 - 1
            
            # Loop over filter columns
            for k1 in range(F_cols):
                kk2 = n2 + M3 - 1
                
                # Loop over filter rows
                for k2 in range(F_rows):
                    # Flipped indices
                    f1 = F_cols - 1 - k1
                    f2 = F_rows - 1 - k2
                    
                    # Accumulate
                    total += h_flipped[f2, f1].item() * x[kk2, kk1].item()
                    kk2 += M3
                
                kk1 += M0
            
            result[n2, n1] = total
    
    return result


def _convolve_upsampled(x: torch.Tensor, f: torch.Tensor, 
                       mup: Union[int, float, torch.Tensor], 
                       is_rec: bool = False) -> torch.Tensor:
    """
    Helper for convolution with an upsampled filter, handling reconstruction.
    
    Args:
        x: Input tensor
        f: Filter tensor
        mup: Upsampling factor or matrix
        is_rec: Whether this is reconstruction (uses time-reversed filter)
    
    Returns:
        Convolution result
    """
    # If the filter is all zeros, return zeros
    if not torch.any(f):
        return torch.zeros_like(x)
    
    # Convert mup to matrix form
    if isinstance(mup, (int, float)):
        mup_mat = torch.tensor([[mup, 0], [0, mup]], 
                              dtype=torch.int64, device=x.device)
    else:
        mup_mat = mup.to(torch.int64)
    
    # For reconstruction, use time-reversed filter
    f_to_use = torch.rot90(f, 2) if is_rec else f
    
    # Use zconv2 for periodic convolution with upsampled filter
    return _zconv2_torch(x, f_to_use, mup_mat)


def nssfbdec(x: torch.Tensor, f1: torch.Tensor, f2: torch.Tensor, 
             mup: Union[torch.Tensor, None] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Two-channel nonsubsampled filter bank decomposition.
    
    Args:
        x: Input tensor
        f1: First filter
        f2: Second filter
        mup: Upsampling matrix (optional)
    
    Returns:
        Tuple of (y1, y2) filtered outputs
    """
    if mup is None:
        y1 = efilter2(x, f1)
        y2 = efilter2(x, f2)
    else:
        y1 = _convolve_upsampled(x, f1, mup, is_rec=False)
        y2 = _convolve_upsampled(x, f2, mup, is_rec=False)
    return y1, y2


def nssfbrec(x1: torch.Tensor, x2: torch.Tensor, 
             f1: torch.Tensor, f2: torch.Tensor,
             mup: Union[torch.Tensor, None] = None) -> torch.Tensor:
    """
    Two-channel nonsubsampled filter bank reconstruction.
    
    Args:
        x1: First input tensor
        x2: Second input tensor
        f1: First filter
        f2: Second filter
        mup: Upsampling matrix (optional)
    
    Returns:
        Reconstructed tensor
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


def nsfbdec(x: torch.Tensor, h0: torch.Tensor, h1: torch.Tensor, 
           lev: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Nonsubsampled filter bank decomposition at a given level.
    PyTorch translation of nsfbdec.m.
    
    Args:
        x: Input image at finer scale
        h0: Lowpass à trous filter
        h1: Highpass à trous filter
        lev: Decomposition level (0 for first level, >0 for subsequent levels)
    
    Returns:
        Tuple of (y0, y1):
            y0: Image at coarser scale (lowpass output)
            y1: Wavelet highpass output (bandpass output)
    """
    if lev != 0:
        # For levels > 0, use upsampled filters
        I2 = torch.eye(2, dtype=torch.int64, device=x.device)
        shift = [-2**(lev-1), -2**(lev-1)]
        shift = [s + 2 for s in shift]
        L = 2**lev
        
        # Upsample filters
        h0_up = upsample2df(h0, lev)
        h1_up = upsample2df(h1, lev)
        
        # Symmetric extension
        x_ext_h0 = symext(x, h0_up, shift)
        x_ext_h1 = symext(x, h1_up, shift)
        
        # Atrous convolution
        mup = I2 * L
        y0 = _atrousc_torch(x_ext_h0, h0, mup)
        y1 = _atrousc_torch(x_ext_h1, h1, mup)
    else:
        # First level (lev == 0)
        shift = [1, 1]
        
        # Symmetric extension
        x_ext_h0 = symext(x, h0, shift)
        x_ext_h1 = symext(x, h1, shift)
        
        # Regular convolution with 'valid' mode
        # Use PyTorch conv2d
        x_ext_h0_4d = x_ext_h0.unsqueeze(0).unsqueeze(0)
        x_ext_h1_4d = x_ext_h1.unsqueeze(0).unsqueeze(0)
        h0_4d = h0.unsqueeze(0).unsqueeze(0)
        h1_4d = h1.unsqueeze(0).unsqueeze(0)
        
        y0 = F.conv2d(x_ext_h0_4d, h0_4d, padding=0).squeeze()
        y1 = F.conv2d(x_ext_h1_4d, h1_4d, padding=0).squeeze()
    
    return y0, y1


def nsfbrec(y0: torch.Tensor, y1: torch.Tensor, 
           g0: torch.Tensor, g1: torch.Tensor, 
           lev: int) -> torch.Tensor:
    """
    Nonsubsampled filter bank reconstruction at a given level.
    PyTorch translation of nsfbrec.m.
    
    Args:
        y0: Lowpass image (coarse scale)
        y1: Highpass image (wavelet details)
        g0: Lowpass synthesis filter
        g1: Highpass synthesis filter
        lev: Reconstruction level
    
    Returns:
        Reconstructed image at finer scale
    """
    I2 = torch.eye(2, dtype=torch.int64, device=y0.device)
    
    if lev != 0:
        # Higher levels: use upsampled filters with atrous convolution
        shift = -2**(lev-1) * torch.tensor([1, 1], device=y0.device) + 2
        L = 2**lev
        
        # Upsample filters
        g0_up = upsample2df(g0, lev)
        g1_up = upsample2df(g1, lev)
        
        # Extend inputs
        y0_ext = symext(y0, g0_up, shift.tolist())
        y1_ext = symext(y1, g1_up, shift.tolist())
        
        # Apply atrous convolution and sum
        x = _atrousc_torch(y0_ext, g0, L * I2) + \
            _atrousc_torch(y1_ext, g1, L * I2)
    else:
        # Level 0: use regular convolution
        shift = [1, 1]
        
        # Extend inputs
        y0_ext = symext(y0, g0, shift)
        y1_ext = symext(y1, g1, shift)
        
        # Convolve and sum
        y0_ext_4d = y0_ext.unsqueeze(0).unsqueeze(0)
        y1_ext_4d = y1_ext.unsqueeze(0).unsqueeze(0)
        g0_4d = g0.unsqueeze(0).unsqueeze(0)
        g1_4d = g1.unsqueeze(0).unsqueeze(0)
        
        x = F.conv2d(y0_ext_4d, g0_4d, padding=0).squeeze() + \
            F.conv2d(y1_ext_4d, g1_4d, padding=0).squeeze()
    
    return x


def nsdfbdec(x: torch.Tensor, dfilter: Union[str, Dict], 
            clevels: int = 0, device: str = 'cpu') -> List[torch.Tensor]:
    """
    Nonsubsampled Directional Filter Bank (NSDFB) decomposition.
    PyTorch translation of nsdfbdec.m.
    
    Args:
        x: Input image (2D tensor)
        dfilter: Either str (filter name) or dict with filters
        clevels: Number of decomposition levels
        device: Device to use for tensor creation
    
    Returns:
        List of output subbands (length 2^clevels)
    """
    # Input validation
    if clevels != round(clevels) or clevels < 0:
        raise ValueError('Number of decomposition levels must be a non-negative integer')
    
    # No decomposition case
    if clevels == 0:
        return [x]
    
    # Get filters
    if isinstance(dfilter, str):
        h1, h2 = dfilters(dfilter, 'd', device=device)
        h1 = h1 / torch.sqrt(torch.tensor(2.0, device=device))
        h2 = h2 / torch.sqrt(torch.tensor(2.0, device=device))
        
        k1 = modulate2(h1, 'c')
        k2 = modulate2(h2, 'c')
        
        f1, f2 = parafilters(h1, h2)
    elif isinstance(dfilter, dict):
        if not all(key in dfilter for key in ['k1', 'k2', 'f1', 'f2']):
            raise ValueError("Filter dict must contain keys: 'k1', 'k2', 'f1', 'f2'")
        k1 = dfilter['k1']
        k2 = dfilter['k2']
        f1 = dfilter['f1']
        f2 = dfilter['f2']
    else:
        raise TypeError('dfilter must be a string or dict')
    
    # Quincunx sampling matrix
    q1 = torch.tensor([[1, -1], [1, 1]], dtype=torch.int64, device=device)
    
    # First-level decomposition
    if clevels == 1:
        y1, y2 = nssfbdec(x, k1, k2)
        return [y1, y2]
    
    # Multi-level decomposition (clevels >= 2)
    x1, x2 = nssfbdec(x, k1, k2)
    
    y: List[torch.Tensor] = [torch.empty(0)] * 4
    y[0], y[1] = nssfbdec(x1, k1, k2, q1)
    y[2], y[3] = nssfbdec(x2, k1, k2, q1)
    
    # Third and higher levels decomposition
    for l in range(3, clevels + 1):
        y_old = y
        y = [torch.empty(0)] * (2 ** l)
        
        # The first half channels
        for k in range(1, 2 ** (l - 2) + 1):
            slk = 2 * ((k - 1) // 2) - 2 ** (l - 3) + 1
            mkl = 2 * torch.tensor([[2 ** (l - 3), 0], [0, 1]], 
                                  dtype=torch.int64, device=device) @ \
                  torch.tensor([[1, 0], [-slk, 1]], dtype=torch.int64, device=device)
            
            i = ((k - 1) % 2)
            y[2 * k - 2], y[2 * k - 1] = nssfbdec(y_old[k - 1], f1[i], f2[i], mkl)
        
        # The second half channels
        for k in range(2 ** (l - 2) + 1, 2 ** (l - 1) + 1):
            slk = 2 * ((k - 2 ** (l - 2) - 1) // 2) - 2 ** (l - 3) + 1
            mkl = 2 * torch.tensor([[1, 0], [0, 2 ** (l - 3)]], 
                                  dtype=torch.int64, device=device) @ \
                  torch.tensor([[1, -slk], [0, 1]], dtype=torch.int64, device=device)
            
            i = ((k - 1) % 2) + 2
            y[2 * k - 2], y[2 * k - 1] = nssfbdec(y_old[k - 1], f1[i], f2[i], mkl)
    
    return y


def nsdfbrec(y: List[torch.Tensor], dfilter: Union[str, Dict],
            device: str = 'cpu') -> torch.Tensor:
    """
    Nonsubsampled directional filter bank reconstruction.
    PyTorch translation of nsdfbrec.m.
    
    Args:
        y: List of directional subbands (2^clevels subbands)
        dfilter: Filter specification (str or dict)
        device: Device to use
    
    Returns:
        Reconstructed image
    """
    # Check input
    if len(y) == 0:
        raise ValueError('Number of subbands must be a power of 2')
    
    # Determine clevels
    clevels = int(torch.log2(torch.tensor(len(y))).item())
    if 2**clevels != len(y):
        raise ValueError('Number of subbands must be a power of 2')
    
    # No reconstruction case
    if clevels == 0:
        return y[0]
    
    # Get filters (synthesis filters 'r')
    if isinstance(dfilter, str):
        h1, h2 = dfilters(dfilter, 'r', device=device)
        h1 = h1 / torch.sqrt(torch.tensor(2.0, device=device))
        h2 = h2 / torch.sqrt(torch.tensor(2.0, device=device))
        
        k1 = modulate2(h1, 'c')
        k2 = modulate2(h2, 'c')
        
        f1, f2 = parafilters(h1, h2)
    elif isinstance(dfilter, dict):
        if not all(key in dfilter for key in ['k1', 'k2', 'f1', 'f2']):
            raise ValueError("Filter dict must contain keys: 'k1', 'k2', 'f1', 'f2'")
        k1 = dfilter['k1']
        k2 = dfilter['k2']
        f1 = dfilter['f1']
        f2 = dfilter['f2']
    else:
        raise TypeError('dfilter must be a string or dict')
    
    # Quincunx sampling matrix
    q1 = torch.tensor([[1, -1], [1, 1]], dtype=torch.int64, device=device)
    
    # First-level reconstruction
    if clevels == 1:
        return nssfbrec(y[0], y[1], k1, k2)
    
    # Multi-level reconstruction
    x = y.copy()
    
    # Third and higher levels reconstructions (from highest to lowest)
    for l in range(clevels, 2, -1):
        # The first half channels
        for k in range(1, 2 ** (l - 2) + 1):
            slk = 2 * ((k - 1) // 2) - 2 ** (l - 3) + 1
            mkl = 2 * torch.tensor([[2 ** (l - 3), 0], [0, 1]], 
                                  dtype=torch.int64, device=device) @ \
                  torch.tensor([[1, 0], [-slk, 1]], dtype=torch.int64, device=device)
            
            i = ((k - 1) % 2)
            x[k - 1] = nssfbrec(x[2 * k - 2], x[2 * k - 1], f1[i], f2[i], mkl)
        
        # The second half channels
        for k in range(2 ** (l - 2) + 1, 2 ** (l - 1) + 1):
            slk = 2 * ((k - 2 ** (l - 2) - 1) // 2) - 2 ** (l - 3) + 1
            mkl = 2 * torch.tensor([[1, 0], [0, 2 ** (l - 3)]], 
                                  dtype=torch.int64, device=device) @ \
                  torch.tensor([[1, -slk], [0, 1]], dtype=torch.int64, device=device)
            
            i = ((k - 1) % 2) + 2
            x[k - 1] = nssfbrec(x[2 * k - 2], x[2 * k - 1], f1[i], f2[i], mkl)
    
    # Second-level reconstruction
    x[0] = nssfbrec(x[0], x[1], k1, k2, q1)
    x[1] = nssfbrec(x[2], x[3], k1, k2, q1)
    
    # First-level reconstruction
    result = nssfbrec(x[0], x[1], k1, k2)
    
    return result


def nsctdec(x: torch.Tensor, levels: List[int], dfilt: str = 'dmaxflat7',
           pfilt: str = 'maxflat', device: str = 'cpu') -> List:
    """
    Nonsubsampled Contourlet Transform Decomposition.
    PyTorch translation of nsctdec.m.
    
    Args:
        x: Input image (2D tensor)
        levels: List of directional decomposition levels at each pyramidal level
        dfilt: Filter name for directional decomposition
        pfilt: Filter name for pyramidal decomposition
        device: Device to use
    
    Returns:
        List where y[0] is lowpass and y[1:] are bandpass directional subbands
    """
    # Input validation
    levels_array = torch.tensor(levels, dtype=torch.int64)
    if not torch.all(levels_array == torch.round(levels_array.to(torch.float32))):
        raise ValueError('The decomposition levels shall be integers')
    
    if torch.any(levels_array < 0):
        raise ValueError('The decomposition levels shall be non-negative integers')
    
    # Get filters
    h1, h2 = dfilters(dfilt, 'd', device=device)
    h1 = h1 / torch.sqrt(torch.tensor(2.0, device=device))
    h2 = h2 / torch.sqrt(torch.tensor(2.0, device=device))
    
    k1 = modulate2(h1, 'c')
    k2 = modulate2(h2, 'c')
    
    f1, f2 = parafilters(h1, h2)
    
    filters = {
        'k1': k1,
        'k2': k2,
        'f1': f1,
        'f2': f2
    }
    
    # Get pyramid filters
    h1_pyr, h2_pyr, g1_pyr, g2_pyr = atrousfilters(pfilt, device=device)
    
    # Number of levels
    clevels = len(levels)
    nIndex = clevels
    
    # Initialize output
    y: List[Any] = [None] * (clevels + 1)
    
    # Nonsubsampled pyramid decomposition
    for i in range(clevels):
        xlo, xhi = nsfbdec(x, h1_pyr, h2_pyr, i)
        
        if levels[nIndex - 1] > 0:
            xhi_dir = nsdfbdec(xhi, filters, levels[nIndex - 1], device=device)
            y[nIndex] = xhi_dir
        else:
            y[nIndex] = xhi
        
        nIndex = nIndex - 1
        x = xlo
    
    # The lowpass output
    y[0] = x
    
    return y


def nsctrec(y: List, dfilt: str = 'dmaxflat7', pfilt: str = 'maxflat',
           device: str = 'cpu') -> torch.Tensor:
    """
    Nonsubsampled Contourlet Reconstruction.
    PyTorch translation of nsctrec.m.
    
    Args:
        y: List of subbands from nsctdec
        dfilt: Filter name for directional reconstruction
        pfilt: Filter name for pyramidal reconstruction
        device: Device to use
    
    Returns:
        Reconstructed image
    """
    # Get filters (synthesis filters 'r')
    h1_dir, h2_dir = dfilters(dfilt, 'r', device=device)
    h1_dir = h1_dir / torch.sqrt(torch.tensor(2.0, device=device))
    h2_dir = h2_dir / torch.sqrt(torch.tensor(2.0, device=device))
    
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
    h1_pyr, h2_pyr, g1_pyr, g2_pyr = atrousfilters(pfilt, device=device)
    
    # Number of pyramid levels
    n = len(y) - 1
    
    # Special case: no pyramid levels
    if n == 0:
        return y[0]
    
    # Start with lowpass
    xlo = y[0]
    nIndex = n - 1
    
    # Reconstruct from coarsest to finest level
    for i in range(n):
        if isinstance(y[i + 1], list):
            xhi = nsdfbrec(y[i + 1], filters, device=device)
        else:
            xhi = y[i + 1]
        
        x = nsfbrec(xlo, xhi, g1_pyr, g2_pyr, nIndex)
        xlo = x
        nIndex = nIndex - 1
    
    return x
