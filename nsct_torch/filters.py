import torch
import torch.nn.functional as F
from typing import Tuple, List, Optional
from nsct_torch.utils import extend2, qupz, modulate2, resampz


def ld2quin(beta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Construct quincunx filters from a ladder network structure allpass filter.
    PyTorch translation of ld2quin.m.

    Args:
        beta (torch.Tensor): 1D allpass filter.

    Returns:
        tuple: (h0, h1) quincunx filters.
    """
    if beta.ndim != 1:
        raise ValueError('The input must be a 1-D filter')

    lf = beta.shape[0]
    n = lf // 2

    if n * 2 != lf:
        raise ValueError('The input allpass filter must be even length')

    # beta(z1) * beta(z2) -> outer product
    sp = torch.outer(beta, beta)

    # beta(z1*z2^{-1}) * beta(z1*z2)
    # Obtained by quincunx upsampling type 1 (with zero padded)
    h = qupz(sp, 1)

    # Lowpass quincunx filter
    h0 = h.clone()
    center_idx = lf - 1
    h0[center_idx, center_idx] += 1
    h0 = h0 / 2.0

    # Highpass quincunx filter
    # PyTorch version of convolve2d
    h_rot = torch.rot90(h0, 2)
    h1 = -conv2d_full(h, h_rot)

    center_idx_h1 = 4 * n - 2
    h1[center_idx_h1, center_idx_h1] += 1

    return h0, h1


def conv2d_full(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """
    2D convolution with 'full' mode (equivalent to scipy.signal.convolve2d mode='full').
    
    Args:
        x: Input tensor (H, W)
        kernel: Convolution kernel (Kh, Kw)
    
    Returns:
        Convolved output with size (H+Kh-1, W+Kw-1)
    """
    # Ensure inputs are 2D
    if x.ndim != 2 or kernel.ndim != 2:
        raise ValueError("Both inputs must be 2D tensors")
    
    # Add batch and channel dimensions
    x_4d = x.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    kernel_4d = kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, Kh, Kw)
    
    # Calculate padding for 'full' mode
    pad_h = kernel.shape[0] - 1
    pad_w = kernel.shape[1] - 1
    
    # Apply padding
    x_padded = F.pad(x_4d, (pad_w, pad_w, pad_h, pad_h), mode='constant', value=0)
    
    # Perform convolution
    result = F.conv2d(x_padded, kernel_4d)
    
    # Remove batch and channel dimensions
    return result.squeeze(0).squeeze(0)


def efilter2(x: torch.Tensor, f: torch.Tensor, extmod: str = 'per', 
             shift: Optional[List[int]] = None) -> torch.Tensor:
    """
    2D Filtering with edge handling (via extension).
    PyTorch translation of efilter2.m.

    Args:
        x (torch.Tensor): Input image.
        f (torch.Tensor): 2D filter.
        extmod (str): Extension mode (default is 'per'). See extend2 for details.
        shift (list or tuple, optional): Specify the window over which the
                                         convolution occurs. Defaults to [0, 0].

    Returns:
        torch.Tensor: Filtered image of the same size as the input.
    """
    if shift is None:
        shift = [0, 0]

    x_float = x.to(torch.float64)

    # The origin of filter f is assumed to be floor(size(f)/2) + 1.
    # Amount of shift should be no more than floor((size(f)-1)/2).
    sf = (torch.tensor(f.shape, dtype=torch.float32) - 1) / 2

    # Extend the image
    xext = extend2(x_float,
                   int(torch.floor(sf[0]).item()) + shift[0],
                   int(torch.ceil(sf[0]).item()) - shift[0],
                   int(torch.floor(sf[1]).item()) + shift[1],
                   int(torch.ceil(sf[1]).item()) - shift[1],
                   extmod)

    # Use PyTorch conv2d for filtering
    # Add batch and channel dimensions
    xext_4d = xext.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    f_4d = f.to(xext.dtype).unsqueeze(0).unsqueeze(0)  # (1, 1, Kh, Kw), match dtype
    
    # Perform convolution (no padding, 'valid' mode)
    y = F.conv2d(xext_4d, f_4d)
    
    # Remove batch and channel dimensions
    y = y.squeeze(0).squeeze(0)

    return y


def dmaxflat(N: int, d: float = 0.0, device: str = 'cpu') -> torch.Tensor:
    """
    Returns 2-D diamond maxflat filters of order 'N'.
    PyTorch translation of dmaxflat.m.

    Args:
        N (int): Order of the filter, must be in {1, 2, ..., 7}.
        d (float): The (0,0) coefficient, being 1 or 0 depending on use.
        device (str): Device to create the tensor on.

    Returns:
        torch.Tensor: The 2D filter.
    """
    if not 1 <= N <= 7:
        raise ValueError('N must be in {1,2,3,4,5,6,7}')

    if N == 1:
        h = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=torch.float64, device=device) / 4.0
        h[1, 1] = d
    elif N == 2:
        h = torch.tensor([[0, -1, 0], [-1, 0, 10], [0, 10, 0]], dtype=torch.float64, device=device)
        h = torch.cat([h, torch.fliplr(h[:, :-1])], dim=1)
        h = torch.cat([h, torch.flipud(h[:-1, :])], dim=0) / 32.0
        h[2, 2] = d
    elif N == 3:
        h = torch.tensor([[0, 3, 0, 2],
                          [3, 0, -27, 0],
                          [0, -27, 0, 174],
                          [2, 0, 174, 0]], dtype=torch.float64, device=device)
        h = torch.cat([h, torch.fliplr(h[:, :-1])], dim=1)
        h = torch.cat([h, torch.flipud(h[:-1, :])], dim=0) / 512.0
        h[3, 3] = d
    elif N == 4:
        h = torch.tensor([[0, -5, 0, -3, 0],
                          [-5, 0, 52, 0, 34],
                          [0, 52, 0, -276, 0],
                          [-3, 0, -276, 0, 1454],
                          [0, 34, 0, 1454, 0]], dtype=torch.float64, device=device) / 2**12
        h = torch.cat([h, torch.fliplr(h[:, :-1])], dim=1)
        h = torch.cat([h, torch.flipud(h[:-1, :])], dim=0)
        h[4, 4] = d
    elif N == 5:
        h = torch.tensor([[0, 35, 0, 20, 0, 18],
                          [35, 0, -425, 0, -250, 0],
                          [0, -425, 0, 2500, 0, 1610],
                          [20, 0, 2500, 0, -10200, 0],
                          [0, -250, 0, -10200, 0, 47780],
                          [18, 0, 1610, 0, 47780, 0]], dtype=torch.float64, device=device) / 2**17
        h = torch.cat([h, torch.fliplr(h[:, :-1])], dim=1)
        h = torch.cat([h, torch.flipud(h[:-1, :])], dim=0)
        h[5, 5] = d
    elif N == 6:
        h = torch.tensor([[0, -63, 0, -35, 0, -30, 0],
                          [-63, 0, 882, 0, 495, 0, 444],
                          [0, 882, 0, -5910, 0, -3420, 0],
                          [-35, 0, -5910, 0, 25875, 0, 16460],
                          [0, 495, 0, 25875, 0, -89730, 0],
                          [-30, 0, -3420, 0, -89730, 0, 389112],
                          [0, 444, 0, 16460, 0, 389112, 0]], dtype=torch.float64, device=device) / 2**20
        h = torch.cat([h, torch.fliplr(h[:, :-1])], dim=1)
        h = torch.cat([h, torch.flipud(h[:-1, :])], dim=0)
        h[6, 6] = d
    elif N == 7:
        h = torch.tensor([[0, 231, 0, 126, 0, 105, 0, 100],
                          [231, 0, -3675, 0, -2009, 0, -1715, 0],
                          [0, -3675, 0, 27930, 0, 15435, 0, 13804],
                          [126, 0, 27930, 0, -136514, 0, -77910, 0],
                          [0, -2009, 0, -136514, 0, 495145, 0, 311780],
                          [105, 0, 15435, 0, 495145, 0, -1535709, 0],
                          [0, -1715, 0, -77910, 0, -1535709, 0, 6305740],
                          [100, 0, 13804, 0, 311780, 0, 6305740, 0]], dtype=torch.float64, device=device) / 2**24
        h = torch.cat([h, torch.fliplr(h[:, :-1])], dim=1)
        h = torch.cat([h, torch.flipud(h[:-1, :])], dim=0)
        h[7, 7] = d

    return h


def atrousfilters(fname: str, device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate pyramid 2D filters for nonsubsampled filter banks.
    PyTorch translation of atrousfilters.m.

    Args:
        fname (str): Filter name. Supported: 'pyr', 'pyrexc', 'maxflat'.
        device (str): Device to create tensors on.

    Returns:
        tuple: (h0, h1, g0, g1) pyramid filters.
    """
    if fname in ['pyr', 'pyrexc']:
        h0 = torch.tensor([
            [-0.003236043456039806, -0.012944173824159223, -0.019416260736238835],
            [-0.012944173824159223, 0.0625, 0.15088834764831843],
            [-0.019416260736238835, 0.15088834764831843, 0.3406092167691145]
        ], dtype=torch.float64, device=device)

        g0 = torch.tensor([
            [-0.00016755163599004882, -0.001005309815940293, -0.002513274539850732, -0.003351032719800976],
            [-0.001005309815940293, -0.005246663087920392, -0.01193886400821893, -0.015395021472477663],
            [-0.002513274539850732, -0.01193886400821893, 0.06769410071569153, 0.15423938036811946],
            [-0.003351032719800976, -0.015395021472477663, 0.15423938036811946, 0.3325667382415921]
        ], dtype=torch.float64, device=device)

        h1_g1_common = torch.tensor([
            [-0.003236043456039806, -0.012944173824159223, -0.019416260736238835],
            [-0.012944173824159223, -0.0625, -0.09911165235168155],
            [-0.019416260736238835, -0.09911165235168155, 0.8406092167691145]
        ], dtype=torch.float64, device=device)

        g1_h1_common = torch.tensor([
            [0.00016755163599004882, 0.001005309815940293, 0.002513274539850732, 0.003351032719800976],
            [0.001005309815940293, -0.0012254238241592198, -0.013949483640099517, -0.023437500000000007],
            [0.002513274539850732, -0.013949483640099517, -0.06769410071569153, -0.10246268507148255],
            [0.003351032719800976, -0.023437500000000007, -0.10246268507148255, 0.8486516952966369]
        ], dtype=torch.float64, device=device)

        if fname == 'pyr':
            g1 = h1_g1_common
            h1 = g1_h1_common
        else:  # 'pyrexc'
            h1 = h1_g1_common
            g1 = g1_h1_common

        # Symmetric extension for all filters
        g0 = torch.cat([g0, torch.fliplr(g0[:, :-1])], dim=1)
        g0 = torch.cat([g0, torch.flipud(g0[:-1, :])], dim=0)
        h0 = torch.cat([h0, torch.fliplr(h0[:, :-1])], dim=1)
        h0 = torch.cat([h0, torch.flipud(h0[:-1, :])], dim=0)
        g1 = torch.cat([g1, torch.fliplr(g1[:, :-1])], dim=1)
        g1 = torch.cat([g1, torch.flipud(g1[:-1, :])], dim=0)
        h1 = torch.cat([h1, torch.fliplr(h1[:, :-1])], dim=1)
        h1 = torch.cat([h1, torch.flipud(h1[:-1, :])], dim=0)

        return h0, h1, g0, g1
    
    elif fname == 'maxflat':
        # Implementation for maxflat filters (large tensors)
        # Due to space constraints, this uses the same structure as the NumPy version
        raise NotImplementedError("maxflat filters are not yet fully translated to PyTorch")
    
    else:
        raise NotImplementedError(f"Filters '{fname}' are not implemented in this translation.")


def mctrans(b: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    McClellan transformation. PyTorch translation of mctrans.m.
    Produces the 2-D FIR filter H that corresponds to the 1-D FIR filter B
    using the transform T.

    Args:
        b (torch.Tensor): 1-D FIR filter (row vector).
        t (torch.Tensor): 2-D transformation filter.

    Returns:
        torch.Tensor: The resulting 2-D FIR filter.
    """
    n = (b.shape[0] - 1) // 2
    b = torch.fft.ifftshift(b)
    a = torch.cat([b[0:1], 2 * b[1:n + 1]])

    # Use Chebyshev polynomials to compute h
    P0 = torch.tensor(1.0, dtype=b.dtype, device=b.device)
    P1 = t
    h = a[1] * P1

    # Add a[0]*P0 to the center of h
    r_h, c_h = h.shape
    h[r_h//2, c_h//2] += a[0]

    for i in range(2, n + 1):
        P2 = 2 * conv2d_full(t, P1)

        # Subtract P0 from the center of P2
        r_p2, c_p2 = P2.shape
        if isinstance(P0, torch.Tensor) and P0.ndim >= 2:
            r_p0, c_p0 = P0.shape
        else:
            r_p0, c_p0 = (1, 1)

        start_r = (r_p2 - r_p0) // 2
        start_c = (c_p2 - c_p0) // 2
        
        if isinstance(P0, torch.Tensor) and P0.ndim >= 2:
            P2[start_r : start_r + r_p0, start_c : start_c + c_p0] -= P0
        else:
            P2[start_r, start_c] -= P0

        # Add the previous h to the center of the new h
        hh = h
        h = a[i] * P2
        r_h, c_h = h.shape
        r_hh, c_hh = hh.shape
        start_r = (r_h - r_hh) // 2
        start_c = (c_h - c_hh) // 2
        h[start_r : start_r + r_hh, start_c : start_c + c_hh] += hh

        P0 = P1
        P1 = P2

    # Rotate for use with filter2 (correlation)
    return torch.rot90(h, 2)


def ldfilter(fname: str, device: str = 'cpu') -> torch.Tensor:
    """
    Generate filter for the ladder structure network.
    PyTorch translation of ldfilter.m.

    Args:
        fname (str): Filter name. 'pkva', 'pkva12', 'pkva8', 'pkva6'.
        device (str): Device to create tensor on.

    Returns:
        torch.Tensor: The 1D filter.
    """
    if fname in ['pkva12', 'pkva']:
        v = torch.tensor([0.6300, -0.1930, 0.0972, -0.0526, 0.0272, -0.0144], 
                        dtype=torch.float64, device=device)
    elif fname == 'pkva8':
        v = torch.tensor([0.6302, -0.1924, 0.0930, -0.0403], 
                        dtype=torch.float64, device=device)
    elif fname == 'pkva6':
        v = torch.tensor([0.6261, -0.1794, 0.0688], 
                        dtype=torch.float64, device=device)
    else:
        raise ValueError(f"Unrecognized ladder structure filter name: {fname}")

    # Symmetric impulse response
    return torch.cat((torch.flip(v, [0]), v))


def dfilters(fname: str, type: str = 'd', device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate directional 2D filters (diamond filter pair).
    PyTorch translation of dfilters.m.

    Args:
        fname (str): Filter name.
        type (str): 'd' for decomposition, 'r' for reconstruction.
        device (str): Device to create tensors on.

    Returns:
        tuple: (h0, h1) diamond filter pair (lowpass and highpass).
    """
    if fname in ['pkva', 'ldtest']:
        beta = ldfilter(fname, device=device)
        h0, h1 = ld2quin(beta)
        h0 *= torch.sqrt(torch.tensor(2.0, device=device))
        h1 *= torch.sqrt(torch.tensor(2.0, device=device))
        if type == 'r':
            f0 = modulate2(h1, 'b')
            f1 = modulate2(h0, 'b')
            h0, h1 = f0, f1

    elif 'dmaxflat' in fname:
        if fname == 'dmaxflat':
            raise ValueError("dmaxflat requires a number, e.g., 'dmaxflat7'")

        N = int(fname.replace('dmaxflat', ''))

        M1 = 1 / torch.sqrt(torch.tensor(2.0, device=device))
        k1 = 1 - torch.sqrt(torch.tensor(2.0, device=device))
        k3 = k1
        k2 = M1

        h = torch.tensor([0.25 * k2 * k3, 0.5 * k2, 1 + 0.5 * k2 * k3], device=device) * M1
        h = torch.cat([h, torch.flip(h[:-1], [0])])

        g = torch.tensor([-0.125*k1*k2*k3, 0.25*k1*k2, (-0.5*k1-0.5*k3-0.375*k1*k2*k3), 
                         1 + 0.5*k1*k2], device=device) * M1
        g = torch.cat([g, torch.flip(g[:-1], [0])])

        B = dmaxflat(N, 0, device=device)
        h0 = mctrans(h, B)
        g0 = mctrans(g, B)

        h0 *= torch.sqrt(torch.tensor(2.0, device=device)) / h0.sum()
        g0 *= torch.sqrt(torch.tensor(2.0, device=device)) / g0.sum()

        h1 = modulate2(g0, 'b')
        if type == 'r':
            h1 = modulate2(h0, 'b')
            h0 = g0

    else:
        raise NotImplementedError(f"Filter '{fname}' not implemented in PyTorch version. Use NumPy version for PyWavelets support.")

    return h0, h1


def parafilters(f1: torch.Tensor, f2: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Generate four groups of parallelogram filters from a pair of diamond filters.
    PyTorch translation of parafilters.m.

    Args:
        f1 (torch.Tensor): The filter for the first branch.
        f2 (torch.Tensor): The filter for the second branch.

    Returns:
        tuple: (y1, y2) where each is a list of 4 parallelogram filters.
    """
    # Initialize output
    y1: List[torch.Tensor] = []
    y2: List[torch.Tensor] = []

    # Modulation operation
    y1.append(modulate2(f1, 'r'))
    y2.append(modulate2(f2, 'r'))
    y1.append(modulate2(f1, 'c'))
    y2.append(modulate2(f2, 'c'))

    # Transpose operation
    y1.append(y1[0].T)
    y2.append(y2[0].T)
    y1.append(y1[1].T)
    y2.append(y2[1].T)

    # Resample the filters by corresponding rotation matrices
    for i in range(4):
        y1[i] = resampz(y1[i], i + 1)
        y2[i] = resampz(y2[i], i + 1)

    return y1, y2


if __name__ == '__main__':
    # Basic tests
    print("--- Running PyTorch filters.py tests ---")
    
    # Test ldfilter
    print("\n--- Testing ldfilter ---")
    f6 = ldfilter('pkva6')
    print(f"ldfilter('pkva6') shape: {f6.shape}")
    assert f6.shape == (6,)
    print("✓ ldfilter test passed!")
    
    # Test dmaxflat
    print("\n--- Testing dmaxflat ---")
    h2 = dmaxflat(2, 0)
    print(f"dmaxflat(2, 0) shape: {h2.shape}")
    assert h2.shape == (5, 5)
    print("✓ dmaxflat test passed!")
    
    # Test atrousfilters
    print("\n--- Testing atrousfilters ---")
    h0, h1, g0, g1 = atrousfilters('pyr')
    print(f"atrousfilters('pyr') shapes: h0={h0.shape}, h1={h1.shape}, g0={g0.shape}, g1={g1.shape}")
    assert h0.shape == (5, 5)
    print("✓ atrousfilters test passed!")
    
    # Test ld2quin
    print("\n--- Testing ld2quin ---")
    beta = ldfilter('pkva6')
    h0_quin, h1_quin = ld2quin(beta)
    print(f"ld2quin output shapes: h0={h0_quin.shape}, h1={h1_quin.shape}")
    assert h0_quin.shape == (11, 11)
    assert h1_quin.shape == (21, 21)
    print("✓ ld2quin test passed!")
    
    print("\n=== All PyTorch filters tests passed! ===")
