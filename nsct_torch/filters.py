import torch
import torch.nn.functional as F
from typing import Tuple, List, Optional
import pywt
import math

# Import the translated utility functions
from nsct_torch.utils import extend2, qupz, modulate2, resampz

def _conv2d(x: torch.Tensor, k: torch.Tensor, mode: str) -> torch.Tensor:
    """Helper for 2D convolution, matching scipy.signal.convolve2d behavior."""
    x = x.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
    k = k.unsqueeze(0).unsqueeze(0)  # Add out_channel and in_channel dims

    # F.conv2d performs cross-correlation. To perform convolution, we flip the kernel.
    k_flipped = torch.rot90(k, 2, [2, 3])

    padding = 0
    if mode == 'full':
        # Pad to get output size of (Hin + Hk - 1, Win + Wk - 1)
        padding = (k.shape[2] - 1, k.shape[3] - 1)
    elif mode == 'same':
        # Pad to get output size equal to input size
        # This is more complex to replicate exactly, but 'valid' and 'full' are what's needed.
        raise NotImplementedError("Mode 'same' is not implemented for _conv2d helper.")

    result = F.conv2d(x, k_flipped, padding=padding)
    return result.squeeze(0).squeeze(0)


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
    # In original: -convolve2d(h, np.rot90(h0, 2), 'full')
    # convolve(a, rot(b)) is equivalent to correlate(a, b).
    # F.conv2d is correlation, so we pass h0 directly.
    h1 = -_conv2d(h, torch.rot90(h0, 2), 'full')

    center_idx_h1 = 4 * n - 2
    h1[center_idx_h1, center_idx_h1] += 1

    return h0, h1


def efilter2(x: torch.Tensor, f: torch.Tensor, extmod: str = 'per', shift: Optional[List[int]] = None) -> torch.Tensor:
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

    sf = (torch.tensor(f.shape, device=x.device) - 1) / 2

    # Extend the image
    xext = extend2(x_float,
                   int(torch.floor(sf[0]) + shift[0]),
                   int(torch.ceil(sf[0]) - shift[0]),
                   int(torch.floor(sf[1]) + shift[1]),
                   int(torch.ceil(sf[1]) - shift[1]),
                   extmod)

    # Perform 'valid' convolution.
    return _conv2d(xext, f, 'valid')


def dmaxflat(N: int, d: float = 0.0, device='cpu') -> torch.Tensor:
    """
    Returns 2-D diamond maxflat filters of order 'N'.
    PyTorch translation of dmaxflat.m.

    Args:
        N (int): Order of the filter, must be in {1, 2, ..., 7}.
        d (float): The (0,0) coefficient, being 1 or 0 depending on use.
        device (str): The torch device to create tensors on.

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
    else:
        raise NotImplementedError(f"dmaxflat for N={N} is not implemented in this translation.")

    return h


def atrousfilters(fname: str, device='cpu') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate pyramid 2D filters for nonsubsampled filter banks.
    PyTorch translation of atrousfilters.m.
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
        else: # 'pyrexc'
            h1 = h1_g1_common
            g1 = g1_h1_common

        g0 = torch.cat([g0, torch.fliplr(g0[:, :-1])], dim=1)
        g0 = torch.cat([g0, torch.flipud(g0[:-1, :])], dim=0)
        h0 = torch.cat([h0, torch.fliplr(h0[:, :-1])], dim=1)
        h0 = torch.cat([h0, torch.flipud(h0[:-1, :])], dim=0)
        g1 = torch.cat([g1, torch.fliplr(g1[:, :-1])], dim=1)
        g1 = torch.cat([g1, torch.flipud(g1[:-1, :])], dim=0)
        h1 = torch.cat([h1, torch.fliplr(h1[:, :-1])], dim=1)
        h1 = torch.cat([h1, torch.flipud(h1[:-1, :])], dim=0)

        return h0, h1, g0, g1

    else:
        raise NotImplementedError(f"Filters '{fname}' are not implemented in this translation.")


def mctrans(b: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    McClellan transformation. PyTorch translation of mctrans.m.
    """
    n = (b.shape[0] - 1) // 2
    b = torch.fft.ifftshift(b)
    # Ensure the first element is a tensor for concatenation
    a = torch.cat((b[0].unsqueeze(0), 2 * b[1:n + 1]))

    # Use Chebyshev polynomials to compute h
    P0 = torch.tensor(1.0, dtype=t.dtype, device=t.device)
    P1 = t
    h = a[1] * P1

    # Add a[0]*P0 to the center of h
    r_h, c_h = h.shape
    h[r_h//2, c_h//2] += a[0]

    for i in range(2, n + 1):
        P2 = 2 * _conv2d(t, P1, 'full')

        # Subtract P0 from the center of P2
        r_p2, c_p2 = P2.shape
        if P0.ndim > 0:
            r_p0, c_p0 = P0.shape
        else:
            r_p0, c_p0 = (1, 1)

        start_r = (r_p2 - r_p0) // 2
        start_c = (c_p2 - c_p0) // 2
        P2[start_r : start_r + r_p0, start_c : start_c + c_p0] -= P0

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
    return torch.rot90(h, 2, [0, 1])


def ldfilter(fname: str, device='cpu') -> torch.Tensor:
    """
    Generate filter for the ladder structure network.
    PyTorch translation of ldfilter.m.
    """
    if fname in ['pkva12', 'pkva']:
        v = torch.tensor([0.6300, -0.1930, 0.0972, -0.0526, 0.0272, -0.0144], dtype=torch.float64, device=device)
    elif fname == 'pkva8':
        v = torch.tensor([0.6302, -0.1924, 0.0930, -0.0403], dtype=torch.float64, device=device)
    elif fname == 'pkva6':
        v = torch.tensor([0.6261, -0.1794, 0.0688], dtype=torch.float64, device=device)
    else:
        raise ValueError(f"Unrecognized ladder structure filter name: {fname}")

    return torch.cat((torch.flip(v, [0]), v))


def dfilters(fname: str, type: str = 'd', device='cpu') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate directional 2D filters (diamond filter pair).
    PyTorch translation of dfilters.m.
    """
    if fname in ['pkva', 'ldtest']:
        beta = ldfilter(fname, device=device)
        h0, h1 = ld2quin(beta)
        h0 *= math.sqrt(2)
        h1 *= math.sqrt(2)
        if type == 'r':
            f0 = modulate2(h1, 'b')
            f1 = modulate2(h0, 'b')
            h0, h1 = f0, f1

    elif 'dmaxflat' in fname:
        if fname == 'dmaxflat':
            raise ValueError("dmaxflat requires a number, e.g., 'dmaxflat7'")

        N = int(fname.replace('dmaxflat', ''))

        M1 = 1 / math.sqrt(2)
        k1 = 1 - math.sqrt(2)
        k3 = k1
        k2 = M1

        h = torch.tensor([0.25 * k2 * k3, 0.5 * k2, 1 + 0.5 * k2 * k3], dtype=torch.float64, device=device) * M1
        h = torch.cat([h, torch.flip(h[:-1], [0])])

        g = torch.tensor([-0.125*k1*k2*k3, 0.25*k1*k2, (-0.5*k1-0.5*k3-0.375*k1*k2*k3), 1 + 0.5*k1*k2], dtype=torch.float64, device=device) * M1
        g = torch.cat([g, torch.flip(g[:-1], [0])])

        B = dmaxflat(N, 0, device=device)
        h0 = mctrans(h, B)
        g0 = mctrans(g, B)

        h0 *= math.sqrt(2) / h0.sum()
        g0 *= math.sqrt(2) / g0.sum()

        h1 = modulate2(g0, 'b')
        if type == 'r':
            h1 = modulate2(h0, 'b')
            h0 = g0

    elif 'pkva-half' in fname:
        raise NotImplementedError("Filters 'pkva-half' are not implemented due to missing 'ldfilterhalf'")

    else:
        try:
            wavelet = pywt.Wavelet(fname)  # type: ignore
            if type == 'd':
                h0 = torch.tensor(wavelet.dec_lo, dtype=torch.float64, device=device)
                h1 = torch.tensor(wavelet.dec_hi, dtype=torch.float64, device=device)
            else: # 'r'
                h0 = torch.tensor(wavelet.rec_lo, dtype=torch.float64, device=device)
                h1 = torch.tensor(wavelet.rec_hi, dtype=torch.float64, device=device)
        except ValueError:
            raise ValueError(f"Unrecognized filter name: {fname}")

    return h0, h1


def parafilters(f1: torch.Tensor, f2: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Generate four groups of parallelogram filters from a pair of diamond filters.
    PyTorch translation of parafilters.m.
    """
    y1: List[torch.Tensor] = []
    y2: List[torch.Tensor] = []

    y1.append(modulate2(f1, 'r'))
    y2.append(modulate2(f2, 'r'))
    y1.append(modulate2(f1, 'c'))
    y2.append(modulate2(f2, 'c'))

    y1.append(y1[0].T)
    y2.append(y2[0].T)
    y1.append(y1[1].T)
    y2.append(y2[1].T)

    for i in range(4):
        y1[i] = resampz(y1[i], i + 1)
        y2[i] = resampz(y2[i], i + 1)

    return y1, y2