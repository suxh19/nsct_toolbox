import torch
import torch.nn.functional as F
from typing import Tuple, List, Optional
from nsct_torch.utils_torch import extend2, qupz, modulate2, resampz


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
    # Compute full convolution: convolve h with rot90(h0, 2)
    h_rot = torch.rot90(h0, 2)
    
    # For full convolution in PyTorch, we need to manually implement it
    # Result size should be h.shape[0] + h_rot.shape[0] - 1 for each dimension
    result_h = h.shape[0] + h_rot.shape[0] - 1
    result_w = h.shape[1] + h_rot.shape[1] - 1
    
    # Add batch and channel dimensions
    h_4d = h.unsqueeze(0).unsqueeze(0)
    h_rot_4d = h_rot.unsqueeze(0).unsqueeze(0)
    
    # Pad input to get 'full' convolution
    pad_h = h_rot.shape[0] - 1
    pad_w = h_rot.shape[1] - 1
    h_padded = F.pad(h_4d, (pad_w, pad_w, pad_h, pad_h), mode='constant', value=0)
    
    # Perform convolution
    h1 = -F.conv2d(h_padded, h_rot_4d).squeeze(0).squeeze(0)
    
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

    # The origin of filter f is assumed to be floor(size(f)/2) + 1.
    # Amount of shift should be no more than floor((size(f)-1)/2).
    sf = (torch.tensor(f.shape, dtype=torch.float64) - 1) / 2

    # Extend the image
    xext = extend2(x_float,
                   int(torch.floor(sf[0]).item()) + shift[0],
                   int(torch.ceil(sf[0]).item()) - shift[0],
                   int(torch.floor(sf[1]).item()) + shift[1],
                   int(torch.ceil(sf[1]).item()) - shift[1],
                   extmod)

    # Use F.conv2d for correlation (need to flip filter for convolution)
    # Add batch and channel dimensions
    xext_4d = xext.unsqueeze(0).unsqueeze(0)
    f_4d = f.unsqueeze(0).unsqueeze(0)
    
    # conv2d performs correlation when we don't flip the kernel
    y = F.conv2d(xext_4d, f_4d).squeeze(0).squeeze(0)

    return y


def dmaxflat(N: int, d: float = 0.0) -> torch.Tensor:
    """
    Returns 2-D diamond maxflat filters of order 'N'.
    PyTorch translation of dmaxflat.m.

    Args:
        N (int): Order of the filter, must be in {1, 2, ..., 7}.
        d (float): The (0,0) coefficient, being 1 or 0 depending on use.

    Returns:
        torch.Tensor: The 2D filter.
    """
    if not 1 <= N <= 7:
        raise ValueError('N must be in {1,2,3,4,5,6,7}')

    if N == 1:
        h = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=torch.float64) / 4.0
        h[1, 1] = d
    elif N == 2:
        h = torch.tensor([[0, -1, 0], [-1, 0, 10], [0, 10, 0]], dtype=torch.float64)
        h = torch.cat([h, torch.fliplr(h[:, :-1])], dim=1)
        h = torch.cat([h, torch.flipud(h[:-1, :])], dim=0) / 32.0
        h[2, 2] = d
    elif N == 3:
        h = torch.tensor([[0, 3, 0, 2],
                          [3, 0, -27, 0],
                          [0, -27, 0, 174],
                          [2, 0, 174, 0]], dtype=torch.float64)
        h = torch.cat([h, torch.fliplr(h[:, :-1])], dim=1)
        h = torch.cat([h, torch.flipud(h[:-1, :])], dim=0) / 512.0
        h[3, 3] = d
    elif N == 4:
        h = torch.tensor([[0, -5, 0, -3, 0],
                          [-5, 0, 52, 0, 34],
                          [0, 52, 0, -276, 0],
                          [-3, 0, -276, 0, 1454],
                          [0, 34, 0, 1454, 0]], dtype=torch.float64) / 2**12
        h = torch.cat([h, torch.fliplr(h[:, :-1])], dim=1)
        h = torch.cat([h, torch.flipud(h[:-1, :])], dim=0)
        h[4, 4] = d
    elif N == 5:
        h = torch.tensor([[0, 35, 0, 20, 0, 18],
                          [35, 0, -425, 0, -250, 0],
                          [0, -425, 0, 2500, 0, 1610],
                          [20, 0, 2500, 0, -10200, 0],
                          [0, -250, 0, -10200, 0, 47780],
                          [18, 0, 1610, 0, 47780, 0]], dtype=torch.float64) / 2**17
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
                          [0, 444, 0, 16460, 0, 389112, 0]], dtype=torch.float64) / 2**20
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
                          [100, 0, 13804, 0, 311780, 0, 6305740, 0]], dtype=torch.float64) / 2**24
        h = torch.cat([h, torch.fliplr(h[:, :-1])], dim=1)
        h = torch.cat([h, torch.flipud(h[:-1, :])], dim=0)
        h[7, 7] = d

    return h


def atrousfilters(fname: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate pyramid 2D filters for nonsubsampled filter banks.
    PyTorch translation of atrousfilters.m.

    Args:
        fname (str): Filter name. Supported: 'pyr', 'pyrexc', 'maxflat'.

    Returns:
        tuple: (h0, h1, g0, g1) pyramid filters.
    """
    if fname in ['pyr', 'pyrexc']:
        h0 = torch.tensor([
            [-0.003236043456039806, -0.012944173824159223, -0.019416260736238835],
            [-0.012944173824159223, 0.0625, 0.15088834764831843],
            [-0.019416260736238835, 0.15088834764831843, 0.3406092167691145]
        ], dtype=torch.float64)

        g0 = torch.tensor([
            [-0.00016755163599004882, -0.001005309815940293, -0.002513274539850732, -0.003351032719800976],
            [-0.001005309815940293, -0.005246663087920392, -0.01193886400821893, -0.015395021472477663],
            [-0.002513274539850732, -0.01193886400821893, 0.06769410071569153, 0.15423938036811946],
            [-0.003351032719800976, -0.015395021472477663, 0.15423938036811946, 0.3325667382415921]
        ], dtype=torch.float64)

        h1_g1_common = torch.tensor([
            [-0.003236043456039806, -0.012944173824159223, -0.019416260736238835],
            [-0.012944173824159223, -0.0625, -0.09911165235168155],
            [-0.019416260736238835, -0.09911165235168155, 0.8406092167691145]
        ], dtype=torch.float64)

        g1_h1_common = torch.tensor([
            [0.00016755163599004882, 0.001005309815940293, 0.002513274539850732, 0.003351032719800976],
            [0.001005309815940293, -0.0012254238241592198, -0.013949483640099517, -0.023437500000000007],
            [0.002513274539850732, -0.013949483640099517, -0.06769410071569153, -0.10246268507148255],
            [0.003351032719800976, -0.023437500000000007, -0.10246268507148255, 0.8486516952966369]
        ], dtype=torch.float64)

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
        # Quasi-tight frame filters
        h0 = torch.tensor([
            [-7.900496718847182e-07, 0., 0.000014220894093924927, 0.000025281589500310983, -0.000049773129328737247, -0.00022753430550279883, -0.00033182086219158167],
            [0, 0, 0, 0, 0, 0, 0],
            [0.000014220894093924927, 0., -0.0002559760936906487, -0.00045506861100559767, 0.0008959163279172705, 0.004095617499050379, 0.00597277551944847],
            [0.000025281589500310983, 0., -0.00045506861100559767, 0.0009765625, 0.0015927401385195919, -0.0087890625, -0.01795090623402861],
            [-0.000049773129328737247, 0., 0.0008959163279172705, 0.0015927401385195919, -0.0031357071477104465, -0.014334661246676327, -0.020904714318069645],
            [-0.00022753430550279883, 0., 0.004095617499050379, -0.0087890625, -0.014334661246676327, 0.0791015625, 0.16155815610625748],
            [-0.00033182086219158167, 0., 0.00597277551944847, -0.01795090623402861, -0.020904714318069645, 0.16155815610625748, 0.3177420190660832]
        ], dtype=torch.float64)
        
        g0 = torch.tensor([
            [-6.391587676622346e-010, 0., 1.7257286726880333e-08, 3.067962084778726e-08, -1.3805829381504267e-07, -5.522331752601707e-07, -3.3747582932565985e-07, 1.9328161134105974e-06, 5.6949046198705095e-06, 7.649452131381623e-06],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [1.7257286726880333e-08, 0., -4.65946741625769e-07, -8.283497628902559e-07, 3.727573933006152e-06, 0.000014910295732024608, 9.111847391792816e-06, -0.000052186035062086126, -0.00015376242473650378, -0.00020653520754730382],
            [3.067962084778726e-08, 0., -8.283497628902559e-07, -1.2809236054493144e-06, 6.6267981031220475e-06, 0.00002305662489808766, 0.000010064497559808503, -0.0000806981871433068, -0.00021814634152337594, -0.00028666046030363884],
            [-1.3805829381504267e-07, 0., 3.727573933006152e-06, 6.6267981031220475e-06, -0.000029820591464049215, -0.00011928236585619686, -0.00007289477913434253, 0.000417488280496689, 0.0012300993978920302, 0.0016522816603784306],
            [-5.522331752601707e-07, 0., 0.000014910295732024608, 0.00002305662489808766, -0.00011928236585619686, -0.00041501924816557786, -0.00018116095607655303, 0.0014525673685795225, 0.0039266341474207675, 0.005159888285465499],
            [-3.3747582932565985e-07, 0., 9.111847391792816e-06, 0.000010064497559808503, -0.00007289477913434253, -0.00018116095607655303, 0.001468581806076247, 0.0006340633462679356, -0.01181401175635013, -0.021745034491193898],
            [1.9328161134105974e-06, 0., -0.000052186035062086126, -0.0000806981871433068, 0.000417488280496689, 0.0014525673685795225, 0.0006340633462679356, -0.005083985790028328, -0.013743219515972684, -0.018059608999129246],
            [5.6949046198705095e-06, 0., -0.00015376242473650378, -0.00021814634152337594, 0.0012300993978920302, 0.0039266341474207675, -0.01181401175635013, -0.013743219515972684, 0.0826466923977296, 0.1638988884584603],
            [7.649452131381623e-06, 0., -0.00020653520754730382, -0.00028666046030363884, 0.0016522816603784306, 0.005159888285465499, -0.021745034491193898, -0.018059608999129246, 0.1638988884584603, 0.31358726209239235]
        ], dtype=torch.float64)
        
        g1 = torch.tensor([
            [-7.900496718847182e-07, 0., 0.000014220894093924927, 0.000025281589500310983, -0.000049773129328737247, -0.00022753430550279883, -0.00033182086219158167],
            [0, 0, 0, 0, 0, 0, 0],
            [0.000014220894093924927, 0., -0.0002559760936906487, -0.00045506861100559767, 0.0008959163279172705, 0.004095617499050379, 0.00597277551944847],
            [0.000025281589500310983, 0., -0.00045506861100559767, -0.0009765625, 0.0015927401385195919, 0.0087890625, 0.01329909376597139],
            [-0.000049773129328737247, 0., 0.0008959163279172705, 0.0015927401385195919, -0.0031357071477104465, -0.014334661246676327, -0.020904714318069645],
            [-0.00022753430550279883, 0., 0.004095617499050379, 0.0087890625, -0.014334661246676327, -0.0791015625, -0.1196918438937425],
            [-0.00033182086219158167, 0., 0.00597277551944847, 0.01329909376597139, -0.020904714318069645, -0.1196918438937425, 0.8177420190660831]
        ], dtype=torch.float64)
        
        h1 = torch.tensor([
            [6.391587676622346e-010, 0., -1.7257286726880333e-08, -3.067962084778726e-08, 1.3805829381504267e-07, 5.522331752601707e-07, 3.3747582932565985e-07, -1.9328161134105974e-06, -5.6949046198705095e-06, -7.649452131381623e-06],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [-1.7257286726880333e-08, 0., 4.65946741625769e-07, 8.283497628902559e-07, -3.727573933006152e-06, -0.000014910295732024608, -9.111847391792816e-06, 0.000052186035062086126, 0.00015376242473650378, 0.00020653520754730382],
            [-3.067962084778726e-08, 0., 8.283497628902559e-07, -2.9917573832012203e-07, -6.6267981031220475e-06, 5.3851632897621965e-06, 0.00004049868144081346, -0.00001884807151416769, -0.00023692226948222173, -0.0003769812640795245],
            [1.3805829381504267e-07, 0., -3.727573933006152e-06, -6.6267981031220475e-06, 0.000029820591464049215, 0.00011928236585619686, 0.00007289477913434253, -0.000417488280496689, -0.0012300993978920302, -0.0016522816603784306],
            [5.522331752601707e-07, 0., -0.000014910295732024608, 5.3851632897621965e-06, 0.00011928236585619686, -0.00009693293921571956, -0.0007289762659346422, 0.00033926528725501844, 0.004264600850679991, 0.006785662753431441],
            [3.3747582932565985e-07, 0., -9.111847391792816e-06, 0.00004049868144081346, 0.00007289477913434253, -0.0007289762659346422, -0.001468581806076247, 0.002551416930771248, 0.01181401175635013, 0.017093222023136675],
            [-1.9328161134105974e-06, 0., 0.000052186035062086126, -0.00001884807151416769, -0.000417488280496689, 0.00033926528725501844, 0.002551416930771248, -0.0011874285053925643, -0.01492610297737997, -0.023749819637010044],
            [-5.6949046198705095e-06, 0., 0.00015376242473650378, -0.00023692226948222173, -0.0012300993978920302, 0.004264600850679991, 0.01181401175635013, -0.01492610297737997, -0.0826466923977296, -0.12203257624594532],
            [-7.649452131381623e-06, 0., 0.00020653520754730382, -0.0003769812640795245, -0.0016522816603784306, 0.006785662753431441, 0.017093222023136675, -0.023749819637010044, -0.12203257624594532, 0.821896776039774]
        ], dtype=torch.float64)
        
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
        # Convolution using F.conv2d
        t_4d = t.unsqueeze(0).unsqueeze(0)
        P1_4d = P1.unsqueeze(0).unsqueeze(0)
        
        pad_h = t.shape[0] - 1
        pad_w = t.shape[1] - 1
        P1_padded = F.pad(P1_4d, (pad_w, pad_w, pad_h, pad_h), mode='constant', value=0)
        P2 = 2 * F.conv2d(P1_padded, t_4d).squeeze(0).squeeze(0)

        # Subtract P0 from the center of P2
        r_p2, c_p2 = P2.shape
        if P0.ndim == 2:
            r_p0, c_p0 = P0.shape
        else:
            r_p0, c_p0 = (1, 1)

        start_r = (r_p2 - r_p0) // 2
        start_c = (c_p2 - c_p0) // 2
        if P0.ndim == 2:
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


def ldfilter(fname: str) -> torch.Tensor:
    """
    Generate filter for the ladder structure network.
    PyTorch translation of ldfilter.m.

    Args:
        fname (str): Filter name. 'pkva', 'pkva12', 'pkva8', 'pkva6'.

    Returns:
        torch.Tensor: The 1D filter.
    """
    if fname in ['pkva12', 'pkva']:
        v = torch.tensor([0.6300, -0.1930, 0.0972, -0.0526, 0.0272, -0.0144], dtype=torch.float64)
    elif fname == 'pkva8':
        v = torch.tensor([0.6302, -0.1924, 0.0930, -0.0403], dtype=torch.float64)
    elif fname == 'pkva6':
        v = torch.tensor([0.6261, -0.1794, 0.0688], dtype=torch.float64)
    else:
        raise ValueError(f"Unrecognized ladder structure filter name: {fname}")

    # Symmetric impulse response
    return torch.cat([torch.flip(v, [0]), v])


def dfilters(fname: str, type: str = 'd') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate directional 2D filters (diamond filter pair).
    PyTorch translation of dfilters.m.

    Args:
        fname (str): Filter name.
        type (str): 'd' for decomposition, 'r' for reconstruction.

    Returns:
        tuple: (h0, h1) diamond filter pair (lowpass and highpass).
    """
    if fname in ['pkva', 'ldtest']:
        beta = ldfilter(fname)
        h0, h1 = ld2quin(beta)
        h0 *= torch.sqrt(torch.tensor(2.0, dtype=h0.dtype))
        h1 *= torch.sqrt(torch.tensor(2.0, dtype=h1.dtype))
        if type == 'r':
            f0 = modulate2(h1, 'b')
            f1 = modulate2(h0, 'b')
            h0, h1 = f0, f1

    elif 'dmaxflat' in fname:
        if fname == 'dmaxflat':
            raise ValueError("dmaxflat requires a number, e.g., 'dmaxflat7'")

        N = int(fname.replace('dmaxflat', ''))

        M1 = 1 / torch.sqrt(torch.tensor(2.0, dtype=torch.float64))
        k1 = 1 - torch.sqrt(torch.tensor(2.0, dtype=torch.float64))
        k3 = k1
        k2 = M1

        h = torch.tensor([0.25 * k2 * k3, 0.5 * k2, 1 + 0.5 * k2 * k3], dtype=torch.float64) * M1
        h = torch.cat([h, torch.flip(h[:-1], [0])])

        g = torch.tensor([-0.125*k1*k2*k3, 0.25*k1*k2, (-0.5*k1-0.5*k3-0.375*k1*k2*k3), 1 + 0.5*k1*k2], dtype=torch.float64) * M1
        g = torch.cat([g, torch.flip(g[:-1], [0])])

        B = dmaxflat(N, 0)
        h0 = mctrans(h, B)
        g0 = mctrans(g, B)

        h0 *= torch.sqrt(torch.tensor(2.0, dtype=h0.dtype)) / h0.sum()
        g0 *= torch.sqrt(torch.tensor(2.0, dtype=g0.dtype)) / g0.sum()

        h1 = modulate2(g0, 'b')
        if type == 'r':
            h1 = modulate2(h0, 'b')
            h0 = g0

    elif 'pkva-half' in fname:
        raise NotImplementedError("Filters 'pkva-half' are not implemented due to missing 'ldfilterhalf'")

    else:
        # Fallback to 1D wavelet filters
        # Note: PyTorch doesn't have built-in wavelet support, so we need to use numpy/pywt and convert
        import pywt
        import numpy as np
        
        try:
            wavelet = pywt.Wavelet(fname)  # type: ignore
            if type == 'd':
                h0 = torch.from_numpy(np.array(wavelet.dec_lo)).to(torch.float64)
                h1 = torch.from_numpy(np.array(wavelet.dec_hi)).to(torch.float64)
            else:  # 'r'
                h0 = torch.from_numpy(np.array(wavelet.rec_lo)).to(torch.float64)
                h1 = torch.from_numpy(np.array(wavelet.rec_hi)).to(torch.float64)
        except ValueError:
            raise ValueError(f"Unrecognized filter name: {fname}")

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
