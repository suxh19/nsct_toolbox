"""
Generate directional 2D filters (diamond filter pair).
PyTorch translation of dfilters.m
"""

import torch
from typing import Tuple
from nsct_torch.utils import modulate2
from nsct_torch.filters.ldfilter import ldfilter
from nsct_torch.filters.ld2quin import ld2quin
from nsct_torch.filters.dmaxflat import dmaxflat
from nsct_torch.filters.mctrans import mctrans


def dfilters(fname: str, type: str = 'd', device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate directional 2D filters (diamond filter pair).
    
    Args:
        fname: Filter name.
        type: 'd' for decomposition, 'r' for reconstruction.
        device: Device to create tensors on ('cpu' or 'cuda').
    
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
        
        h = torch.tensor([0.25 * k2.item() * k3.item(), 
                         0.5 * k2.item(), 
                         1 + 0.5 * k2.item() * k3.item()], 
                        dtype=torch.float64, device=device) * M1
        h = torch.cat([h, torch.flip(h[:-1], dims=[0])])
        
        g = torch.tensor([-0.125 * k1.item() * k2.item() * k3.item(), 
                         0.25 * k1.item() * k2.item(), 
                         (-0.5 * k1.item() - 0.5 * k3.item() - 0.375 * k1.item() * k2.item() * k3.item()), 
                         1 + 0.5 * k1.item() * k2.item()], 
                        dtype=torch.float64, device=device) * M1
        g = torch.cat([g, torch.flip(g[:-1], dims=[0])])
        
        B = dmaxflat(N, 0, device=device)
        h0 = mctrans(h, B)
        g0 = mctrans(g, B)
        
        h0 *= torch.sqrt(torch.tensor(2.0, device=device)) / h0.sum()
        g0 *= torch.sqrt(torch.tensor(2.0, device=device)) / g0.sum()
        
        h1 = modulate2(g0, 'b')
        if type == 'r':
            h1 = modulate2(h0, 'b')
            h0 = g0
    
    elif 'pkva-half' in fname:
        raise NotImplementedError("Filters 'pkva-half' are not implemented due to missing 'ldfilterhalf'")
    
    else:
        # Fallback to 1D wavelet filters from PyWavelets
        try:
            import pywt
            wavelet = pywt.Wavelet(fname)
            if type == 'd':
                h0 = torch.tensor(wavelet.dec_lo, dtype=torch.float64, device=device)
                h1 = torch.tensor(wavelet.dec_hi, dtype=torch.float64, device=device)
            else:  # 'r'
                h0 = torch.tensor(wavelet.rec_lo, dtype=torch.float64, device=device)
                h1 = torch.tensor(wavelet.rec_hi, dtype=torch.float64, device=device)
        except (ValueError, ImportError) as e:
            raise ValueError(f"Unrecognized filter name: {fname}. Error: {e}")
    
    return h0, h1
