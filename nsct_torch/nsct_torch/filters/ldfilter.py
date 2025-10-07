"""
Generate filter for the ladder structure network.
PyTorch translation of ldfilter.m
"""

import torch


def ldfilter(fname: str, device: str = 'cpu') -> torch.Tensor:
    """
    Generate filter for the ladder structure network.
    
    Args:
        fname: Filter name. 'pkva', 'pkva12', 'pkva8', 'pkva6'.
        device: Device to create tensor on ('cpu' or 'cuda').
    
    Returns:
        The 1D filter tensor.
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
    return torch.cat([torch.flip(v, dims=[0]), v])
