"""
Returns 2-D diamond maxflat filters of order 'N'.
PyTorch translation of dmaxflat.m
"""

import torch


def dmaxflat(N: int, d: float = 0.0, device: str = 'cpu') -> torch.Tensor:
    """
    Returns 2-D diamond maxflat filters of order 'N'.
    
    Args:
        N: Order of the filter, must be in {1, 2, ..., 7}.
        d: The (0,0) coefficient, being 1 or 0 depending on use.
        device: Device to create tensor on ('cpu' or 'cuda').
    
    Returns:
        The 2D filter tensor.
    """
    if not 1 <= N <= 7:
        raise ValueError('N must be in {1,2,3,4,5,6,7}')
    
    if N == 1:
        h = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], 
                        dtype=torch.float64, device=device) / 4.0
        h[1, 1] = d
    elif N == 2:
        h = torch.tensor([[0, -1, 0], [-1, 0, 10], [0, 10, 0]], 
                        dtype=torch.float64, device=device)
        h = torch.cat([h, torch.flip(h[:, :-1], dims=[1])], dim=1)
        h = torch.cat([h, torch.flip(h[:-1, :], dims=[0])], dim=0) / 32.0
        h[2, 2] = d
    elif N == 3:
        h = torch.tensor([[0, 3, 0, 2],
                         [3, 0, -27, 0],
                         [0, -27, 0, 174],
                         [2, 0, 174, 0]], 
                        dtype=torch.float64, device=device)
        h = torch.cat([h, torch.flip(h[:, :-1], dims=[1])], dim=1)
        h = torch.cat([h, torch.flip(h[:-1, :], dims=[0])], dim=0) / 512.0
        h[3, 3] = d
    elif N == 4:
        h = torch.tensor([[0, -5, 0, -3, 0],
                         [-5, 0, 52, 0, 34],
                         [0, 52, 0, -276, 0],
                         [-3, 0, -276, 0, 1454],
                         [0, 34, 0, 1454, 0]], 
                        dtype=torch.float64, device=device) / 2**12
        h = torch.cat([h, torch.flip(h[:, :-1], dims=[1])], dim=1)
        h = torch.cat([h, torch.flip(h[:-1, :], dims=[0])], dim=0)
        h[4, 4] = d
    elif N == 5:
        h = torch.tensor([[0, 35, 0, 20, 0, 18],
                         [35, 0, -425, 0, -250, 0],
                         [0, -425, 0, 2500, 0, 1610],
                         [20, 0, 2500, 0, -10200, 0],
                         [0, -250, 0, -10200, 0, 47780],
                         [18, 0, 1610, 0, 47780, 0]], 
                        dtype=torch.float64, device=device) / 2**17
        h = torch.cat([h, torch.flip(h[:, :-1], dims=[1])], dim=1)
        h = torch.cat([h, torch.flip(h[:-1, :], dims=[0])], dim=0)
        h[5, 5] = d
    elif N == 6:
        h = torch.tensor([[0, -63, 0, -35, 0, -30, 0],
                         [-63, 0, 882, 0, 495, 0, 444],
                         [0, 882, 0, -5910, 0, -3420, 0],
                         [-35, 0, -5910, 0, 25875, 0, 16460],
                         [0, 495, 0, 25875, 0, -89730, 0],
                         [-30, 0, -3420, 0, -89730, 0, 389112],
                         [0, 444, 0, 16460, 0, 389112, 0]], 
                        dtype=torch.float64, device=device) / 2**20
        h = torch.cat([h, torch.flip(h[:, :-1], dims=[1])], dim=1)
        h = torch.cat([h, torch.flip(h[:-1, :], dims=[0])], dim=0)
        h[6, 6] = d
    elif N == 7:
        h = torch.tensor([[0, 231, 0, 126, 0, 105, 0, 100],
                         [231, 0, -3675, 0, -2009, 0, -1715, 0],
                         [0, -3675, 0, 27930, 0, 15435, 0, 13804],
                         [126, 0, 27930, 0, -136514, 0, -77910, 0],
                         [0, -2009, 0, -136514, 0, 495145, 0, 311780],
                         [105, 0, 15435, 0, 495145, 0, -1535709, 0],
                         [0, -1715, 0, -77910, 0, -1535709, 0, 6305740],
                         [100, 0, 13804, 0, 311780, 0, 6305740, 0]], 
                        dtype=torch.float64, device=device) / 2**24
        h = torch.cat([h, torch.flip(h[:, :-1], dims=[1])], dim=1)
        h = torch.cat([h, torch.flip(h[:-1, :], dims=[0])], dim=0)
        h[7, 7] = d
    
    return h
