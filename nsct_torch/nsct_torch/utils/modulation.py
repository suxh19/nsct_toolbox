"""
Modulation operations for NSCT
包含2D调制函数
"""

import torch
from typing import Optional, List


def modulate2(x: torch.Tensor, mode: str = 'b', center: Optional[List[int]] = None) -> torch.Tensor:
    """
    2D modulation. PyTorch translation of modulate2.m.

    Args:
        x: Input matrix tensor.
        mode: 'r' for row, 'c' for column, 'b' for both.
        center: Modulation center offset. Defaults to None.

    Returns:
        Modulated matrix tensor.
    """
    s = x.shape
    if center is None:
        center = [0, 0]

    o = torch.floor(torch.tensor(s, dtype=torch.float64) / 2) + 1 + torch.tensor(center, dtype=torch.float64)

    n1 = torch.arange(1, s[0] + 1, dtype=torch.float64, device=x.device) - o[0]
    n2 = torch.arange(1, s[1] + 1, dtype=torch.float64, device=x.device) - o[1]

    y = x.to(dtype=torch.float64).clone()
    
    if mode in ['r', 'b']:
        m1 = (-1) ** n1
        y *= m1.unsqueeze(1)

    if mode in ['c', 'b']:
        m2 = (-1) ** n2
        y *= m2.unsqueeze(0)

    return y
