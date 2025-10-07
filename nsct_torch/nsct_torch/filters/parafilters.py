"""
Generate four groups of parallelogram filters from a pair of diamond filters.
PyTorch translation of parafilters.m
"""

import torch
from typing import List, Tuple
from nsct_torch.utils import modulate2, resampz


def parafilters(f1: torch.Tensor, f2: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Generate four groups of parallelogram filters from a pair of diamond filters.
    
    Args:
        f1: The filter for the first branch tensor.
        f2: The filter for the second branch tensor.
    
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
