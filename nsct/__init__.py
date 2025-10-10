"""
NSCT - Batched PyTorch implementation of the Nonsubsampled Contourlet Transform.

The refactored package reuses CUDA kernels from the legacy `nsct_torch` module
but extends them to operate on batched tensors. High-level helpers for batch
decomposition and reconstruction are exposed via :mod:`nsct.api`.
"""

from __future__ import annotations

from .api import nsct_batch_dec, nsct_batch_rec

__all__ = ["nsct_batch_dec", "nsct_batch_rec"]
