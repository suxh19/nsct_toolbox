"""
High-level batched API for the NSCT package.

These helpers reshape 4D tensors of shape ``[B, C, H, W]`` into the flattened
form ``[B * C, H, W]`` consumed by the core NSCT implementation, invoke the
transform, and reshape the results back so the caller retains batch and
channel semantics.
"""

from __future__ import annotations

from typing import Any, Iterable, List, Sequence, Tuple

import torch

from . import core

CoeffTree = Sequence[Any]


def _flatten_batch(x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
    if x.dim() != 4:
        raise ValueError(f"Expected input of shape [B, C, H, W], got {tuple(x.shape)}")
    b, c, h, w = x.shape
    return x.reshape(b * c, h, w), b, c


def _reshape_coeff_tree(tree: CoeffTree, batch: int, channels: int) -> List[Any]:
    def reshape_tensor(t: torch.Tensor) -> torch.Tensor:
        if t.dim() < 3:
            raise ValueError("Coefficient tensors must have at least 3 dimensions")
        spatial = t.shape[1:]
        return t.reshape(batch, channels, *spatial)

    def recurse(node: Any) -> Any:
        if isinstance(node, torch.Tensor):
            return reshape_tensor(node)
        if isinstance(node, Iterable) and not isinstance(node, (torch.Tensor, bytes, str)):
            return [recurse(child) for child in node]
        return node

    return [recurse(item) for item in tree]


def _extract_batch_channel(coeffs: Sequence[Any]) -> Tuple[int, int]:
    if not coeffs:
        raise ValueError("Coefficient list must not be empty")
    first = coeffs[0]
    if not isinstance(first, torch.Tensor) or first.dim() < 3:
        raise ValueError("Low-pass coefficient must be a tensor with shape [B, C, ...]")
    return first.shape[0], first.shape[1]


def _flatten_coeff_tree(tree: Sequence[Any], batch: int, channels: int) -> List[Any]:
    def flatten_tensor(t: torch.Tensor) -> torch.Tensor:
        if t.shape[0] != batch or t.shape[1] != channels:
            raise ValueError("All coefficient tensors must share the same batch/channel sizes")
        trailing = t.shape[2:]
        return t.reshape(batch * channels, *trailing)

    def recurse(node: Any) -> Any:
        if isinstance(node, torch.Tensor):
            return flatten_tensor(node)
        if isinstance(node, Iterable) and not isinstance(node, (torch.Tensor, bytes, str)):
            return [recurse(child) for child in node]
        return node

    return [recurse(item) for item in tree]


def nsct_batch_dec(
    x: torch.Tensor,
    nlevs: Sequence[int],
    dfilt: str = "dmaxflat7",
    pfilt: str = "maxflat",
    dtype: torch.dtype | None = None,
) -> List[Any]:
    """
    Batched NSCT decomposition.

    Args:
        x: Input tensor with shape ``[B, C, H, W]`` on a CUDA device.
        nlevs: Directional decomposition levels per pyramid stage (see ``nsct.core.nsctdec``).
        dfilt: Directional filter identifier.
        pfilt: Pyramid filter identifier.
        dtype: Optional computation dtype override.
    """
    x_flat, batch, channels = _flatten_batch(x)
    coeffs = core.nsctdec(x_flat, nlevs, dfilt=dfilt, pfilt=pfilt, dtype=dtype)
    return _reshape_coeff_tree(coeffs, batch, channels)


def nsct_batch_rec(
    coeffs: Sequence[Any],
    dfilt: str = "dmaxflat7",
    pfilt: str = "maxflat",
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """
    Batched NSCT reconstruction.

    Args:
        coeffs: Coefficient tree produced by :func:`nsct_batch_dec`.
        dfilt: Directional filter identifier.
        pfilt: Pyramid filter identifier.
        dtype: Optional computation dtype override.
    """
    batch, channels = _extract_batch_channel(coeffs)
    coeffs_flat = _flatten_coeff_tree(coeffs, batch, channels)
    recon = core.nsctrec(coeffs_flat, dfilt=dfilt, pfilt=pfilt, dtype=dtype)
    spatial = recon.shape[1:]
    return recon.reshape(batch, channels, *spatial)


__all__ = ["nsct_batch_dec", "nsct_batch_rec"]
