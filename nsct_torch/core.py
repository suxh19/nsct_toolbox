import torch
import torch.nn.functional as F
from typing import Union, Tuple, Any

from nsct_torch.filters import efilter2
from nsct_torch.utils import extend2

def _upsample_and_find_origin(f: torch.Tensor, mup: Union[int, float, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Upsamples a filter and returns the upsampled filter and its new origin.
    PyTorch translation.
    """
    if (isinstance(mup, (int, float)) and mup == 1) or \
       (isinstance(mup, torch.Tensor) and torch.equal(mup, torch.eye(2, device=f.device))):
        origin = (torch.tensor(f.shape, device=f.device) - 1) // 2
        return f, origin

    if isinstance(mup, (int, float)):
        mup = torch.tensor([[mup, 0], [0, mup]], dtype=torch.long, device=f.device)
    mup = mup.to(dtype=torch.long, device=f.device)

    # Use as_tuple=True to get row and column indices separately
    taps_r, taps_c = torch.nonzero(f, as_tuple=True)
    tap_coords = torch.stack([taps_r, taps_c], dim=0).to(torch.long)

    upsampled_coords = mup @ tap_coords

    orig_origin = (torch.tensor(f.shape, device=f.device) - 1) // 2
    new_origin_coord = mup @ orig_origin

    min_coords, _ = torch.min(upsampled_coords, dim=1)
    max_coords, _ = torch.max(upsampled_coords, dim=1)

    new_size = max_coords - min_coords + 1
    f_up = torch.zeros(tuple(new_size), dtype=f.dtype, device=f.device)

    shifted_coords = upsampled_coords - min_coords.unsqueeze(1)
    f_up[shifted_coords[0, :], shifted_coords[1, :]] = f[taps_r, taps_c]

    f_up_origin = new_origin_coord - min_coords

    return f_up, f_up_origin

def _correlate_upsampled(x: torch.Tensor, f: torch.Tensor, mup: Any, is_rec: bool = False) -> torch.Tensor:
    """
    Helper for correlation with an upsampled filter, handling reconstruction.
    This replaces `_convolve_upsampled` and uses correlation directly.
    """
    if not torch.any(f):
        return torch.zeros_like(x)

    f_up, f_up_origin = _upsample_and_find_origin(f, mup)

    if is_rec:
        f_up = torch.rot90(f_up, 2, [0, 1])
        f_up_origin = torch.tensor(f_up.shape, device=f.device) - 1 - f_up_origin

    pad_top = f_up_origin[0].item()
    pad_bottom = (f_up.shape[0] - 1 - f_up_origin[0]).item()
    pad_left = f_up_origin[1].item()
    pad_right = (f_up.shape[1] - 1 - f_up_origin[1]).item()

    x_ext = extend2(x, pad_top, pad_bottom, pad_left, pad_right)

    # Unsqueeze for conv2d: (H, W) -> (N, C_in, H, W)
    x_ext = x_ext.unsqueeze(0).unsqueeze(0)
    # Unsqueeze for conv2d: (Hk, Wk) -> (C_out, C_in, Hk, Wk)
    f_up = f_up.unsqueeze(0).unsqueeze(0)

    # F.conv2d performs cross-correlation, which is what the original MATLAB code did.
    # The original python code used convolve2d(x, rot90(f,2)), which is also correlation.
    result = F.conv2d(x_ext, f_up, padding='valid')

    return result.squeeze(0).squeeze(0)


def nssfbdec(x: torch.Tensor, f1: torch.Tensor, f2: torch.Tensor, mup: Any = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Two-channel nonsubsampled filter bank decomposition.
    PyTorch translation.
    """
    if mup is None:
        y1 = efilter2(x, f1)
        y2 = efilter2(x, f2)
    else:
        y1 = _correlate_upsampled(x, f1, mup, is_rec=False)
        y2 = _correlate_upsampled(x, f2, mup, is_rec=False)
    return y1, y2


def nssfbrec(x1: torch.Tensor, x2: torch.Tensor, f1: torch.Tensor, f2: torch.Tensor, mup: Any = None) -> torch.Tensor:
    """
    Two-channel nonsubsampled filter bank reconstruction.
    PyTorch translation.
    """
    if x1.shape != x2.shape:
        raise ValueError("Input sizes for the two branches must be the same")

    if mup is None:
        y1 = efilter2(x1, f1)
        y2 = efilter2(x2, f2)
    else:
        y1 = _correlate_upsampled(x1, f1, mup, is_rec=True)
        y2 = _correlate_upsampled(x2, f2, mup, is_rec=True)

    return y1 + y2