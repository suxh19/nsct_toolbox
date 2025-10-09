#!/usr/bin/env python
"""Benchmark and consistency checks between nsct_python and nsct_torch backends."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nsct_python.core import nsctdec as nsctdec_numpy, nsctrec as nsctrec_numpy
from nsct_torch.core_torch import nsctdec as nsctdec_torch, nsctrec as nsctrec_torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare nsct_python and nsct_torch on decomposition shapes, numerical "
            "consistency, and timing benchmarks."
        )
    )
    parser.add_argument(
        "image",
        type=Path,
        help="Path to the grayscale test image (will be converted if necessary).",
    )
    parser.add_argument(
        "--levels",
        type=str,
        default="[2];[2,2];[2,2,2]",
        help=(
            "Semicolon-separated list of decomposition level lists. "
            "Example: \"[2];[2,2]\" (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--dfilt",
        type=str,
        default="dmaxflat7",
        help="Directional filter name (default: %(default)s).",
    )
    parser.add_argument(
        "--pfilt",
        type=str,
        default="maxflat",
        help="Pyramidal filter name (default: %(default)s).",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float64",
        help="Floating point precision to use for both backends (default: %(default)s).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of timed repetitions per backend after warm-up (default: %(default)s).",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional path to dump the raw results as JSON.",
    )
    return parser.parse_args()


def parse_levels(spec: str) -> List[List[int]]:
    """Parse level specification like \"[2];[2,2]\" into a list of integer lists."""
    levels: List[List[int]] = []
    for chunk in spec.split(";"):
        token = chunk.strip()
        if not token:
            continue
        if not (token.startswith("[") and token.endswith("]")):
            raise ValueError(f"Invalid levels chunk: {token!r}")
        items = token[1:-1].strip()
        if not items:
            levels.append([])
            continue
        levels.append([int(part.strip()) for part in items.split(",") if part.strip()])
    if not levels:
        raise ValueError("No levels provided.")
    return levels


def load_image(image_path: Path, dtype: str) -> Tuple[np.ndarray, torch.Tensor]:
    """Load image as grayscale, normalise to [0,1], and return NumPy/Torch copies."""
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = Image.open(image_path).convert("F")
    arr = np.asarray(img, dtype=np.float64) / 255.0
    if dtype == "float32":
        arr = arr.astype(np.float32, copy=False)
    torch_dtype = torch.float32 if dtype == "float32" else torch.float64
    torch_tensor = torch.from_numpy(np.ascontiguousarray(arr)).to(torch_dtype)
    return arr, torch_tensor


def flatten_nested(struct: Any) -> Iterable[np.ndarray]:
    """Yield all numpy arrays within a nested coefficient structure."""
    if isinstance(struct, (list, tuple)):
        for item in struct:
            yield from flatten_nested(item)
    else:
        yield struct


def torch_to_numpy_structure(struct: Any) -> Any:
    if isinstance(struct, torch.Tensor):
        return struct.detach().cpu().numpy()
    if isinstance(struct, (list, tuple)):
        return [torch_to_numpy_structure(elem) for elem in struct]
    return struct


def compare_structures(
    coeffs_numpy: Any, coeffs_torch: Any
) -> Tuple[bool, float, float]:
    """Check nesting/shape equality and compute max absolute/relative coefficient diffs."""
    torch_np = torch_to_numpy_structure(coeffs_torch)
    structure_ok = True
    max_abs = 0.0
    max_rel = 0.0

    def _walk(a: Any, b: Any) -> None:
        nonlocal structure_ok, max_abs, max_rel
        if isinstance(a, list):
            if not isinstance(b, list) or len(a) != len(b):
                structure_ok = False
                return
            for sa, sb in zip(a, b):
                _walk(sa, sb)
        else:
            if not isinstance(b, np.ndarray) or a.shape != b.shape:
                structure_ok = False
                return
            diff = np.abs(a - b)
            if diff.size:
                max_abs = max(max_abs, float(diff.max()))
                denom = np.maximum(np.abs(a), np.abs(b))
                with np.errstate(divide="ignore", invalid="ignore"):
                    rel = np.where(denom > 0.0, diff / denom, 0.0)
                if rel.size and not np.all(np.isnan(rel)):
                    max_rel = max(max_rel, float(np.nanmax(rel)))

    _walk(coeffs_numpy, torch_np)
    return structure_ok, max_abs, max_rel


def time_call(fn, *args, repeats: int) -> Tuple[float, float]:
    samples: List[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn(*args)
        samples.append(time.perf_counter() - start)
    return statistics.mean(samples), statistics.pstdev(samples)


def benchmark_backends(
    image_np: np.ndarray,
    image_torch: torch.Tensor,
    levels: Sequence[int],
    dfilt: str,
    pfilt: str,
    repeats: int,
) -> dict[str, Any]:
    coeffs_numpy = nsctdec_numpy(image_np, levels, dfilt=dfilt, pfilt=pfilt)
    coeffs_torch = nsctdec_torch(image_torch, levels, dfilt=dfilt, pfilt=pfilt)

    structure_ok, max_abs, max_rel = compare_structures(coeffs_numpy, coeffs_torch)

    recon_numpy = nsctrec_numpy(coeffs_numpy, dfilt=dfilt, pfilt=pfilt)
    recon_torch = nsctrec_torch(coeffs_torch, dfilt=dfilt, pfilt=pfilt)
    recon_torch_np = recon_torch.detach().cpu().numpy()

    recon_err_numpy = float(np.max(np.abs(image_np - recon_numpy)))
    recon_err_torch = float(np.max(np.abs(image_np - recon_torch_np)))
    recon_gap = float(np.max(np.abs(recon_numpy - recon_torch_np)))

    # Warm-up done above; repeat timing on fresh runs.
    dec_np_mean, dec_np_std = time_call(
        nsctdec_numpy, image_np, levels, dfilt, pfilt, repeats=repeats
    )
    rec_np_mean, rec_np_std = time_call(
        nsctrec_numpy, coeffs_numpy, dfilt, pfilt, repeats=repeats
    )
    dec_torch_mean, dec_torch_std = time_call(
        nsctdec_torch, image_torch, levels, dfilt, pfilt, repeats=repeats
    )
    rec_torch_mean, rec_torch_std = time_call(
        nsctrec_torch, coeffs_torch, dfilt, pfilt, repeats=repeats
    )

    return {
        "levels": list(levels),
        "structure_match": structure_ok,
        "max_abs_coeff_diff": max_abs,
        "max_rel_coeff_diff": max_rel if structure_ok else math.nan,
        "recon_error_numpy": recon_err_numpy,
        "recon_error_torch": recon_err_torch,
        "recon_difference": recon_gap,
        "timings": {
            "numpy_dec_mean": dec_np_mean,
            "numpy_dec_std": dec_np_std,
            "numpy_rec_mean": rec_np_mean,
            "numpy_rec_std": rec_np_std,
            "torch_dec_mean": dec_torch_mean,
            "torch_dec_std": dec_torch_std,
            "torch_rec_mean": rec_torch_mean,
            "torch_rec_std": rec_torch_std,
        },
    }


def main() -> None:
    args = parse_args()
    levels_list = parse_levels(args.levels)

    np_image, torch_image = load_image(args.image, dtype=args.dtype)
    print(f"Loaded image {args.image} → shape={np_image.shape}, dtype={np_image.dtype}")
    print(f"Level configurations: {levels_list}")
    print(f"Filters: dfilt='{args.dfilt}', pfilt='{args.pfilt}', repeats={args.repeats}")

    all_results = {}
    for idx, levels in enumerate(levels_list, start=1):
        print(f"\n=== Run {idx}: levels={levels} ===")
        result = benchmark_backends(
            np_image, torch_image, levels, args.dfilt, args.pfilt, repeats=args.repeats
        )
        all_results[str(levels)] = result

        print(f"Structure match: {result['structure_match']}")
        print(
            f"Coeff diffs max abs={result['max_abs_coeff_diff']:.3e}, "
            f"max rel={result['max_rel_coeff_diff']:.3e}"
        )
        print(
            f"Reconstruction errors NumPy={result['recon_error_numpy']:.3e}, "
            f"Torch={result['recon_error_torch']:.3e}, "
            f"gap={result['recon_difference']:.3e}"
        )
        timings = result["timings"]
        print(
            "Timings (s): "
            f"np dec {timings['numpy_dec_mean']:.3f}±{timings['numpy_dec_std']:.3f}, "
            f"np rec {timings['numpy_rec_mean']:.3f}±{timings['numpy_rec_std']:.3f}, "
            f"torch dec {timings['torch_dec_mean']:.3f}±{timings['torch_dec_std']:.3f}, "
            f"torch rec {timings['torch_rec_mean']:.3f}±{timings['torch_rec_std']:.3f}"
        )

    if args.json:
        args.json.write_text(json.dumps(all_results, indent=2))
        print(f"\nWrote JSON report to {args.json}")


if __name__ == "__main__":
    main()
