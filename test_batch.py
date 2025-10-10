"""
Benchmark script comparing the batched NSCT implementation with the legacy
per-sample pipeline.

Run after building the CUDA extensions, e.g.:

    python setup.py build_ext --inplace
    python test_batch.py --data-dir data
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
from PIL import Image

from nsct.api import nsct_batch_dec, nsct_batch_rec
from nsct_torch import nsctdec as nsctdec_single
from nsct_torch import nsctrec as nsctrec_single


def _load_image_paths(data_dir: Path) -> List[Path]:
    if not data_dir.is_dir():
        raise SystemExit(f"Dataset directory not found: {data_dir}")
    image_paths = sorted(
        [path for path in data_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    )
    if not image_paths:
        raise SystemExit(f"No image files found in dataset directory: {data_dir}")
    return image_paths


def _load_dataset(data_dir: Path, device: torch.device) -> torch.Tensor:
    image_tensors = []
    for path in _load_image_paths(data_dir):
        with Image.open(path) as img:
            image = img.convert("F")
        tensor = torch.from_numpy(np.array(image, dtype=np.float32))
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        elif tensor.dim() == 3:
            tensor = tensor.permute(2, 0, 1)
        image_tensors.append(tensor)

    data = torch.stack(image_tensors, dim=0)
    return (data / 255.0).to(device=device)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark batched NSCT using a real dataset.")
    default_data = Path(__file__).resolve().parent / "data"
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_data,
        help=f"Directory containing test images (default: {default_data})",
    )
    parser.add_argument(
        "--nlevs",
        type=int,
        nargs="+",
        default=[3, 3, 2],
        help="Directional decomposition levels per pyramid stage.",
    )
    return parser.parse_args()


def _synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> None:
    args = _parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required to run this benchmark")

    device = torch.device("cuda")

    x = _load_dataset(args.data_dir, device)
    batch, channels, height, width = x.shape
    nlevs: Sequence[int] = args.nlevs

    # Batched pipeline
    _synchronize()
    start = time.perf_counter()
    coeffs_batched = nsct_batch_dec(x, nlevs)
    _synchronize()
    dec_time = time.perf_counter() - start

    _synchronize()
    start = time.perf_counter()
    recon_batched = nsct_batch_rec(coeffs_batched)
    _synchronize()
    rec_time = time.perf_counter() - start

    max_err_batched = (recon_batched - x).abs().max().item()

    # Legacy sequential pipeline
    x_flat = x.view(batch * channels, height, width)
    coeffs_seq: List = []

    _synchronize()
    start = time.perf_counter()
    for idx in range(x_flat.size(0)):
        coeffs_seq.append(nsctdec_single(x_flat[idx], nlevs))
    _synchronize()
    seq_dec_time = time.perf_counter() - start

    recon_seq_list = []
    _synchronize()
    start = time.perf_counter()
    for coeff in coeffs_seq:
        recon_seq_list.append(nsctrec_single(coeff))
    _synchronize()
    seq_rec_time = time.perf_counter() - start

    recon_seq = torch.stack(recon_seq_list, dim=0).view(batch, channels, height, width)
    max_diff_between_methods = (recon_seq - recon_batched).abs().max().item()
    max_err_seq = (recon_seq - x).abs().max().item()

    print("=== Batched NSCT ===")
    print(f"Decomposition time : {dec_time * 1e3:.2f} ms")
    print(f"Reconstruction time: {rec_time * 1e3:.2f} ms")
    print(f"Max error vs input : {max_err_batched:.3e}")

    print("\n=== Sequential NSCT (nsct_torch) ===")
    print(f"Decomposition time : {seq_dec_time * 1e3:.2f} ms")
    print(f"Reconstruction time: {seq_rec_time * 1e3:.2f} ms")
    print(f"Max error vs input : {max_err_seq:.3e}")

    print("\n=== Comparison ===")
    print(f"Max diff between methods: {max_diff_between_methods:.3e}")

    # 保存处理结果
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    # 创建子文件夹
    batched_dir = results_dir / "batched"
    sequential_dir = results_dir / "sequential"
    batched_dir.mkdir(exist_ok=True)
    sequential_dir.mkdir(exist_ok=True)

    # 保存批处理结果
    torch.save(
        {
            "coeffs": coeffs_batched,
            "reconstruction": recon_batched,
            "original": x,
            "metrics": {
                "dec_time_ms": dec_time * 1e3,
                "rec_time_ms": rec_time * 1e3,
                "max_error": max_err_batched,
            },
        },
        batched_dir / "results.pt",
    )
    print(f"\n✓ Batched results saved to {batched_dir / 'results.pt'}")

    # 保存顺序处理结果
    torch.save(
        {
            "coeffs": coeffs_seq,
            "reconstruction": recon_seq,
            "original": x,
            "metrics": {
                "dec_time_ms": seq_dec_time * 1e3,
                "rec_time_ms": seq_rec_time * 1e3,
                "max_error": max_err_seq,
            },
        },
        sequential_dir / "results.pt",
    )
    print(f"✓ Sequential results saved to {sequential_dir / 'results.pt'}")

    # 保存对比信息
    comparison_info = {
        "max_diff_between_methods": max_diff_between_methods,
        "batched_metrics": {
            "dec_time_ms": dec_time * 1e3,
            "rec_time_ms": rec_time * 1e3,
            "max_error": max_err_batched,
        },
        "sequential_metrics": {
            "dec_time_ms": seq_dec_time * 1e3,
            "rec_time_ms": seq_rec_time * 1e3,
            "max_error": max_err_seq,
        },
        "speedup": {
            "decomposition": seq_dec_time / dec_time,
            "reconstruction": seq_rec_time / rec_time,
        },
    }
    torch.save(comparison_info, results_dir / "comparison.pt")
    print(f"✓ Comparison results saved to {results_dir / 'comparison.pt'}")

    # 保存PNG图像
    print("\n=== Saving PNG images ===")
    
    def save_tensor_as_png(tensor: torch.Tensor, filepath: Path, prefix: str) -> None:
        """将张量保存为PNG图像"""
        tensor_np = (tensor.cpu().clamp(0, 1) * 255).to(torch.uint8).numpy()
        
        for i in range(tensor_np.shape[0]):
            for c in range(tensor_np.shape[1]):
                img_data = tensor_np[i, c]
                img = Image.fromarray(img_data, mode="L")
                filename = f"{prefix}_batch{i}_ch{c}.png"
                img.save(filepath / filename)
    
    # 保存原始图像到主目录
    save_tensor_as_png(x, results_dir, "original")
    print(f"✓ Original images saved to {results_dir}/original_*.png")
    
    # 保存批处理重建图像到批处理文件夹
    save_tensor_as_png(recon_batched, batched_dir, "reconstruction")
    print(f"✓ Batched reconstruction saved to {batched_dir}/reconstruction_*.png")
    
    # 保存顺序处理重建图像到顺序处理文件夹
    save_tensor_as_png(recon_seq, sequential_dir, "reconstruction")
    print(f"✓ Sequential reconstruction saved to {sequential_dir}/reconstruction_*.png")
    
    # 保存误差图
    error_batched = (recon_batched - x).abs()
    error_seq = (recon_seq - x).abs()
    diff_methods = (recon_batched - recon_seq).abs()
    
    # 归一化误差以便可视化
    if error_batched.max() > 0:
        error_batched_vis = error_batched / error_batched.max()
        save_tensor_as_png(error_batched_vis, batched_dir, "error")
        print(f"✓ Batched error maps saved to {batched_dir}/error_*.png")
    
    if error_seq.max() > 0:
        error_seq_vis = error_seq / error_seq.max()
        save_tensor_as_png(error_seq_vis, sequential_dir, "error")
        print(f"✓ Sequential error maps saved to {sequential_dir}/error_*.png")
    
    if diff_methods.max() > 0:
        diff_methods_vis = diff_methods / diff_methods.max()
        save_tensor_as_png(diff_methods_vis, results_dir, "diff_methods")
        print(f"✓ Method difference maps saved to {results_dir}/diff_methods_*.png")


if __name__ == "__main__":
    main()
