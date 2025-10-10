"""
Benchmark script comparing the batched NSCT implementation with the legacy
per-sample pipeline.

Run after building the CUDA extensions, e.g.:

    python setup.py build_ext --inplace
    python test_batch.py
"""

from __future__ import annotations

import time
from typing import List

import torch

from nsct.api import nsct_batch_dec, nsct_batch_rec
from nsct_torch import nsctdec as nsctdec_single
from nsct_torch import nsctrec as nsctrec_single


def _synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required to run this benchmark")

    device = torch.device("cuda")
    torch.manual_seed(1234)

    batch, channels, height, width = 320, 3, 128, 128
    nlevs = [3, 3, 2]

    x = torch.randn(batch, channels, height, width, device=device)

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


if __name__ == "__main__":
    main()
