# NSCT Toolbox

This repository provides CUDA-accelerated implementations of the Nonsubsampled
Contourlet Transform (NSCT) for PyTorch. The refactored `nsct` package extends
the legacy `nsct_torch` module with native batch support so tensors shaped
`[batch, channel, height, width]` can be transformed in parallel.

## Building the Extensions

The batched implementation relies on custom CUDA kernels. Build them in-place
before running any examples:

```bash
python setup.py build_ext --inplace
```

Ensure you are using a CUDA-enabled PyTorch installation that matches the
toolkit available on your system.

## Batched API Usage

```python
import torch
from nsct.api import nsct_batch_dec, nsct_batch_rec

device = torch.device("cuda")
x = torch.randn(4, 3, 256, 256, device=device)
nlevs = [3, 3, 2]  # directional decomposition per pyramid level

coeffs = nsct_batch_dec(x, nlevs)
recon = nsct_batch_rec(coeffs)

print(f"Max reconstruction error: {(recon - x).abs().max().item():.3e}")
```

The returned coefficient tree mirrors the structure of the original
`nsct_torch` API, but each tensor now preserves the batch and channel
dimensions.

## Benchmark Script

After building the extensions you can compare the new batched pipeline against
the legacy per-sample approach:

```bash
python test_batch.py
```

The script reports decomposition/reconstruction timings for both methods and
verifies perfect reconstruction relative to the input tensor.
