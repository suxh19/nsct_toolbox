# NSCT Batch Processing Refactoring Plan

## Objective
Refactor the `@nsct_torch` library to support batch processing of 4D tensors in the format `[Batch, Channel, Height, Width]`. This will be achieved by modifying the underlying CUDA kernels for true parallel processing, significantly improving performance for AI training workflows. The new, refactored code will reside in a new `nsct` directory.

## Core Strategy
The core strategy is to treat a `[B, C, H, W]` tensor as a larger batch of `[B * C, H, W]` 2D images. This reshaped tensor will be processed in parallel by modified CUDA kernels. The final output will be reshaped back to `[B, C, ...]` to maintain a consistent API.

## Detailed Steps

1.  **Initialize Project Structure**:
    *   Create the main `nsct` directory.
    *   Copy the CUDA source directories (`zconv2`, `atrousc`) from `nsct_torch` into `nsct`.
    *   Create placeholder Python files (`__init__.py`, `core.py`, `utils.py`, `filters.py`, `api.py`) within `nsct`.

2.  **Modify `zconv2` CUDA/C++ Kernel**:
    *   **`zconv2.cpp`**:
        *   Update the input tensor check to accept 3D tensors `[N, H, W]`.
        *   Extract `batch_size` (N) from the input tensor shape.
        *   Modify the CUDA kernel launch configuration to use a 3D grid, where `gridDim.z = batch_size`.
        *   Pass `batch_size`, `height`, and `width` to the kernel launcher.
    *   **`zconv2_kernel.cu`**:
        *   Add a `batch_idx` calculated from `blockIdx.z`.
        *   Calculate the memory offset for the current image in the batch (`batch_idx * height * width`).
        *   Apply this offset to all pointers accessing the input (`x`) and output (`y`) tensors.

3.  **Modify `atrousc` CUDA/C++ Kernel**:
    *   **`atrousc.cpp`**:
        *   Apply the same modifications as in `zconv2.cpp` (accept 3D tensor, extract batch size, launch 3D grid).
    *   **`atrousc_kernel.cu`**:
        *   Apply the same modifications as in `zconv2_kernel.cu` (add `batch_idx`, calculate memory offsets for `x` and `out`).

4.  **Update `setup.py` Script**:
    *   Create a new `setup.py` at the root level (or modify the existing one).
    *   Point the `CUDAExtension` to the new source files within the `nsct` directory.
    *   Ensure the extension is named appropriately (e.g., `nsct.ops`).

5.  **Adapt `utils.py` Helper Functions**:
    *   Copy relevant functions from `nsct_torch/utils.py` to `nsct/utils.py`.
    *   Modify functions like `symext` to operate on 3D tensors `[N, H, W]`. This will likely involve iterating over the batch dimension and applying the 2D logic to each slice.

6.  **Refactor `core.py` Logic**:
    *   Copy the core transformation logic from `nsct_torch/core.py` to `nsct/core.py`.
    *   Update functions like `nsctdec`, `nsdfbdec`, etc., to work with 3D tensors `[N, H, W]`.
    *   Ensure that all internal calls to filtering and convolution functions pass the 3D tensors correctly to the newly compiled CUDA extensions.

7.  **Create High-Level `api.py`**:
    *   Create a user-facing function, e.g., `nsct_batch_dec(x: torch.Tensor, ...)`.
    *   This function will take a `[B, C, H, W]` tensor as input.
    *   Inside the function, perform the reshape: `x_reshaped = x.view(-1, H, W)`.
    *   Call the refactored core functions from `nsct.core` with `x_reshaped`.
    *   Take the output coefficients and reshape them back to include batch and channel dimensions, e.g., `[B, C, ...]`.
    *   Create a corresponding `nsct_batch_rec` for reconstruction.

8.  **Write Test Script**:
    *   Create a `test_batch.py` script.
    *   Generate a random `[B, C, H, W]` tensor.
    *   Run it through the full decomposition and reconstruction pipeline (`nsct.api.nsct_batch_dec` -> `nsct.api.nsct_batch_rec`).
    *   Verify that the reconstructed tensor is close to the original (perfect reconstruction check).
    *   Compare the execution time with a loop-based approach using the old `nsct_torch` library to quantify the performance improvement.

9.  **Update Documentation**:
    *   Update the main `README.md` to explain the new batch processing feature and how to use the new API.
    *   Add a note about the requirement of a CUDA build environment.
    *   Update `.gitignore` if necessary to exclude build artifacts.