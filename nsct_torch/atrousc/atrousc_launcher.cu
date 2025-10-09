/*
 * atrousc_launcher.cu - CUDA kernel launcher for atrousc
 * 
 * This file contains the host-side code that launches the CUDA kernel
 * and manages GPU memory through PyTorch tensors.
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdexcept>

// Declare the CUDA kernel (defined in atrousc_kernel.cu)
__global__ void atrousc_kernel(
    const float* __restrict__ x,
    const float* __restrict__ h,
    float* __restrict__ out,
    const int S_rows,
    const int S_cols,
    const int F_rows,
    const int F_cols,
    const int O_rows,
    const int O_cols,
    const int M0,
    const int M3
);

/**
 * CUDA kernel launcher function.
 * 
 * This function is called from C++ code and launches the CUDA kernel
 * with appropriate grid and block dimensions.
 * 
 * @param x: Extended input tensor (on GPU)
 * @param h: Filter tensor (on GPU)
 * @param out: Output tensor (on GPU, pre-allocated)
 * @param M0: Upsampling factor for rows
 * @param M3: Upsampling factor for columns
 */
void atrousc_launcher(
    const torch::Tensor& x,
    const torch::Tensor& h,
    torch::Tensor& out,
    int M0,
    int M3
) {
    // Get dimensions from tensors
    const int S_rows = x.size(0);
    const int S_cols = x.size(1);
    const int F_rows = h.size(0);
    const int F_cols = h.size(1);
    const int O_rows = out.size(0);
    const int O_cols = out.size(1);

    // Define thread block dimensions (16x16 = 256 threads per block)
    // This is a good balance for most GPUs
    const dim3 threads(16, 16);
    
    // Calculate grid dimensions (number of blocks needed)
    // Round up to ensure full coverage
    // Note: blocks.x for columns (n1), blocks.y for rows (n2)
    const dim3 blocks(
        (O_cols + threads.x - 1) / threads.x,
        (O_rows + threads.y - 1) / threads.y
    );

    // Launch CUDA kernel
    atrousc_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        h.data_ptr<float>(),
        out.data_ptr<float>(),
        S_rows, S_cols,
        F_rows, F_cols,
        O_rows, O_cols,
        M0, M3
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err)
        );
    }
}
