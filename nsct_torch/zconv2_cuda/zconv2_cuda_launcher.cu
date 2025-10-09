/*
 * zconv2_cuda_launcher.cu - CUDA kernel launcher for zconv2
 * 
 * This file contains the host-side code that launches the CUDA kernel
 * and manages GPU memory through PyTorch tensors.
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdexcept>

// Declare the CUDA kernel (defined in zconv2_cuda_kernel.cu)
__global__ void zconv2_kernel(
    const double* __restrict__ x,
    const double* __restrict__ h,
    double* __restrict__ y,
    const int s_row_len, 
    const int s_col_len,
    const int f_row_len, 
    const int f_col_len,
    const int M0, 
    const int M1, 
    const int M2, 
    const int M3,
    const int mn1_init, 
    const int mn2_save
);

/**
 * CUDA kernel launcher function.
 * 
 * This function is called from C++ code and launches the CUDA kernel
 * with appropriate grid and block dimensions.
 * 
 * @param x: Input tensor (on GPU)
 * @param h: Filter tensor (on GPU)
 * @param y: Output tensor (on GPU, pre-allocated)
 * @param M0, M1, M2, M3: Upsampling matrix elements
 * @param mn1_init: Initial row offset
 * @param mn2_save: Initial column offset
 */
void zconv2_cuda_launcher(
    const torch::Tensor& x,
    const torch::Tensor& h,
    torch::Tensor& y,
    int M0, int M1, int M2, int M3,
    int mn1_init, int mn2_save
) {
    // Get dimensions from tensors
    const int s_row_len = x.size(0);
    const int s_col_len = x.size(1);
    const int f_row_len = h.size(0);
    const int f_col_len = h.size(1);

    // Define thread block dimensions (16x16 = 256 threads per block)
    // This is a good balance for most GPUs
    const dim3 threads(16, 16);
    
    // Calculate grid dimensions (number of blocks needed)
    // Round up to ensure full coverage
    const dim3 blocks(
        (s_col_len + threads.x - 1) / threads.x,
        (s_row_len + threads.y - 1) / threads.y
    );

    // Launch CUDA kernel
    zconv2_kernel<<<blocks, threads>>>(
        x.data_ptr<double>(),
        h.data_ptr<double>(),
        y.data_ptr<double>(),
        s_row_len, s_col_len,
        f_row_len, f_col_len,
        M0, M1, M2, M3,
        mn1_init, mn2_save
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err)
        );
    }
}
