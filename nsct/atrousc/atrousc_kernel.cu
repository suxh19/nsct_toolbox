/*
 * atrousc_kernel.cu - CUDA kernel for à trous convolution
 * 
 * This CUDA implementation provides GPU-accelerated computation for the NSCT
 * (Nonsubsampled Contourlet Transform) à trous convolution operation.
 * 
 * Each CUDA thread computes one output pixel, providing massive parallelism
 * for large images.
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Macro for converting 2D position to linear index (row-major)
#define LINPOS(row, col, collen) ((row) * (collen) + (col))

/**
 * CUDA kernel for à trous convolution with upsampled filter.
 * 
 * Each thread computes one output pixel out[n2, n1] by:
 * 1. Starting from position (n2, n1) in the extended input
 * 2. Stepping through the input with strides M0 and M3
 * 3. Accumulating weighted sums with the flipped filter
 * 
 * The algorithm follows the CPU implementation exactly:
 * - Filter is conceptually flipped before use
 * - Iteration order: outer loop over columns (n1), inner loop over rows (n2)
 * - Filter iteration: outer loop over filter columns (k1), inner loop over filter rows (k2)
 * 
 * @param x: Extended input signal (flattened 2D array)
 * @param h: Original filter, not upsampled (flattened 2D array)
 * @param out: Output array (flattened 2D array)
 * @param S_rows: Number of rows in extended input
 * @param S_cols: Number of columns in extended input
 * @param F_rows: Number of rows in filter
 * @param F_cols: Number of columns in filter
 * @param O_rows: Number of rows in output
 * @param O_cols: Number of columns in output
 * @param M0: Upsampling factor for rows (from matrix M[0,0])
 * @param M3: Upsampling factor for columns (from matrix M[1,1])
 */
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
) {
    // Calculate global thread indices (output pixel coordinates)
    // Note: blockIdx.x corresponds to columns (n1), blockIdx.y to rows (n2)
    const int batch_idx = blockIdx.z;
    const int n1 = blockIdx.x * blockDim.x + threadIdx.x;  // Column index
    const int n2 = blockIdx.y * blockDim.y + threadIdx.y;  // Row index

    // Boundary check: ensure thread is within output dimensions
    if (n1 >= O_cols || n2 >= O_rows) {
        return;
    }

    const int input_plane_stride = S_rows * S_cols;
    const int output_plane_stride = O_rows * O_cols;
    const float* x_plane = x + batch_idx * input_plane_stride;
    float* out_plane = out + batch_idx * output_plane_stride;

    // Accumulate convolution result
    // Note: Following CPU implementation exactly
    // n1 is column index, n2 is row index
    // kk1 corresponds to column position in x, kk2 to row position
    // M0 is used with columns (kk1, k1), M3 is used with rows (kk2, k2)
    float total = 0.0f;
    int kk1 = n1 + M0 - 1;  // Column initial position

    // Loop over filter columns (outer loop matches CPU implementation)
    for (int k1 = 0; k1 < F_cols; ++k1) {
        int kk2 = n2 + M3 - 1;  // Row initial position (reset for each column)

        // Loop over filter rows (inner loop matches CPU implementation)
        for (int k2 = 0; k2 < F_rows; ++k2) {
            // CPU implementation uses double flip which cancels out
            // So we access h directly without flipping
            // h[k2, k1] * x[kk2, kk1]
            // Note: x is indexed as [row, col] = [kk2, kk1]
            total += h[LINPOS(k2, k1, F_cols)] * x_plane[LINPOS(kk2, kk1, S_cols)];

            // Step through input with M3 (row stride)
            kk2 += M3;
        }

        // Step through input with M0 (column stride)
        kk1 += M0;
    }

    // Write result to output
    // Output is stored as [n2, n1] (row-major order)
    out_plane[LINPOS(n2, n1, O_cols)] = total;
}
