/*
 * zconv2_kernel.cu - CUDA kernel for 2D convolution with upsampled filter
 * 
 * This CUDA implementation provides GPU-accelerated computation for the NSCT
 * (Nonsubsampled Contourlet Transform) zconv2 operation.
 * 
 * Each CUDA thread computes one output pixel, providing massive parallelism
 * for large images.
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Macro for converting 2D position to linear index (row-major)
#define LINPOS(row, col, collen) ((row) * (collen) + (col))

/**
 * CUDA kernel for 2D convolution with upsampled filter using periodic boundary.
 * 
 * Each thread computes one output pixel y[n1, n2] by:
 * 1. Starting from a calculated position in the input
 * 2. Stepping through the input with strides defined by mup matrix
 * 3. Accumulating weighted sums with the filter
 * 
 * @param x: Input signal (flattened 2D array)
 * @param h: Filter (flattened 2D array)
 * @param y: Output array (flattened 2D array, same size as x)
 * @param s_row_len: Number of rows in input/output
 * @param s_col_len: Number of columns in input/output
 * @param f_row_len: Number of rows in filter
 * @param f_col_len: Number of columns in filter
 * @param M0, M1, M2, M3: Upsampling matrix elements [[M0, M1], [M2, M3]]
 * @param mn1_init: Initial row offset
 * @param mn2_save: Initial column offset
 */
__global__ void zconv2_kernel(
    const float* __restrict__ x,
    const float* __restrict__ h,
    float* __restrict__ y,
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
) {
    // Calculate global thread indices (output pixel coordinates)
    const int n1 = blockIdx.y * blockDim.y + threadIdx.y;
    const int n2 = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check: ensure thread is within output dimensions
    if (n1 >= s_row_len || n2 >= s_col_len) {
        return;
    }

    // Calculate starting position for this output pixel (with periodic boundary)
    int mn1 = (mn1_init + n1) % s_row_len;
    int mn2 = (mn2_save + n2) % s_col_len;

    float sum = 0.0f;
    int out_index_x = mn1;
    int out_index_y = mn2;

    // Loop over filter rows
    for (int l1 = 0; l1 < f_row_len; ++l1) {
        int index_x = out_index_x;
        int index_y = out_index_y;

        // Loop over filter columns
        for (int l2 = 0; l2 < f_col_len; ++l2) {
            // Accumulate: x[index_x, index_y] * h[l1, l2]
            sum += x[LINPOS(index_x, index_y, s_col_len)] * 
                   h[LINPOS(l1, l2, f_col_len)];

            // Step through input with M2, M3 (periodic boundary)
            index_x -= M2;
            if (index_x < 0) 
                index_x += s_row_len;
            if (index_x >= s_row_len)
                index_x -= s_row_len;

            index_y -= M3;
            if (index_y < 0) 
                index_y += s_col_len;
            if (index_y >= s_col_len)
                index_y -= s_col_len;
        }

        // Step for outer filter loop with M0, M1 (periodic boundary)
        out_index_x -= M0;
        if (out_index_x < 0) 
            out_index_x += s_row_len;
        if (out_index_x >= s_row_len)
            out_index_x -= s_row_len;

        out_index_y -= M1;
        if (out_index_y < 0) 
            out_index_y += s_col_len;
        if (out_index_y >= s_col_len)
            out_index_y -= s_col_len;
    }

    // Write result to output
    y[LINPOS(n1, n2, s_col_len)] = sum;
}
