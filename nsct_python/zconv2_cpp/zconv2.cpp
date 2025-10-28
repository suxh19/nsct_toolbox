/*
 * zconv2.cpp - High-performance 2D convolution with upsampled filter
 * 
 * This C++ implementation replaces the pure Python _zconv2 function
 * for significant performance improvements.
 * 
 * Computes convolution with an upsampled filter using periodic boundary.
 * This is critical for the NSCT (Nonsubsampled Contourlet Transform) implementation.
 * 
 * Original C version by Jason Laska, converted to C++/pybind11: 2025
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <algorithm>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

/*MACRO for converting positions to linear index (row-major for NumPy)*/
#define LINPOS(row, col, collen) ((row) * (collen) + (col))

/**
 * Performs 2D convolution with upsampled filter using periodic boundary.
 * 
 * This computes convolution as if the filter had been upsampled by matrix mup,
 * but without actually upsampling the filter (efficient stepping through zeros).
 * 
 * @param x_input: Input signal (2D numpy array)
 * @param h_input: Filter (2D numpy array)
 * @param mup_input: Upsampling matrix (2x2 numpy array) [[M0, M1], [M2, M3]]
 * @return: Convolution output (same size as x)
 */
py::array_t<double> zconv2_cpp(
    py::array_t<double> x_input,
    py::array_t<double> h_input,
    py::array_t<double> mup_input
) {
    // Get buffer info
    auto x_buf = x_input.request();
    auto h_buf = h_input.request();
    auto mup_buf = mup_input.request();
    
    // Validate dimensions
    if (x_buf.ndim != 2) {
        throw std::runtime_error("Input signal x must be 2-dimensional");
    }
    if (h_buf.ndim != 2) {
        throw std::runtime_error("Filter h must be 2-dimensional");
    }
    if (mup_buf.ndim != 2 || mup_buf.shape[0] != 2 || mup_buf.shape[1] != 2) {
        throw std::runtime_error("Upsampling matrix mup must be 2x2");
    }
    
    // Get pointers to data
    double* x_ptr = static_cast<double*>(x_buf.ptr);
    double* h_ptr = static_cast<double*>(h_buf.ptr);
    double* mup_ptr = static_cast<double*>(mup_buf.ptr);
    
    // Get dimensions
    const int s_row_len = static_cast<int>(x_buf.shape[0]);
    const int s_col_len = static_cast<int>(x_buf.shape[1]);
    const int f_row_len = static_cast<int>(h_buf.shape[0]);
    const int f_col_len = static_cast<int>(h_buf.shape[1]);
    
    // Extract upsampling matrix elements
    // mup = [[M0, M1], [M2, M3]]
    // NumPy uses row-major (C-style) storage by default:
    // For 2x2 matrix [[a,b],[c,d]], memory layout is: a, b, c, d
    const int M0 = static_cast<int>(mup_ptr[0]);  // mup[0,0]
    const int M1 = static_cast<int>(mup_ptr[1]);  // mup[0,1]
    const int M2 = static_cast<int>(mup_ptr[2]);  // mup[1,0]
    const int M3 = static_cast<int>(mup_ptr[3]);  // mup[1,1]
    
    // Calculate upsampled filter dimensions
    const int new_f_row_len = (M0 - 1) * (f_row_len - 1) + M2 * (f_col_len - 1) + f_row_len - 1;
    const int new_f_col_len = (M3 - 1) * (f_col_len - 1) + M1 * (f_row_len - 1) + f_col_len - 1;
    
    // Create output array with same shape as input x
    auto result = py::array_t<double>({s_row_len, s_col_len});
    auto result_buf = result.request();
    double* out_ptr = static_cast<double*>(result_buf.ptr);
    
    // Initialize
    const int start1 = new_f_row_len / 2;
    const int start2 = new_f_col_len / 2;
    const int mn1_init = start1 % s_row_len;
    const int mn2_save = start2 % s_col_len;
    
    // Main convolution loop - parallelized over rows (n1) for performance
    // Each thread works on independent rows, writing to separate output locations
    // This ensures deterministic results (no race conditions)
    #pragma omp parallel for schedule(static) if(s_row_len * s_col_len > 1024)
    for (int n1 = 0; n1 < s_row_len; ++n1) {
        // Each thread computes its own mn1 based on row index
        int mn1 = (mn1_init + n1) % s_row_len;
        int mn2 = mn2_save;
        
        for (int n2 = 0; n2 < s_col_len; ++n2) {
            int out_index_x = mn1;
            int out_index_y = mn2;
            double sum = 0.0;
            
            // Loop over filter rows
            for (int l1 = 0; l1 < f_row_len; ++l1) {
                int index_x = out_index_x;
                int index_y = out_index_y;
                
                // Loop over filter columns
                for (int l2 = 0; l2 < f_col_len; ++l2) {
                    // Accumulate: x[index_x, index_y] * h[l1, l2]
                    sum += x_ptr[LINPOS(index_x, index_y, s_col_len)] * 
                           h_ptr[LINPOS(l1, l2, f_col_len)];
                    
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
                
                // Step for outer filter loop with M0, M1
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
            
            // Store result
            out_ptr[LINPOS(n1, n2, s_col_len)] = sum;
            
            // Update mn2 for next iteration
            mn2++;
            if (mn2 >= s_col_len)
                mn2 -= s_col_len;
        }
    }
    
    return result;
}


// Python module definition
PYBIND11_MODULE(zconv2_cpp, m) {
    m.doc() = "High-performance C++ implementation of 2D convolution with upsampled filter for NSCT";
    
    m.def("zconv2", &zconv2_cpp,
          "Performs 2D convolution with upsampled filter using periodic boundary",
          py::arg("x"), py::arg("h"), py::arg("mup"));
}
