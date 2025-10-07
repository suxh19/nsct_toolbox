/*
 * atrousc.cpp - High-performance à trous convolution implementation
 * 
 * This C++ implementation replaces the pure Python atrousc function
 * for significant performance improvements.
 * 
 * Computes convolution with an upsampled filter without actually upsampling.
 * This is critical for the NSCT (Nonsubsampled Contourlet Transform) implementation.
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

/**
 * Performs à trous convolution with an upsampled filter.
 * 
 * @param x_input: Extended input signal (2D numpy array)
 * @param h_input: Original filter, not upsampled (2D numpy array)
 * @param M_input: Upsampling matrix (2x2 numpy array or scalar)
 * @return: Result of convolution in 'valid' mode (2D numpy array)
 */
py::array_t<double> atrousc_cpp(
    py::array_t<double> x_input,
    py::array_t<double> h_input,
    py::array_t<double> M_input
) {
    // Get buffer info
    auto x_buf = x_input.request();
    auto h_buf = h_input.request();
    auto M_buf = M_input.request();
    
    // Validate dimensions
    if (x_buf.ndim != 2) {
        throw std::runtime_error("Input signal x must be 2-dimensional");
    }
    if (h_buf.ndim != 2) {
        throw std::runtime_error("Filter h must be 2-dimensional");
    }
    
    // Get pointers to data
    double* x_ptr = static_cast<double*>(x_buf.ptr);
    double* h_ptr = static_cast<double*>(h_buf.ptr);
    double* M_ptr = static_cast<double*>(M_buf.ptr);
    
    // Get dimensions
    const int S_rows = static_cast<int>(x_buf.shape[0]);
    const int S_cols = static_cast<int>(x_buf.shape[1]);
    const int F_rows = static_cast<int>(h_buf.shape[0]);
    const int F_cols = static_cast<int>(h_buf.shape[1]);
    
    // Extract upsampling factors from matrix M
    // For diagonal matrix M = [[M0, 0], [0, M3]]
    int M0, M3;
    
    if (M_buf.ndim == 0 || (M_buf.ndim == 2 && M_buf.shape[0] == 1 && M_buf.shape[1] == 1)) {
        // Scalar case
        M0 = M3 = static_cast<int>(M_ptr[0]);
    } else if (M_buf.ndim == 2 && M_buf.shape[0] == 2 && M_buf.shape[1] == 2) {
        // 2x2 matrix case - extract diagonal elements
        M0 = static_cast<int>(M_ptr[0]);      // M[0,0]
        M3 = static_cast<int>(M_ptr[3]);      // M[1,1]
    } else {
        throw std::runtime_error("M must be a scalar or 2x2 matrix");
    }
    
    // Calculate output dimensions (matching Python implementation)
    const int O_rows = S_rows - M0 * F_rows + 1;
    const int O_cols = S_cols - M3 * F_cols + 1;
    
    // Handle edge case
    if (O_rows <= 0 || O_cols <= 0) {
        auto result = py::array_t<double>({std::max(0, O_rows), std::max(0, O_cols)});
        auto result_buf = result.request();
        double* result_ptr = static_cast<double*>(result_buf.ptr);
        std::fill(result_ptr, result_ptr + std::max(0, O_rows * O_cols), 0.0);
        return result;
    }
    
    // Allocate output array
    auto result = py::array_t<double>({O_rows, O_cols});
    auto result_buf = result.request();
    double* out_ptr = static_cast<double*>(result_buf.ptr);
    
    // Flip the filter for convolution (h_flipped = flipud(fliplr(h)))
    std::vector<double> h_flipped(F_rows * F_cols);
    for (int i = 0; i < F_rows; ++i) {
        for (int j = 0; j < F_cols; ++j) {
            int src_idx = i * F_cols + j;
            int dst_idx = (F_rows - 1 - i) * F_cols + (F_cols - 1 - j);
            h_flipped[dst_idx] = h_ptr[src_idx];
        }
    }
    
    // Main convolution loop - matching Python implementation exactly
    // Python loops over columns first (n1), then rows (n2)
    // Parallelize outer loop for performance (threshold to avoid overhead on small images)
    #pragma omp parallel for schedule(static) if(O_cols * O_rows > 1024)
    for (int n1 = 0; n1 < O_cols; ++n1) {  // Column loop (matches Python)
        for (int n2 = 0; n2 < O_rows; ++n2) {  // Row loop (matches Python)
            double total = 0.0;
            int kk1 = n1 + M0 - 1;
            
            // Loop over filter columns
            for (int k1 = 0; k1 < F_cols; ++k1) {
                int kk2 = n2 + M3 - 1;
                
                // Loop over filter rows
                for (int k2 = 0; k2 < F_rows; ++k2) {
                    // Flipped indices
                    int f1 = F_cols - 1 - k1;
                    int f2 = F_rows - 1 - k2;
                    
                    // Access input and filter
                    total += h_flipped[f2 * F_cols + f1] * x_ptr[kk2 * S_cols + kk1];
                    kk2 += M3;
                }
                kk1 += M0;
            }
            
            // Store result (note: output index is [n2, n1])
            out_ptr[n2 * O_cols + n1] = total;
        }
    }
    
    return result;
}




// Python module definition
PYBIND11_MODULE(atrousc_cpp, m) {
    m.doc() = "High-performance C++ implementation of à trous convolution for NSCT";
    
    m.def("atrousc", &atrousc_cpp,
          "Performs à trous convolution with an upsampled filter",
          py::arg("x"), py::arg("h"), py::arg("M"));
}
