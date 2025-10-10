/*
 * atrousc.cpp - PyTorch C++ extension interface for CUDA-accelerated atrousc
 * 
 * This file provides the bridge between Python/PyTorch and the CUDA kernel.
 * It handles tensor validation, data type conversion, and parameter extraction.
 */

#include <torch/extension.h>
#include <vector>
#include <stdexcept>
#include <algorithm>

// Declare CUDA kernel launcher (defined in atrousc_launcher.cu)
void atrousc_launcher(
    const torch::Tensor& x,
    const torch::Tensor& h,
    torch::Tensor& out,
    int M0,
    int M3
);

/**
 * PyTorch C++ extension main function for CUDA-accelerated atrousc.
 * 
 * This function:
 * 1. Validates input tensors
 * 2. Ensures proper data types and memory layout
 * 3. Extracts upsampling matrix parameters
 * 4. Calculates output dimensions
 * 5. Allocates output tensor
 * 6. Launches CUDA kernel
 * 7. Returns the result
 * 
 * @param x_input: Extended input signal tensor (2D, on CUDA device)
 * @param h_input: Filter tensor (2D, on CUDA device)
 * @param M_input: Upsampling matrix (2x2 or scalar, can be on CPU or CUDA)
 * @return: Output tensor (on CUDA device)
 */
torch::Tensor atrousc_torch(
    torch::Tensor x_input,
    torch::Tensor h_input,
    torch::Tensor M_input
) {
    // ===== Input Validation =====
    TORCH_CHECK(x_input.is_cuda(), "Input tensor x must be a CUDA tensor");
    TORCH_CHECK(h_input.is_cuda(), "Filter tensor h must be a CUDA tensor");
    TORCH_CHECK(x_input.dim() == 2 || x_input.dim() == 3, "Input tensor x must be 2D or 3D");
    TORCH_CHECK(h_input.dim() == 2, "Filter tensor h must be 2D");

    // ===== Data Type and Layout Preparation =====
    // Ensure single precision and contiguous memory layout
    bool squeeze_batch_dim = false;
    if (x_input.dim() == 2) {
        x_input = x_input.unsqueeze(0);
        squeeze_batch_dim = true;
    }

    auto x = x_input.to(torch::kFloat32).contiguous();
    auto h = h_input.to(torch::kFloat32).contiguous();
    
    // Move M to CPU for parameter extraction (int32)
    auto M_cpu = M_input.to(torch::kCPU, torch::kInt32).contiguous();

    // ===== Extract Dimensions =====
    const int batch_size = x.size(0);
    const int S_rows = x.size(1);
    const int S_cols = x.size(2);
    const int F_rows = h.size(0);
    const int F_cols = h.size(1);

    // ===== Extract Upsampling Matrix Parameters =====
    // Handle both scalar and 2x2 matrix cases
    int M0, M3;
    
    if (M_cpu.dim() == 0 || (M_cpu.dim() == 2 && M_cpu.size(0) == 1 && M_cpu.size(1) == 1)) {
        // Scalar case: M0 = M3 = M
        auto M_scalar = M_cpu.item<int>();
        M0 = M_scalar;
        M3 = M_scalar;
    } else if (M_cpu.dim() == 2 && M_cpu.size(0) == 2 && M_cpu.size(1) == 2) {
        // 2x2 matrix case: extract diagonal elements
        // NOTE: Swapped to fix semantic bug - M0 is used for columns, M3 for rows
        // M = [[row_factor, 0], [0, col_factor]]
        const int* M_ptr = M_cpu.data_ptr<int>();
        M0 = M_ptr[3];  // M[1, 1] - column upsampling factor
        M3 = M_ptr[0];  // M[0, 0] - row upsampling factor
    } else {
        throw std::runtime_error("Upsampling matrix M must be a scalar or 2x2 matrix");
    }

    // ===== Calculate Output Dimensions =====
    // M0 is used for columns, M3 for rows (semantically corrected)
    const int O_rows = S_rows - M3 * F_rows + 1;
    const int O_cols = S_cols - M0 * F_cols + 1;

    // ===== Handle Edge Case =====
    if (O_rows <= 0 || O_cols <= 0) {
        // Return empty tensor with non-negative dimensions
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());
        auto result = torch::zeros(
            {batch_size, std::max(0, O_rows), std::max(0, O_cols)},
            options
        );
        if (squeeze_batch_dim) {
            return result.squeeze(0);
        }
        return result;
    }

    // ===== Allocate Output Tensor =====
    auto out = torch::empty(
        {batch_size, O_rows, O_cols},
        torch::TensorOptions().dtype(torch::kFloat32).device(x.device())
    );

    // ===== Launch CUDA Kernel =====
    atrousc_launcher(x, h, out, M0, M3);

    if (squeeze_batch_dim) {
        return out.squeeze(0);
    }
    return out;
}

// ===== Python Module Binding =====
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "CUDA-accelerated à trous convolution for NSCT";
    
    m.def(
        "atrousc", 
        &atrousc_torch,
        "Performs à trous convolution with an upsampled filter using CUDA",
        py::arg("x"), 
        py::arg("h"), 
        py::arg("M")
    );
}
