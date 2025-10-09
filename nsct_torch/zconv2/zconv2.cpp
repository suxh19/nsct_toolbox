/*
 * zconv2.cpp - PyTorch C++ extension interface for CUDA-accelerated zconv2
 * 
 * This file provides the bridge between Python/PyTorch and the CUDA kernel.
 * It handles tensor validation, data type conversion, and parameter extraction.
 */

#include <torch/extension.h>
#include <vector>
#include <stdexcept>

// Declare CUDA kernel launcher (defined in zconv2_launcher.cu)
void zconv2_launcher(
    const torch::Tensor& x,
    const torch::Tensor& h,
    torch::Tensor& y,
    int M0, int M1, int M2, int M3,
    int mn1_init, int mn2_save
);

/**
 * PyTorch C++ extension main function for CUDA-accelerated zconv2.
 * 
 * This function:
 * 1. Validates input tensors
 * 2. Ensures proper data types and memory layout
 * 3. Extracts upsampling matrix parameters
 * 4. Calculates initial offsets
 * 5. Launches CUDA kernel
 * 6. Returns the result
 * 
 * @param x: Input signal tensor (2D, on CUDA device)
 * @param h: Filter tensor (2D, on CUDA device)
 * @param mup: Upsampling matrix (2x2, can be on CPU or CUDA)
 * @return: Output tensor (same size as x, on CUDA device)
 */
torch::Tensor zconv2_torch(
    torch::Tensor x,
    torch::Tensor h,
    torch::Tensor mup
) {
    // ===== Input Validation =====
    TORCH_CHECK(x.is_cuda(), "Input tensor x must be a CUDA tensor");
    TORCH_CHECK(h.is_cuda(), "Filter tensor h must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 2, "Input tensor x must be 2D");
    TORCH_CHECK(h.dim() == 2, "Filter tensor h must be 2D");
    TORCH_CHECK(
        mup.dim() == 2 && mup.size(0) == 2 && mup.size(1) == 2, 
        "Upsampling matrix mup must be 2x2"
    );

    // ===== Data Type and Layout Preparation =====
    // Ensure single precision and contiguous memory layout
    x = x.to(torch::kFloat32).contiguous();
    h = h.to(torch::kFloat32).contiguous();
    
    // Move mup to CPU for parameter extraction (int32)
    auto mup_cpu = mup.to(torch::kCPU, torch::kInt32).contiguous();

    // ===== Extract Dimensions =====
    const int s_row_len = x.size(0);
    const int s_col_len = x.size(1);
    const int f_row_len = h.size(0);
    const int f_col_len = h.size(1);

    // ===== Extract Upsampling Matrix Parameters =====
    // mup = [[M0, M1], [M2, M3]]
    const int* mup_ptr = mup_cpu.data_ptr<int>();
    const int M0 = mup_ptr[0];  // mup[0, 0]
    const int M1 = mup_ptr[1];  // mup[0, 1]
    const int M2 = mup_ptr[2];  // mup[1, 0]
    const int M3 = mup_ptr[3];  // mup[1, 1]

    // ===== Calculate Initial Offsets =====
    // These match the original zconv2.cpp algorithm
    const int new_f_row_len = (M0 - 1) * (f_row_len - 1) + 
                              M2 * (f_col_len - 1) + f_row_len - 1;
    const int new_f_col_len = (M3 - 1) * (f_col_len - 1) + 
                              M1 * (f_row_len - 1) + f_col_len - 1;
    
    const int start1 = new_f_row_len / 2;
    const int start2 = new_f_col_len / 2;
    const int mn1_init = start1 % s_row_len;
    const int mn2_save = start2 % s_col_len;

    // ===== Allocate Output Tensor =====
    auto y = torch::empty_like(x);

    // ===== Launch CUDA Kernel =====
    zconv2_launcher(x, h, y, M0, M1, M2, M3, mn1_init, mn2_save);

    return y;
}

// ===== Python Module Binding =====
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "CUDA-accelerated 2D convolution with upsampled filter for NSCT";
    
    m.def(
        "zconv2", 
        &zconv2_torch, 
        "Performs 2D convolution with upsampled filter using CUDA (periodic boundary)",
        py::arg("x"),
        py::arg("h"),
        py::arg("mup")
    );
}
