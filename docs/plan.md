# `nsct_python` 到 `nsct_torch` 的迁移计划

## 1. 目标

本项目旨在将 `nsct_python`（一个基于 NumPy 和 C++ 的非下采样轮廓波变换库）迁移到一个新的库 `nsct_torch`。新的实现将利用 PyTorch 和 CUDA 技术，以实现 GPU 加速，从而显著提升大规模图像处理的性能。

## 2. 核心策略

- **后端重写**: 将性能瓶颈，即自定义的 C++ 卷积函数 (`atrousc` 和 `zconv2`)，用 CUDA C++ 重写，以在 GPU 上实现大规模并行计算。
- **前端迁移**: 将所有基于 NumPy 的 Python 代码（包括核心算法、滤波器生成和工具函数）迁移到 PyTorch，使用张量 (Tensors) 作为基本数据结构。
- **无缝集成**: 通过 PyTorch 的 C++ 扩展机制，将后端的 CUDA 内核编译成可直接在 Python 中调用的模块，实现与 PyTorch 生态的无缝集成。
- **确保一致性**: 通过严格的单元测试和端到端测试，确保 `nsct_torch` 的计算结果与 `nsct_python` 在数值上保持一致（在浮点精度误差范围内）。

## 3. 详细步骤 (TODO List)

---

### **阶段一: 项目基础与 CUDA 核心**

#### `[ ]` 1. 项目初始化
- **任务**: 创建 `nsct_torch` 的目录结构和基础环境配置。
- **细节**:
    - 创建根目录 `nsct_torch/`。
    - 创建 Python 包目录: `nsct_torch/nsct_torch/`。
    - 创建 C++/CUDA 源码目录: `nsct_torch/csrc/`。
    - 创建测试目录: `nsct_torch/tests/`。
    - 初始化 `setup.py` 用于编译 C++/CUDA 扩展。
    - 创建 `requirements.txt`，包含 `torch`, `numpy` (用于测试对比) 等依赖。
    - 创建 `.gitignore` 文件。

#### `[ ]` 2. CUDA 核心函数开发
- **任务**: 将 `zconv2.cpp` 和 `atrousc.cpp` 的核心逻辑重写为 CUDA 内核。
- **`atrousc` CUDA 内核**:
    - 设计一个 2D CUDA 内核，每个线程计算输出图像的一个像素值。
    - 通过全局内存高效读取输入图像和滤波器。
    - 在内核中直接实现 "à trous" 算法的步进逻辑，避免创建稀疏矩阵。
- **`zconv2` CUDA 内核**:
    - 设计一个 2D CUDA 内核，处理周期性边界条件。
    - 索引计算将是关键，需要精确地将 C++ 中的 `%` (modulo) 逻辑映射到 CUDA。
    - 线程块和网格的划分需要优化，以最大化 GPU 的占用率。

#### `[ ]` 3. PyTorch C++/CUDA 扩展绑定
- **任务**: 使用 PyTorch 的 C++ 扩展功能，将 CUDA 内核封装成可被 Python 调用的 PyTorch 模块。
- **细节**:
    - 创建一个 C++ 接口文件 (`nsct_torch/csrc/nsct_cuda.cpp`)。
    - 在此文件中，使用 `torch/extension.h` 定义 PyTorch 张量与 C++ 数据类型之间的转换。
    - 编写一个 CUDA 包装函数 (`.cu` 文件)，该函数负责调用实际的 CUDA 内核，并处理诸如内存分配、数据拷贝 (Host to Device, Device to Host) 等操作。
    - 使用 `PYBIND11_MODULE` 将 C++ 函数绑定到 Python 模块。
    - 更新 `setup.py` 以使用 `CUDAExtension` 来编译和链接所有 C++/CUDA 文件。

---

### **阶段二: Python 逻辑层迁移**

#### `[ ]` 4. 工具函数迁移
- **任务**: 将 `nsct_python/utils.py` 中的 NumPy 函数等价地迁移到 PyTorch。
- **主要函数**: `symext`, `upsample2df`, `modulate2`, `resampz`, `qupz` 等。
- **策略**:
    - 将所有 `np.array` 操作替换为 `torch.Tensor` 操作。
    - 利用 PyTorch 内置函数，如 `torch.nn.functional.pad` 来替代 `np.pad`。
    - 确保数据类型 (`dtype`) 和设备 (`device`) 在所有操作中保持一致。

#### `[ ]` 5. 滤波器函数迁移
- **任务**: 将 `nsct_python/filters.py` 中的滤波器生成和处理函数迁移到 PyTorch。
- **主要函数**: `dfilters`, `atrousfilters`, `parafilters`, `dmaxflat` 等。
- **策略**:
    - 这些函数主要是数值计算，可以直接将 NumPy 的数学运算替换为等价的 PyTorch 运算。
    - 滤波器应被创建为 `torch.Tensor`，以便后续直接在 GPU 上使用。

#### `[ ]` 6. 核心逻辑迁移
- **任务**: 将 `nsct_python/core.py` 中的分解和重构算法迁移到 PyTorch，并调用新的 CUDA 模块。
- **主要函数**: `nsctdec`, `nsctrec`, `nsfbdec`, `nsdfbdec` 等。
- **策略**:
    - 重写所有函数，使其接受并返回 PyTorch 张量。
    - 在 `nsfbdec` 和 `nssfbdec` 等函数中，将对 `_atrousc_cpp` 和 `_zconv2_cpp` 的调用，替换为对我们新创建的 PyTorch CUDA 模块的调用。
    - 确保所有中间变量（如 `xlo`, `xhi`）都是 PyTorch 张量，并驻留在正确的设备上（CPU 或 GPU）。

---

### **阶段三: 验证、优化与发布**

#### `[ ]` 7. 编写测试用例
- **任务**: 为 `nsct_torch` 库编写单元测试和端到端测试。
- **策略**:
    - **单元测试**: 为 `utils.py` 和 `filters.py` 中的每个函数编写测试，确保其输出与 NumPy 版本一致。
    - **集成测试**: 编写一个端到端的测试，加载一张图像，分别使用 `nsct_python` 和 `nsct_torch` 进行 `nsctdec` 分解和 `nsctrec` 重构。
    - **一致性验证**: 断言两个版本的重构图像之间的差异（例如，均方误差）小于一个非常小的值 (e.g., `1e-6`)，以验证完美重构属性和数值一致性。

#### `[ ]` 8. 性能基准测试
- **任务**: 对比 `nsct_torch` (GPU) 与 `nsct_python` (CPU) 的性能。
- **策略**:
    - 创建一个基准测试脚本 (`benchmark.py`)。
    - 在不同尺寸（例如 256x256, 512x512, 1024x1024）的图像上，多次运行 `nsctdec` 和 `nsctrec` 函数。
    - 记录并打印 CPU (NumPy/C++) 和 GPU (PyTorch/CUDA) 的平均执行时间。
    - 计算并报告性能提升的倍数。

#### `[ ]` 9. 文档和示例
- **任务**: 创建 `README.md` 和示例代码。
- **`README.md` 内容**:
    - 项目简介。
    - **安装指南**: 如何编译 C++/CUDA 扩展 (e.g., `python setup.py install`)。
    - **快速入门**: 一个简单的代码示例，展示如何加载图像、执行 NSCT 分解和重构。
    - **API 参考** (可选): 简要介绍主要函数的功能。
- **示例脚本**: 提供一个或多个 `.py` 文件，作为用户可以直接运行的例子。
