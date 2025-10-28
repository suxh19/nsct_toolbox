# atrousc_cpp - 高性能 À Trous 卷积 C++ 实现

这是一个针对 NSCT (Nonsubsampled Contourlet Transform) 中 à trous 卷积算法的高性能 C++ 实现。

## 特性

- ⚡ **高性能**: C++ 实现比纯 Python 快 200-300 倍
- 🔄 **自动回退**: 如果 C++ 扩展不可用，自动回退到 Python 实现
- 🔌 **易于集成**: 无缝集成到现有的 Python 代码中

## 性能对比

| 实现方式 | 256x256 图像 | 512x512 图像 | 1024x1024 图像 | 加速比 |
|---------|------------|------------|---------------|--------|
| Python  | ~478ms     | ~2172ms    | ~9000ms       | 1.0× |
| C++     | ~1.63ms    | ~8.19ms    | ~43.5ms       | **220-315×** |

*基准测试环境: Windows, Python 3.13, 测试日期: 2025年10月6日*

### 详细性能数据

完整的性能测试结果请参见:
- [TEST_REPORT.md](TEST_REPORT.md) - 详细测试报告
- [PYTHON_IMPLEMENTATION_SUMMARY.md](PYTHON_IMPLEMENTATION_SUMMARY.md) - Python 实现总结

## 安装要求

### 基础要求
- Python 3.7+
- NumPy >= 1.19.0
- pybind11 >= 2.6.0
- C++ 编译器 (支持 C++17)

### Windows 编译器选项
- **推荐**: Visual Studio 2019 或更新版本 (包含 MSVC)
- MinGW-w64 (GCC)

### Linux 编译器
- GCC 7+ 或 Clang 5+

### macOS 编译器
- Xcode Command Line Tools (Clang)

## 编译步骤

### 1. 安装依赖

```bash
# 安装 Python 依赖
pip install numpy pybind11

# Windows: 确保已安装 Visual Studio 或 MinGW
# Linux: 安装 build-essential
# sudo apt-get install build-essential  # Ubuntu/Debian
# sudo yum groupinstall "Development Tools"  # CentOS/RHEL

# macOS: 安装 Xcode Command Line Tools
# xcode-select --install
```

### 2. 编译 C++ 扩展

进入 `atrousc_cpp` 目录并编译：

```bash
cd nsct_python/atrousc_cpp
python setup.py build_ext --inplace
```

### 3. 验证编译

```python
import sys
sys.path.append('path/to/nsct_python/atrousc_cpp')
from atrousc_cpp import atrousc_cpp

# 检查后端信息
from atrousc_cpp import get_backend_info
print(get_backend_info())
# 输出: {'cpp_available': True, 'backend': 'C++', 'import_error': None}
```

## 使用方法

### 方法 1: 直接使用 C++ 模块

```python
import numpy as np
from atrousc_cpp import atrousc

# 准备数据
x = np.random.rand(256, 256)
h = np.random.rand(5, 5)
M = np.array([[2, 0], [0, 2]])

# 使用 C++ 实现（默认）
result = atrousc(x, h, M)

# 或显式指定
result = atrousc(x, h, M, use_cpp=True)

# 强制使用 Python 实现
result = atrousc(x, h, M, use_cpp=False)
```

### 方法 2: 修改 core.py 以使用 C++ 实现

在 `nsct_python/core.py` 中，替换 `_atrousc_equivalent` 函数的调用：

```python
# 在文件顶部添加导入
try:
    from nsct_python.atrousc_cpp import atrousc as _atrousc_cpp
    USE_CPP = True
except ImportError:
    USE_CPP = False

# 在 nsfbdec 和 nsfbrec 函数中，替换调用
if USE_CPP:
    y0 = _atrousc_cpp(x_ext_h0, h0, mup)
    y1 = _atrousc_cpp(x_ext_h1, h1, mup)
else:
    y0 = _atrousc_equivalent(x_ext_h0, h0, mup)  # 原始 Python 实现
    y1 = _atrousc_equivalent(x_ext_h1, h1, mup)
```

## 性能基准测试

### 方法 1: 运行完整测试套件

```bash
cd nsct_python/atrousc_cpp
python benchmark.py
```

这将运行:
- ✅ 功能正确性测试
- ✅ 数值精度对比 (C++ vs Python)
- ✅ 输出大小分析
- ✅ 性能对比测试
- ✅ 不同图像大小的性能测试
- ✅ 不同滤波器大小的性能测试

### 方法 2: 快速测试

```bash
cd nsct_python/atrousc_cpp
python quick_test.py
```

快速验证 C++ 和 Python 实现的正确性和性能差异。

### 方法 3: 自定义测试脚本

```python
import numpy as np
import time
from atrousc_cpp import atrousc, get_backend_info

# 检查后端
print("Backend info:", get_backend_info())

# 测试数据
sizes = [(256, 256), (512, 512), (1024, 1024)]
h = np.random.rand(7, 7)
M = np.array([[4, 0], [0, 4]])

for size in sizes:
    x = np.random.rand(*size)
    
    # C++ 实现
    start = time.time()
    result = atrousc(x, h, M)
    time_cpp = (time.time() - start) * 1000
    
    print(f"图像大小: {size}, 时间: {time_cpp:.2f} ms")
```

### 测试报告

详细的测试结果和分析请参见:
- **[TEST_REPORT.md](TEST_REPORT.md)** - 完整的测试报告，包含所有测试数据
- **[PYTHON_IMPLEMENTATION_SUMMARY.md](PYTHON_IMPLEMENTATION_SUMMARY.md)** - Python 实现的详细分析和对比

## 故障排除

### Windows 常见问题

**问题**: "error: Microsoft Visual C++ 14.0 is required"
**解决方案**: 
- 安装 [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/)
- 或安装完整的 Visual Studio Community Edition

**问题**: 找不到 `cl.exe`
**解决方案**:
```bash
# 使用 Visual Studio Developer Command Prompt
# 或在 PowerShell 中运行:
& "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
```

### Linux 常见问题

**问题**: 缺少编译器
**解决方案**:
```bash
# Ubuntu/Debian
sudo apt-get install build-essential python3-dev

# CentOS/RHEL
sudo yum groupinstall "Development Tools"
sudo yum install python3-devel
```


## 技术细节

### 算法优化

1. **内存连续性**: 使用 C-style 连续数组，优化缓存命中率
2. **循环展开**: 编译器自动优化的友好代码结构
3. **指针算术**: 直接使用指针访问，避免索引计算开销

### 数据类型

- 所有浮点运算使用 `double` (64位)
- 索引和维度使用 `int`
- NumPy 数组自动转换为连续存储

## 贡献

欢迎提交问题和改进建议！

## 许可

与 NSCT Toolbox 主项目保持一致。

## 参考

- NSCT 原始论文: Cunha, A. L., Zhou, J., & Do, M. N. (2006). The Nonsubsampled Contourlet Transform
- pybind11 文档: https://pybind11.readthedocs.io/
