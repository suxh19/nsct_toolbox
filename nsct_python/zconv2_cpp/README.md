# zconv2_cpp - High-Performance 2D Convolution with Upsampled Filter

This module provides a high-performance C++ implementation of 2D convolution with upsampled filter using periodic boundary conditions, optimized for the Nonsubsampled Contourlet Transform (NSCT).

## Overview

The `zconv2` function computes 2D convolution as if the filter had been upsampled by a parallelogram matrix, but without actually upsampling the filter. This approach is significantly more efficient by stepping through zeros intelligently.

## Features

- **High Performance**: C++ implementation using pybind11 for Python integration
- **Periodic Boundary**: Uses periodic boundary conditions for seamless tiling
- **Optimized**: Avoids explicit upsampling by computing convolution directly
- **Compatible**: Drop-in replacement for the pure Python implementation

## Building

### Prerequisites

- Python >= 3.7
- NumPy >= 1.19.0
- pybind11 >= 2.6.0
- C++17 compatible compiler (MSVC on Windows, GCC/Clang on Linux/macOS)

### Installation

```bash
cd nsct_python/zconv2_cpp
python setup.py build_ext --inplace
```

Or install as a package:

```bash
pip install .
```

## Usage

```python
import numpy as np
from nsct_python.zconv2_cpp import zconv2

# Create test data
x = np.random.rand(64, 64)
h = np.random.rand(5, 5)
mup = np.array([[1, -1], [1, 1]])  # Quincunx upsampling matrix

# Compute convolution
y = zconv2(x, h, mup)
```

## Performance

The C++ implementation provides significant speedup compared to pure Python:

- **10-100x faster** depending on input size and upsampling matrix
- Optimized memory access patterns
- Compiler optimizations (O2/O3, fast-math)

## Technical Details

### Algorithm

This implementation is based on the original MATLAB MEX function `zconv2.c` by Jason Laska. The key insight is that upsampling a filter introduces many zeros, and we can skip these zeros during convolution by adjusting the indexing pattern.

### Input Parameters

- `x` (ndarray): Input signal, 2D array
- `h` (ndarray): Filter, 2D array  
- `mup` (ndarray): Upsampling matrix, 2×2 array `[[M0, M1], [M2, M3]]`

### Output

- `y` (ndarray): Convolution result, same size as input `x`

### Boundary Conditions

The implementation uses **periodic boundary conditions**, meaning the input is treated as if it wraps around at the edges.

## Integration with NSCT

This module is designed to work with the NSCT Python implementation in `nsct_python/core.py`. The `_zconv2` function in `core.py` will automatically use the C++ version if available, falling back to pure Python if not.

## Development

### Testing

```bash
python -c "from nsct_python.zconv2_cpp import is_cpp_available, get_backend_info; print(get_backend_info())"
```

### Debugging

To build with debug symbols:

```bash
python setup.py build_ext --inplace --debug
```

## References

- Original C implementation: `zconv2.c` by Jason Laska (2004)
- Nonsubsampled Contourlet Transform: da Cunha, Zhou, and Do (2006)

## License

This code is part of the NSCT Toolbox and follows the same license.
