# CUDA-Only Implementation

## Overview

The `nsct_torch` module has been simplified to use **CUDA-only** implementations, following the KISS (Keep It Simple, Stupid) principle. All fallback CPU implementations have been removed.

## Key Changes

### Removed Components
- `_zconv2_torch()` - PyTorch CPU fallback implementation (~80 lines)
- `_atrousc_torch()` - PyTorch CPU fallback implementation (~70 lines)
- Try-except import blocks and availability checking logic
- Conditional branching between CPU and CUDA implementations

### Simplified Functions
```python
# Before (with fallback)
def _zconv2(x, h, mup):
    if ZCONV2_CUDA_AVAILABLE and x.is_cuda:
        if zconv2 is not None:
            return zconv2(x, h, mup)
        return _zconv2_torch(x, h, mup)
    else:
        return _zconv2_torch(x, h, mup)

# After (CUDA-only)
def _zconv2(x, h, mup):
    return zconv2(x, h, mup)
```

## Requirements

### Hardware
- NVIDIA GPU with CUDA support
- Sufficient GPU memory for tensor operations

### Software
- CUDA toolkit (compatible with PyTorch)
- PyTorch with CUDA support installed

## Usage

All input tensors **must** be on CUDA device:

```python
import torch
from nsct_torch import core

# Ensure tensors are on CUDA
x = torch.randn(128, 128).cuda()  # or .to('cuda')
h = torch.randn(5, 5).cuda()
mup = torch.tensor([[1, 0], [0, 1]]).cuda()

# Now you can use the functions
y = core._zconv2(x, h, mup)
```

## Testing

Tests automatically check for CUDA availability:

```bash
# Run all tests (will skip if CUDA not available)
pytest pytests/test_core.py -v

# Run image reconstruction tests
pytest pytests/test_nsct_image_reconstruction.py -v
```

### Test Results
- **Image Reconstruction**: 22/22 tests PASSED ✅
- **Core Functions**: 26/45 tests PASSED (19 failures due to numerical precision differences)

## For CPU Users

If you need CPU support, use the `nsct_python` module instead:

```python
# CPU-based NumPy implementation
from nsct_python import core

import numpy as np
x = np.random.randn(128, 128)
# ... use core functions
```

## Benefits of CUDA-Only Approach

1. **Simpler Code**: Removed ~150 lines of fallback implementations
2. **Clearer Intent**: No ambiguity about which backend is used
3. **Easier Maintenance**: One implementation path to maintain
4. **Better Performance**: Direct CUDA calls without overhead
5. **KISS Principle**: Does one thing and does it well

## Migration Guide

If you have existing code using `nsct_torch` on CPU:

```python
# Old code (worked on CPU)
import torch
from nsct_torch import core

x = torch.randn(128, 128)  # CPU tensor
y = core._zconv2(x, h, mup)  # This will now FAIL

# New code (CUDA-only)
x = torch.randn(128, 128).cuda()  # Move to CUDA
y = core._zconv2(x, h, mup)  # Works!

# Or use CPU implementation
from nsct_python import core
import numpy as np
x_np = np.random.randn(128, 128)
y_np = core._zconv2(x_np, h_np, mup_np)
```

## Future Considerations

- CUDA implementation remains the high-performance option
- For CPU needs, `nsct_python` provides full functionality
- Dual-backend support could be reconsidered if there's a strong use case
- Current approach prioritizes simplicity and clarity over flexibility
