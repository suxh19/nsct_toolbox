"""
Simple demonstration of symext function
"""

import numpy as np
from nsct_python.utils import symext

print("=" * 60)
print("symext (Symmetric Extension) Function Demonstration")
print("=" * 60)

# Example 1: Basic usage
print("\nExample 1: Basic 4x4 image with 3x3 filter")
print("-" * 60)
x = np.arange(16).reshape(4, 4)
h = np.ones((3, 3))
shift = [1, 1]

print("Input image (4x4):")
print(x)
print(f"\nFilter size: {h.shape}")
print(f"Shift: {shift}")

result = symext(x, h, shift)
print(f"\nOutput size: {result.shape}")
print("Extended image:")
print(result)

# Example 2: Verify the symmetry property
print("\n" + "=" * 60)
print("Example 2: Verify symmetry at boundaries")
print("-" * 60)

x = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12],
              [13, 14, 15, 16]])
h = np.ones((3, 3))
shift = [1, 1]

result = symext(x, h, shift)

print("Original image:")
print(x)
print("\nExtended image:")
print(result)

print(f"\nLeft edge reflects right:")
print(f"  result[2, 0] = {result[2, 0]}, should reflect x[2, 0] = {x[2, 0]}")
print(f"\nTop edge reflects:")
print(f"  result[0, 2] = {result[0, 2]}, should reflect x[0, 2] = {x[0, 2]}")

# Example 3: Use case - prepare for convolution
print("\n" + "=" * 60)
print("Example 3: Typical use case - preparing for convolution")
print("-" * 60)

from scipy.signal import convolve2d

x = np.random.rand(8, 8)
h = np.array([[1, 2, 1],
              [2, 4, 2],
              [1, 2, 1]]) / 16  # Gaussian-like filter
shift = [1, 1]

# Extend the image
x_ext = symext(x, h, shift)

# Perform convolution with 'valid' mode
# The result will have the same size as the original image
result = convolve2d(x_ext, h, mode='valid')

print(f"Original image size: {x.shape}")
print(f"Extended image size: {x_ext.shape}")
print(f"Convolution result size: {result.shape}")
print(f"\nThe extended size is exactly right so that")
print(f"conv2(extended, filter, 'valid') gives original size!")

# Example 4: Different shift values
print("\n" + "=" * 60)
print("Example 4: Effect of different shift values")
print("-" * 60)

x = np.arange(16).reshape(4, 4)
h = np.ones((3, 3))

for shift_val in [[0, 0], [1, 1], [-1, -1], [0, 1]]:
    result = symext(x, h, shift_val)
    print(f"shift={shift_val}: output size = {result.shape}")

print("\n" + "=" * 60)
print("All examples completed successfully!")
print("=" * 60)
