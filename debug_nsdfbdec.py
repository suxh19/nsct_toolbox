"""Debug script for nsdfbdec level 2 decomposition."""
import numpy as np
from nsct_python.core import nsdfbdec
import scipy.io as sio

# Load MATLAB test data
mat_data = sio.loadmat('data/test_nsdfbdec_results.mat')

# Test 3: Level 2 decomposition
print("=== Test Level 2 Decomposition ===")
x = mat_data['x3']
clevels = int(mat_data['clevels3'][0, 0])
num_subbands = 2 ** clevels

print(f"Input shape: {x.shape}")
print(f"Clevels: {clevels}")
print(f"Number of subbands: {num_subbands}")

# Run Python implementation
y_python = nsdfbdec(x, 'pkva', clevels)

print(f"Python output: {len(y_python)} subbands")
print()

# Compare each subband
for i in range(num_subbands):
    y_py = y_python[i]
    y_mat = mat_data[f'y3_{i+1}']
    
    print(f"Subband {i}:")
    print(f"  Python shape: {y_py.shape}")
    print(f"  MATLAB shape: {y_mat.shape}")
    print(f"  Python mean: {np.mean(y_py):.6f}")
    print(f"  MATLAB mean: {np.mean(y_mat):.6f}")
    print(f"  Max diff: {np.max(np.abs(y_py - y_mat)):.10f}")
    print(f"  Shape match: {y_py.shape == y_mat.shape}")
    print()
