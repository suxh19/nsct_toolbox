"""Detailed debug for nsdfbdec - compare filter outputs."""
import numpy as np
from nsct_python.core import nsdfbdec, nssfbdec
from nsct_python.filters import dfilters, parafilters
from nsct_python.utils import modulate2
import scipy.io as sio

# Load MATLAB test data
mat_data = sio.loadmat('data/test_nsdfbdec_results.mat')

# Test 2: Level 1 decomposition (which passes)
print("=== Level 1 Decomposition (PASSES) ===")
x = mat_data['x2']
clevels = 1

# Get filters (same as in nsdfbdec)
h1, h2 = dfilters('pkva', 'd')
h1 = h1 / np.sqrt(2)
h2 = h2 / np.sqrt(2)
k1 = modulate2(h1, 'c')
k2 = modulate2(h2, 'c')

# First level decomposition
y1_py, y2_py = nssfbdec(x, k1, k2)
y1_mat = mat_data['y2_1']
y2_mat = mat_data['y2_2']

print(f"Subband 1 max diff: {np.max(np.abs(y1_py - y1_mat)):.12f}")
print(f"Subband 2 max diff: {np.max(np.abs(y2_py - y2_mat)):.12f}")
print()

# Test 3: Level 2 decomposition (which fails)
print("=== Level 2 Decomposition (FAILS) ===")
x = mat_data['x3']
clevels = 2

# Get filters
h1, h2 = dfilters('pkva', 'd')
h1 = h1 / np.sqrt(2)
h2 = h2 / np.sqrt(2)
k1 = modulate2(h1, 'c')
k2 = modulate2(h2, 'c')
f1, f2 = parafilters(h1, h2)

# Quincunx sampling matrix
q1 = np.array([[1, -1], [1, 1]])

# First level
x1, x2 = nssfbdec(x, k1, k2)
print(f"After first level:")
print(f"  x1 shape: {x1.shape}, mean: {np.mean(x1):.6f}")
print(f"  x2 shape: {x2.shape}, mean: {np.mean(x2):.6f}")

# Second level - decompose x1
y_0_py, y_1_py = nssfbdec(x1, k1, k2, q1)
y_0_mat = mat_data['y3_1']
y_1_mat = mat_data['y3_2']

print(f"\nDecomposing x1 with nssfbdec(x1, k1, k2, q1):")
print(f"  Python y[0] shape: {y_0_py.shape}, mean: {np.mean(y_0_py):.6f}")
print(f"  MATLAB y{1} shape: {y_0_mat.shape}, mean: {np.mean(y_0_mat):.6f}")
print(f"  Max diff: {np.max(np.abs(y_0_py - y_0_mat)):.12f}")
print(f"  Python y[1] shape: {y_1_py.shape}, mean: {np.mean(y_1_py):.6f}")
print(f"  MATLAB y{2} shape: {y_1_mat.shape}, mean: {np.mean(y_1_mat):.6f}")
print(f"  Max diff: {np.max(np.abs(y_1_py - y_1_mat)):.12f}")

# Second level - decompose x2
y_2_py, y_3_py = nssfbdec(x2, k1, k2, q1)
y_2_mat = mat_data['y3_3']
y_3_mat = mat_data['y3_4']

print(f"\nDecomposing x2 with nssfbdec(x2, k1, k2, q1):")
print(f"  Python y[2] shape: {y_2_py.shape}, mean: {np.mean(y_2_py):.6f}")
print(f"  MATLAB y{3} shape: {y_2_mat.shape}, mean: {np.mean(y_2_mat):.6f}")
print(f"  Max diff: {np.max(np.abs(y_2_py - y_2_mat)):.12f}")
print(f"  Python y[3] shape: {y_3_py.shape}, mean: {np.mean(y_3_py):.6f}")
print(f"  MATLAB y{4} shape: {y_3_mat.shape}, mean: {np.mean(y_3_mat):.6f}")
print(f"  Max diff: {np.max(np.abs(y_3_py - y_3_mat)):.12f}")

# Check q1 matrix
print(f"\nQuincunx matrix q1:")
print(q1)
print(f"det(q1) = {np.linalg.det(q1):.1f}")
