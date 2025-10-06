"""Compare Python nssfbdec with MATLAB zconv2."""
import numpy as np
from nsct_python.core import nssfbdec
import scipy.io as sio

# Load MATLAB data
mat_data = sio.loadmat('data/test_nssfbdec_direct.mat')

x = mat_data['x']
k1 = mat_data['k1']
k2 = mat_data['k2']
q1 = mat_data['q1']

print("Input:")
print(f"  x shape: {x.shape}, mean: {np.mean(x):.6f}")
print(f"  k1 shape: {k1.shape}")
print(f"  k2 shape: {k2.shape}")
print(f"  q1: {q1.flatten()}")
print()

# No upsampling
print("=== No upsampling ===")
y1_no_py, y2_no_py = nssfbdec(x, k1, k2)
y1_no_mat = mat_data['y1_no']
y2_no_mat = mat_data['y2_no']

print(f"Python y1: shape={y1_no_py.shape}, mean={np.mean(y1_no_py):.6f}")
print(f"MATLAB y1: shape={y1_no_mat.shape}, mean={np.mean(y1_no_mat):.6f}")
print(f"Max diff: {np.max(np.abs(y1_no_py - y1_no_mat)):.12f}")
print(f"Python y2: shape={y2_no_py.shape}, mean={np.mean(y2_no_py):.6f}")
print(f"MATLAB y2: shape={y2_no_mat.shape}, mean={np.mean(y2_no_mat):.6f}")
print(f"Max diff: {np.max(np.abs(y2_no_py - y2_no_mat)):.12f}")
print()

# With upsampling
print("=== With quincunx upsampling ===")
y1_q_py, y2_q_py = nssfbdec(x, k1, k2, q1)
y1_q_mat = mat_data['y1_q']
y2_q_mat = mat_data['y2_q']

print(f"Python y1: shape={y1_q_py.shape}, mean={np.mean(y1_q_py):.6f}")
print(f"MATLAB y1: shape={y1_q_mat.shape}, mean={np.mean(y1_q_mat):.6f}")
print(f"Max diff: {np.max(np.abs(y1_q_py - y1_q_mat)):.12f}")
print(f"Python y2: shape={y2_q_py.shape}, mean={np.mean(y2_q_py):.6f}")
print(f"MATLAB y2: shape={y2_q_mat.shape}, mean={np.mean(y2_q_mat):.6f}")
print(f"Max diff: {np.max(np.abs(y2_q_py - y2_q_mat)):.12f}")

# Detailed comparison of a few values
print("\n=== Detailed value comparison (y1_q) ===")
print("Python y1[0:3, 0:3]:")
print(y1_q_py[0:3, 0:3])
print("MATLAB y1[0:3, 0:3]:")
print(y1_q_mat[0:3, 0:3])
print(f"Difference [0:3, 0:3]:")
print(y1_q_py[0:3, 0:3] - y1_q_mat[0:3, 0:3])
