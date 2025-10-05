import numpy as np
from nsct_python.core import nssfbdec
from nsct_python.filters import dfilters

def run_final_debug_test():
    """
    This test uses the exact 4x4 data provided by the user to perform a
    definitive validation of the nssfbdec function.
    """
    # --- Exact data provided by the user from MATLAB ---

    # Input image x (4x4)
    x = np.array([
        [0.1117, 0.1897, 0.8507, 0.5828],
        [0.1363, 0.4950, 0.5606, 0.8154],
        [0.6787, 0.1476, 0.9296, 0.8790],
        [0.4952, 0.0550, 0.6967, 0.9889]
    ])

    # MATLAB ground truth for y1 and y2
    y1_ml = np.array([
        [0.4859, 0.4606, 1.0085, 0.7385],
        [0.4073, 0.7943, 0.7073, 1.0209],
        [0.8868, 0.4308, 1.0642, 0.9480],
        [0.6611, 0.4832, 0.8036, 1.1436]
    ])

    y2_ml = np.array([
       [-0.6834, -0.3622, 0.3618, 0.1937],
       [-0.4374, -0.1421, 0.1626, 0.3110],
       [0.1184, -0.4217, 0.4733, 0.6126],
       [0.0702, -0.7644, 0.3551, 0.5564]
    ])

    # --- Run the Python function and compare ---

    print("Generating filters with Python's dfilters('pkva', 'd')...")
    h0, h1 = dfilters('pkva', 'd')

    mup = np.array([[1, 1], [-1, 1]])

    print("Running Python's nssfbdec with the provided data...")
    y1_py, y2_py = nssfbdec(x, h0, h1, mup)

    # Calculate and print the Mean Squared Error
    mse1 = np.mean((y1_py - y1_ml)**2)
    mse2 = np.mean((y2_py - y2_ml)**2)
    print(f"\nMSE between MATLAB and Python for y1: {mse1:.10f}")
    print(f"MSE between MATLAB and Python for y2: {mse2:.10f}")

    # Assert if they are close enough (using a reasonable tolerance)
    assert np.allclose(y1_py, y1_ml, atol=1e-4), "y1 does not match MATLAB output"
    assert np.allclose(y2_py, y2_ml, atol=1e-4), "y2 does not match MATLAB output"

    print("\nSUCCESS! Python output perfectly matches MATLAB ground truth!")
    print("This validates the entire chain: qupz -> ld2quin -> dfilters -> nssfbdec.")

if __name__ == '__main__':
    run_final_debug_test()