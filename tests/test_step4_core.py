import pytest
import scipy.io as sio
import numpy as np
from nsct_python import core, filters

# Define the path to the test data
TEST_DATA_PATH = "test_data/"

def test_nssfbdec():
    """
    Tests the nssfbdec function (two-channel nonsubsampled filter bank decomposition).
    Corresponds to test_data/step4_core_decomposition.mat
    """
    data = sio.loadmat(TEST_DATA_PATH + 'step4_core_decomposition.mat')

    # Load inputs and expected outputs from the .mat file
    test_image = data['test_image']
    y0_exp = data['y0']
    y1_exp = data['y1']

    # Replicate the filter generation from the MATLAB test script
    h0, h1 = filters.dfilters('pkva', 'd')

    # Run the Python implementation
    # The MATLAB script calls nssfbdec(image, h0, h1) without an upsampling matrix,
    # so we do the same here.
    y0_act, y1_act = core.nssfbdec(test_image, h0, h1)

    # Compare the results
    assert np.allclose(y0_act, y0_exp, atol=1e-9)
    assert np.allclose(y1_act, y1_exp, atol=1e-9)