import pytest
import scipy.io as sio
import numpy as np
from nsct_python import filters

# Define the path to the test data
TEST_DATA_PATH = "test_data/"

def test_efilter2():
    """
    Tests the efilter2 function for convolution with edge handling.
    Corresponds to test_data/step2_efilter2.mat
    """
    data = sio.loadmat(TEST_DATA_PATH + 'step2_efilter2.mat')

    test_image = data['test_image']
    test_filter = data['small_filter']

    # Test Case 1: Periodic extension (matches 'result1')
    expected_result1 = data['result1']
    actual_result1 = filters.efilter2(test_image, test_filter, 'per')
    assert np.allclose(actual_result1, expected_result1, atol=1e-9)

    # Test Case 2: Quincunx periodic row extension (matches 'result3')
    expected_result3 = data['result3']
    actual_result3 = filters.efilter2(test_image, test_filter, 'qper_row')
    assert np.allclose(actual_result3, expected_result3, atol=1e-9)

def test_dmaxflat():
    """
    Tests the dmaxflat function for generating diamond maxflat filters.
    Corresponds to test_data/step3_dmaxflat.mat
    """
    data = sio.loadmat(TEST_DATA_PATH + 'step3_dmaxflat.mat')

    # Test for N=1, 2, 3 with d=1.0, which matches the MATLAB test data
    for N in range(1, 4):
        expected_h = data[f'h{N}']
        actual_h = filters.dmaxflat(N, d=1.0)
        assert np.allclose(actual_h, expected_h, atol=1e-9)

    # Test for N=1 with d=0
    expected_h1_d0 = data['h1_d0']
    actual_h1_d0 = filters.dmaxflat(1, d=0.0)
    assert np.allclose(actual_h1_d0, expected_h1_d0, atol=1e-9)

def test_atrousfilters():
    """
    Tests the atrousfilters function for generating pyramid filters.
    Corresponds to test_data/step3_atrousfilters.mat
    """
    data = sio.loadmat(TEST_DATA_PATH + 'step3_atrousfilters.mat')

    # --- Test Case 1: 'pyr' filter ---
    fname_pyr = data['fname3'][0]
    h0_exp_pyr, h1_exp_pyr = data['h0_3'], data['h1_3']
    g0_exp_pyr, g1_exp_pyr = data['g0_3'], data['g1_3']

    h0_act_pyr, h1_act_pyr, g0_act_pyr, g1_act_pyr = filters.atrousfilters(fname_pyr)

    assert np.allclose(h0_act_pyr, h0_exp_pyr, atol=1e-9)
    assert np.allclose(h1_act_pyr, h1_exp_pyr, atol=1e-9)
    assert np.allclose(g0_act_pyr, g0_exp_pyr, atol=1e-9)
    assert np.allclose(g1_act_pyr, g1_exp_pyr, atol=1e-9)

    # --- Test Case 2: 'pyrexc' filter ---
    fname_pyrexc = data['fname4'][0]
    h0_exp_pyrexc, h1_exp_pyrexc = data['h0_4'], data['h1_4']
    g0_exp_pyrexc, g1_exp_pyrexc = data['g0_4'], data['g1_4']

    h0_act_pyrexc, h1_act_pyrexc, g0_act_pyrexc, g1_act_pyrexc = filters.atrousfilters(fname_pyrexc)

    assert np.allclose(h0_act_pyrexc, h0_exp_pyrexc, atol=1e-9)
    assert np.allclose(h1_act_pyrexc, h1_exp_pyrexc, atol=1e-9)
    assert np.allclose(g0_act_pyrexc, g0_exp_pyrexc, atol=1e-9)
    assert np.allclose(g1_act_pyrexc, g1_exp_pyrexc, atol=1e-9)

def test_dfilters():
    """
    Tests the dfilters function for generating directional filters.
    Corresponds to test_data/step3_dfilters.mat
    """
    data = sio.loadmat(TEST_DATA_PATH + 'step3_dfilters.mat')

    # Test the 'pkva' case, which is fully implemented in Python
    fname = data['fname1'][0]
    ftype = data['type1'][0]
    h0_exp = data['h0_1']
    h1_exp = data['h1_1']

    h0_act, h1_act = filters.dfilters(fname, ftype)

    assert np.allclose(h0_act, h0_exp, atol=1e-9)
    assert np.allclose(h1_act, h1_exp, atol=1e-9)

def test_parafilters():
    """
    Tests the parafilters function for generating parallelogram filters.
    Corresponds to test_data/step3_parafilters.mat
    """
    data = sio.loadmat(TEST_DATA_PATH + 'step3_parafilters.mat')

    # Use the first test case from the MATLAB script
    h1_in = data['h1']
    h2_in = data['h2']
    p0_exp = data['p0'].ravel() # .ravel() to flatten the object array
    p1_exp = data['p1'].ravel()

    p0_act, p1_act = filters.parafilters(h1_in, h2_in)

    # Compare the lists of filters
    assert len(p0_act) == len(p0_exp)
    for act, exp in zip(p0_act, p0_exp):
        assert np.allclose(act, exp, atol=1e-9)

    assert len(p1_act) == len(p1_exp)
    for act, exp in zip(p1_act, p1_exp):
        assert np.allclose(act, exp, atol=1e-9)