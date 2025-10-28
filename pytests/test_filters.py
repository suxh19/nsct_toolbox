"""
Test suite for filters.py functions
Tests Python implementation against MATLAB reference outputs
"""

import numpy as np
import pytest
from scipy.io import loadmat
import sys
from pathlib import Path

# Add parent directory to path to import nsct_python module
sys.path.insert(0, str(Path(__file__).parent.parent))

from nsct_python.filters import (
    ldfilter, ld2quin, dmaxflat, atrousfilters, 
    mctrans, efilter2, parafilters
)


# Load MATLAB test results
MATLAB_RESULTS_PATH = Path(__file__).parent.parent / 'data/test_filters_results.mat'


@pytest.fixture(scope='module')
def matlab_results():
    """Load MATLAB reference results"""
    if not MATLAB_RESULTS_PATH.exists():
        pytest.skip(f"MATLAB results file not found: {MATLAB_RESULTS_PATH}")
    
    mat_data = loadmat(str(MATLAB_RESULTS_PATH))
    return mat_data['test_results'][0, 0]


class TestLdfilter:
    """Test cases for ldfilter function"""
    
    def test_ldfilter_pkva12(self, matlab_results):
        """Test ldfilter with pkva12"""
        test_data = matlab_results['test1'][0, 0]
        
        fname = str(test_data['fname'][0])
        expected_output = test_data['output'][0]
        
        result = ldfilter(fname)
        
        assert result.shape == expected_output.shape, \
            f"Shape mismatch: {result.shape} vs {expected_output.shape}"
        np.testing.assert_array_almost_equal(result, expected_output, decimal=10)
        print(f"✓ Test 1 passed: ldfilter pkva12 (length {len(result)})")
    
    def test_ldfilter_pkva8(self, matlab_results):
        """Test ldfilter with pkva8"""
        test_data = matlab_results['test2'][0, 0]
        
        fname = str(test_data['fname'][0])
        expected_output = test_data['output'][0]
        
        result = ldfilter(fname)
        
        assert result.shape == expected_output.shape, \
            f"Shape mismatch: {result.shape} vs {expected_output.shape}"
        np.testing.assert_array_almost_equal(result, expected_output, decimal=10)
        print(f"✓ Test 2 passed: ldfilter pkva8 (length {len(result)})")
    
    def test_ldfilter_pkva6(self, matlab_results):
        """Test ldfilter with pkva6"""
        test_data = matlab_results['test3'][0, 0]
        
        fname = str(test_data['fname'][0])
        expected_output = test_data['output'][0]
        
        result = ldfilter(fname)
        
        assert result.shape == expected_output.shape, \
            f"Shape mismatch: {result.shape} vs {expected_output.shape}"
        np.testing.assert_array_almost_equal(result, expected_output, decimal=10)
        print(f"✓ Test 3 passed: ldfilter pkva6 (length {len(result)})")


class TestLd2quin:
    """Test cases for ld2quin function"""
    
    def test_ld2quin_pkva6(self, matlab_results):
        """Test ld2quin with pkva6"""
        test_data = matlab_results['test4'][0, 0]
        
        beta = test_data['beta'][0]
        expected_h0 = test_data['h0']
        expected_h1 = test_data['h1']
        
        h0, h1 = ld2quin(beta)
        
        assert h0.shape == expected_h0.shape, \
            f"h0 shape mismatch: {h0.shape} vs {expected_h0.shape}"
        assert h1.shape == expected_h1.shape, \
            f"h1 shape mismatch: {h1.shape} vs {expected_h1.shape}"
        np.testing.assert_array_almost_equal(h0, expected_h0, decimal=10)
        np.testing.assert_array_almost_equal(h1, expected_h1, decimal=10)
        print(f"✓ Test 4 passed: ld2quin pkva6 (h0: {h0.shape}, h1: {h1.shape})")
    
    def test_ld2quin_pkva12(self, matlab_results):
        """Test ld2quin with pkva12"""
        test_data = matlab_results['test5'][0, 0]
        
        beta = test_data['beta'][0]
        expected_h0 = test_data['h0']
        expected_h1 = test_data['h1']
        
        h0, h1 = ld2quin(beta)
        
        assert h0.shape == expected_h0.shape, \
            f"h0 shape mismatch: {h0.shape} vs {expected_h0.shape}"
        assert h1.shape == expected_h1.shape, \
            f"h1 shape mismatch: {h1.shape} vs {expected_h1.shape}"
        np.testing.assert_array_almost_equal(h0, expected_h0, decimal=10)
        np.testing.assert_array_almost_equal(h1, expected_h1, decimal=10)
        print(f"✓ Test 5 passed: ld2quin pkva12 (h0: {h0.shape}, h1: {h1.shape})")


class TestDmaxflat:
    """Test cases for dmaxflat function"""
    
    def test_dmaxflat_N1_d0(self, matlab_results):
        """Test dmaxflat with N=1, d=0"""
        test_data = matlab_results['test6'][0, 0]
        
        N = int(test_data['N'][0, 0])
        d = float(test_data['d'][0, 0])
        expected_output = test_data['output']
        
        result = dmaxflat(N, d)
        
        assert result.shape == expected_output.shape, \
            f"Shape mismatch: {result.shape} vs {expected_output.shape}"
        np.testing.assert_array_almost_equal(result, expected_output, decimal=10)
        print(f"✓ Test 6 passed: dmaxflat N=1, d=0 ({result.shape})")
    
    def test_dmaxflat_N2_d1(self, matlab_results):
        """Test dmaxflat with N=2, d=1"""
        test_data = matlab_results['test7'][0, 0]
        
        N = int(test_data['N'][0, 0])
        d = float(test_data['d'][0, 0])
        expected_output = test_data['output']
        
        result = dmaxflat(N, d)
        
        assert result.shape == expected_output.shape, \
            f"Shape mismatch: {result.shape} vs {expected_output.shape}"
        np.testing.assert_array_almost_equal(result, expected_output, decimal=10)
        print(f"✓ Test 7 passed: dmaxflat N=2, d=1 ({result.shape})")
    
    def test_dmaxflat_N3_d0(self, matlab_results):
        """Test dmaxflat with N=3, d=0"""
        test_data = matlab_results['test8'][0, 0]
        
        N = int(test_data['N'][0, 0])
        d = float(test_data['d'][0, 0])
        expected_output = test_data['output']
        
        result = dmaxflat(N, d)
        
        assert result.shape == expected_output.shape, \
            f"Shape mismatch: {result.shape} vs {expected_output.shape}"
        np.testing.assert_array_almost_equal(result, expected_output, decimal=10)
        print(f"✓ Test 8 passed: dmaxflat N=3, d=0 ({result.shape})")


class TestAtrousfilters:
    """Test cases for atrousfilters function"""
    
    def test_atrousfilters_pyr(self, matlab_results):
        """Test atrousfilters with pyr"""
        test_data = matlab_results['test9'][0, 0]
        
        fname = str(test_data['fname'][0])
        expected_h0 = test_data['h0']
        expected_h1 = test_data['h1']
        expected_g0 = test_data['g0']
        expected_g1 = test_data['g1']
        
        h0, h1, g0, g1 = atrousfilters(fname)
        
        assert h0.shape == expected_h0.shape
        assert h1.shape == expected_h1.shape
        assert g0.shape == expected_g0.shape
        assert g1.shape == expected_g1.shape
        
        np.testing.assert_array_almost_equal(h0, expected_h0, decimal=10)
        np.testing.assert_array_almost_equal(h1, expected_h1, decimal=10)
        np.testing.assert_array_almost_equal(g0, expected_g0, decimal=10)
        np.testing.assert_array_almost_equal(g1, expected_g1, decimal=10)
        print(f"✓ Test 9 passed: atrousfilters pyr (h0: {h0.shape}, h1: {h1.shape}, g0: {g0.shape}, g1: {g1.shape})")
    
    def test_atrousfilters_pyrexc(self, matlab_results):
        """Test atrousfilters with pyrexc"""
        test_data = matlab_results['test10'][0, 0]
        
        fname = str(test_data['fname'][0])
        expected_h0 = test_data['h0']
        expected_h1 = test_data['h1']
        expected_g0 = test_data['g0']
        expected_g1 = test_data['g1']
        
        h0, h1, g0, g1 = atrousfilters(fname)
        
        assert h0.shape == expected_h0.shape
        assert h1.shape == expected_h1.shape
        assert g0.shape == expected_g0.shape
        assert g1.shape == expected_g1.shape
        
        np.testing.assert_array_almost_equal(h0, expected_h0, decimal=10)
        np.testing.assert_array_almost_equal(h1, expected_h1, decimal=10)
        np.testing.assert_array_almost_equal(g0, expected_g0, decimal=10)
        np.testing.assert_array_almost_equal(g1, expected_g1, decimal=10)
        print(f"✓ Test 10 passed: atrousfilters pyrexc (h0: {h0.shape}, h1: {h1.shape}, g0: {g0.shape}, g1: {g1.shape})")


class TestMctrans:
    """Test cases for mctrans function"""
    
    def test_mctrans_simple(self, matlab_results):
        """Test mctrans with simple case"""
        test_data = matlab_results['test11'][0, 0]
        
        b = test_data['b'][0]
        t = test_data['t']
        expected_output = test_data['output']
        
        result = mctrans(b, t)
        
        assert result.shape == expected_output.shape, \
            f"Shape mismatch: {result.shape} vs {expected_output.shape}"
        np.testing.assert_array_almost_equal(result, expected_output, decimal=10)
        print(f"✓ Test 11 passed: mctrans simple ({result.shape})")
    
    def test_mctrans_larger(self, matlab_results):
        """Test mctrans with larger filter"""
        test_data = matlab_results['test12'][0, 0]
        
        b = test_data['b'][0]
        t = test_data['t']
        expected_output = test_data['output']
        
        result = mctrans(b, t)
        
        assert result.shape == expected_output.shape, \
            f"Shape mismatch: {result.shape} vs {expected_output.shape}"
        np.testing.assert_array_almost_equal(result, expected_output, decimal=10)
        print(f"✓ Test 12 passed: mctrans larger ({result.shape})")


class TestEfilter2:
    """Test cases for efilter2 function"""
    
    def test_efilter2_basic(self, matlab_results):
        """Test efilter2 basic filtering"""
        test_data = matlab_results['test13'][0, 0]
        
        x = test_data['x']
        f = test_data['f']
        extmod = str(test_data['extmod'][0])
        shift = test_data['shift'].flatten()
        expected_output = test_data['output']
        
        result = efilter2(x, f, extmod, shift)
        
        assert result.shape == expected_output.shape, \
            f"Shape mismatch: {result.shape} vs {expected_output.shape}"
        np.testing.assert_array_almost_equal(result, expected_output, decimal=10)
        print(f"✓ Test 13 passed: efilter2 basic ({result.shape})")
    
    def test_efilter2_with_shift(self, matlab_results):
        """Test efilter2 with shift"""
        test_data = matlab_results['test14'][0, 0]
        
        x = test_data['x']
        f = test_data['f']
        extmod = str(test_data['extmod'][0])
        shift = test_data['shift'].flatten()
        expected_output = test_data['output']
        
        result = efilter2(x, f, extmod, shift)
        
        assert result.shape == expected_output.shape, \
            f"Shape mismatch: {result.shape} vs {expected_output.shape}"
        np.testing.assert_array_almost_equal(result, expected_output, decimal=10)
        print(f"✓ Test 14 passed: efilter2 with shift ({result.shape})")


class TestParafilters:
    """Test cases for parafilters function"""
    
    def test_parafilters_basic(self, matlab_results):
        """Test parafilters basic case"""
        test_data = matlab_results['test15'][0, 0]
        
        f1 = test_data['f1']
        f2 = test_data['f2']
        expected_y1_1 = test_data['y1_1']
        expected_y1_2 = test_data['y1_2']
        expected_y1_3 = test_data['y1_3']
        expected_y1_4 = test_data['y1_4']
        expected_y2_1 = test_data['y2_1']
        expected_y2_2 = test_data['y2_2']
        expected_y2_3 = test_data['y2_3']
        expected_y2_4 = test_data['y2_4']
        
        y1, y2 = parafilters(f1, f2)
        
        assert len(y1) == 4 and len(y2) == 4
        
        np.testing.assert_array_almost_equal(y1[0], expected_y1_1, decimal=10)
        np.testing.assert_array_almost_equal(y1[1], expected_y1_2, decimal=10)
        np.testing.assert_array_almost_equal(y1[2], expected_y1_3, decimal=10)
        np.testing.assert_array_almost_equal(y1[3], expected_y1_4, decimal=10)
        
        np.testing.assert_array_almost_equal(y2[0], expected_y2_1, decimal=10)
        np.testing.assert_array_almost_equal(y2[1], expected_y2_2, decimal=10)
        np.testing.assert_array_almost_equal(y2[2], expected_y2_3, decimal=10)
        np.testing.assert_array_almost_equal(y2[3], expected_y2_4, decimal=10)
        
        print(f"✓ Test 15 passed: parafilters basic (4 outputs each)")
    
    def test_parafilters_dmaxflat(self, matlab_results):
        """Test parafilters with dmaxflat filters"""
        test_data = matlab_results['test16'][0, 0]
        
        f1 = test_data['f1']
        f2 = test_data['f2']
        expected_y1_1 = test_data['y1_1']
        expected_y1_2 = test_data['y1_2']
        expected_y1_3 = test_data['y1_3']
        expected_y1_4 = test_data['y1_4']
        expected_y2_1 = test_data['y2_1']
        expected_y2_2 = test_data['y2_2']
        expected_y2_3 = test_data['y2_3']
        expected_y2_4 = test_data['y2_4']
        
        y1, y2 = parafilters(f1, f2)
        
        assert len(y1) == 4 and len(y2) == 4
        
        np.testing.assert_array_almost_equal(y1[0], expected_y1_1, decimal=10)
        np.testing.assert_array_almost_equal(y1[1], expected_y1_2, decimal=10)
        np.testing.assert_array_almost_equal(y1[2], expected_y1_3, decimal=10)
        np.testing.assert_array_almost_equal(y1[3], expected_y1_4, decimal=10)
        
        np.testing.assert_array_almost_equal(y2[0], expected_y2_1, decimal=10)
        np.testing.assert_array_almost_equal(y2[1], expected_y2_2, decimal=10)
        np.testing.assert_array_almost_equal(y2[2], expected_y2_3, decimal=10)
        np.testing.assert_array_almost_equal(y2[3], expected_y2_4, decimal=10)
        
        print(f"✓ Test 16 passed: parafilters dmaxflat (4 outputs each)")


if __name__ == '__main__':
    # Run tests with verbose output
    pytest.main([__file__, '-v', '-s'])
