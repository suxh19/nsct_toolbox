"""
Test suite for utils.py functions
Tests Python implementation against MATLAB reference outputs
"""

import numpy as np
import pytest
from scipy.io import loadmat
import sys
from pathlib import Path

# Add parent directory to path to import nsct_python module
sys.path.insert(0, str(Path(__file__).parent.parent))

from nsct_python.utils import extend2, upsample2df, modulate2, resampz, qupz


# Load MATLAB test results
MATLAB_RESULTS_PATH = Path(__file__).parent.parent / 'data/test_utils_results.mat'


@pytest.fixture(scope='module')
def matlab_results():
    """Load MATLAB reference results"""
    if not MATLAB_RESULTS_PATH.exists():
        pytest.skip(f"MATLAB results file not found: {MATLAB_RESULTS_PATH}")
    
    mat_data = loadmat(str(MATLAB_RESULTS_PATH))
    return mat_data['test_results'][0, 0]


class TestExtend2:
    """Test cases for extend2 function"""
    
    def test_extend2_periodic_basic(self, matlab_results):
        """Test extend2 with periodic extension (basic)"""
        test_data = matlab_results['test1'][0, 0]
        
        input_mat = test_data['input']
        ru = int(test_data['ru'][0, 0])
        rd = int(test_data['rd'][0, 0])
        cl = int(test_data['cl'][0, 0])
        cr = int(test_data['cr'][0, 0])
        extmod = str(test_data['extmod'][0])
        expected_output = test_data['output']
        
        result = extend2(input_mat, ru, rd, cl, cr, extmod)
        
        assert result.shape == expected_output.shape, \
            f"Shape mismatch: {result.shape} vs {expected_output.shape}"
        np.testing.assert_array_equal(result, expected_output)
        print(f"✓ Test 1 passed: extend2 periodic basic ({result.shape})")
    
    def test_extend2_periodic_small(self, matlab_results):
        """Test extend2 with periodic extension (small)"""
        test_data = matlab_results['test2'][0, 0]
        
        input_mat = test_data['input']
        ru = int(test_data['ru'][0, 0])
        rd = int(test_data['rd'][0, 0])
        cl = int(test_data['cl'][0, 0])
        cr = int(test_data['cr'][0, 0])
        extmod = str(test_data['extmod'][0])
        expected_output = test_data['output']
        
        result = extend2(input_mat, ru, rd, cl, cr, extmod)
        
        assert result.shape == expected_output.shape, \
            f"Shape mismatch: {result.shape} vs {expected_output.shape}"
        np.testing.assert_array_equal(result, expected_output)
        print(f"✓ Test 2 passed: extend2 periodic small ({result.shape})")
    
    def test_extend2_qper_row(self, matlab_results):
        """Test extend2 with quincunx periodic extension (row)"""
        test_data = matlab_results['test3'][0, 0]
        
        input_mat = test_data['input']
        ru = int(test_data['ru'][0, 0])
        rd = int(test_data['rd'][0, 0])
        cl = int(test_data['cl'][0, 0])
        cr = int(test_data['cr'][0, 0])
        extmod = str(test_data['extmod'][0])
        expected_output = test_data['output']
        
        result = extend2(input_mat, ru, rd, cl, cr, extmod)
        
        assert result.shape == expected_output.shape, \
            f"Shape mismatch: {result.shape} vs {expected_output.shape}"
        np.testing.assert_array_equal(result, expected_output)
        print(f"✓ Test 3 passed: extend2 qper_row ({result.shape})")
    
    def test_extend2_qper_col(self, matlab_results):
        """Test extend2 with quincunx periodic extension (col)"""
        test_data = matlab_results['test4'][0, 0]
        
        input_mat = test_data['input']
        ru = int(test_data['ru'][0, 0])
        rd = int(test_data['rd'][0, 0])
        cl = int(test_data['cl'][0, 0])
        cr = int(test_data['cr'][0, 0])
        extmod = str(test_data['extmod'][0])
        expected_output = test_data['output']
        
        result = extend2(input_mat, ru, rd, cl, cr, extmod)
        
        assert result.shape == expected_output.shape, \
            f"Shape mismatch: {result.shape} vs {expected_output.shape}"
        np.testing.assert_array_equal(result, expected_output)
        print(f"✓ Test 4 passed: extend2 qper_col ({result.shape})")


class TestUpsample2df:
    """Test cases for upsample2df function"""
    
    def test_upsample2df_power1(self, matlab_results):
        """Test upsample2df with power=1"""
        test_data = matlab_results['test5'][0, 0]
        
        input_mat = test_data['input']
        power = int(test_data['power'][0, 0])
        expected_output = test_data['output']
        
        result = upsample2df(input_mat, power)
        
        assert result.shape == expected_output.shape, \
            f"Shape mismatch: {result.shape} vs {expected_output.shape}"
        np.testing.assert_array_equal(result, expected_output)
        print(f"✓ Test 5 passed: upsample2df power=1 ({result.shape})")
    
    def test_upsample2df_power2(self, matlab_results):
        """Test upsample2df with power=2"""
        test_data = matlab_results['test6'][0, 0]
        
        input_mat = test_data['input']
        power = int(test_data['power'][0, 0])
        expected_output = test_data['output']
        
        result = upsample2df(input_mat, power)
        
        assert result.shape == expected_output.shape, \
            f"Shape mismatch: {result.shape} vs {expected_output.shape}"
        np.testing.assert_array_equal(result, expected_output)
        print(f"✓ Test 6 passed: upsample2df power=2 ({result.shape})")


class TestModulate2:
    """Test cases for modulate2 function"""
    
    def test_modulate2_row(self, matlab_results):
        """Test modulate2 with row modulation"""
        test_data = matlab_results['test7'][0, 0]
        
        input_mat = test_data['input']
        mod_type = str(test_data['type'][0])
        center = test_data['center'][0].tolist()
        expected_output = test_data['output']
        
        result = modulate2(input_mat, mod_type, center)
        
        assert result.shape == expected_output.shape, \
            f"Shape mismatch: {result.shape} vs {expected_output.shape}"
        np.testing.assert_array_almost_equal(result, expected_output, decimal=10)
        print(f"✓ Test 7 passed: modulate2 row ({result.shape})")
    
    def test_modulate2_column(self, matlab_results):
        """Test modulate2 with column modulation"""
        test_data = matlab_results['test8'][0, 0]
        
        input_mat = test_data['input']
        mod_type = str(test_data['type'][0])
        center = test_data['center'][0].tolist()
        expected_output = test_data['output']
        
        result = modulate2(input_mat, mod_type, center)
        
        assert result.shape == expected_output.shape, \
            f"Shape mismatch: {result.shape} vs {expected_output.shape}"
        np.testing.assert_array_almost_equal(result, expected_output, decimal=10)
        print(f"✓ Test 8 passed: modulate2 column ({result.shape})")
    
    def test_modulate2_both(self, matlab_results):
        """Test modulate2 with both directions"""
        test_data = matlab_results['test9'][0, 0]
        
        input_mat = test_data['input']
        mod_type = str(test_data['type'][0])
        center = test_data['center'][0].tolist()
        expected_output = test_data['output']
        
        result = modulate2(input_mat, mod_type, center)
        
        assert result.shape == expected_output.shape, \
            f"Shape mismatch: {result.shape} vs {expected_output.shape}"
        np.testing.assert_array_almost_equal(result, expected_output, decimal=10)
        print(f"✓ Test 9 passed: modulate2 both ({result.shape})")
    
    def test_modulate2_both_with_center(self, matlab_results):
        """Test modulate2 with both directions and center offset"""
        test_data = matlab_results['test10'][0, 0]
        
        input_mat = test_data['input']
        mod_type = str(test_data['type'][0])
        center = test_data['center'][0].tolist()
        expected_output = test_data['output']
        
        result = modulate2(input_mat, mod_type, center)
        
        assert result.shape == expected_output.shape, \
            f"Shape mismatch: {result.shape} vs {expected_output.shape}"
        np.testing.assert_array_almost_equal(result, expected_output, decimal=10)
        print(f"✓ Test 10 passed: modulate2 both with center ({result.shape})")


class TestResampz:
    """Test cases for resampz function"""
    
    def test_resampz_type1(self, matlab_results):
        """Test resampz type 1 (R1 = [1,1;0,1])"""
        test_data = matlab_results['test11'][0, 0]
        
        input_mat = test_data['input']
        resamp_type = int(test_data['type'][0, 0])
        shift = int(test_data['shift'][0, 0])
        expected_output = test_data['output']
        
        result = resampz(input_mat, resamp_type, shift)
        
        assert result.shape == expected_output.shape, \
            f"Shape mismatch: {result.shape} vs {expected_output.shape}"
        np.testing.assert_array_equal(result, expected_output)
        print(f"✓ Test 11 passed: resampz type 1 ({result.shape})")
    
    def test_resampz_type2(self, matlab_results):
        """Test resampz type 2 (R2 = [1,-1;0,1])"""
        test_data = matlab_results['test12'][0, 0]
        
        input_mat = test_data['input']
        resamp_type = int(test_data['type'][0, 0])
        shift = int(test_data['shift'][0, 0])
        expected_output = test_data['output']
        
        result = resampz(input_mat, resamp_type, shift)
        
        assert result.shape == expected_output.shape, \
            f"Shape mismatch: {result.shape} vs {expected_output.shape}"
        np.testing.assert_array_equal(result, expected_output)
        print(f"✓ Test 12 passed: resampz type 2 ({result.shape})")
    
    def test_resampz_type3(self, matlab_results):
        """Test resampz type 3 (R3 = [1,0;1,1])"""
        test_data = matlab_results['test13'][0, 0]
        
        input_mat = test_data['input']
        resamp_type = int(test_data['type'][0, 0])
        shift = int(test_data['shift'][0, 0])
        expected_output = test_data['output']
        
        result = resampz(input_mat, resamp_type, shift)
        
        assert result.shape == expected_output.shape, \
            f"Shape mismatch: {result.shape} vs {expected_output.shape}"
        np.testing.assert_array_equal(result, expected_output)
        print(f"✓ Test 13 passed: resampz type 3 ({result.shape})")
    
    def test_resampz_type4(self, matlab_results):
        """Test resampz type 4 (R4 = [1,0;-1,1])"""
        test_data = matlab_results['test14'][0, 0]
        
        input_mat = test_data['input']
        resamp_type = int(test_data['type'][0, 0])
        shift = int(test_data['shift'][0, 0])
        expected_output = test_data['output']
        
        result = resampz(input_mat, resamp_type, shift)
        
        assert result.shape == expected_output.shape, \
            f"Shape mismatch: {result.shape} vs {expected_output.shape}"
        np.testing.assert_array_equal(result, expected_output)
        print(f"✓ Test 14 passed: resampz type 4 ({result.shape})")
    
    def test_resampz_type1_shift2(self, matlab_results):
        """Test resampz type 1 with shift=2"""
        test_data = matlab_results['test15'][0, 0]
        
        input_mat = test_data['input']
        resamp_type = int(test_data['type'][0, 0])
        shift = int(test_data['shift'][0, 0])
        expected_output = test_data['output']
        
        result = resampz(input_mat, resamp_type, shift)
        
        assert result.shape == expected_output.shape, \
            f"Shape mismatch: {result.shape} vs {expected_output.shape}"
        np.testing.assert_array_equal(result, expected_output)
        print(f"✓ Test 15 passed: resampz type 1 shift=2 ({result.shape})")


class TestQupz:
    """Test cases for qupz function"""
    
    def test_qupz_type1_small(self, matlab_results):
        """Test qupz type 1 with 2x2 matrix"""
        test_data = matlab_results['test16'][0, 0]
        
        input_mat = test_data['input']
        qupz_type = int(test_data['type'][0, 0])
        expected_output = test_data['output']
        
        result = qupz(input_mat, qupz_type)
        
        assert result.shape == expected_output.shape, \
            f"Shape mismatch: {result.shape} vs {expected_output.shape}"
        np.testing.assert_array_equal(result, expected_output)
        print(f"✓ Test 16 passed: qupz type 1 2x2 ({result.shape})")
    
    def test_qupz_type2_small(self, matlab_results):
        """Test qupz type 2 with 2x2 matrix"""
        test_data = matlab_results['test17'][0, 0]
        
        input_mat = test_data['input']
        qupz_type = int(test_data['type'][0, 0])
        expected_output = test_data['output']
        
        result = qupz(input_mat, qupz_type)
        
        assert result.shape == expected_output.shape, \
            f"Shape mismatch: {result.shape} vs {expected_output.shape}"
        np.testing.assert_array_equal(result, expected_output)
        print(f"✓ Test 17 passed: qupz type 2 2x2 ({result.shape})")
    
    def test_qupz_type1_large(self, matlab_results):
        """Test qupz type 1 with 3x3 matrix"""
        test_data = matlab_results['test18'][0, 0]
        
        input_mat = test_data['input']
        qupz_type = int(test_data['type'][0, 0])
        expected_output = test_data['output']
        
        result = qupz(input_mat, qupz_type)
        
        assert result.shape == expected_output.shape, \
            f"Shape mismatch: {result.shape} vs {expected_output.shape}"
        np.testing.assert_array_equal(result, expected_output)
        print(f"✓ Test 18 passed: qupz type 1 3x3 ({result.shape})")
    
    def test_qupz_type2_large(self, matlab_results):
        """Test qupz type 2 with 3x3 matrix"""
        test_data = matlab_results['test19'][0, 0]
        
        input_mat = test_data['input']
        qupz_type = int(test_data['type'][0, 0])
        expected_output = test_data['output']
        
        result = qupz(input_mat, qupz_type)
        
        assert result.shape == expected_output.shape, \
            f"Shape mismatch: {result.shape} vs {expected_output.shape}"
        np.testing.assert_array_equal(result, expected_output)
        print(f"✓ Test 19 passed: qupz type 2 3x3 ({result.shape})")


if __name__ == '__main__':
    # Run tests with verbose output
    pytest.main([__file__, '-v', '-s'])
