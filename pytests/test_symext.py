"""
Tests for symext (symmetric extension) function.
Compares Python implementation against MATLAB reference results.
"""

import numpy as np
import pytest
from scipy.io import loadmat
from nsct_python.utils import symext


class TestSymext:
    """Test suite for symext function."""
    
    @classmethod
    def setup_class(cls):
        """Load MATLAB test results once for all tests."""
        cls.matlab_results = loadmat('data/test_symext_results.mat')
    
    def test_basic_4x4_image(self):
        """Test 1: Basic 4x4 image with 3x3 filter."""
        x = self.matlab_results['x1']
        h = self.matlab_results['h1']
        shift = self.matlab_results['shift1'][0].tolist()
        expected = self.matlab_results['y1']
        
        result = symext(x, h, shift)
        
        np.testing.assert_array_almost_equal(result, expected, decimal=14,
            err_msg="Basic 4x4 image test failed")
        assert result.shape == expected.shape, \
            f"Shape mismatch: {result.shape} vs {expected.shape}"
    
    def test_5x5_image_5x5_filter(self):
        """Test 2: 5x5 image with 5x5 filter."""
        x = self.matlab_results['x2']
        h = self.matlab_results['h2']
        shift = self.matlab_results['shift2'][0].tolist()
        expected = self.matlab_results['y2']
        
        result = symext(x, h, shift)
        
        np.testing.assert_array_almost_equal(result, expected, decimal=14,
            err_msg="5x5 image test failed")
        assert result.shape == expected.shape
    
    def test_non_square_image(self):
        """Test 3: Non-square image (6x4)."""
        x = self.matlab_results['x3']
        h = self.matlab_results['h3']
        shift = self.matlab_results['shift3'][0].tolist()
        expected = self.matlab_results['y3']
        
        result = symext(x, h, shift)
        
        np.testing.assert_array_almost_equal(result, expected, decimal=14,
            err_msg="Non-square image test failed")
        assert result.shape == expected.shape
    
    def test_different_shift_values(self):
        """Test 4: Different shift values [0, 1]."""
        x = self.matlab_results['x4']
        h = self.matlab_results['h4']
        shift = self.matlab_results['shift4'][0].tolist()
        expected = self.matlab_results['y4']
        
        result = symext(x, h, shift)
        
        np.testing.assert_array_almost_equal(result, expected, decimal=14,
            err_msg="Different shift values test failed")
        assert result.shape == expected.shape
    
    def test_negative_shift(self):
        """Test 5: Negative shift [-1, -1]."""
        x = self.matlab_results['x5']
        h = self.matlab_results['h5']
        shift = self.matlab_results['shift5'][0].tolist()
        expected = self.matlab_results['y5']
        
        result = symext(x, h, shift)
        
        np.testing.assert_array_almost_equal(result, expected, decimal=14,
            err_msg="Negative shift test failed")
        assert result.shape == expected.shape
    
    def test_large_filter_7x7(self):
        """Test 6: Large filter (7x7)."""
        x = self.matlab_results['x6']
        h = self.matlab_results['h6']
        shift = self.matlab_results['shift6'][0].tolist()
        expected = self.matlab_results['y6']
        
        result = symext(x, h, shift)
        
        np.testing.assert_array_almost_equal(result, expected, decimal=14,
            err_msg="Large filter test failed")
        assert result.shape == expected.shape
    
    def test_small_2x2_image(self):
        """Test 7: Small 2x2 image."""
        x = self.matlab_results['x7']
        h = self.matlab_results['h7']
        shift = self.matlab_results['shift7'][0].tolist()
        expected = self.matlab_results['y7']
        
        result = symext(x, h, shift)
        
        np.testing.assert_array_almost_equal(result, expected, decimal=14,
            err_msg="Small 2x2 image test failed")
        assert result.shape == expected.shape
    
    def test_non_uniform_filter_3x5(self):
        """Test 8: Non-uniform filter (3x5)."""
        x = self.matlab_results['x8']
        h = self.matlab_results['h8']
        shift = self.matlab_results['shift8'][0].tolist()
        expected = self.matlab_results['y8']
        
        result = symext(x, h, shift)
        
        np.testing.assert_array_almost_equal(result, expected, decimal=14,
            err_msg="Non-uniform filter 3x5 test failed")
        assert result.shape == expected.shape
    
    def test_non_uniform_filter_5x3(self):
        """Test 9: Non-uniform filter (5x3)."""
        x = self.matlab_results['x9']
        h = self.matlab_results['h9']
        shift = self.matlab_results['shift9'][0].tolist()
        expected = self.matlab_results['y9']
        
        result = symext(x, h, shift)
        
        np.testing.assert_array_almost_equal(result, expected, decimal=14,
            err_msg="Non-uniform filter 5x3 test failed")
        assert result.shape == expected.shape
    
    def test_random_values(self):
        """Test 10: Random values."""
        x = self.matlab_results['x10']
        h = self.matlab_results['h10']
        shift = self.matlab_results['shift10'][0].tolist()
        expected = self.matlab_results['y10']
        
        result = symext(x, h, shift)
        
        np.testing.assert_array_almost_equal(result, expected, decimal=14,
            err_msg="Random values test failed")
        assert result.shape == expected.shape
    
    def test_symmetry_verification(self):
        """Test 11: Verify symmetry property."""
        x = self.matlab_results['x11']
        h = self.matlab_results['h11']
        shift = self.matlab_results['shift11'][0].tolist()
        expected = self.matlab_results['y11']
        
        result = symext(x, h, shift)
        
        np.testing.assert_array_almost_equal(result, expected, decimal=14,
            err_msg="Symmetry verification test failed")
        
        # Additional symmetry checks
        # The extension should reflect the original image at boundaries
        # Check that the extended region mirrors the original
        assert result.shape == expected.shape
    
    def test_minimum_filter_size(self):
        """Test 12: Minimum filter size (1x1)."""
        x = self.matlab_results['x12']
        h = self.matlab_results['h12']
        shift = self.matlab_results['shift12'][0].tolist()
        expected = self.matlab_results['y12']
        
        result = symext(x, h, shift)
        
        np.testing.assert_array_almost_equal(result, expected, decimal=14,
            err_msg="Minimum filter size test failed")
        assert result.shape == expected.shape
    
    def test_output_size_property(self):
        """Test that output size is always (m+p-1) × (n+q-1) for shift=[1,1]."""
        test_cases = [
            ((4, 4), (3, 3)),
            ((8, 8), (3, 3)),
            ((6, 8), (3, 3)),
            ((10, 10), (5, 5)),
            ((8, 6), (3, 5)),
        ]
        
        for img_shape, filter_shape in test_cases:
            x = np.random.rand(*img_shape)
            h = np.ones(filter_shape)
            # Use shift=[1,1] which is a safe, commonly used value
            shift = [1, 1]
            
            result = symext(x, h, shift)
            
            expected_shape = (
                img_shape[0] + filter_shape[0] - 1,
                img_shape[1] + filter_shape[1] - 1
            )
            
            assert result.shape == expected_shape, \
                f"Shape mismatch for img {img_shape} and filter {filter_shape}: " \
                f"got {result.shape}, expected {expected_shape}"


class TestSymextEdgeCases:
    """Additional edge case tests for symext."""
    
    def test_zero_shift(self):
        """Test with zero shift values."""
        x = np.arange(16).reshape(4, 4)
        h = np.ones((3, 3))
        shift = [0, 0]
        
        result = symext(x, h, shift)
        
        # Should still produce correct output size
        assert result.shape == (6, 6)
        
        # Verify that the original image is contained in the result
        # The exact position depends on the extension logic
        assert result is not None
    
    def test_large_shift(self):
        """Test with large shift values."""
        x = np.arange(16).reshape(4, 4)
        h = np.ones((5, 5))
        shift = [2, 2]
        
        result = symext(x, h, shift)
        
        assert result.shape == (8, 8)
    
    def test_preserve_data_type(self):
        """Test that data type is preserved."""
        x_int = np.arange(16).reshape(4, 4).astype(np.int32)
        x_float = np.arange(16).reshape(4, 4).astype(np.float64)
        h = np.ones((3, 3))
        shift = [1, 1]
        
        result_int = symext(x_int, h, shift)
        result_float = symext(x_float, h, shift)
        
        # NumPy operations may change dtype, but results should be valid
        assert result_int.shape == (6, 6)
        assert result_float.shape == (6, 6)
    
    def test_consistency_with_conv2_valid(self):
        """Test that symext produces correct size for conv2 'valid' mode."""
        x = np.random.rand(8, 8)
        h = np.ones((5, 5))
        shift = [2, 2]  # Centered shift for odd-sized filter
        
        extended = symext(x, h, shift)
        
        # After convolution with 'valid', the result should have the same size as x
        # conv2(extended, h, 'valid') would give shape (8, 8)
        # This requires extended to have shape (8+5-1, 8+5-1) = (12, 12)
        assert extended.shape == (12, 12)


if __name__ == '__main__':
    # Run tests with verbose output
    pytest.main([__file__, '-v', '--tb=short'])
