"""
Test suite for nsfbdec function against MATLAB reference implementation.

This module tests the Python implementation of nsfbdec (Nonsubsampled Filter Bank Decomposition)
against MATLAB reference results to ensure numerical accuracy.
"""

import pytest
import numpy as np
import scipy.io as sio
from nsct_python.core import nsfbdec
from nsct_python.filters import atrousfilters


class TestNsfbdec:
    """Test nsfbdec function against MATLAB reference implementation."""
    
    @classmethod
    def setup_class(cls):
        """Load MATLAB test results once for all tests."""
        mat_data = sio.loadmat('data/test_nsfbdec_results.mat')
        cls.mat_data = mat_data
        
        # Store filters
        cls.h0 = mat_data['h0']
        cls.h1 = mat_data['h1']
        cls.g0 = mat_data['g0']
        cls.g1 = mat_data['g1']
        
        cls.h0_97 = mat_data['h0_97']
        cls.h1_97 = mat_data['h1_97']
        cls.g0_97 = mat_data['g0_97']
        cls.g1_97 = mat_data['g1_97']
    
    def test_level_0_decomposition(self):
        """Test 1: Level 0 decomposition (32x32 image)."""
        x = self.mat_data['x1']
        lev = int(self.mat_data['lev1'][0, 0])
        y0_matlab = self.mat_data['y0_1']
        y1_matlab = self.mat_data['y1_1']
        
        y0, y1 = nsfbdec(x, self.h0, self.h1, lev)
        
        # Check shapes
        assert y0.shape == y0_matlab.shape, f"y0 shape mismatch: {y0.shape} vs {y0_matlab.shape}"
        assert y1.shape == y1_matlab.shape, f"y1 shape mismatch: {y1.shape} vs {y1_matlab.shape}"
        
        # Check numerical accuracy
        np.testing.assert_allclose(y0, y0_matlab, rtol=1e-10, atol=1e-12,
                                   err_msg="y0 values don't match MATLAB")
        np.testing.assert_allclose(y1, y1_matlab, rtol=1e-10, atol=1e-12,
                                   err_msg="y1 values don't match MATLAB")
    
    def test_level_1_decomposition(self):
        """Test 2: Level 1 decomposition (64x64 image)."""
        x = self.mat_data['x2']
        lev = int(self.mat_data['lev2'][0, 0])
        y0_matlab = self.mat_data['y0_2']
        y1_matlab = self.mat_data['y1_2']
        
        y0, y1 = nsfbdec(x, self.h0, self.h1, lev)
        
        assert y0.shape == y0_matlab.shape
        assert y1.shape == y1_matlab.shape
        
        np.testing.assert_allclose(y0, y0_matlab, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(y1, y1_matlab, rtol=1e-10, atol=1e-12)
    
    def test_level_2_decomposition(self):
        """Test 3: Level 2 decomposition (128x128 image)."""
        x = self.mat_data['x3']
        lev = int(self.mat_data['lev3'][0, 0])
        y0_matlab = self.mat_data['y0_3']
        y1_matlab = self.mat_data['y1_3']
        
        y0, y1 = nsfbdec(x, self.h0, self.h1, lev)
        
        assert y0.shape == y0_matlab.shape
        assert y1.shape == y1_matlab.shape
        
        np.testing.assert_allclose(y0, y0_matlab, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(y1, y1_matlab, rtol=1e-10, atol=1e-12)
    
    def test_level_3_decomposition(self):
        """Test 4: Level 3 decomposition (256x256 image)."""
        x = self.mat_data['x4']
        lev = int(self.mat_data['lev4'][0, 0])
        y0_matlab = self.mat_data['y0_4']
        y1_matlab = self.mat_data['y1_4']
        
        y0, y1 = nsfbdec(x, self.h0, self.h1, lev)
        
        assert y0.shape == y0_matlab.shape
        assert y1.shape == y1_matlab.shape
        
        np.testing.assert_allclose(y0, y0_matlab, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(y1, y1_matlab, rtol=1e-10, atol=1e-12)
    
    def test_non_square_level_0(self):
        """Test 5: Non-square image (32x48) at level 0."""
        x = self.mat_data['x5']
        lev = int(self.mat_data['lev5'][0, 0])
        y0_matlab = self.mat_data['y0_5']
        y1_matlab = self.mat_data['y1_5']
        
        y0, y1 = nsfbdec(x, self.h0, self.h1, lev)
        
        assert y0.shape == y0_matlab.shape
        assert y1.shape == y1_matlab.shape
        
        np.testing.assert_allclose(y0, y0_matlab, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(y1, y1_matlab, rtol=1e-10, atol=1e-12)
    
    def test_non_square_level_1(self):
        """Test 6: Non-square image (64x96) at level 1."""
        x = self.mat_data['x6']
        lev = int(self.mat_data['lev6'][0, 0])
        y0_matlab = self.mat_data['y0_6']
        y1_matlab = self.mat_data['y1_6']
        
        y0, y1 = nsfbdec(x, self.h0, self.h1, lev)
        
        assert y0.shape == y0_matlab.shape
        assert y1.shape == y1_matlab.shape
        
        np.testing.assert_allclose(y0, y0_matlab, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(y1, y1_matlab, rtol=1e-10, atol=1e-12)
    
    def test_9_7_filters_level_0(self):
        """Test 7: Using 9-7 filters at level 0."""
        x = self.mat_data['x7']
        lev = int(self.mat_data['lev7'][0, 0])
        y0_matlab = self.mat_data['y0_7']
        y1_matlab = self.mat_data['y1_7']
        
        y0, y1 = nsfbdec(x, self.h0_97, self.h1_97, lev)
        
        assert y0.shape == y0_matlab.shape
        assert y1.shape == y1_matlab.shape
        
        np.testing.assert_allclose(y0, y0_matlab, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(y1, y1_matlab, rtol=1e-10, atol=1e-12)
    
    def test_9_7_filters_level_1(self):
        """Test 8: Using 9-7 filters at level 1."""
        x = self.mat_data['x8']
        lev = int(self.mat_data['lev8'][0, 0])
        y0_matlab = self.mat_data['y0_8']
        y1_matlab = self.mat_data['y1_8']
        
        y0, y1 = nsfbdec(x, self.h0_97, self.h1_97, lev)
        
        assert y0.shape == y0_matlab.shape
        assert y1.shape == y1_matlab.shape
        
        np.testing.assert_allclose(y0, y0_matlab, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(y1, y1_matlab, rtol=1e-10, atol=1e-12)
    
    def test_fixed_pattern(self):
        """Test 9: Fixed pattern image at level 0."""
        x = self.mat_data['x9']
        lev = int(self.mat_data['lev9'][0, 0])
        y0_matlab = self.mat_data['y0_9']
        y1_matlab = self.mat_data['y1_9']
        
        y0, y1 = nsfbdec(x, self.h0, self.h1, lev)
        
        assert y0.shape == y0_matlab.shape
        assert y1.shape == y1_matlab.shape
        
        np.testing.assert_allclose(y0, y0_matlab, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(y1, y1_matlab, rtol=1e-10, atol=1e-12)
    
    def test_energy_conservation(self):
        """Test 10: Energy conservation check."""
        x = self.mat_data['x10']
        lev = int(self.mat_data['lev10'][0, 0])
        energy_in_matlab = float(self.mat_data['energy_in'].item())
        energy_out_matlab = float(self.mat_data['energy_out'].item())
        
        y0, y1 = nsfbdec(x, self.h0, self.h1, lev)
        
        # Calculate energy
        energy_in = np.sum(x**2)
        energy_out = np.sum(y0**2) + np.sum(y1**2)
        energy_ratio = energy_out / energy_in
        
        # Check against MATLAB values
        np.testing.assert_allclose(energy_in, energy_in_matlab, rtol=1e-10,
                                   err_msg="Input energy doesn't match MATLAB")
        np.testing.assert_allclose(energy_out, energy_out_matlab, rtol=1e-10,
                                   err_msg="Output energy doesn't match MATLAB")
        
        # Energy should be approximately conserved (ratio close to 1)
        # Note: Perfect conservation may not hold due to filter design
        assert 0.9 < energy_ratio < 1.1, f"Energy ratio {energy_ratio} out of expected range"


class TestNsfbdecEdgeCases:
    """Test edge cases and special conditions for nsfbdec."""
    
    @classmethod
    def setup_class(cls):
        """Setup filters for edge case tests."""
        # Note: Using atrousfilters from MATLAB-generated data instead 
        # since 'maxflat' is not fully implemented in atrousfilters yet
        mat_data = sio.loadmat('data/test_nsfbdec_results.mat')
        cls.h0 = mat_data['h0']
        cls.h1 = mat_data['h1']
    
    def test_minimum_image_size_level_0(self):
        """Test with minimum recommended image size for level 0."""
        x = np.random.rand(32, 32)
        
        y0, y1 = nsfbdec(x, self.h0, self.h1, 0)
        
        assert y0.shape == x.shape
        assert y1.shape == x.shape
        assert np.isfinite(y0).all()
        assert np.isfinite(y1).all()
    
    def test_minimum_image_size_level_1(self):
        """Test with minimum recommended image size for level 1."""
        x = np.random.rand(64, 64)
        
        y0, y1 = nsfbdec(x, self.h0, self.h1, 1)
        
        assert y0.shape == x.shape
        assert y1.shape == x.shape
        assert np.isfinite(y0).all()
        assert np.isfinite(y1).all()
    
    def test_zero_image(self):
        """Test with zero image."""
        x = np.zeros((64, 64))
        
        y0, y1 = nsfbdec(x, self.h0, self.h1, 0)
        
        # Output should also be zero (or very close)
        np.testing.assert_allclose(y0, 0, atol=1e-14)
        np.testing.assert_allclose(y1, 0, atol=1e-14)
    
    def test_constant_image(self):
        """Test with constant image."""
        x = np.ones((64, 64)) * 5.0
        
        y0, y1 = nsfbdec(x, self.h0, self.h1, 0)
        
        # y0 (lowpass) should contain most of the constant value
        # y1 (highpass) should be close to zero for constant input
        assert np.abs(np.mean(y0) - 5.0) < 1.0
        assert np.abs(np.mean(y1)) < 1.0
    
    def test_impulse_response(self):
        """Test with impulse (single pixel) to verify filter behavior."""
        x = np.zeros((64, 64))
        x[32, 32] = 1.0  # Single impulse at center
        
        y0, y1 = nsfbdec(x, self.h0, self.h1, 0)
        
        # Both outputs should have energy concentrated around the center
        assert np.sum(y0**2) > 0
        assert np.sum(y1**2) > 0
        
        # Find peak locations
        y0_peak_idx = np.unravel_index(np.argmax(np.abs(y0)), y0.shape)
        y1_peak_idx = np.unravel_index(np.argmax(np.abs(y1)), y1.shape)
        
        # Peaks should be near the center
        assert abs(y0_peak_idx[0] - 32) < 10
        assert abs(y0_peak_idx[1] - 32) < 10
        assert abs(y1_peak_idx[0] - 32) < 10
        assert abs(y1_peak_idx[1] - 32) < 10
    
    def test_different_filter_sizes(self):
        """Test that filters with different sizes work correctly."""
        # Use 9-7 filters from MATLAB data
        mat_data = sio.loadmat('data/test_nsfbdec_results.mat')
        h0_97 = mat_data['h0_97']
        h1_97 = mat_data['h1_97']
        x = np.random.rand(64, 64)
        
        y0, y1 = nsfbdec(x, h0_97, h1_97, 0)
        
        assert y0.shape == x.shape
        assert y1.shape == x.shape
        assert np.isfinite(y0).all()
        assert np.isfinite(y1).all()
    
    def test_multiple_levels_consistency(self):
        """Test that higher levels produce consistent output shapes."""
        x = np.random.rand(128, 128)
        
        for lev in range(3):
            y0, y1 = nsfbdec(x, self.h0, self.h1, lev)
            assert y0.shape == x.shape, f"Level {lev} y0 shape mismatch"
            assert y1.shape == x.shape, f"Level {lev} y1 shape mismatch"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
