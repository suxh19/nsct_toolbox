"""
Test suite for nsfbrec function against MATLAB reference implementation.

This module tests the Python implementation of nsfbrec (Nonsubsampled Filter Bank Reconstruction)
against MATLAB reference results to ensure numerical accuracy and perfect reconstruction property.
"""

import pytest
import numpy as np
import scipy.io as sio
from nsct_python.core import nsfbdec, nsfbrec
from nsct_python.filters import atrousfilters


class TestNsfbrec:
    """Test nsfbrec function against MATLAB reference implementation."""
    
    @classmethod
    def setup_class(cls):
        """Load MATLAB test results once for all tests."""
        mat_data = sio.loadmat('data/test_nsfbrec_results.mat')
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
    
    def test_perfect_reconstruction_level_0(self):
        """Test 1: Perfect reconstruction at level 0 (32x32)."""
        x = self.mat_data['x1']
        y0 = self.mat_data['y0_1']
        y1 = self.mat_data['y1_1']
        lev = int(self.mat_data['lev1'][0, 0])
        x_rec_matlab = self.mat_data['x_rec_1']
        err_matlab = float(self.mat_data['err1'][0, 0])
        
        # Reconstruct using Python
        x_rec = nsfbrec(y0, y1, self.g0, self.g1, lev)
        
        # Check shape
        assert x_rec.shape == x_rec_matlab.shape, f"Shape mismatch: {x_rec.shape} vs {x_rec_matlab.shape}"
        
        # Check against MATLAB reconstruction
        np.testing.assert_allclose(x_rec, x_rec_matlab, rtol=1e-10, atol=1e-12,
                                   err_msg="Reconstruction doesn't match MATLAB")
        
        # Check perfect reconstruction property
        err_python = np.linalg.norm(x.ravel() - x_rec.ravel()) / np.linalg.norm(x.ravel())
        assert err_python < 1e-14, f"Reconstruction error too large: {err_python}"
        print(f"  Reconstruction error: {err_python:.6e} (MATLAB: {err_matlab:.6e})")
    
    def test_perfect_reconstruction_level_1(self):
        """Test 2: Perfect reconstruction at level 1 (64x64)."""
        x = self.mat_data['x2']
        y0 = self.mat_data['y0_2']
        y1 = self.mat_data['y1_2']
        lev = int(self.mat_data['lev2'][0, 0])
        x_rec_matlab = self.mat_data['x_rec_2']
        
        x_rec = nsfbrec(y0, y1, self.g0, self.g1, lev)
        
        assert x_rec.shape == x_rec_matlab.shape
        np.testing.assert_allclose(x_rec, x_rec_matlab, rtol=1e-10, atol=1e-12)
        
        err = np.linalg.norm(x.ravel() - x_rec.ravel()) / np.linalg.norm(x.ravel())
        assert err < 1e-14
    
    def test_perfect_reconstruction_level_2(self):
        """Test 3: Perfect reconstruction at level 2 (128x128)."""
        x = self.mat_data['x3']
        y0 = self.mat_data['y0_3']
        y1 = self.mat_data['y1_3']
        lev = int(self.mat_data['lev3'][0, 0])
        x_rec_matlab = self.mat_data['x_rec_3']
        
        x_rec = nsfbrec(y0, y1, self.g0, self.g1, lev)
        
        assert x_rec.shape == x_rec_matlab.shape
        np.testing.assert_allclose(x_rec, x_rec_matlab, rtol=1e-10, atol=1e-12)
        
        err = np.linalg.norm(x.ravel() - x_rec.ravel()) / np.linalg.norm(x.ravel())
        assert err < 1e-14
    
    def test_perfect_reconstruction_level_3(self):
        """Test 4: Perfect reconstruction at level 3 (256x256)."""
        x = self.mat_data['x4']
        y0 = self.mat_data['y0_4']
        y1 = self.mat_data['y1_4']
        lev = int(self.mat_data['lev4'][0, 0])
        x_rec_matlab = self.mat_data['x_rec_4']
        
        x_rec = nsfbrec(y0, y1, self.g0, self.g1, lev)
        
        assert x_rec.shape == x_rec_matlab.shape
        np.testing.assert_allclose(x_rec, x_rec_matlab, rtol=1e-10, atol=1e-12)
        
        err = np.linalg.norm(x.ravel() - x_rec.ravel()) / np.linalg.norm(x.ravel())
        assert err < 1e-14
    
    def test_non_square_level_0(self):
        """Test 5: Non-square reconstruction at level 0 (32x48)."""
        x = self.mat_data['x5']
        y0 = self.mat_data['y0_5']
        y1 = self.mat_data['y1_5']
        lev = int(self.mat_data['lev5'][0, 0])
        x_rec_matlab = self.mat_data['x_rec_5']
        
        x_rec = nsfbrec(y0, y1, self.g0, self.g1, lev)
        
        assert x_rec.shape == x_rec_matlab.shape
        np.testing.assert_allclose(x_rec, x_rec_matlab, rtol=1e-10, atol=1e-12)
        
        err = np.linalg.norm(x.ravel() - x_rec.ravel()) / np.linalg.norm(x.ravel())
        assert err < 1e-14
    
    def test_non_square_level_1(self):
        """Test 6: Non-square reconstruction at level 1 (64x96)."""
        x = self.mat_data['x6']
        y0 = self.mat_data['y0_6']
        y1 = self.mat_data['y1_6']
        lev = int(self.mat_data['lev6'][0, 0])
        x_rec_matlab = self.mat_data['x_rec_6']
        
        x_rec = nsfbrec(y0, y1, self.g0, self.g1, lev)
        
        assert x_rec.shape == x_rec_matlab.shape
        np.testing.assert_allclose(x_rec, x_rec_matlab, rtol=1e-10, atol=1e-12)
        
        err = np.linalg.norm(x.ravel() - x_rec.ravel()) / np.linalg.norm(x.ravel())
        assert err < 1e-14
    
    def test_9_7_filters_level_0(self):
        """Test 7: Using 9-7 filters at level 0 (32x32)."""
        x = self.mat_data['x7']
        y0 = self.mat_data['y0_7']
        y1 = self.mat_data['y1_7']
        lev = int(self.mat_data['lev7'][0, 0])
        x_rec_matlab = self.mat_data['x_rec_7']
        
        x_rec = nsfbrec(y0, y1, self.g0_97, self.g1_97, lev)
        
        assert x_rec.shape == x_rec_matlab.shape
        np.testing.assert_allclose(x_rec, x_rec_matlab, rtol=1e-10, atol=1e-12)
        
        err = np.linalg.norm(x.ravel() - x_rec.ravel()) / np.linalg.norm(x.ravel())
        assert err < 1e-14
    
    def test_9_7_filters_level_1(self):
        """Test 8: Using 9-7 filters at level 1 (64x64)."""
        x = self.mat_data['x8']
        y0 = self.mat_data['y0_8']
        y1 = self.mat_data['y1_8']
        lev = int(self.mat_data['lev8'][0, 0])
        x_rec_matlab = self.mat_data['x_rec_8']
        
        x_rec = nsfbrec(y0, y1, self.g0_97, self.g1_97, lev)
        
        assert x_rec.shape == x_rec_matlab.shape
        np.testing.assert_allclose(x_rec, x_rec_matlab, rtol=1e-10, atol=1e-12)
        
        err = np.linalg.norm(x.ravel() - x_rec.ravel()) / np.linalg.norm(x.ravel())
        assert err < 1e-14
    
    def test_direct_reconstruction(self):
        """Test 9: Direct reconstruction test."""
        x = self.mat_data['x9']
        y0 = self.mat_data['y0_9']
        y1 = self.mat_data['y1_9']
        lev = int(self.mat_data['lev9'][0, 0])
        x_rec_matlab = self.mat_data['x_rec_9']
        
        x_rec = nsfbrec(y0, y1, self.g0, self.g1, lev)
        
        assert x_rec.shape == x_rec_matlab.shape
        np.testing.assert_allclose(x_rec, x_rec_matlab, rtol=1e-10, atol=1e-12)
        
        err = np.linalg.norm(x.ravel() - x_rec.ravel()) / np.linalg.norm(x.ravel())
        assert err < 1e-14
    
    def test_multilevel_consistency(self):
        """Test 10: Multi-level consistency check."""
        x = self.mat_data['x10']
        y0 = self.mat_data['y0_10']
        y1 = self.mat_data['y1_10']
        lev = int(self.mat_data['lev10'][0, 0])
        x_rec_matlab = self.mat_data['x_rec_10']
        
        x_rec = nsfbrec(y0, y1, self.g0, self.g1, lev)
        
        assert x_rec.shape == x_rec_matlab.shape
        np.testing.assert_allclose(x_rec, x_rec_matlab, rtol=1e-10, atol=1e-12)
        
        err = np.linalg.norm(x.ravel() - x_rec.ravel()) / np.linalg.norm(x.ravel())
        assert err < 1e-14


class TestNsfbRoundTrip:
    """Test perfect reconstruction property with forward and inverse transforms."""
    
    @classmethod
    def setup_class(cls):
        """Setup filters for round-trip tests."""
        # Load filters from MATLAB data instead
        mat_data = sio.loadmat('data/test_nsfbrec_results.mat')
        cls.h0 = mat_data['h0']
        cls.h1 = mat_data['h1']
        cls.g0 = mat_data['g0']
        cls.g1 = mat_data['g1']
    
    def test_round_trip_level_0(self):
        """Test round-trip: x -> nsfbdec -> nsfbrec -> x (level 0)."""
        np.random.seed(123)
        x = np.random.rand(64, 64)
        
        # Forward transform
        y0, y1 = nsfbdec(x, self.h0, self.h1, 0)
        
        # Inverse transform
        x_rec = nsfbrec(y0, y1, self.g0, self.g1, 0)
        
        # Check perfect reconstruction
        err = np.linalg.norm(x.ravel() - x_rec.ravel()) / np.linalg.norm(x.ravel())
        assert err < 1e-14, f"Round-trip error: {err}"
        print(f"  Round-trip error (level 0): {err:.6e}")
    
    def test_round_trip_level_1(self):
        """Test round-trip at level 1."""
        np.random.seed(456)
        x = np.random.rand(64, 64)
        
        y0, y1 = nsfbdec(x, self.h0, self.h1, 1)
        x_rec = nsfbrec(y0, y1, self.g0, self.g1, 1)
        
        err = np.linalg.norm(x.ravel() - x_rec.ravel()) / np.linalg.norm(x.ravel())
        assert err < 1e-14
        print(f"  Round-trip error (level 1): {err:.6e}")
    
    def test_round_trip_level_2(self):
        """Test round-trip at level 2."""
        np.random.seed(789)
        x = np.random.rand(128, 128)
        
        y0, y1 = nsfbdec(x, self.h0, self.h1, 2)
        x_rec = nsfbrec(y0, y1, self.g0, self.g1, 2)
        
        err = np.linalg.norm(x.ravel() - x_rec.ravel()) / np.linalg.norm(x.ravel())
        assert err < 1e-14
        print(f"  Round-trip error (level 2): {err:.6e}")
    
    def test_round_trip_constant_image(self):
        """Test round-trip with constant image."""
        x = np.ones((64, 64)) * 3.14159
        
        y0, y1 = nsfbdec(x, self.h0, self.h1, 0)
        x_rec = nsfbrec(y0, y1, self.g0, self.g1, 0)
        
        # For constant image, should have perfect reconstruction
        np.testing.assert_allclose(x, x_rec, rtol=1e-14, atol=1e-14)
    
    def test_round_trip_impulse(self):
        """Test round-trip with impulse."""
        x = np.zeros((64, 64))
        x[32, 32] = 1.0
        
        y0, y1 = nsfbdec(x, self.h0, self.h1, 0)
        x_rec = nsfbrec(y0, y1, self.g0, self.g1, 0)
        
        err = np.linalg.norm(x.ravel() - x_rec.ravel()) / np.linalg.norm(x.ravel())
        assert err < 1e-14


class TestNsfbrecEdgeCases:
    """Test edge cases for nsfbrec."""
    
    @classmethod
    def setup_class(cls):
        """Setup filters for edge case tests."""
        mat_data = sio.loadmat('data/test_nsfbrec_results.mat')
        cls.h0 = mat_data['h0']
        cls.h1 = mat_data['h1']
        cls.g0 = mat_data['g0']
        cls.g1 = mat_data['g1']
    
    def test_zero_inputs(self):
        """Test reconstruction with zero inputs."""
        y0 = np.zeros((64, 64))
        y1 = np.zeros((64, 64))
        
        x_rec = nsfbrec(y0, y1, self.g0, self.g1, 0)
        
        # Output should also be zero
        np.testing.assert_allclose(x_rec, 0, atol=1e-14)
    
    def test_minimum_size_level_0(self):
        """Test with minimum image size for level 0."""
        np.random.seed(999)
        x = np.random.rand(32, 32)
        
        y0, y1 = nsfbdec(x, self.h0, self.h1, 0)
        x_rec = nsfbrec(y0, y1, self.g0, self.g1, 0)
        
        err = np.linalg.norm(x.ravel() - x_rec.ravel()) / np.linalg.norm(x.ravel())
        assert err < 1e-14
    
    def test_minimum_size_level_1(self):
        """Test with minimum image size for level 1."""
        np.random.seed(111)
        x = np.random.rand(64, 64)
        
        y0, y1 = nsfbdec(x, self.h0, self.h1, 1)
        x_rec = nsfbrec(y0, y1, self.g0, self.g1, 1)
        
        err = np.linalg.norm(x.ravel() - x_rec.ravel()) / np.linalg.norm(x.ravel())
        assert err < 1e-14


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
