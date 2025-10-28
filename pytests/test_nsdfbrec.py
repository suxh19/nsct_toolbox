"""
Test suite for nsdfbrec (Nonsubsampled Directional Filter Bank Reconstruction).
Tests against MATLAB reference implementation.
"""

import pytest
import numpy as np
import scipy.io as sio
from nsct_python.core import nsdfbdec, nsdfbrec


class TestNsdfbrec:
    """Test nsdfbrec against MATLAB implementation."""
    
    @classmethod
    def setup_class(cls):
        """Load MATLAB test data once for all tests."""
        cls.mat_data = sio.loadmat('data/test_nsdfbrec_results.mat')
    
    def test_perfect_reconstruction_level_1(self):
        """Test 1: Perfect reconstruction - Level 1."""
        x = self.mat_data['x1']
        clevels = int(self.mat_data['clevels1'][0, 0])
        x_rec_matlab = self.mat_data['x1_rec']
        
        # Decompose and reconstruct
        y = nsdfbdec(x, 'pkva', clevels)
        x_rec = nsdfbrec(y, 'pkva')
        
        # Check perfect reconstruction
        assert x_rec.shape == x.shape
        np.testing.assert_allclose(x_rec, x, rtol=1e-10, atol=1e-12,
                                   err_msg="Perfect reconstruction failed")
        np.testing.assert_allclose(x_rec, x_rec_matlab, rtol=1e-10, atol=1e-12,
                                   err_msg="Reconstruction doesn't match MATLAB")
    
    def test_perfect_reconstruction_level_2(self):
        """Test 2: Perfect reconstruction - Level 2."""
        x = self.mat_data['x2']
        clevels = int(self.mat_data['clevels2'][0, 0])
        x_rec_matlab = self.mat_data['x2_rec']
        
        # Decompose and reconstruct
        y = nsdfbdec(x, 'pkva', clevels)
        x_rec = nsdfbrec(y, 'pkva')
        
        # Check perfect reconstruction
        assert x_rec.shape == x.shape
        np.testing.assert_allclose(x_rec, x, rtol=1e-10, atol=1e-12,
                                   err_msg="Perfect reconstruction failed")
        np.testing.assert_allclose(x_rec, x_rec_matlab, rtol=1e-10, atol=1e-12,
                                   err_msg="Reconstruction doesn't match MATLAB")
    
    def test_perfect_reconstruction_level_3(self):
        """Test 3: Perfect reconstruction - Level 3."""
        x = self.mat_data['x3']
        clevels = int(self.mat_data['clevels3'][0, 0])
        x_rec_matlab = self.mat_data['x3_rec']
        
        # Decompose and reconstruct
        y = nsdfbdec(x, 'pkva', clevels)
        x_rec = nsdfbrec(y, 'pkva')
        
        # Check perfect reconstruction
        assert x_rec.shape == x.shape
        np.testing.assert_allclose(x_rec, x, rtol=1e-10, atol=1e-12,
                                   err_msg="Perfect reconstruction failed")
        np.testing.assert_allclose(x_rec, x_rec_matlab, rtol=1e-10, atol=1e-12,
                                   err_msg="Reconstruction doesn't match MATLAB")
    
    def test_dmaxflat7_filter(self):
        """Test 4: Perfect reconstruction - dmaxflat7 filter."""
        x = self.mat_data['x4']
        clevels = int(self.mat_data['clevels4'][0, 0])
        x_rec_matlab = self.mat_data['x4_rec']
        
        # Decompose and reconstruct
        y = nsdfbdec(x, 'dmaxflat7', clevels)
        x_rec = nsdfbrec(y, 'dmaxflat7')
        
        # Check perfect reconstruction
        assert x_rec.shape == x.shape
        np.testing.assert_allclose(x_rec, x, rtol=1e-10, atol=1e-12,
                                   err_msg="Perfect reconstruction failed")
        np.testing.assert_allclose(x_rec, x_rec_matlab, rtol=1e-10, atol=1e-12,
                                   err_msg="Reconstruction doesn't match MATLAB")
    
    def test_non_square_image(self):
        """Test 5: Non-square image."""
        x = self.mat_data['x5']
        clevels = int(self.mat_data['clevels5'][0, 0])
        x_rec_matlab = self.mat_data['x5_rec']
        
        # Decompose and reconstruct
        y = nsdfbdec(x, 'pkva', clevels)
        x_rec = nsdfbrec(y, 'pkva')
        
        # Check perfect reconstruction
        assert x_rec.shape == x.shape
        np.testing.assert_allclose(x_rec, x, rtol=1e-10, atol=1e-12,
                                   err_msg="Perfect reconstruction failed")
        np.testing.assert_allclose(x_rec, x_rec_matlab, rtol=1e-10, atol=1e-12,
                                   err_msg="Reconstruction doesn't match MATLAB")
    
    def test_level_0_no_decomposition(self):
        """Test 6: Level 0 (no decomposition)."""
        x = self.mat_data['x6']
        clevels = int(self.mat_data['clevels6'][0, 0])
        x_rec_matlab = self.mat_data['x6_rec']
        
        # Decompose and reconstruct
        y = nsdfbdec(x, 'pkva', clevels)
        x_rec = nsdfbrec(y, 'pkva')
        
        # Check perfect reconstruction
        assert x_rec.shape == x.shape
        np.testing.assert_allclose(x_rec, x, rtol=1e-10, atol=1e-14,
                                   err_msg="Perfect reconstruction failed")
        np.testing.assert_allclose(x_rec, x_rec_matlab, rtol=1e-10, atol=1e-14,
                                   err_msg="Reconstruction doesn't match MATLAB")
    
    @pytest.mark.skip(reason="Large image test (Level 4) skipped per user request")
    def test_large_image_level_4(self):
        """Test 7: Large image - Level 4."""
        x = self.mat_data['x7']
        clevels = int(self.mat_data['clevels7'][0, 0])
        x_rec_matlab = self.mat_data['x7_rec']
        
        # Decompose and reconstruct
        y = nsdfbdec(x, 'pkva', clevels)
        x_rec = nsdfbrec(y, 'pkva')
        
        # Check perfect reconstruction
        assert x_rec.shape == x.shape
        np.testing.assert_allclose(x_rec, x, rtol=1e-10, atol=1e-12,
                                   err_msg="Perfect reconstruction failed")
        np.testing.assert_allclose(x_rec, x_rec_matlab, rtol=1e-10, atol=1e-12,
                                   err_msg="Reconstruction doesn't match MATLAB")
    
    def test_constant_image(self):
        """Test 8: Constant image."""
        x = self.mat_data['x8']
        clevels = int(self.mat_data['clevels8'][0, 0])
        x_rec_matlab = self.mat_data['x8_rec']
        
        # Decompose and reconstruct
        y = nsdfbdec(x, 'pkva', clevels)
        x_rec = nsdfbrec(y, 'pkva')
        
        # Check perfect reconstruction
        assert x_rec.shape == x.shape
        np.testing.assert_allclose(x_rec, x, rtol=1e-10, atol=1e-12,
                                   err_msg="Perfect reconstruction failed")
        np.testing.assert_allclose(x_rec, x_rec_matlab, rtol=1e-10, atol=1e-12,
                                   err_msg="Reconstruction doesn't match MATLAB")


class TestNsdfbrecEdgeCases:
    """Test edge cases and error handling for nsdfbrec."""
    
    def test_invalid_number_of_subbands(self):
        """Test with invalid number of subbands (not power of 2)."""
        x = np.random.rand(32, 32)
        y = [x, x, x]  # 3 subbands (not power of 2)
        
        with pytest.raises(ValueError, match="Number of subbands must be a power of 2"):
            nsdfbrec(y, 'pkva')
    
    def test_empty_subbands(self):
        """Test with empty subbands list."""
        y = []
        
        with pytest.raises((ValueError, OverflowError)):
            nsdfbrec(y, 'pkva')
    
    def test_invalid_filter_type(self):
        """Test with invalid filter type."""
        x = np.random.rand(32, 32)
        # Need at least 2 subbands to trigger filter usage
        y = [x, x]
        
        with pytest.raises((TypeError, AttributeError)):
            nsdfbrec(y, 12345)
    
    def test_reconstruction_energy_conservation(self):
        """Test that reconstruction conserves energy."""
        x = np.random.rand(64, 64)
        
        # Decompose and reconstruct
        y = nsdfbdec(x, 'pkva', 2)
        x_rec = nsdfbrec(y, 'pkva')
        
        # Check energy conservation
        energy_in = np.sum(x**2)
        energy_out = np.sum(x_rec**2)
        
        np.testing.assert_allclose(energy_in, energy_out, rtol=1e-10)
    
    def test_different_decomposition_levels(self):
        """Test reconstruction at different decomposition levels."""
        x = np.random.rand(128, 128)
        
        for clevels in [0, 1, 2, 3]:
            y = nsdfbdec(x, 'pkva', clevels)
            x_rec = nsdfbrec(y, 'pkva')
            
            assert x_rec.shape == x.shape
            np.testing.assert_allclose(x_rec, x, rtol=1e-10, atol=1e-12,
                                       err_msg=f"Reconstruction failed for clevels={clevels}")
    
    def test_zero_image(self):
        """Test reconstruction of zero image."""
        x = np.zeros((64, 64))
        
        y = nsdfbdec(x, 'pkva', 2)
        x_rec = nsdfbrec(y, 'pkva')
        
        assert x_rec.shape == x.shape
        np.testing.assert_allclose(x_rec, 0, atol=1e-14)
    
    def test_different_image_sizes(self):
        """Test reconstruction with different image sizes."""
        sizes = [(32, 32), (64, 64), (128, 128), (64, 96)]
        
        for size in sizes:
            x = np.random.rand(*size)
            y = nsdfbdec(x, 'pkva', 2)
            x_rec = nsdfbrec(y, 'pkva')
            
            assert x_rec.shape == x.shape
            np.testing.assert_allclose(x_rec, x, rtol=1e-10, atol=1e-12,
                                       err_msg=f"Reconstruction failed for size {size}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
