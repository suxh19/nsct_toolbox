"""
Test suite for nsdfbdec function against MATLAB reference implementation.

This module tests the Python implementation of nsdfbdec (Nonsubsampled Directional
Filter Bank Decomposition) against MATLAB reference results to ensure numerical accuracy.
"""

import pytest
import numpy as np
import scipy.io as sio
from nsct_python.core import nsdfbdec


class TestNsdfbdec:
    """Test nsdfbdec function against MATLAB reference implementation."""
    
    @classmethod
    def setup_class(cls):
        """Load MATLAB test results once for all tests."""
        cls.mat_data = sio.loadmat('data/test_nsdfbdec_results.mat')
    
    def test_level_0_no_decomposition(self):
        """Test 1: Level 0 (no decomposition) - 32x32 image."""
        x = self.mat_data['x1']
        clevels = int(self.mat_data['clevels1'][0, 0])
        y_matlab = [self.mat_data['y1_1']]
        
        y = nsdfbdec(x, 'pkva', clevels)
        
        assert len(y) == 2**clevels, f"Expected {2**clevels} subbands, got {len(y)}"
        assert len(y) == len(y_matlab)
        
        for i in range(len(y)):
            assert y[i].shape == y_matlab[i].shape
            np.testing.assert_allclose(y[i], y_matlab[i], rtol=1e-10, atol=1e-12,
                                       err_msg=f"Subband {i+1} doesn't match MATLAB")
    
    def test_level_1_decomposition(self):
        """Test 2: Level 1 decomposition - 64x64 image."""
        x = self.mat_data['x2']
        clevels = int(self.mat_data['clevels2'][0, 0])
        y_matlab = [self.mat_data['y2_1'], self.mat_data['y2_2']]
        
        y = nsdfbdec(x, 'pkva', clevels)
        
        assert len(y) == 2**clevels
        assert len(y) == len(y_matlab)
        
        for i in range(len(y)):
            assert y[i].shape == y_matlab[i].shape
            np.testing.assert_allclose(y[i], y_matlab[i], rtol=1e-10, atol=1e-12,
                                       err_msg=f"Subband {i+1} doesn't match MATLAB")
    
    def test_level_2_decomposition(self):
        """Test 3: Level 2 decomposition - 64x64 image."""
        x = self.mat_data['x3']
        clevels = int(self.mat_data['clevels3'][0, 0])
        y_matlab = [self.mat_data[f'y3_{i}'] for i in range(1, 5)]
        
        y = nsdfbdec(x, 'pkva', clevels)
        
        assert len(y) == 2**clevels
        assert len(y) == len(y_matlab)
        
        for i in range(len(y)):
            assert y[i].shape == y_matlab[i].shape
            np.testing.assert_allclose(y[i], y_matlab[i], rtol=1e-10, atol=1e-12,
                                       err_msg=f"Subband {i+1} doesn't match MATLAB")
    
    def test_level_3_decomposition(self):
        """Test 4: Level 3 decomposition - 128x128 image."""
        x = self.mat_data['x4']
        clevels = int(self.mat_data['clevels4'][0, 0])
        y_matlab = [self.mat_data[f'y4_{i}'] for i in range(1, 9)]
        
        y = nsdfbdec(x, 'pkva', clevels)
        
        assert len(y) == 2**clevels
        assert len(y) == len(y_matlab)
        
        for i in range(len(y)):
            assert y[i].shape == y_matlab[i].shape
            np.testing.assert_allclose(y[i], y_matlab[i], rtol=1e-10, atol=1e-12,
                                       err_msg=f"Subband {i+1} doesn't match MATLAB")
    
    def test_dmaxflat7_filter(self):
        """Test 5: Level 2 with dmaxflat7 filter - 64x64 image."""
        x = self.mat_data['x5']
        clevels = int(self.mat_data['clevels5'][0, 0])
        y_matlab = [self.mat_data[f'y5_{i}'] for i in range(1, 5)]
        
        y = nsdfbdec(x, 'dmaxflat7', clevels)
        
        assert len(y) == 2**clevels
        
        for i in range(len(y)):
            assert y[i].shape == y_matlab[i].shape
            np.testing.assert_allclose(y[i], y_matlab[i], rtol=1e-10, atol=1e-12,
                                       err_msg=f"Subband {i+1} doesn't match MATLAB")
    
    def test_non_square_image(self):
        """Test 6: Non-square image (64x96), level 2."""
        x = self.mat_data['x6']
        clevels = int(self.mat_data['clevels6'][0, 0])
        y_matlab = [self.mat_data[f'y6_{i}'] for i in range(1, 5)]
        
        y = nsdfbdec(x, 'pkva', clevels)
        
        assert len(y) == 2**clevels
        
        for i in range(len(y)):
            assert y[i].shape == y_matlab[i].shape
            np.testing.assert_allclose(y[i], y_matlab[i], rtol=1e-10, atol=1e-12,
                                       err_msg=f"Subband {i+1} doesn't match MATLAB")
    
    def test_energy_conservation(self):
        """Test 7: Energy conservation check."""
        x = self.mat_data['x7']
        clevels = int(self.mat_data['clevels7'][0, 0])
        energy_in_matlab = self.mat_data['energy_in'].item()
        energy_out_matlab = self.mat_data['energy_out'].item()
        
        y = nsdfbdec(x, 'pkva', clevels)
        
        energy_in = np.sum(x**2)
        energy_out = sum(np.sum(yi**2) for yi in y)
        
        # Check against MATLAB values
        np.testing.assert_allclose(energy_in, energy_in_matlab, rtol=1e-10)
        np.testing.assert_allclose(energy_out, energy_out_matlab, rtol=1e-10)
    
    @pytest.mark.skip(reason="Level 4 decomposition test skipped per user request")
    def test_level_4_decomposition(self):
        """Test 8: Level 4 decomposition - 256x256 image."""
        x = self.mat_data['x8']
        clevels = int(self.mat_data['clevels8'][0, 0])
        y_matlab = [self.mat_data[f'y8_{i}'] for i in range(1, 17)]
        
        y = nsdfbdec(x, 'pkva', clevels)
        
        assert len(y) == 2**clevels
        assert len(y) == 16
        
        for i in range(len(y)):
            assert y[i].shape == y_matlab[i].shape
            np.testing.assert_allclose(y[i], y_matlab[i], rtol=1e-10, atol=1e-12,
                                       err_msg=f"Subband {i+1} doesn't match MATLAB")
    
    def test_small_image(self):
        """Test 9: Small image (32x32), level 1."""
        x = self.mat_data['x9']
        clevels = int(self.mat_data['clevels9'][0, 0])
        y_matlab = [self.mat_data['y9_1'], self.mat_data['y9_2']]
        
        y = nsdfbdec(x, 'pkva', clevels)
        
        assert len(y) == 2**clevels
        
        for i in range(len(y)):
            assert y[i].shape == y_matlab[i].shape
            np.testing.assert_allclose(y[i], y_matlab[i], rtol=1e-10, atol=1e-12)
    
    def test_constant_image(self):
        """Test 10: Constant image (64x64), level 2."""
        x = self.mat_data['x10']
        clevels = int(self.mat_data['clevels10'][0, 0])
        y_matlab = [self.mat_data[f'y10_{i}'] for i in range(1, 5)]
        
        y = nsdfbdec(x, 'pkva', clevels)
        
        assert len(y) == 2**clevels
        
        for i in range(len(y)):
            assert y[i].shape == y_matlab[i].shape
            np.testing.assert_allclose(y[i], y_matlab[i], rtol=1e-10, atol=1e-12)


class TestNsdfbdecEdgeCases:
    """Test edge cases and special conditions for nsdfbdec."""
    
    def test_invalid_clevels_negative(self):
        """Test that negative clevels raises error."""
        x = np.random.rand(64, 64)
        with pytest.raises(ValueError, match="non-negative integer"):
            nsdfbdec(x, 'pkva', -1)
    
    def test_invalid_clevels_float(self):
        """Test that non-integer clevels raises error."""
        x = np.random.rand(64, 64)
        with pytest.raises(ValueError, match="non-negative integer"):
            nsdfbdec(x, 'pkva', 1.5)
    
    def test_invalid_filter_type(self):
        """Test that invalid filter type raises error."""
        x = np.random.rand(64, 64)
        with pytest.raises(TypeError, match="string or dict"):
            nsdfbdec(x, 123, 1)
    
    def test_output_shape_preservation(self):
        """Test that all output subbands have the same shape as input."""
        x = np.random.rand(64, 64)
        y = nsdfbdec(x, 'pkva', 2)
        
        for i, yi in enumerate(y):
            assert yi.shape == x.shape, f"Subband {i} shape {yi.shape} != input shape {x.shape}"
    
    def test_correct_number_of_subbands(self):
        """Test that the number of subbands is 2^clevels."""
        x = np.random.rand(64, 64)
        
        for clevels in range(5):
            y = nsdfbdec(x, 'pkva', clevels)
            expected_count = 2**clevels if clevels > 0 else 1
            assert len(y) == expected_count, f"Level {clevels}: expected {expected_count}, got {len(y)}"
    
    def test_different_image_sizes(self):
        """Test with various image sizes."""
        sizes = [(32, 32), (64, 64), (128, 128), (64, 96)]
        
        for size in sizes:
            x = np.random.rand(*size)
            y = nsdfbdec(x, 'pkva', 2)
            
            assert len(y) == 4
            for yi in y:
                assert yi.shape == size
    
    def test_zero_image(self):
        """Test with zero image."""
        x = np.zeros((64, 64))
        y = nsdfbdec(x, 'pkva', 2)
        
        # All subbands should be close to zero
        for i, yi in enumerate(y):
            assert np.allclose(yi, 0, atol=1e-14), f"Subband {i} not zero"
    
    def test_directional_selectivity(self):
        """Test that different subbands capture different orientations."""
        # Create a simple oriented pattern (horizontal lines)
        x = np.zeros((64, 64))
        x[::4, :] = 1.0  # Horizontal lines
        
        y = nsdfbdec(x, 'pkva', 2)
        
        # Check that subbands have different energies (directional selectivity)
        energies = [np.sum(yi**2) for yi in y]
        
        # At least one subband should capture more energy than others
        max_energy = max(energies)
        min_energy = min(energies)
        
        # There should be significant variation in energies
        assert max_energy > min_energy * 1.1, "No directional selectivity detected"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
