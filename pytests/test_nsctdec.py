"""
Test suite for nsctdec (Nonsubsampled Contourlet Transform Decomposition).
Tests against MATLAB reference implementation.
"""

import pytest
import numpy as np
import scipy.io as sio
from nsct_python.core import nsctdec


class TestNsctdec:
    """Test nsctdec against MATLAB implementation."""
    
    @classmethod
    def setup_class(cls):
        """Load MATLAB test data once for all tests."""
        cls.mat_data = sio.loadmat('data/test_nsctdec_results.mat')
    
    def test_single_pyramid_level_2_directional(self):
        """Test 1: Single pyramid level, 2 directional levels."""
        x = self.mat_data['x1']
        levels = [int(v) for v in self.mat_data['levels1'][0]]
        y_lowpass_matlab = self.mat_data['y1_lowpass']
        y_num_dir = int(self.mat_data['y1_num_dir'][0, 0])
        
        # Run Python implementation
        y = nsctdec(x, levels, 'pkva', 'maxflat')
        
        # Check structure
        assert len(y) == 2, f"Expected 2 output subbands, got {len(y)}"
        
        # Check lowpass
        assert y[0].shape == y_lowpass_matlab.shape
        np.testing.assert_allclose(y[0], y_lowpass_matlab, rtol=1e-10, atol=1e-12,
                                   err_msg="Lowpass subband doesn't match MATLAB")
        
        # Check bandpass directional subbands
        assert len(y[1]) == y_num_dir, f"Expected {y_num_dir} directional subbands"
        
        for k in range(y_num_dir):
            y_band_matlab = self.mat_data[f'y1_band_{k+1}']
            assert y[1][k].shape == y_band_matlab.shape
            np.testing.assert_allclose(y[1][k], y_band_matlab, rtol=1e-10, atol=1e-12,
                                       err_msg=f"Bandpass subband {k+1} doesn't match MATLAB")
    
    def test_two_pyramid_levels(self):
        """Test 2: Two pyramid levels, [2, 3] directional levels."""
        x = self.mat_data['x2']
        levels = [int(v) for v in self.mat_data['levels2'][0]]
        y_lowpass_matlab = self.mat_data['y2_lowpass']
        y_num_dir1 = int(self.mat_data['y2_num_dir1'][0, 0])
        y_num_dir2 = int(self.mat_data['y2_num_dir2'][0, 0])
        
        # Run Python implementation
        y = nsctdec(x, levels, 'pkva', 'maxflat')
        
        # Check structure
        assert len(y) == 3, f"Expected 3 output subbands, got {len(y)}"
        
        # Check lowpass
        assert y[0].shape == y_lowpass_matlab.shape
        np.testing.assert_allclose(y[0], y_lowpass_matlab, rtol=1e-10, atol=1e-12)
        
        # Check first pyramid level (4 directional subbands)
        assert len(y[1]) == y_num_dir1
        for k in range(y_num_dir1):
            y_band_matlab = self.mat_data[f'y2_band1_{k+1}']
            np.testing.assert_allclose(y[1][k], y_band_matlab, rtol=1e-10, atol=1e-12,
                                       err_msg=f"Level 1 subband {k+1} doesn't match")
        
        # Check second pyramid level (8 directional subbands)
        assert len(y[2]) == y_num_dir2
        for k in range(y_num_dir2):
            y_band_matlab = self.mat_data[f'y2_band2_{k+1}']
            np.testing.assert_allclose(y[2][k], y_band_matlab, rtol=1e-10, atol=1e-12,
                                       err_msg=f"Level 2 subband {k+1} doesn't match")
    
    def test_three_pyramid_levels(self):
        """Test 3: Three pyramid levels, [2, 3, 4] directional levels."""
        x = self.mat_data['x3']
        levels = [int(v) for v in self.mat_data['levels3'][0]]
        y_lowpass_matlab = self.mat_data['y3_lowpass']
        
        # Run Python implementation
        y = nsctdec(x, levels, 'pkva', 'maxflat')
        
        # Check structure
        assert len(y) == 4, f"Expected 4 output subbands, got {len(y)}"
        
        # Check lowpass
        assert y[0].shape == y_lowpass_matlab.shape
        np.testing.assert_allclose(y[0], y_lowpass_matlab, rtol=1e-10, atol=1e-12)
        
        # Check number of directional subbands at each level
        assert len(y[1]) == 4, "First level should have 4 directional subbands"
        assert len(y[2]) == 8, "Second level should have 8 directional subbands"
        assert len(y[3]) == 16, "Third level should have 16 directional subbands"
    
    def test_level_0_no_directional(self):
        """Test 4: Level 0 (no directional decomposition)."""
        x = self.mat_data['x4']
        levels = [int(v) for v in self.mat_data['levels4'][0]]
        y_lowpass_matlab = self.mat_data['y4_lowpass']
        y_bandpass_matlab = self.mat_data['y4_bandpass']
        
        # Run Python implementation
        y = nsctdec(x, levels, 'pkva', 'maxflat')
        
        # Check structure
        assert len(y) == 2
        
        # Check lowpass
        np.testing.assert_allclose(y[0], y_lowpass_matlab, rtol=1e-10, atol=1e-12)
        
        # Check bandpass (single image, no directionality)
        assert isinstance(y[1], np.ndarray), "Level 0 should return a single array"
        np.testing.assert_allclose(y[1], y_bandpass_matlab, rtol=1e-10, atol=1e-12)
    
    def test_mixed_levels(self):
        """Test 5: Mixed levels [0, 2]."""
        x = self.mat_data['x5']
        levels = [int(v) for v in self.mat_data['levels5'][0]]
        y_lowpass_matlab = self.mat_data['y5_lowpass']
        y_bandpass_matlab = self.mat_data['y5_bandpass']
        y_num_dir = int(self.mat_data['y5_num_dir'][0, 0])
        
        # Run Python implementation
        y = nsctdec(x, levels, 'pkva', 'maxflat')
        
        # Check structure
        assert len(y) == 3
        
        # Check lowpass
        np.testing.assert_allclose(y[0], y_lowpass_matlab, rtol=1e-10, atol=1e-12)
        
        # Check first level (no directionality)
        assert isinstance(y[1], np.ndarray)
        np.testing.assert_allclose(y[1], y_bandpass_matlab, rtol=1e-10, atol=1e-12)
        
        # Check second level (4 directional subbands)
        assert len(y[2]) == y_num_dir
    
    def test_dmaxflat7_filter(self):
        """Test 6: dmaxflat7 filter."""
        x = self.mat_data['x6']
        levels = [int(v) for v in self.mat_data['levels6'][0]]
        y_lowpass_matlab = self.mat_data['y6_lowpass']
        
        # Run Python implementation
        y = nsctdec(x, levels, 'dmaxflat7', 'maxflat')
        
        # Check structure
        assert len(y) == 3
        
        # Check lowpass
        np.testing.assert_allclose(y[0], y_lowpass_matlab, rtol=1e-10, atol=1e-12)
        
        # Check number of directional subbands
        assert len(y[1]) == 4
        assert len(y[2]) == 8
    
    def test_non_square_image(self):
        """Test 7: Non-square image."""
        x = self.mat_data['x7']
        levels = [int(v) for v in self.mat_data['levels7'][0]]
        y_lowpass_matlab = self.mat_data['y7_lowpass']
        
        # Run Python implementation
        y = nsctdec(x, levels, 'pkva', 'maxflat')
        
        # Check structure
        assert len(y) == 3
        
        # Check lowpass shape
        assert y[0].shape == y_lowpass_matlab.shape
        np.testing.assert_allclose(y[0], y_lowpass_matlab, rtol=1e-10, atol=1e-12)
        
        # Check directional subbands
        assert len(y[1]) == 4
        assert len(y[2]) == 8
    
    def test_constant_image(self):
        """Test 8: Constant image."""
        x = self.mat_data['x8']
        levels = [int(v) for v in self.mat_data['levels8'][0]]
        y_lowpass_matlab = self.mat_data['y8_lowpass']
        
        # Run Python implementation
        y = nsctdec(x, levels, 'pkva', 'maxflat')
        
        # Check lowpass (should be constant)
        np.testing.assert_allclose(y[0], y_lowpass_matlab, rtol=1e-10, atol=1e-12)
        
        # Check bandpass (should be near zero)
        for k in range(len(y[1])):
            assert np.abs(np.mean(y[1][k])) < 1e-14, \
                f"Bandpass subband {k+1} should be near zero for constant input"


class TestNsctdecEdgeCases:
    """Test edge cases and error handling for nsctdec."""
    
    def test_invalid_levels_type(self):
        """Test with invalid levels type."""
        x = np.random.rand(64, 64)
        
        with pytest.raises(TypeError):
            nsctdec(x, "invalid", 'pkva', 'maxflat')  # type: ignore[reportArgumentType]
    
    def test_invalid_levels_negative(self):
        """Test with negative levels."""
        x = np.random.rand(64, 64)
        
        with pytest.raises(ValueError, match="non-negative"):
            nsctdec(x, [-1, 2], 'pkva', 'maxflat')
    
    @pytest.mark.skip(reason="Float validation not enforced to match MATLAB behavior")
    def test_invalid_levels_float(self):
        """Test with float levels."""
        x = np.random.rand(64, 64)
        
        with pytest.raises(ValueError, match="integers"):
            nsctdec(x, [1.5, 2], 'pkva', 'maxflat')  # type: ignore[reportArgumentType]
    
    def test_single_level_decomposition(self):
        """Test with single pyramid level."""
        x = np.random.rand(64, 64)
        levels = [2]
        
        y = nsctdec(x, levels, 'pkva', 'maxflat')
        
        assert len(y) == 2
        assert y[0].shape == x.shape  # Lowpass
        assert len(y[1]) == 4  # 2^2 = 4 directional subbands
    
    def test_multiple_level_decomposition(self):
        """Test with multiple pyramid levels."""
        x = np.random.rand(128, 128)
        levels = [2, 3]
        
        y = nsctdec(x, levels, 'pkva', 'maxflat')
        
        assert len(y) == 3
        assert len(y[1]) == 4  # 2^2 = 4
        assert len(y[2]) == 8  # 2^3 = 8
    
    def test_output_shape_preservation(self):
        """Test that all subbands have same size as input."""
        x = np.random.rand(64, 64)
        levels = [2, 3]
        
        y = nsctdec(x, levels, 'pkva', 'maxflat')
        
        # Check lowpass
        assert y[0].shape == x.shape
        
        # Check all directional subbands
        for i in range(1, len(y)):
            for subband in y[i]:
                assert subband.shape == x.shape
    
    def test_zero_image(self):
        """Test decomposition of zero image."""
        x = np.zeros((64, 64))
        levels = [2]
        
        y = nsctdec(x, levels, 'pkva', 'maxflat')
        
        # All subbands should be near zero
        assert np.allclose(y[0], 0, atol=1e-14)
        for subband in y[1]:
            assert np.allclose(subband, 0, atol=1e-14)
    
    def test_energy_conservation(self):
        """Test that energy is approximately conserved."""
        x = np.random.rand(64, 64)
        levels = [2]
        
        y = nsctdec(x, levels, 'pkva', 'maxflat')
        
        # Calculate energy
        energy_in = np.sum(x**2)
        energy_out = np.sum(y[0]**2)
        for subband in y[1]:
            energy_out += np.sum(subband**2)
        
        # Energy should be approximately conserved (within 50% for NSCT)
        ratio = energy_out / energy_in
        assert 0.5 < ratio < 2.0, f"Energy ratio {ratio} is out of expected range"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
