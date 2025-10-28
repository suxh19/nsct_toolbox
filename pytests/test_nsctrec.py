"""
Test suite for nsctrec() - NSCT Reconstruction

Tests complete decomposition-reconstruction cycle to verify perfect reconstruction.
Compares Python implementation against MATLAB reference data.
"""

import numpy as np
import pytest
import scipy.io as sio
from pathlib import Path

from nsct_python.core import nsctdec, nsctrec


class TestNsctrec:
    """Test nsctrec() against MATLAB reference data."""
    
    @classmethod
    def setup_class(cls):
        """Load MATLAB test results once for all tests."""
        mat_file = Path(__file__).parent.parent / 'data' / 'test_nsctrec_results.mat'
        cls.matlab_data = sio.loadmat(str(mat_file), struct_as_record=False, squeeze_me=True)
        cls.results = cls.matlab_data['results']
    
    def test_single_pyramid_level_2_directional(self):
        """Test single pyramid level [2] with pkva/maxflat."""
        test_data = self.results.test1
        
        x_orig = test_data.x_orig
        levels = [int(test_data.levels)]
        dfilt = str(test_data.dfilt)
        pfilt = str(test_data.pfilt)
        
        # Python decomposition and reconstruction
        y = nsctdec(x_orig, levels, dfilt, pfilt)
        x_rec = nsctrec(y, dfilt, pfilt)
        
        # Compare with original (perfect reconstruction)
        error = np.max(np.abs(x_orig - x_rec))
        
        # Also compare with MATLAB reconstruction
        x_rec_matlab = test_data.x_rec
        matlab_diff = np.max(np.abs(x_rec - x_rec_matlab))
        
        print(f"\nTest 1 - Single level [2]:")
        print(f"  Python reconstruction error: {error:.15e}")
        print(f"  Difference from MATLAB: {matlab_diff:.15e}")
        
        assert error < 1e-12, f"Reconstruction error {error} exceeds tolerance"
        assert matlab_diff < 1e-12, f"MATLAB difference {matlab_diff} exceeds tolerance"
    
    def test_two_pyramid_levels(self):
        """Test two pyramid levels [2,3] with pkva/maxflat."""
        test_data = self.results.test2
        
        x_orig = test_data.x_orig
        levels = [int(l) for l in test_data.levels]
        dfilt = str(test_data.dfilt)
        pfilt = str(test_data.pfilt)
        
        # Python decomposition and reconstruction
        y = nsctdec(x_orig, levels, dfilt, pfilt)
        x_rec = nsctrec(y, dfilt, pfilt)
        
        # Compare with original
        error = np.max(np.abs(x_orig - x_rec))
        
        # Compare with MATLAB
        x_rec_matlab = test_data.x_rec
        matlab_diff = np.max(np.abs(x_rec - x_rec_matlab))
        
        print(f"\nTest 2 - Two levels [2,3]:")
        print(f"  Python reconstruction error: {error:.15e}")
        print(f"  Difference from MATLAB: {matlab_diff:.15e}")
        
        assert error < 1e-12, f"Reconstruction error {error} exceeds tolerance"
        assert matlab_diff < 1e-12, f"MATLAB difference {matlab_diff} exceeds tolerance"
    
    def test_three_pyramid_levels(self):
        """Test three pyramid levels [2,3,4] with pkva/maxflat."""
        test_data = self.results.test3
        
        x_orig = test_data.x_orig
        levels = [int(l) for l in test_data.levels]
        dfilt = str(test_data.dfilt)
        pfilt = str(test_data.pfilt)
        
        # Python decomposition and reconstruction
        y = nsctdec(x_orig, levels, dfilt, pfilt)
        x_rec = nsctrec(y, dfilt, pfilt)
        
        # Compare with original
        error = np.max(np.abs(x_orig - x_rec))
        
        # Compare with MATLAB
        x_rec_matlab = test_data.x_rec
        matlab_diff = np.max(np.abs(x_rec - x_rec_matlab))
        
        print(f"\nTest 3 - Three levels [2,3,4]:")
        print(f"  Python reconstruction error: {error:.15e}")
        print(f"  Difference from MATLAB: {matlab_diff:.15e}")
        
        assert error < 1e-12, f"Reconstruction error {error} exceeds tolerance"
        assert matlab_diff < 1e-12, f"MATLAB difference {matlab_diff} exceeds tolerance"
    
    def test_level_0_no_directional(self):
        """Test level 0 (no directional decomposition)."""
        test_data = self.results.test4
        
        x_orig = test_data.x_orig
        levels = [int(test_data.levels)]
        dfilt = str(test_data.dfilt)
        pfilt = str(test_data.pfilt)
        
        # Python decomposition and reconstruction
        y = nsctdec(x_orig, levels, dfilt, pfilt)
        x_rec = nsctrec(y, dfilt, pfilt)
        
        # Compare with original
        error = np.max(np.abs(x_orig - x_rec))
        
        # Compare with MATLAB
        x_rec_matlab = test_data.x_rec
        matlab_diff = np.max(np.abs(x_rec - x_rec_matlab))
        
        print(f"\nTest 4 - Level 0 (no directional):")
        print(f"  Python reconstruction error: {error:.15e}")
        print(f"  Difference from MATLAB: {matlab_diff:.15e}")
        
        assert error < 1e-12, f"Reconstruction error {error} exceeds tolerance"
        assert matlab_diff < 1e-12, f"MATLAB difference {matlab_diff} exceeds tolerance"
    
    def test_mixed_levels(self):
        """Test mixed levels [0,2]."""
        test_data = self.results.test5
        
        x_orig = test_data.x_orig
        levels = [int(l) for l in test_data.levels]
        dfilt = str(test_data.dfilt)
        pfilt = str(test_data.pfilt)
        
        # Python decomposition and reconstruction
        y = nsctdec(x_orig, levels, dfilt, pfilt)
        x_rec = nsctrec(y, dfilt, pfilt)
        
        # Compare with original
        error = np.max(np.abs(x_orig - x_rec))
        
        # Compare with MATLAB
        x_rec_matlab = test_data.x_rec
        matlab_diff = np.max(np.abs(x_rec - x_rec_matlab))
        
        print(f"\nTest 5 - Mixed levels [0,2]:")
        print(f"  Python reconstruction error: {error:.15e}")
        print(f"  Difference from MATLAB: {matlab_diff:.15e}")
        
        assert error < 1e-12, f"Reconstruction error {error} exceeds tolerance"
        assert matlab_diff < 1e-12, f"MATLAB difference {matlab_diff} exceeds tolerance"
    
    def test_dmaxflat7_filter(self):
        """Test dmaxflat7 directional filter."""
        test_data = self.results.test6
        
        x_orig = test_data.x_orig
        levels = [int(l) for l in test_data.levels]
        dfilt = str(test_data.dfilt)
        pfilt = str(test_data.pfilt)
        
        # Python decomposition and reconstruction
        y = nsctdec(x_orig, levels, dfilt, pfilt)
        x_rec = nsctrec(y, dfilt, pfilt)
        
        # Compare with original
        error = np.max(np.abs(x_orig - x_rec))
        
        # Compare with MATLAB
        x_rec_matlab = test_data.x_rec
        matlab_diff = np.max(np.abs(x_rec - x_rec_matlab))
        
        print(f"\nTest 6 - dmaxflat7 filter:")
        print(f"  Python reconstruction error: {error:.15e}")
        print(f"  Difference from MATLAB: {matlab_diff:.15e}")
        
        assert error < 1e-12, f"Reconstruction error {error} exceeds tolerance"
        assert matlab_diff < 1e-12, f"MATLAB difference {matlab_diff} exceeds tolerance"
    
    def test_non_square_image(self):
        """Test non-square image (64x96)."""
        test_data = self.results.test7
        
        x_orig = test_data.x_orig
        levels = [int(l) for l in test_data.levels]
        dfilt = str(test_data.dfilt)
        pfilt = str(test_data.pfilt)
        
        # Python decomposition and reconstruction
        y = nsctdec(x_orig, levels, dfilt, pfilt)
        x_rec = nsctrec(y, dfilt, pfilt)
        
        # Compare with original
        error = np.max(np.abs(x_orig - x_rec))
        
        # Compare with MATLAB
        x_rec_matlab = test_data.x_rec
        matlab_diff = np.max(np.abs(x_rec - x_rec_matlab))
        
        print(f"\nTest 7 - Non-square image (64x96):")
        print(f"  Python reconstruction error: {error:.15e}")
        print(f"  Difference from MATLAB: {matlab_diff:.15e}")
        
        assert error < 1e-12, f"Reconstruction error {error} exceeds tolerance"
        assert matlab_diff < 1e-12, f"MATLAB difference {matlab_diff} exceeds tolerance"
    
    def test_constant_image(self):
        """Test constant image (special case)."""
        test_data = self.results.test8
        
        x_orig = test_data.x_orig
        levels = [int(test_data.levels)]
        dfilt = str(test_data.dfilt)
        pfilt = str(test_data.pfilt)
        
        # Python decomposition and reconstruction
        y = nsctdec(x_orig, levels, dfilt, pfilt)
        x_rec = nsctrec(y, dfilt, pfilt)
        
        # Compare with original
        error = np.max(np.abs(x_orig - x_rec))
        
        # Compare with MATLAB
        x_rec_matlab = test_data.x_rec
        matlab_diff = np.max(np.abs(x_rec - x_rec_matlab))
        
        print(f"\nTest 8 - Constant image (value=5.0):")
        print(f"  Python reconstruction error: {error:.15e}")
        print(f"  Difference from MATLAB: {matlab_diff:.15e}")
        
        assert error < 1e-12, f"Reconstruction error {error} exceeds tolerance"
        assert matlab_diff < 1e-12, f"MATLAB difference {matlab_diff} exceeds tolerance"


class TestNsctrecEdgeCases:
    """Test edge cases and error handling for nsctrec()."""
    
    @pytest.mark.skip(reason="Type validation not enforced to match MATLAB behavior")
    def test_invalid_y_type(self):
        """Test that non-list y raises an error."""
        x = np.random.rand(64, 64)
        
        with pytest.raises((TypeError, AttributeError)):
            nsctrec(x, 'pkva', 'maxflat')
    
    def test_empty_y_list(self):
        """Test that empty y list raises an error."""
        y = []
        
        with pytest.raises((IndexError, ValueError)):
            nsctrec(y, 'pkva', 'maxflat')
    
    def test_single_element_y(self):
        """Test single element (just lowpass, no bandpass)."""
        x_orig = np.random.rand(64, 64)
        
        # Create a y with just lowpass (should work - no pyramid levels)
        # This is essentially the trivial case where n=0
        y = [x_orig]
        
        # This should return the lowpass directly
        x_rec = nsctrec(y, 'pkva', 'maxflat')
        
        # Should be identical (no processing)
        error = np.max(np.abs(x_orig - x_rec))
        assert error < 1e-15, "Single element reconstruction failed"
    
    def test_perfect_reconstruction_random(self):
        """Test perfect reconstruction with random image."""
        np.random.seed(123)
        x_orig = np.random.rand(64, 64)
        levels = [2, 3]
        
        y = nsctdec(x_orig, levels, 'pkva', 'maxflat')
        x_rec = nsctrec(y, 'pkva', 'maxflat')
        
        error = np.max(np.abs(x_orig - x_rec))
        
        print(f"\nRandom image reconstruction error: {error:.15e}")
        assert error < 1e-12, f"Perfect reconstruction failed: error = {error}"
    
    def test_energy_conservation(self):
        """Test that energy is conserved in decomposition-reconstruction."""
        np.random.seed(456)
        x_orig = np.random.rand(64, 64)
        levels = [2]
        
        # Decompose
        y = nsctdec(x_orig, levels, 'pkva', 'maxflat')
        
        # Reconstruct
        x_rec = nsctrec(y, 'pkva', 'maxflat')
        
        # Compare energy
        energy_orig = np.sum(x_orig ** 2)
        energy_rec = np.sum(x_rec ** 2)
        energy_diff = abs(energy_orig - energy_rec)
        
        print(f"\nEnergy conservation test:")
        print(f"  Original energy: {energy_orig:.10e}")
        print(f"  Reconstructed energy: {energy_rec:.10e}")
        print(f"  Energy difference: {energy_diff:.10e}")
        
        assert energy_diff < 1e-10, f"Energy not conserved: diff = {energy_diff}"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
