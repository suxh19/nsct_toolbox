"""
Test suite for filters_torch.py - PyTorch version of filters.py
Tests numerical consistency, precision, and shape compatibility between NumPy and PyTorch implementations.
"""

import pytest
import numpy as np
import torch
from nsct_python.filters import (
    ld2quin as ld2quin_np,
    efilter2 as efilter2_np,
    dmaxflat as dmaxflat_np,
    atrousfilters as atrousfilters_np,
    mctrans as mctrans_np,
    ldfilter as ldfilter_np,
    dfilters as dfilters_np,
    parafilters as parafilters_np
)
from nsct_python.filters_torch import (
    ld2quin as ld2quin_torch,
    efilter2 as efilter2_torch,
    dmaxflat as dmaxflat_torch,
    atrousfilters as atrousfilters_torch,
    mctrans as mctrans_torch,
    ldfilter as ldfilter_torch,
    dfilters as dfilters_torch,
    parafilters as parafilters_torch
)


# Tolerance for numerical comparison
RTOL = 1e-12
ATOL = 1e-14


def to_torch(arr: np.ndarray) -> torch.Tensor:
    """Convert numpy array to torch tensor with float64 precision."""
    return torch.from_numpy(arr).to(torch.float64)


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert torch tensor to numpy array."""
    return tensor.detach().cpu().numpy()


class TestLdfilter:
    """Test ldfilter function."""
    
    @pytest.mark.parametrize("fname", ['pkva', 'pkva12', 'pkva8', 'pkva6'])
    def test_ldfilter_values(self, fname):
        """Test that ldfilter produces identical values for NumPy and PyTorch."""
        result_np = ldfilter_np(fname)
        result_torch = ldfilter_torch(fname)
        
        # Convert to numpy for comparison
        result_torch_np = to_numpy(result_torch)
        
        # Check shape
        assert result_np.shape == result_torch_np.shape, \
            f"Shape mismatch for {fname}: NumPy {result_np.shape} vs Torch {result_torch_np.shape}"
        
        # Check values
        np.testing.assert_allclose(result_np, result_torch_np, rtol=RTOL, atol=ATOL,
                                   err_msg=f"Values mismatch for {fname}")
    
    def test_ldfilter_invalid(self):
        """Test that invalid filter names raise ValueError."""
        with pytest.raises(ValueError):
            ldfilter_np('invalid')
        with pytest.raises(ValueError):
            ldfilter_torch('invalid')


class TestLd2quin:
    """Test ld2quin function."""
    
    @pytest.mark.parametrize("fname", ['pkva6', 'pkva8'])
    def test_ld2quin_consistency(self, fname):
        """Test ld2quin consistency between NumPy and PyTorch."""
        beta_np = ldfilter_np(fname)
        beta_torch = ldfilter_torch(fname)
        
        h0_np, h1_np = ld2quin_np(beta_np)
        h0_torch, h1_torch = ld2quin_torch(beta_torch)
        
        # Check shapes
        assert h0_np.shape == to_numpy(h0_torch).shape, \
            f"h0 shape mismatch for {fname}"
        assert h1_np.shape == to_numpy(h1_torch).shape, \
            f"h1 shape mismatch for {fname}"
        
        # Check values
        np.testing.assert_allclose(h0_np, to_numpy(h0_torch), rtol=RTOL, atol=ATOL,
                                   err_msg=f"h0 values mismatch for {fname}")
        np.testing.assert_allclose(h1_np, to_numpy(h1_torch), rtol=RTOL, atol=ATOL,
                                   err_msg=f"h1 values mismatch for {fname}")
    
    def test_ld2quin_shape_properties(self):
        """Test shape properties of ld2quin output."""
        beta_np = ldfilter_np('pkva6')
        beta_torch = ldfilter_torch('pkva6')
        
        h0_np, h1_np = ld2quin_np(beta_np)
        h0_torch, h1_torch = ld2quin_torch(beta_torch)
        
        # For lf=6, n=3, h0 should be (11, 11) and h1 should be (21, 21)
        assert h0_np.shape == (11, 11)
        assert to_numpy(h0_torch).shape == (11, 11)
        assert h1_np.shape == (21, 21)
        assert to_numpy(h1_torch).shape == (21, 21)


class TestDmaxflat:
    """Test dmaxflat function."""
    
    @pytest.mark.parametrize("N", [1, 2, 3, 4, 5, 6, 7])
    @pytest.mark.parametrize("d", [0.0, 1.0])
    def test_dmaxflat_consistency(self, N, d):
        """Test dmaxflat consistency between NumPy and PyTorch."""
        h_np = dmaxflat_np(N, d)
        h_torch = dmaxflat_torch(N, d)
        
        # Check shape
        assert h_np.shape == to_numpy(h_torch).shape, \
            f"Shape mismatch for N={N}, d={d}"
        
        # Check values
        np.testing.assert_allclose(h_np, to_numpy(h_torch), rtol=RTOL, atol=ATOL,
                                   err_msg=f"Values mismatch for N={N}, d={d}")
    
    def test_dmaxflat_center_value(self):
        """Test that the center value is correctly set."""
        N = 3
        d = 1.0
        h_np = dmaxflat_np(N, d)
        h_torch = dmaxflat_torch(N, d)
        
        # Center should be at [3, 3] for N=3
        assert h_np[3, 3] == d
        assert to_numpy(h_torch)[3, 3] == d
    
    def test_dmaxflat_invalid(self):
        """Test that invalid N raises ValueError."""
        with pytest.raises(ValueError):
            dmaxflat_np(0, 0)
        with pytest.raises(ValueError):
            dmaxflat_torch(0, 0)
        with pytest.raises(ValueError):
            dmaxflat_np(8, 0)
        with pytest.raises(ValueError):
            dmaxflat_torch(8, 0)


class TestEfilter2:
    """Test efilter2 function."""
    
    def test_efilter2_basic(self):
        """Test basic efilter2 functionality."""
        img_np = np.arange(9).reshape(3, 3).astype(np.float64)
        filt_np = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
        
        img_torch = to_torch(img_np)
        filt_torch = to_torch(filt_np)
        
        y_np = efilter2_np(img_np, filt_np, 'per')
        y_torch = efilter2_torch(img_torch, filt_torch, 'per')
        
        # Check shape
        assert y_np.shape == to_numpy(y_torch).shape
        assert y_np.shape == (3, 3)
        
        # Check values
        np.testing.assert_allclose(y_np, to_numpy(y_torch), rtol=RTOL, atol=ATOL,
                                   err_msg="Values mismatch for basic efilter2")
    
    @pytest.mark.parametrize("extmod", ['per'])
    def test_efilter2_extension_modes(self, extmod):
        """Test efilter2 with different extension modes."""
        img_np = np.random.rand(5, 5)
        filt_np = np.ones((3, 3)) / 9.0
        
        img_torch = to_torch(img_np)
        filt_torch = to_torch(filt_np)
        
        y_np = efilter2_np(img_np, filt_np, extmod)
        y_torch = efilter2_torch(img_torch, filt_torch, extmod)
        
        # Check shape
        assert y_np.shape == to_numpy(y_torch).shape
        
        # Check values
        np.testing.assert_allclose(y_np, to_numpy(y_torch), rtol=RTOL, atol=ATOL,
                                   err_msg=f"Values mismatch for extmod={extmod}")
    
    def test_efilter2_with_shift(self):
        """Test efilter2 with shift parameter."""
        img_np = np.random.rand(7, 7)
        filt_np = np.ones((3, 3)) / 9.0
        shift = [1, 1]
        
        img_torch = to_torch(img_np)
        filt_torch = to_torch(filt_np)
        
        y_np = efilter2_np(img_np, filt_np, 'per', shift)
        y_torch = efilter2_torch(img_torch, filt_torch, 'per', shift)
        
        # Check shape
        assert y_np.shape == to_numpy(y_torch).shape
        
        # Check values
        np.testing.assert_allclose(y_np, to_numpy(y_torch), rtol=RTOL, atol=ATOL,
                                   err_msg="Values mismatch with shift")


class TestAtrousfilters:
    """Test atrousfilters function."""
    
    @pytest.mark.parametrize("fname", ['pyr', 'pyrexc', 'maxflat'])
    def test_atrousfilters_consistency(self, fname):
        """Test atrousfilters consistency between NumPy and PyTorch."""
        h0_np, h1_np, g0_np, g1_np = atrousfilters_np(fname)
        h0_torch, h1_torch, g0_torch, g1_torch = atrousfilters_torch(fname)
        
        # Check shapes
        assert h0_np.shape == to_numpy(h0_torch).shape, f"h0 shape mismatch for {fname}"
        assert h1_np.shape == to_numpy(h1_torch).shape, f"h1 shape mismatch for {fname}"
        assert g0_np.shape == to_numpy(g0_torch).shape, f"g0 shape mismatch for {fname}"
        assert g1_np.shape == to_numpy(g1_torch).shape, f"g1 shape mismatch for {fname}"
        
        # Check values
        np.testing.assert_allclose(h0_np, to_numpy(h0_torch), rtol=RTOL, atol=ATOL,
                                   err_msg=f"h0 values mismatch for {fname}")
        np.testing.assert_allclose(h1_np, to_numpy(h1_torch), rtol=RTOL, atol=ATOL,
                                   err_msg=f"h1 values mismatch for {fname}")
        np.testing.assert_allclose(g0_np, to_numpy(g0_torch), rtol=RTOL, atol=ATOL,
                                   err_msg=f"g0 values mismatch for {fname}")
        np.testing.assert_allclose(g1_np, to_numpy(g1_torch), rtol=RTOL, atol=ATOL,
                                   err_msg=f"g1 values mismatch for {fname}")
    
    def test_atrousfilters_shape_pyr(self):
        """Test that pyr filters have expected shape."""
        h0_np, h1_np, g0_np, g1_np = atrousfilters_np('pyr')
        h0_torch, _, _, _ = atrousfilters_torch('pyr')
        
        assert h0_np.shape == (5, 5)
        assert to_numpy(h0_torch).shape == (5, 5)
    
    def test_atrousfilters_invalid(self):
        """Test that invalid filter name raises error."""
        with pytest.raises(NotImplementedError):
            atrousfilters_np('invalid')
        with pytest.raises(NotImplementedError):
            atrousfilters_torch('invalid')


class TestMctrans:
    """Test mctrans function."""
    
    def test_mctrans_basic(self):
        """Test basic mctrans functionality."""
        b_np = np.array([1, 2, 1], dtype=np.float64) / 4.0
        t_np = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float64) / 4.0
        
        b_torch = to_torch(b_np)
        t_torch = to_torch(t_np)
        
        h_np = mctrans_np(b_np, t_np)
        h_torch = mctrans_torch(b_torch, t_torch)
        
        # Check shape
        assert h_np.shape == to_numpy(h_torch).shape
        
        # Check values
        np.testing.assert_allclose(h_np, to_numpy(h_torch), rtol=RTOL, atol=ATOL,
                                   err_msg="Values mismatch for mctrans")
    
    def test_mctrans_with_dmaxflat(self):
        """Test mctrans with dmaxflat input."""
        b_np = np.array([1, 2, 1], dtype=np.float64) / 4.0
        t_np = dmaxflat_np(2, 0)
        
        b_torch = to_torch(b_np)
        t_torch = dmaxflat_torch(2, 0)
        
        h_np = mctrans_np(b_np, t_np)
        h_torch = mctrans_torch(b_torch, t_torch)
        
        # Check shape
        assert h_np.shape == to_numpy(h_torch).shape
        
        # Check values
        np.testing.assert_allclose(h_np, to_numpy(h_torch), rtol=1e-10, atol=1e-12,
                                   err_msg="Values mismatch for mctrans with dmaxflat")


class TestDfilters:
    """Test dfilters function."""
    
    @pytest.mark.parametrize("fname", ['pkva', 'dmaxflat3', 'dmaxflat5'])
    @pytest.mark.parametrize("ftype", ['d', 'r'])
    def test_dfilters_consistency(self, fname, ftype):
        """Test dfilters consistency between NumPy and PyTorch."""
        h0_np, h1_np = dfilters_np(fname, ftype)
        h0_torch, h1_torch = dfilters_torch(fname, ftype)
        
        # Check shapes
        assert h0_np.shape == to_numpy(h0_torch).shape, \
            f"h0 shape mismatch for {fname}, type={ftype}"
        assert h1_np.shape == to_numpy(h1_torch).shape, \
            f"h1 shape mismatch for {fname}, type={ftype}"
        
        # Check values
        np.testing.assert_allclose(h0_np, to_numpy(h0_torch), rtol=1e-10, atol=1e-12,
                                   err_msg=f"h0 values mismatch for {fname}, type={ftype}")
        np.testing.assert_allclose(h1_np, to_numpy(h1_torch), rtol=1e-10, atol=1e-12,
                                   err_msg=f"h1 values mismatch for {fname}, type={ftype}")
    
    def test_dfilters_pkva_shape(self):
        """Test that pkva filters have expected shape."""
        h0_np, h1_np = dfilters_np('pkva', 'd')
        h0_torch, h1_torch = dfilters_torch('pkva', 'd')
        
        assert h0_np.shape == (23, 23)
        assert to_numpy(h0_torch).shape == (23, 23)
        assert h1_np.shape == (45, 45)
        assert to_numpy(h1_torch).shape == (45, 45)
    
    @pytest.mark.parametrize("wavelet", ['db2', 'haar'])
    def test_dfilters_pywt_fallback(self, wavelet):
        """Test dfilters with PyWavelets fallback."""
        h0_np, h1_np = dfilters_np(wavelet, 'd')
        h0_torch, h1_torch = dfilters_torch(wavelet, 'd')
        
        # Check shapes
        assert h0_np.shape == to_numpy(h0_torch).shape
        assert h1_np.shape == to_numpy(h1_torch).shape
        
        # Check values
        np.testing.assert_allclose(h0_np, to_numpy(h0_torch), rtol=RTOL, atol=ATOL,
                                   err_msg=f"h0 values mismatch for {wavelet}")
        np.testing.assert_allclose(h1_np, to_numpy(h1_torch), rtol=RTOL, atol=ATOL,
                                   err_msg=f"h1 values mismatch for {wavelet}")


class TestParafilters:
    """Test parafilters function."""
    
    def test_parafilters_consistency(self):
        """Test parafilters consistency between NumPy and PyTorch."""
        f1_np = np.random.rand(3, 3)
        f2_np = np.random.rand(3, 3)
        
        f1_torch = to_torch(f1_np)
        f2_torch = to_torch(f2_np)
        
        y1_np, y2_np = parafilters_np(f1_np, f2_np)
        y1_torch, y2_torch = parafilters_torch(f1_torch, f2_torch)
        
        # Check that we get 4 filters in each list
        assert len(y1_np) == 4
        assert len(y1_torch) == 4
        assert len(y2_np) == 4
        assert len(y2_torch) == 4
        
        # Check shapes and values for each filter
        for i in range(4):
            assert y1_np[i].shape == to_numpy(y1_torch[i]).shape, \
                f"y1[{i}] shape mismatch"
            assert y2_np[i].shape == to_numpy(y2_torch[i]).shape, \
                f"y2[{i}] shape mismatch"
            
            np.testing.assert_allclose(y1_np[i], to_numpy(y1_torch[i]), rtol=1e-10, atol=1e-12,
                                       err_msg=f"y1[{i}] values mismatch")
            np.testing.assert_allclose(y2_np[i], to_numpy(y2_torch[i]), rtol=1e-10, atol=1e-12,
                                       err_msg=f"y2[{i}] values mismatch")
    
    def test_parafilters_shape_properties(self):
        """Test shape properties of parafilters output."""
        f1_np = np.ones((3, 3))
        f2_np = np.ones((3, 3)) * 2
        
        f1_torch = to_torch(f1_np)
        f2_torch = to_torch(f2_np)
        
        y1_np, y2_np = parafilters_np(f1_np, f2_np)
        y1_torch, y2_torch = parafilters_torch(f1_torch, f2_torch)
        
        # Check that resampz produces the expected shape for type 1
        assert y1_np[0].shape == (5, 3)
        assert to_numpy(y1_torch[0]).shape == (5, 3)


class TestPrecision:
    """Test numerical precision and edge cases."""
    
    def test_dtype_preservation(self):
        """Test that float64 precision is preserved."""
        beta = ldfilter_torch('pkva6')
        assert beta.dtype == torch.float64
        
        h0, h1 = ld2quin_torch(beta)
        assert h0.dtype == torch.float64
        assert h1.dtype == torch.float64
    
    def test_gradient_support(self):
        """Test that tensors support gradient computation."""
        beta = ldfilter_torch('pkva6')
        beta.requires_grad = True
        
        h0, h1 = ld2quin_torch(beta)
        
        # Compute a scalar from the output
        scalar = h0.sum()
        scalar.backward()
        
        # Check that gradient was computed
        assert beta.grad is not None
    
    def test_small_values_precision(self):
        """Test handling of very small values."""
        # Test with maxflat filters which have very small coefficients
        h0_np, _, _, _ = atrousfilters_np('maxflat')
        h0_torch, _, _, _ = atrousfilters_torch('maxflat')
        
        # Check that small values are preserved
        small_values_np = h0_np[np.abs(h0_np) < 1e-6]
        small_values_torch = to_numpy(h0_torch)[np.abs(to_numpy(h0_torch)) < 1e-6]
        
        assert len(small_values_np) == len(small_values_torch)
        np.testing.assert_allclose(small_values_np, small_values_torch, rtol=RTOL, atol=ATOL)


class TestEndToEnd:
    """End-to-end integration tests."""
    
    def test_full_filter_pipeline(self):
        """Test a complete filter generation pipeline."""
        # Generate filters using dfilters
        h0_np, h1_np = dfilters_np('pkva', 'd')
        h0_torch, h1_torch = dfilters_torch('pkva', 'd')
        
        # Generate parafilters
        y1_np, y2_np = parafilters_np(h0_np, h1_np)
        y1_torch, y2_torch = parafilters_torch(h0_torch, h1_torch)
        
        # Check consistency throughout the pipeline
        for i in range(4):
            np.testing.assert_allclose(y1_np[i], to_numpy(y1_torch[i]), rtol=1e-10, atol=1e-12,
                                       err_msg=f"Full pipeline y1[{i}] mismatch")
            np.testing.assert_allclose(y2_np[i], to_numpy(y2_torch[i]), rtol=1e-10, atol=1e-12,
                                       err_msg=f"Full pipeline y2[{i}] mismatch")
    
    def test_filter_application(self):
        """Test applying filters to an image."""
        # Create a test image
        img_np = np.random.rand(64, 64)
        img_torch = to_torch(img_np)
        
        # Get filters
        h0_np, h1_np, _, _ = atrousfilters_np('pyr')
        h0_torch, h1_torch, _, _ = atrousfilters_torch('pyr')
        
        # Apply filters
        y0_np = efilter2_np(img_np, h0_np, 'per')
        y0_torch = efilter2_torch(img_torch, h0_torch, 'per')
        
        y1_np = efilter2_np(img_np, h1_np, 'per')
        y1_torch = efilter2_torch(img_torch, h1_torch, 'per')
        
        # Check consistency
        np.testing.assert_allclose(y0_np, to_numpy(y0_torch), rtol=1e-10, atol=1e-12,
                                   err_msg="Filter application y0 mismatch")
        np.testing.assert_allclose(y1_np, to_numpy(y1_torch), rtol=1e-10, atol=1e-12,
                                   err_msg="Filter application y1 mismatch")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
