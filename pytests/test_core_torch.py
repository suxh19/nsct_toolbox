"""
Comprehensive pytest tests for core_torch.py

Tests numerical accuracy, precision, shape consistency, and one-to-one consistency
between NumPy (core.py) and PyTorch (core_torch.py) implementations.
"""

import pytest
import numpy as np
import torch
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from nsct_python import core
from nsct_torch import core_torch
from nsct_python.filters import dfilters as dfilters_np, atrousfilters as atrousfilters_np
from nsct_torch.filters_torch import dfilters as dfilters_torch, atrousfilters as atrousfilters_torch


def numpy_to_torch(arr):
    """Convert numpy array to torch tensor."""
    if isinstance(arr, np.ndarray):
        return torch.from_numpy(arr.copy())
    return arr


def torch_to_numpy(tensor):
    """Convert torch tensor to numpy array."""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor


class TestZconv2Consistency:
    """Test _zconv2 function consistency between NumPy and PyTorch."""
    
    @pytest.mark.parametrize("size", [(16, 16), (32, 32), (64, 64)])
    @pytest.mark.parametrize("filter_size", [(3, 3), (5, 5), (7, 7)])
    def test_zconv2_shape(self, size, filter_size):
        """Test that output shapes match."""
        x_np = np.random.rand(*size)
        h_np = np.random.rand(*filter_size)
        mup_np = np.array([[1, 1], [-1, 1]], dtype=np.float64)
        
        x_torch = numpy_to_torch(x_np)
        h_torch = numpy_to_torch(h_np)
        mup_torch = numpy_to_torch(mup_np)
        
        y_np = core._zconv2(x_np, h_np, mup_np)
        y_torch = core_torch._zconv2(x_torch, h_torch, mup_torch)
        
        assert y_np.shape == torch_to_numpy(y_torch).shape
        assert y_np.shape == x_np.shape
    
    @pytest.mark.parametrize("mup_type", [
        [[1, 1], [-1, 1]],  # Quincunx
        [[2, 0], [0, 1]],   # Row
        [[1, 0], [0, 2]],   # Col
    ])
    def test_zconv2_numerical_accuracy(self, mup_type):
        """Test numerical accuracy between implementations."""
        x_np = np.random.rand(32, 32)
        h_np = np.random.rand(5, 5)
        mup_np = np.array(mup_type, dtype=np.float64)
        
        x_torch = numpy_to_torch(x_np)
        h_torch = numpy_to_torch(h_np)
        mup_torch = numpy_to_torch(mup_np)
        
        y_np = core._zconv2(x_np, h_np, mup_np)
        y_torch = core_torch._zconv2(x_torch, h_torch, mup_torch)
        y_torch_np = torch_to_numpy(y_torch)
        
        # Check close numerical values
        assert np.allclose(y_np, y_torch_np, rtol=1e-5, atol=1e-7)
        
        # Check maximum absolute error
        max_error = np.max(np.abs(y_np - y_torch_np))
        assert max_error < 1e-6
    
    def test_zconv2_zero_filter(self):
        """Test behavior with zero filter."""
        x_np = np.random.rand(32, 32)
        h_np = np.zeros((5, 5))
        mup_np = np.array([[1, 1], [-1, 1]], dtype=np.float64)
        
        x_torch = numpy_to_torch(x_np)
        h_torch = numpy_to_torch(h_np)
        mup_torch = numpy_to_torch(mup_np)
        
        y_np = core._zconv2(x_np, h_np, mup_np)
        y_torch = core_torch._zconv2(x_torch, h_torch, mup_torch)
        
        assert np.allclose(y_np, 0.0)
        assert torch.allclose(y_torch, torch.zeros_like(y_torch))


class TestConvolveUpsampledConsistency:
    """Test _convolve_upsampled function consistency."""
    
    def test_convolve_upsampled_forward(self):
        """Test forward convolution consistency."""
        x_np = np.random.rand(32, 32)
        f_np = np.random.rand(5, 5)
        mup_np = np.array([[1, 1], [-1, 1]])
        
        x_torch = numpy_to_torch(x_np)
        f_torch = numpy_to_torch(f_np)
        mup_torch = numpy_to_torch(mup_np)
        
        y_np = core._convolve_upsampled(x_np, f_np, mup_np, is_rec=False)
        y_torch = core_torch._convolve_upsampled(x_torch, f_torch, mup_torch, is_rec=False)
        
        assert np.allclose(y_np, torch_to_numpy(y_torch), rtol=1e-5, atol=1e-7)
    
    def test_convolve_upsampled_reconstruction(self):
        """Test reconstruction convolution consistency."""
        x_np = np.random.rand(32, 32)
        f_np = np.random.rand(5, 5)
        mup_np = np.array([[1, 1], [-1, 1]])
        
        x_torch = numpy_to_torch(x_np)
        f_torch = numpy_to_torch(f_np)
        mup_torch = numpy_to_torch(mup_np)
        
        y_np = core._convolve_upsampled(x_np, f_np, mup_np, is_rec=True)
        y_torch = core_torch._convolve_upsampled(x_torch, f_torch, mup_torch, is_rec=True)
        
        assert np.allclose(y_np, torch_to_numpy(y_torch), rtol=1e-5, atol=1e-7)


class TestNssfbdecConsistency:
    """Test nssfbdec function consistency."""
    
    def test_nssfbdec_with_mup(self):
        """Test nssfbdec with upsampling matrix."""
        x_np = np.random.rand(32, 32)
        
        # Get filters
        h0_np, h1_np = dfilters_np('pkva', 'd')
        h0_torch, h1_torch = dfilters_torch('pkva', 'd')
        
        mup_np = np.array([[1, 1], [-1, 1]])
        mup_torch = torch.tensor([[1, 1], [-1, 1]])
        
        x_torch = numpy_to_torch(x_np)
        
        y1_np, y2_np = core.nssfbdec(x_np, h0_np, h1_np, mup_np)
        y1_torch, y2_torch = core_torch.nssfbdec(x_torch, h0_torch, h1_torch, mup_torch)
        
        # Check shapes
        assert y1_np.shape == torch_to_numpy(y1_torch).shape
        assert y2_np.shape == torch_to_numpy(y2_torch).shape
        
        # Check numerical accuracy
        assert np.allclose(y1_np, torch_to_numpy(y1_torch), rtol=1e-5, atol=1e-7)
        assert np.allclose(y2_np, torch_to_numpy(y2_torch), rtol=1e-5, atol=1e-7)
    
    def test_nssfbdec_without_mup(self):
        """Test nssfbdec without upsampling matrix."""
        x_np = np.random.rand(32, 32)
        
        h0_np, h1_np = dfilters_np('pkva', 'd')
        h0_torch, h1_torch = dfilters_torch('pkva', 'd')
        
        x_torch = numpy_to_torch(x_np)
        
        y1_np, y2_np = core.nssfbdec(x_np, h0_np, h1_np, mup=None)
        y1_torch, y2_torch = core_torch.nssfbdec(x_torch, h0_torch, h1_torch, mup=None)
        
        assert np.allclose(y1_np, torch_to_numpy(y1_torch), rtol=1e-5, atol=1e-7)
        assert np.allclose(y2_np, torch_to_numpy(y2_torch), rtol=1e-5, atol=1e-7)


class TestNssfbrecConsistency:
    """Test nssfbrec function consistency."""
    
    def test_nssfbrec_perfect_reconstruction(self):
        """Test perfect reconstruction property with proper filter scaling."""
        x_np = np.random.rand(32, 32)
        
        # Analysis filters
        h0_np, h1_np = dfilters_np('pkva', 'd')
        h0_torch, h1_torch = dfilters_torch('pkva', 'd')
        
        # Synthesis filters
        g0_np, g1_np = dfilters_np('pkva', 'r')
        g0_torch, g1_torch = dfilters_torch('pkva', 'r')
        
        # Scale filters by 1/sqrt(2) for nonsubsampled case with quincunx sampling
        # This is necessary for perfect reconstruction
        scale = 1.0 / np.sqrt(2.0)
        h0_np = h0_np * scale
        h1_np = h1_np * scale
        g0_np = g0_np * scale
        g1_np = g1_np * scale
        
        scale_torch = 1.0 / torch.sqrt(torch.tensor(2.0))
        h0_torch = h0_torch * scale_torch
        h1_torch = h1_torch * scale_torch
        g0_torch = g0_torch * scale_torch
        g1_torch = g1_torch * scale_torch
        
        mup_np = np.array([[1, 1], [-1, 1]])
        mup_torch = torch.tensor([[1, 1], [-1, 1]])
        
        x_torch = numpy_to_torch(x_np)
        
        # NumPy path
        y1_np, y2_np = core.nssfbdec(x_np, h0_np, h1_np, mup_np)
        recon_np = core.nssfbrec(y1_np, y2_np, g0_np, g1_np, mup_np)
        
        # PyTorch path
        y1_torch, y2_torch = core_torch.nssfbdec(x_torch, h0_torch, h1_torch, mup_torch)
        recon_torch = core_torch.nssfbrec(y1_torch, y2_torch, g0_torch, g1_torch, mup_torch)
        
        # Check reconstruction accuracy
        assert np.allclose(x_np, recon_np, atol=1e-9)
        assert torch.allclose(x_torch, recon_torch, atol=1e-6)
        
        # Check consistency between implementations
        assert np.allclose(recon_np, torch_to_numpy(recon_torch), rtol=1e-5, atol=1e-7)


class TestNsfbdecConsistency:
    """Test nsfbdec function consistency."""
    
    @pytest.mark.parametrize("level", [0, 1, 2, 3])
    def test_nsfbdec_levels(self, level):
        """Test nsfbdec at different decomposition levels."""
        x_np = np.random.rand(64, 64)
        
        h1_np, h2_np, g1_np, g2_np = atrousfilters_np('maxflat')
        h1_torch, h2_torch, g1_torch, g2_torch = atrousfilters_torch('maxflat')
        
        x_torch = numpy_to_torch(x_np)
        
        y0_np, y1_np = core.nsfbdec(x_np, h1_np, h2_np, level)
        y0_torch, y1_torch = core_torch.nsfbdec(x_torch, h1_torch, h2_torch, level)
        
        # Check shapes
        assert y0_np.shape == torch_to_numpy(y0_torch).shape
        assert y1_np.shape == torch_to_numpy(y1_torch).shape
        assert y0_np.shape == x_np.shape
        
        # Check numerical accuracy
        assert np.allclose(y0_np, torch_to_numpy(y0_torch), rtol=1e-5, atol=1e-7)
        assert np.allclose(y1_np, torch_to_numpy(y1_torch), rtol=1e-5, atol=1e-7)


class TestNsfbrecConsistency:
    """Test nsfbrec function consistency."""
    
    @pytest.mark.parametrize("level", [0, 1, 2])
    def test_nsfbrec_perfect_reconstruction(self, level):
        """Test perfect reconstruction at different levels."""
        x_np = np.random.rand(64, 64)
        
        h1_np, h2_np, g1_np, g2_np = atrousfilters_np('maxflat')
        h1_torch, h2_torch, g1_torch, g2_torch = atrousfilters_torch('maxflat')
        
        x_torch = numpy_to_torch(x_np)
        
        # Decompose
        y0_np, y1_np = core.nsfbdec(x_np, h1_np, h2_np, level)
        y0_torch, y1_torch = core_torch.nsfbdec(x_torch, h1_torch, h2_torch, level)
        
        # Reconstruct
        recon_np = core.nsfbrec(y0_np, y1_np, g1_np, g2_np, level)
        recon_torch = core_torch.nsfbrec(y0_torch, y1_torch, g1_torch, g2_torch, level)
        
        # Check perfect reconstruction
        assert np.allclose(x_np, recon_np, atol=1e-9)
        assert torch.allclose(x_torch, recon_torch, atol=1e-6)
        
        # Check consistency
        assert np.allclose(recon_np, torch_to_numpy(recon_torch), rtol=1e-5, atol=1e-7)


class TestNsdfbdecConsistency:
    """Test nsdfbdec function consistency."""
    
    @pytest.mark.parametrize("n_levels", [0, 1, 2, 3])
    def test_nsdfbdec_levels(self, n_levels):
        """Test nsdfbdec at different directional levels."""
        x_np = np.random.rand(64, 64)
        
        # Get filters
        h1_np, h2_np = dfilters_np('pkva', 'd')
        h1_torch, h2_torch = dfilters_torch('pkva', 'd')
        
        # Scale filters
        h1_np = h1_np / np.sqrt(2)
        h2_np = h2_np / np.sqrt(2)
        h1_torch = h1_torch / torch.sqrt(torch.tensor(2.0))
        h2_torch = h2_torch / torch.sqrt(torch.tensor(2.0))
        
        # Create filter dictionaries
        from nsct_python.filters import modulate2 as modulate2_np, parafilters as parafilters_np
        from nsct_torch.filters_torch import modulate2 as modulate2_torch, parafilters as parafilters_torch
        
        k1_np = modulate2_np(h1_np, 'c')
        k2_np = modulate2_np(h2_np, 'c')
        f1_np, f2_np = parafilters_np(h1_np, h2_np)
        
        k1_torch = modulate2_torch(h1_torch, 'c')
        k2_torch = modulate2_torch(h2_torch, 'c')
        f1_torch, f2_torch = parafilters_torch(h1_torch, h2_torch)
        
        filters_np = {'k1': k1_np, 'k2': k2_np, 'f1': f1_np, 'f2': f2_np}
        filters_torch = {'k1': k1_torch, 'k2': k2_torch, 'f1': f1_torch, 'f2': f2_torch}
        
        x_torch = numpy_to_torch(x_np)
        
        # Decompose
        y_np = core.nsdfbdec(x_np, filters_np, n_levels)
        y_torch = core_torch.nsdfbdec(x_torch, filters_torch, n_levels)
        
        # Check number of subbands
        assert len(y_np) == len(y_torch)
        assert len(y_np) == 2**n_levels
        
        # Check shapes and numerical accuracy
        for i in range(len(y_np)):
            assert y_np[i].shape == torch_to_numpy(y_torch[i]).shape
            assert np.allclose(y_np[i], torch_to_numpy(y_torch[i]), rtol=1e-5, atol=1e-7)


class TestNsdfbrecConsistency:
    """Test nsdfbrec function consistency."""
    
    @pytest.mark.parametrize("n_levels", [1, 2, 3])
    def test_nsdfbrec_perfect_reconstruction(self, n_levels):
        """Test perfect reconstruction with directional filter bank."""
        x_np = np.random.rand(64, 64)
        
        # Get analysis filters
        h1_np, h2_np = dfilters_np('pkva', 'd')
        h1_torch, h2_torch = dfilters_torch('pkva', 'd')
        
        h1_np = h1_np / np.sqrt(2)
        h2_np = h2_np / np.sqrt(2)
        h1_torch = h1_torch / torch.sqrt(torch.tensor(2.0))
        h2_torch = h2_torch / torch.sqrt(torch.tensor(2.0))
        
        from nsct_python.filters import modulate2 as modulate2_np, parafilters as parafilters_np
        from nsct_torch.filters_torch import modulate2 as modulate2_torch, parafilters as parafilters_torch
        
        k1_np = modulate2_np(h1_np, 'c')
        k2_np = modulate2_np(h2_np, 'c')
        f1_np, f2_np = parafilters_np(h1_np, h2_np)
        
        k1_torch = modulate2_torch(h1_torch, 'c')
        k2_torch = modulate2_torch(h2_torch, 'c')
        f1_torch, f2_torch = parafilters_torch(h1_torch, h2_torch)
        
        filters_np = {'k1': k1_np, 'k2': k2_np, 'f1': f1_np, 'f2': f2_np}
        filters_torch = {'k1': k1_torch, 'k2': k2_torch, 'f1': f1_torch, 'f2': f2_torch}
        
        x_torch = numpy_to_torch(x_np)
        
        # Decompose
        y_np = core.nsdfbdec(x_np, filters_np, n_levels)
        y_torch = core_torch.nsdfbdec(x_torch, filters_torch, n_levels)
        
        # Get synthesis filters
        h1_syn_np, h2_syn_np = dfilters_np('pkva', 'r')
        h1_syn_torch, h2_syn_torch = dfilters_torch('pkva', 'r')
        
        h1_syn_np = h1_syn_np / np.sqrt(2)
        h2_syn_np = h2_syn_np / np.sqrt(2)
        h1_syn_torch = h1_syn_torch / torch.sqrt(torch.tensor(2.0))
        h2_syn_torch = h2_syn_torch / torch.sqrt(torch.tensor(2.0))
        
        k1_syn_np = modulate2_np(h1_syn_np, 'c')
        k2_syn_np = modulate2_np(h2_syn_np, 'c')
        f1_syn_np, f2_syn_np = parafilters_np(h1_syn_np, h2_syn_np)
        
        k1_syn_torch = modulate2_torch(h1_syn_torch, 'c')
        k2_syn_torch = modulate2_torch(h2_syn_torch, 'c')
        f1_syn_torch, f2_syn_torch = parafilters_torch(h1_syn_torch, h2_syn_torch)
        
        filters_syn_np = {'k1': k1_syn_np, 'k2': k2_syn_np, 'f1': f1_syn_np, 'f2': f2_syn_np}
        filters_syn_torch = {'k1': k1_syn_torch, 'k2': k2_syn_torch, 'f1': f1_syn_torch, 'f2': f2_syn_torch}
        
        # Reconstruct
        recon_np = core.nsdfbrec(y_np, filters_syn_np)
        recon_torch = core_torch.nsdfbrec(y_torch, filters_syn_torch)
        
        # Check perfect reconstruction
        assert np.allclose(x_np, recon_np, atol=1e-9)
        assert torch.allclose(x_torch, recon_torch, atol=1e-6)
        
        # Check consistency
        assert np.allclose(recon_np, torch_to_numpy(recon_torch), rtol=1e-5, atol=1e-7)


class TestNsctdecConsistency:
    """Test nsctdec function consistency."""
    
    @pytest.mark.parametrize("nlevs", [
        [2],
        [2, 3],
        [2, 3, 4],
    ])
    def test_nsctdec_structure(self, nlevs):
        """Test NSCT decomposition structure."""
        x_np = np.random.rand(128, 128)
        x_torch = numpy_to_torch(x_np)
        
        y_np = core.nsctdec(x_np, nlevs, dfilt='pkva', pfilt='maxflat')
        y_torch = core_torch.nsctdec(x_torch, nlevs, dfilt='pkva', pfilt='maxflat')
        
        # Check structure
        assert len(y_np) == len(y_torch)
        assert len(y_np) == len(nlevs) + 1
        
        # Check lowpass
        assert y_np[0].shape == torch_to_numpy(y_torch[0]).shape
        assert np.allclose(y_np[0], torch_to_numpy(y_torch[0]), rtol=1e-5, atol=1e-7)
        
        # Check directional subbands
        for i in range(1, len(y_np)):
            if isinstance(y_np[i], list):
                assert isinstance(y_torch[i], list)
                assert len(y_np[i]) == len(y_torch[i])
                for j in range(len(y_np[i])):
                    assert y_np[i][j].shape == torch_to_numpy(y_torch[i][j]).shape
                    assert np.allclose(y_np[i][j], torch_to_numpy(y_torch[i][j]), rtol=1e-5, atol=1e-7)
            else:
                assert y_np[i].shape == torch_to_numpy(y_torch[i]).shape
                assert np.allclose(y_np[i], torch_to_numpy(y_torch[i]), rtol=1e-5, atol=1e-7)


class TestNsctrecConsistency:
    """Test nsctrec function consistency."""
    
    @pytest.mark.parametrize("nlevs", [
        [2],
        [2, 3],
        [2, 3, 4],
    ])
    def test_nsctrec_perfect_reconstruction(self, nlevs):
        """Test NSCT perfect reconstruction."""
        x_np = np.random.rand(128, 128)
        x_torch = numpy_to_torch(x_np)
        
        # Decompose
        y_np = core.nsctdec(x_np, nlevs, dfilt='pkva', pfilt='maxflat')
        y_torch = core_torch.nsctdec(x_torch, nlevs, dfilt='pkva', pfilt='maxflat')
        
        # Reconstruct
        recon_np = core.nsctrec(y_np, dfilt='pkva', pfilt='maxflat')
        recon_torch = core_torch.nsctrec(y_torch, dfilt='pkva', pfilt='maxflat')
        
        # Check perfect reconstruction
        recon_error_np = np.max(np.abs(x_np - recon_np))
        recon_error_torch = torch.max(torch.abs(x_torch - recon_torch)).item()
        
        print(f"NumPy reconstruction error: {recon_error_np}")
        print(f"PyTorch reconstruction error: {recon_error_torch}")
        
        assert recon_error_np < 1e-10
        assert recon_error_torch < 1e-6
        
        # Check consistency between implementations
        assert np.allclose(recon_np, torch_to_numpy(recon_torch), rtol=1e-5, atol=1e-7)


class TestPrecisionAndDtype:
    """Test precision and data type handling."""
    
    def test_float32_precision(self):
        """Test with float32 precision."""
        x_np = np.random.rand(32, 32).astype(np.float32)
        x_torch = torch.from_numpy(x_np)
        
        h1_np, h2_np, g1_np, g2_np = atrousfilters_np('maxflat')
        h1_torch, h2_torch, g1_torch, g2_torch = atrousfilters_torch('maxflat')
        
        # Ensure float32
        h1_torch = h1_torch.float()
        h2_torch = h2_torch.float()
        
        y0_torch, y1_torch = core_torch.nsfbdec(x_torch, h1_torch, h2_torch, 0)
        
        assert y0_torch.dtype == torch.float32
        assert y1_torch.dtype == torch.float32
    
    def test_float64_precision(self):
        """Test with float64 precision."""
        x_np = np.random.rand(32, 32).astype(np.float64)
        x_torch = torch.from_numpy(x_np)
        
        h1_np, h2_np, g1_np, g2_np = atrousfilters_np('maxflat')
        h1_torch, h2_torch, g1_torch, g2_torch = atrousfilters_torch('maxflat')
        
        # Ensure float64
        h1_torch = h1_torch.double()
        h2_torch = h2_torch.double()
        
        y0_torch, y1_torch = core_torch.nsfbdec(x_torch, h1_torch, h2_torch, 0)
        
        assert y0_torch.dtype == torch.float64
        assert y1_torch.dtype == torch.float64


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_nssfbrec_shape_mismatch(self):
        """Test error when input shapes don't match."""
        x1 = torch.rand(32, 32)
        x2 = torch.rand(16, 16)
        
        h0, h1 = dfilters_torch('pkva', 'd')
        
        with pytest.raises(ValueError, match="Input sizes.*must be the same"):
            core_torch.nssfbrec(x1, x2, h0, h1)
    
    def test_nsfbrec_shape_mismatch(self):
        """Test error when input shapes don't match."""
        x0 = torch.rand(32, 32)
        x1 = torch.rand(16, 16)
        
        h1, h2, g1, g2 = atrousfilters_torch('maxflat')
        
        with pytest.raises(ValueError, match="Input sizes.*must be the same"):
            core_torch.nsfbrec(x0, x1, g1, g2, 0)
    
    def test_nsdfbrec_invalid_subband_count(self):
        """Test error with invalid number of subbands."""
        # 3 subbands (not a power of 2)
        y = [torch.rand(32, 32) for _ in range(3)]
        
        h1, h2 = dfilters_torch('pkva', 'r')
        h1 = h1 / torch.sqrt(torch.tensor(2.0))
        h2 = h2 / torch.sqrt(torch.tensor(2.0))
        
        from nsct_torch.filters_torch import modulate2, parafilters
        k1 = modulate2(h1, 'c')
        k2 = modulate2(h2, 'c')
        f1, f2 = parafilters(h1, h2)
        
        filters = {'k1': k1, 'k2': k2, 'f1': f1, 'f2': f2}
        
        with pytest.raises(ValueError, match="Number of subbands must be a power of 2"):
            core_torch.nsdfbrec(y, filters)


class TestCUDACompatibility:
    """Test CUDA compatibility if available."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_device(self):
        """Test operations on CUDA device."""
        x_cpu = torch.rand(32, 32)
        x_cuda = x_cpu.cuda()
        
        h1, h2, g1, g2 = atrousfilters_torch('maxflat')
        h1 = h1.cuda()
        h2 = h2.cuda()
        
        y0, y1 = core_torch.nsfbdec(x_cuda, h1, h2, 0)
        
        assert y0.is_cuda
        assert y1.is_cuda
        assert y0.device == x_cuda.device
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cpu_cuda_consistency(self):
        """Test consistency between CPU and CUDA operations."""
        x = torch.rand(32, 32, dtype=torch.float64)  # Use float64 to match filters
        
        h1, h2, g1, g2 = atrousfilters_torch('maxflat')
        
        # CPU
        y0_cpu, y1_cpu = core_torch.nsfbdec(x, h1, h2, 0)
        
        # CUDA
        y0_cuda, y1_cuda = core_torch.nsfbdec(x.cuda(), h1.cuda(), h2.cuda(), 0)
        
        assert torch.allclose(y0_cpu, y0_cuda.cpu(), rtol=1e-5, atol=1e-7)
        assert torch.allclose(y1_cpu, y1_cuda.cpu(), rtol=1e-5, atol=1e-7)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
