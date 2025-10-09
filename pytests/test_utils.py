"""
Pytest tests for nsct_torch/utils.py

Tests numerical accuracy, precision, shape consistency, and one-to-one 
consistency between NumPy (nsct_python/utils.py) and PyTorch (nsct_torch/utils.py) implementations.
"""

import pytest
import numpy as np
import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nsct_python.utils import (
    extend2 as extend2_np,
    symext as symext_np,
    upsample2df as upsample2df_np,
    modulate2 as modulate2_np,
    resampz as resampz_np,
    qupz as qupz_np
)

from nsct_torch.utils import (
    extend2 as extend2_torch,
    symext as symext_torch,
    upsample2df as upsample2df_torch,
    modulate2 as modulate2_torch,
    resampz as resampz_torch,
    qupz as qupz_torch
)


# ============================================================================
# Helper functions for testing
# ============================================================================

def assert_tensors_close(torch_result, np_result, rtol=1e-7, atol=1e-7):
    """Assert that torch and numpy results are close."""
    torch_np = torch_result.cpu().numpy()
    assert torch_np.shape == np_result.shape, \
        f"Shape mismatch: torch {torch_np.shape} vs numpy {np_result.shape}"
    np.testing.assert_allclose(torch_np, np_result, rtol=rtol, atol=atol)


def assert_exact_match(torch_result, np_result):
    """Assert that torch and numpy results match exactly."""
    torch_np = torch_result.cpu().numpy()
    assert torch_np.shape == np_result.shape, \
        f"Shape mismatch: torch {torch_np.shape} vs numpy {np_result.shape}"
    np.testing.assert_array_equal(torch_np, np_result)


# ============================================================================
# Test extend2 function
# ============================================================================

class TestExtend2:
    """Test cases for extend2 function."""
    
    @pytest.mark.parametrize("shape", [(4, 4), (5, 7), (8, 6), (10, 10)])
    @pytest.mark.parametrize("ru,rd,cl,cr", [
        (1, 1, 1, 1),
        (2, 2, 2, 2),
        (1, 2, 3, 4),
        (0, 1, 0, 1),
    ])
    def test_extend2_per_mode(self, shape, ru, rd, cl, cr):
        """Test periodic extension mode with various shapes and padding."""
        x_np = np.random.randn(*shape)
        x_torch = torch.from_numpy(x_np)
        
        result_np = extend2_np(x_np, ru, rd, cl, cr, 'per')
        result_torch = extend2_torch(x_torch, ru, rd, cl, cr, 'per')
        
        # Check shape
        expected_shape = (shape[0] + ru + rd, shape[1] + cl + cr)
        assert result_torch.shape == expected_shape
        
        # Check numerical consistency
        assert_tensors_close(result_torch, result_np)
    
    @pytest.mark.parametrize("shape", [(4, 4), (6, 8), (10, 10)])
    def test_extend2_qper_row_mode(self, shape):
        """Test quincunx periodized extension in row."""
        x_np = np.random.randn(*shape)
        x_torch = torch.from_numpy(x_np)
        
        ru, rd, cl, cr = 2, 2, 2, 2
        
        result_np = extend2_np(x_np, ru, rd, cl, cr, 'qper_row')
        result_torch = extend2_torch(x_torch, ru, rd, cl, cr, 'qper_row')
        
        # Check shape
        assert result_torch.shape == result_np.shape
        
        # Check numerical consistency
        assert_tensors_close(result_torch, result_np)
    
    @pytest.mark.parametrize("shape", [(4, 4), (6, 8), (10, 10)])
    def test_extend2_qper_col_mode(self, shape):
        """Test quincunx periodized extension in column."""
        x_np = np.random.randn(*shape)
        x_torch = torch.from_numpy(x_np)
        
        ru, rd, cl, cr = 2, 2, 2, 2
        
        result_np = extend2_np(x_np, ru, rd, cl, cr, 'qper_col')
        result_torch = extend2_torch(x_torch, ru, rd, cl, cr, 'qper_col')
        
        # Check shape
        assert result_torch.shape == result_np.shape
        
        # Check numerical consistency
        assert_tensors_close(result_torch, result_np)
    
    def test_extend2_invalid_mode(self):
        """Test that invalid mode raises ValueError."""
        x_torch = torch.randn(4, 4)
        with pytest.raises(ValueError):
            extend2_torch(x_torch, 1, 1, 1, 1, 'invalid')
    
    def test_extend2_dtype_preservation(self):
        """Test that dtype is preserved."""
        for dtype in [torch.float32, torch.float64]:
            x_torch = torch.randn(4, 4, dtype=dtype)
            result = extend2_torch(x_torch, 1, 1, 1, 1, 'per')
            assert result.dtype == dtype
    
    def test_extend2_device_preservation(self):
        """Test that device is preserved."""
        x_torch = torch.randn(4, 4)
        result = extend2_torch(x_torch, 1, 1, 1, 1, 'per')
        assert result.device == x_torch.device


# ============================================================================
# Test symext function
# ============================================================================

class TestSymext:
    """Test cases for symext function."""
    
    @pytest.mark.parametrize("img_shape", [(4, 4), (5, 7), (8, 6)])
    @pytest.mark.parametrize("filt_shape", [(3, 3), (5, 5), (3, 5)])
    @pytest.mark.parametrize("shift", [[0, 0], [1, 1], [2, 2]])
    def test_symext_basic(self, img_shape, filt_shape, shift):
        """Test symmetric extension with various configurations."""
        x_np = np.random.randn(*img_shape)
        h_np = np.random.randn(*filt_shape)
        
        x_torch = torch.from_numpy(x_np)
        h_torch = torch.from_numpy(h_np)
        
        result_np = symext_np(x_np, h_np, shift)
        result_torch = symext_torch(x_torch, h_torch, shift)
        
        # Check shape consistency between implementations
        assert result_torch.shape == result_np.shape, \
            f"Shape mismatch: torch {result_torch.shape} vs numpy {result_np.shape}"
        
        # The expected shape should be close to (m+p-1, n+q-1), but the final 
        # size depends on the extension logic and shift parameters
        # So we just verify the implementations are consistent
        
        # Check numerical consistency
        assert_tensors_close(result_torch, result_np)
    
    def test_symext_specific_case(self):
        """Test specific case from documentation."""
        x_np = np.arange(16).reshape(4, 4)
        h_np = np.ones((3, 3))
        shift = [1, 1]
        
        x_torch = torch.from_numpy(x_np.astype(np.float64))
        h_torch = torch.from_numpy(h_np)
        
        result_np = symext_np(x_np, h_np, shift)
        result_torch = symext_torch(x_torch, h_torch, shift)
        
        assert result_torch.shape == (6, 6)
        assert_tensors_close(result_torch, result_np)
    
    def test_symext_dtype_preservation(self):
        """Test that dtype is preserved."""
        x_np = np.random.randn(4, 4)
        h_np = np.random.randn(3, 3)
        
        for dtype in [torch.float32, torch.float64]:
            x_torch = torch.from_numpy(x_np).to(dtype)
            h_torch = torch.from_numpy(h_np).to(dtype)
            result = symext_torch(x_torch, h_torch, [1, 1])
            assert result.dtype == dtype


# ============================================================================
# Test upsample2df function
# ============================================================================

class TestUpsample2df:
    """Test cases for upsample2df function."""
    
    @pytest.mark.parametrize("shape", [(2, 2), (3, 3), (4, 5)])
    @pytest.mark.parametrize("power", [1, 2, 3])
    def test_upsample2df_basic(self, shape, power):
        """Test upsampling with various shapes and powers."""
        h_np = np.random.randn(*shape)
        h_torch = torch.from_numpy(h_np)
        
        result_np = upsample2df_np(h_np, power)
        result_torch = upsample2df_torch(h_torch, power)
        
        # Check shape
        factor = 2 ** power
        expected_shape = (shape[0] * factor, shape[1] * factor)
        assert result_torch.shape == expected_shape
        
        # Check numerical consistency
        assert_tensors_close(result_torch, result_np)
    
    def test_upsample2df_specific_case(self):
        """Test specific case: [[1,2],[3,4]] with power=1."""
        h_np = np.array([[1, 2], [3, 4]], dtype=np.float64)
        h_torch = torch.from_numpy(h_np)
        
        result_np = upsample2df_np(h_np, power=1)
        result_torch = upsample2df_torch(h_torch, power=1)
        
        expected = np.array([[1, 0, 2, 0], 
                            [0, 0, 0, 0], 
                            [3, 0, 4, 0], 
                            [0, 0, 0, 0]], dtype=np.float64)
        
        assert_tensors_close(result_torch, expected)
        assert_exact_match(result_torch, result_np)
    
    def test_upsample2df_zeros_preserved(self):
        """Test that zeros are inserted correctly."""
        h_np = np.ones((2, 2))
        h_torch = torch.from_numpy(h_np)
        
        result_torch = upsample2df_torch(h_torch, power=1)
        
        # Check that zeros are in the right places
        assert result_torch[0, 1].item() == 0
        assert result_torch[1, 0].item() == 0
        assert result_torch[1, 1].item() == 0
        
        # Check that original values are preserved
        assert result_torch[0, 0].item() == 1
        assert result_torch[0, 2].item() == 1
        assert result_torch[2, 0].item() == 1
        assert result_torch[2, 2].item() == 1


# ============================================================================
# Test modulate2 function
# ============================================================================

class TestModulate2:
    """Test cases for modulate2 function."""
    
    @pytest.mark.parametrize("shape", [(3, 4), (4, 4), (5, 6)])
    @pytest.mark.parametrize("mode", ['r', 'c', 'b'])
    def test_modulate2_basic(self, shape, mode):
        """Test modulation with various shapes and modes."""
        x_np = np.random.randn(*shape)
        x_torch = torch.from_numpy(x_np)
        
        result_np = modulate2_np(x_np, mode)
        result_torch = modulate2_torch(x_torch, mode)
        
        # Check shape
        assert result_torch.shape == shape
        
        # Check numerical consistency
        assert_tensors_close(result_torch, result_np)
    
    def test_modulate2_with_center(self):
        """Test modulation with custom center."""
        x_np = np.random.randn(4, 4)
        x_torch = torch.from_numpy(x_np)
        center = [1, 1]
        
        result_np = modulate2_np(x_np, 'b', center)
        result_torch = modulate2_torch(x_torch, 'b', center)
        
        assert_tensors_close(result_torch, result_np)
    
    def test_modulate2_specific_case(self):
        """Test specific case: ones matrix modulated."""
        m_np = np.ones((3, 4))
        m_torch = torch.ones((3, 4), dtype=torch.float64)
        
        result_np = modulate2_np(m_np, 'b')
        result_torch = modulate2_torch(m_torch, 'b')
        
        # Check specific values
        assert result_torch[0, 0].item() == -1.0
        assert result_torch[1, 0].item() == 1.0
        assert result_torch[0, 1].item() == 1.0
        
        assert_tensors_close(result_torch, result_np)
    
    def test_modulate2_alternating_signs(self):
        """Test that modulation creates alternating sign pattern."""
        x_torch = torch.ones((4, 4), dtype=torch.float64)
        result = modulate2_torch(x_torch, 'b')
        
        # Check checkerboard pattern (alternating signs)
        for i in range(4):
            for j in range(4):
                expected_sign = (-1) ** (i + j)
                # Allow for numerical errors
                assert abs(result[i, j].item() - expected_sign) < 1e-10


# ============================================================================
# Test resampz function
# ============================================================================

class TestResampz:
    """Test cases for resampz function."""
    
    @pytest.mark.parametrize("type", [1, 2, 3, 4])
    @pytest.mark.parametrize("shift", [1, 2])
    def test_resampz_basic(self, type, shift):
        """Test resampling with all types."""
        x_np = np.random.randn(3, 4)
        x_torch = torch.from_numpy(x_np)
        
        result_np = resampz_np(x_np, type, shift)
        result_torch = resampz_torch(x_torch, type, shift)
        
        # Check shape
        assert result_torch.shape == result_np.shape
        
        # Check numerical consistency
        assert_tensors_close(result_torch, result_np)
    
    def test_resampz_type1_specific(self):
        """Test type 1 (vertical shearing) with specific case."""
        r_in_np = np.arange(1, 7).reshape(2, 3)
        r_in_torch = torch.from_numpy(r_in_np.astype(np.float64))
        
        result_np = resampz_np(r_in_np, 1, shift=1)
        result_torch = resampz_torch(r_in_torch, 1, shift=1)
        
        expected = np.array([[0, 0, 3], [0, 2, 6], [1, 5, 0], [4, 0, 0]], 
                           dtype=np.float64)
        
        assert_exact_match(result_torch, expected)
        assert_exact_match(result_torch, result_np)
    
    def test_resampz_type3_specific(self):
        """Test type 3 (horizontal shearing) with specific case."""
        r_in_np = np.arange(1, 7).reshape(2, 3)
        r_in_torch = torch.from_numpy(r_in_np.astype(np.float64))
        
        result_np = resampz_np(r_in_np, 3, shift=1)
        result_torch = resampz_torch(r_in_torch, 3, shift=1)
        
        expected = np.array([[0, 1, 2, 3], [4, 5, 6, 0]], dtype=np.float64)
        
        assert_exact_match(result_torch, expected)
        assert_exact_match(result_torch, result_np)
    
    def test_resampz_invalid_type(self):
        """Test that invalid type raises ValueError."""
        x_torch = torch.randn(3, 4)
        with pytest.raises(ValueError):
            resampz_torch(x_torch, 5, shift=1)
    
    def test_resampz_empty_result(self):
        """Test handling of edge cases that might produce empty results."""
        x_np = np.zeros((2, 2))
        x_torch = torch.from_numpy(x_np)
        
        result_np = resampz_np(x_np, 1, shift=1)
        result_torch = resampz_torch(x_torch, 1, shift=1)
        
        # Both should handle zeros gracefully
        assert result_torch.shape == result_np.shape


# ============================================================================
# Test qupz function
# ============================================================================

class TestQupz:
    """Test cases for qupz function."""
    
    @pytest.mark.parametrize("type", [1, 2])
    def test_qupz_basic(self, type):
        """Test quincunx upsampling with both types."""
        x_np = np.random.randn(3, 4)
        x_torch = torch.from_numpy(x_np)
        
        result_np = qupz_np(x_np, type)
        result_torch = qupz_torch(x_torch, type)
        
        # Check shape
        assert result_torch.shape == result_np.shape
        
        # Check numerical consistency
        assert_tensors_close(result_torch, result_np)
    
    def test_qupz_type1_specific(self):
        """Test type 1 with specific case."""
        q_in_np = np.array([[1, 2], [3, 4]], dtype=np.float64)
        q_in_torch = torch.from_numpy(q_in_np)
        
        result_np = qupz_np(q_in_np, 1)
        result_torch = qupz_torch(q_in_torch, 1)
        
        expected = np.array([[0, 2, 0],
                            [1, 0, 4],
                            [0, 3, 0]], dtype=np.float64)
        
        assert_exact_match(result_torch, expected)
        assert_exact_match(result_torch, result_np)
    
    def test_qupz_type2_specific(self):
        """Test type 2 with specific case."""
        q_in_np = np.array([[1, 2], [3, 4]], dtype=np.float64)
        q_in_torch = torch.from_numpy(q_in_np)
        
        result_np = qupz_np(q_in_np, 2)
        result_torch = qupz_torch(q_in_torch, 2)
        
        # Check shape and numerical consistency
        assert result_torch.shape == result_np.shape
        assert_exact_match(result_torch, result_np)
    
    def test_qupz_invalid_type(self):
        """Test that invalid type raises ValueError."""
        x_torch = torch.randn(3, 3)
        with pytest.raises(ValueError):
            qupz_torch(x_torch, 3)
    
    def test_qupz_zeros_placement(self):
        """Test that zeros are placed correctly in output."""
        q_in_torch = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
        result = qupz_torch(q_in_torch, 1)
        
        # Check specific zero positions
        assert result[0, 0].item() == 0
        assert result[1, 1].item() == 0
        assert result[2, 2].item() == 0
        
        # Check non-zero positions
        assert result[0, 1].item() == 2
        assert result[1, 0].item() == 1
        assert result[1, 2].item() == 4
        assert result[2, 1].item() == 3
    
    @pytest.mark.parametrize("shape", [(2, 2), (3, 3), (4, 5), (5, 4)])
    def test_qupz_various_shapes(self, shape):
        """Test qupz with various input shapes."""
        x_np = np.random.randn(*shape)
        x_torch = torch.from_numpy(x_np)
        
        for type in [1, 2]:
            result_np = qupz_np(x_np, type)
            result_torch = qupz_torch(x_torch, type)
            
            assert result_torch.shape == result_np.shape
            assert_tensors_close(result_torch, result_np)


# ============================================================================
# Integration tests
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple functions."""
    
    def test_extend_and_modulate(self):
        """Test combining extend2 and modulate2."""
        x_np = np.random.randn(4, 4)
        x_torch = torch.from_numpy(x_np)
        
        # Extend
        ext_np = extend2_np(x_np, 1, 1, 1, 1, 'per')
        ext_torch = extend2_torch(x_torch, 1, 1, 1, 1, 'per')
        
        # Modulate
        result_np = modulate2_np(ext_np, 'b')
        result_torch = modulate2_torch(ext_torch, 'b')
        
        assert_tensors_close(result_torch, result_np)
    
    def test_upsample_and_resample(self):
        """Test combining upsample2df and resampz."""
        h_np = np.random.randn(2, 2)
        h_torch = torch.from_numpy(h_np)
        
        # Upsample
        up_np = upsample2df_np(h_np, power=1)
        up_torch = upsample2df_torch(h_torch, power=1)
        
        # Resample
        result_np = resampz_np(up_np, 1, shift=1)
        result_torch = resampz_torch(up_torch, 1, shift=1)
        
        assert_tensors_close(result_torch, result_np)
    
    def test_symext_consistency(self):
        """Test symext with realistic filter sizes."""
        x_np = np.random.randn(16, 16)
        h_np = np.random.randn(7, 7)
        x_torch = torch.from_numpy(x_np)
        h_torch = torch.from_numpy(h_np)
        
        result_np = symext_np(x_np, h_np, [3, 3])
        result_torch = symext_torch(x_torch, h_torch, [3, 3])
        
        assert_tensors_close(result_torch, result_np)


# ============================================================================
# Precision tests
# ============================================================================

class TestPrecision:
    """Test numerical precision and stability."""
    
    def test_float32_precision(self):
        """Test that float32 operations maintain acceptable precision."""
        x_np = np.random.randn(8, 8).astype(np.float32)
        x_torch = torch.from_numpy(x_np)
        
        result_np = extend2_np(x_np, 2, 2, 2, 2, 'per')
        result_torch = extend2_torch(x_torch, 2, 2, 2, 2, 'per')
        
        # Float32 should still be very close
        assert_tensors_close(result_torch, result_np, rtol=1e-6, atol=1e-6)
    
    def test_float64_precision(self):
        """Test that float64 operations maintain high precision."""
        x_np = np.random.randn(8, 8).astype(np.float64)
        x_torch = torch.from_numpy(x_np)
        
        result_np = modulate2_np(x_np, 'b')
        result_torch = modulate2_torch(x_torch, 'b')
        
        # Float64 should be very precise
        assert_tensors_close(result_torch, result_np, rtol=1e-12, atol=1e-12)
    
    def test_large_values(self):
        """Test with large magnitude values."""
        x_np = np.random.randn(4, 4) * 1e6
        x_torch = torch.from_numpy(x_np)
        
        result_np = extend2_np(x_np, 1, 1, 1, 1, 'per')
        result_torch = extend2_torch(x_torch, 1, 1, 1, 1, 'per')
        
        # Relative tolerance matters for large values
        assert_tensors_close(result_torch, result_np, rtol=1e-6)
    
    def test_small_values(self):
        """Test with small magnitude values."""
        x_np = np.random.randn(4, 4) * 1e-6
        x_torch = torch.from_numpy(x_np)
        
        result_np = upsample2df_np(x_np, power=1)
        result_torch = upsample2df_torch(x_torch, power=1)
        
        # Absolute tolerance matters for small values
        assert_tensors_close(result_torch, result_np, atol=1e-12)


# ============================================================================
# Edge case tests
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_minimum_size(self):
        """Test with minimum valid input sizes."""
        x_np = np.array([[1.0]])
        x_torch = torch.from_numpy(x_np)
        
        result_np = extend2_np(x_np, 1, 1, 1, 1, 'per')
        result_torch = extend2_torch(x_torch, 1, 1, 1, 1, 'per')
        
        assert_tensors_close(result_torch, result_np)
    
    def test_zero_extension(self):
        """Test with zero extension amounts."""
        x_np = np.random.randn(4, 4)
        x_torch = torch.from_numpy(x_np)
        
        result_np = extend2_np(x_np, 0, 0, 0, 0, 'per')
        result_torch = extend2_torch(x_torch, 0, 0, 0, 0, 'per')
        
        # Should return the same array
        assert_tensors_close(result_torch, x_np)
        assert_exact_match(result_torch, result_np)
    
    def test_asymmetric_shapes(self):
        """Test with highly asymmetric shapes."""
        x_np = np.random.randn(2, 10)
        x_torch = torch.from_numpy(x_np)
        
        result_np = modulate2_np(x_np, 'b')
        result_torch = modulate2_torch(x_torch, 'b')
        
        assert_tensors_close(result_torch, result_np)
    
    def test_power_zero_upsample(self):
        """Test upsample with power=0 (no upsampling)."""
        h_np = np.random.randn(3, 3)
        h_torch = torch.from_numpy(h_np)
        
        result_np = upsample2df_np(h_np, power=0)
        result_torch = upsample2df_torch(h_torch, power=0)
        
        # Should return the same array
        assert_exact_match(result_torch, h_np)
        assert_exact_match(result_torch, result_np)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
