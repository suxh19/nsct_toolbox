"""
test_atrousc_consistency.py
============================

Test suite for verifying consistency between C++ and CUDA implementations of atrousc.

This module tests:
- Numerical point-to-point consistency between implementations
- Precision and accuracy
- Edge cases and boundary conditions
- Performance characteristics
"""

import sys
import pytest
import numpy as np
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Try to import both implementations
try:
    from nsct_python.atrousc_cpp import atrousc_cpp, CPP_AVAILABLE
    atrousc_cpp_module = atrousc_cpp
except ImportError as e:
    CPP_AVAILABLE = False
    atrousc_cpp_module = None
    print(f"Failed to import C++ module: {e}")

try:
    import torch
    from nsct_torch.atrousc_cuda import atrousc_cuda, is_available
    CUDA_AVAILABLE = torch.cuda.is_available() and is_available()
    if not CUDA_AVAILABLE:
        print("CUDA not available: torch.cuda.is_available() =", torch.cuda.is_available())
except ImportError as e:
    CUDA_AVAILABLE = False
    torch = None
    atrousc_cuda = None
    print(f"Failed to import CUDA module: {e}")


# Skip all tests if either implementation is not available
pytestmark = pytest.mark.skipif(
    not (CPP_AVAILABLE and CUDA_AVAILABLE),
    reason="Both C++ and CUDA implementations required for consistency tests"
)


class TestAtrouscConsistency:
    """Test consistency between C++ and CUDA implementations of atrousc."""
    
    @pytest.fixture(scope="class")
    def test_data(self):
        """Generate test data for consistency checks."""
        np.random.seed(42)
        
        test_cases = {
            'small': {
                'x': np.random.randn(50, 50),
                'h': np.random.randn(3, 3),
                'M': np.array([[2, 0], [0, 2]], dtype=np.int32),
                'description': 'Small extended input, 2x upsampling'
            },
            'medium': {
                'x': np.random.randn(150, 150),
                'h': np.random.randn(5, 5),
                'M': np.array([[2, 0], [0, 2]], dtype=np.int32),
                'description': 'Medium extended input, 2x upsampling'
            },
            'large': {
                'x': np.random.randn(300, 300),
                'h': np.random.randn(7, 7),
                'M': np.array([[3, 0], [0, 3]], dtype=np.int32),
                'description': 'Large extended input, 3x upsampling'
            },
            'asymmetric': {
                'x': np.random.randn(100, 150),
                'h': np.random.randn(3, 5),
                'M': np.array([[2, 0], [0, 3]], dtype=np.int32),
                'description': 'Asymmetric dimensions and different upsampling factors'
            },
            'extreme_values': {
                'x': np.random.randn(80, 80) * 1e6,
                'h': np.random.randn(5, 5) * 1e-6,
                'M': np.array([[2, 0], [0, 2]], dtype=np.int32),
                'description': 'Extreme value ranges'
            },
            'scalar_upsampling': {
                'x': np.random.randn(60, 60),
                'h': np.random.randn(3, 3),
                'M': np.array([[2]], dtype=np.int32),
                'description': 'Scalar upsampling matrix'
            }
        }
        
        return test_cases
    
    def test_exact_match_small(self, test_data):
        """Test exact numerical match on small data."""
        case = test_data['small']
        x, h, M = case['x'], case['h'], case['M']
        
        # C++ implementation
        cpp_result = atrousc_cpp_module.atrousc(x, h, M)
        
        # CUDA implementation
        x_cuda = torch.from_numpy(x).cuda()
        h_cuda = torch.from_numpy(h).cuda()
        M_cuda = torch.from_numpy(M)
        cuda_result = atrousc_cuda(x_cuda, h_cuda, M_cuda).cpu().numpy()
        
        # Check exact match
        np.testing.assert_array_almost_equal(
            cpp_result, cuda_result, decimal=10,
            err_msg=f"Mismatch in {case['description']}"
        )
    
    def test_exact_match_medium(self, test_data):
        """Test exact numerical match on medium-sized data."""
        case = test_data['medium']
        x, h, M = case['x'], case['h'], case['M']
        
        # C++ implementation
        cpp_result = atrousc_cpp_module.atrousc(x, h, M)
        
        # CUDA implementation
        x_cuda = torch.from_numpy(x).cuda()
        h_cuda = torch.from_numpy(h).cuda()
        M_cuda = torch.from_numpy(M)
        cuda_result = atrousc_cuda(x_cuda, h_cuda, M_cuda).cpu().numpy()
        
        # Check exact match
        np.testing.assert_array_almost_equal(
            cpp_result, cuda_result, decimal=10,
            err_msg=f"Mismatch in {case['description']}"
        )
    
    def test_exact_match_large(self, test_data):
        """Test exact numerical match on large data."""
        case = test_data['large']
        x, h, M = case['x'], case['h'], case['M']
        
        # C++ implementation
        cpp_result = atrousc_cpp_module.atrousc(x, h, M)
        
        # CUDA implementation
        x_cuda = torch.from_numpy(x).cuda()
        h_cuda = torch.from_numpy(h).cuda()
        M_cuda = torch.from_numpy(M)
        cuda_result = atrousc_cuda(x_cuda, h_cuda, M_cuda).cpu().numpy()
        
        # Check exact match
        np.testing.assert_array_almost_equal(
            cpp_result, cuda_result, decimal=10,
            err_msg=f"Mismatch in {case['description']}"
        )
    
    def test_asymmetric_dimensions(self, test_data):
        """Test with asymmetric input and filter dimensions."""
        case = test_data['asymmetric']
        x, h, M = case['x'], case['h'], case['M']
        
        # C++ implementation
        cpp_result = atrousc_cpp_module.atrousc(x, h, M)
        
        # CUDA implementation
        x_cuda = torch.from_numpy(x).cuda()
        h_cuda = torch.from_numpy(h).cuda()
        M_cuda = torch.from_numpy(M)
        cuda_result = atrousc_cuda(x_cuda, h_cuda, M_cuda).cpu().numpy()
        
        # Check exact match
        np.testing.assert_array_almost_equal(
            cpp_result, cuda_result, decimal=10,
            err_msg=f"Mismatch in {case['description']}"
        )
    
    def test_extreme_values(self, test_data):
        """Test with extreme value ranges."""
        case = test_data['extreme_values']
        x, h, M = case['x'], case['h'], case['M']
        
        # C++ implementation
        cpp_result = atrousc_cpp_module.atrousc(x, h, M)
        
        # CUDA implementation
        x_cuda = torch.from_numpy(x).cuda()
        h_cuda = torch.from_numpy(h).cuda()
        M_cuda = torch.from_numpy(M)
        cuda_result = atrousc_cuda(x_cuda, h_cuda, M_cuda).cpu().numpy()
        
        # Check match with appropriate tolerance for extreme values
        np.testing.assert_array_almost_equal(
            cpp_result, cuda_result, decimal=8,
            err_msg=f"Mismatch in {case['description']}"
        )
    
    def test_scalar_upsampling(self, test_data):
        """Test with scalar upsampling matrix."""
        case = test_data['scalar_upsampling']
        x, h, M = case['x'], case['h'], case['M']
        
        # C++ implementation
        cpp_result = atrousc_cpp_module.atrousc(x, h, M)
        
        # CUDA implementation
        x_cuda = torch.from_numpy(x).cuda()
        h_cuda = torch.from_numpy(h).cuda()
        M_cuda = torch.from_numpy(M)
        cuda_result = atrousc_cuda(x_cuda, h_cuda, M_cuda).cpu().numpy()
        
        # Check exact match
        np.testing.assert_array_almost_equal(
            cpp_result, cuda_result, decimal=10,
            err_msg=f"Mismatch in {case['description']}"
        )
    
    def test_point_wise_consistency(self, test_data):
        """Test point-wise consistency across all test cases."""
        for name, case in test_data.items():
            x, h, M = case['x'], case['h'], case['M']
            
            # C++ implementation
            cpp_result = atrousc_cpp_module.atrousc(x, h, M)
            
            # CUDA implementation
            x_cuda = torch.from_numpy(x).cuda()
            h_cuda = torch.from_numpy(h).cuda()
            M_cuda = torch.from_numpy(M)
            cuda_result = atrousc_cuda(x_cuda, h_cuda, M_cuda).cpu().numpy()
            
            # Check that outputs have same shape
            assert cpp_result.shape == cuda_result.shape, \
                f"Shape mismatch in {name}: CPP {cpp_result.shape} vs CUDA {cuda_result.shape}"
            
            # Check point-wise difference
            abs_diff = np.abs(cpp_result - cuda_result)
            max_diff = np.max(abs_diff)
            mean_diff = np.mean(abs_diff)
            
            # Report statistics
            print(f"\n{name} ({case['description']}):")
            print(f"  Input shape: {x.shape}, Filter shape: {h.shape}")
            print(f"  Output shape: {cpp_result.shape}")
            print(f"  Max absolute difference: {max_diff:.2e}")
            print(f"  Mean absolute difference: {mean_diff:.2e}")
            print(f"  Relative error: {max_diff / (np.abs(cpp_result).max() + 1e-10):.2e}")
            
            # Assert tight tolerance for all cases
            assert max_diff < 1e-9, \
                f"Max difference {max_diff} exceeds threshold in {name}"


class TestAtrouscPrecision:
    """Test numerical precision and accuracy of implementations."""
    
    @pytest.fixture(scope="class")
    def precision_data(self):
        """Generate data for precision testing."""
        np.random.seed(123)
        return {
            'x': np.random.randn(100, 100),
            'h': np.random.randn(5, 5),
            'M': np.array([[2, 0], [0, 2]], dtype=np.int32)
        }
    
    def test_floating_point_precision(self, precision_data):
        """Test that implementations maintain floating-point precision."""
        x, h, M = precision_data['x'], precision_data['h'], precision_data['M']
        
        # C++ implementation
        cpp_result = atrousc_cpp_module.atrousc(x, h, M)
        
        # CUDA implementation
        x_cuda = torch.from_numpy(x).cuda()
        h_cuda = torch.from_numpy(h).cuda()
        M_cuda = torch.from_numpy(M)
        cuda_result = atrousc_cuda(x_cuda, h_cuda, M_cuda).cpu().numpy()
        
        # Check relative error
        relative_error = np.abs(cpp_result - cuda_result) / (np.abs(cpp_result) + 1e-10)
        max_relative_error = np.max(relative_error)
        
        print(f"\nFloating-point precision test:")
        print(f"  Max relative error: {max_relative_error:.2e}")
        
        # Assert that relative error is within acceptable bounds
        assert max_relative_error < 1e-8, \
            f"Relative error {max_relative_error} exceeds acceptable threshold"
    
    def test_deterministic_output(self, precision_data):
        """Test that both implementations produce deterministic output."""
        x, h, M = precision_data['x'], precision_data['h'], precision_data['M']
        
        # Run C++ multiple times
        cpp_results = [atrousc_cpp_module.atrousc(x, h, M) for _ in range(3)]
        
        # Check C++ determinism
        for i in range(1, len(cpp_results)):
            np.testing.assert_array_equal(
                cpp_results[0], cpp_results[i],
                err_msg="C++ implementation is not deterministic"
            )
        
        # Run CUDA multiple times
        x_cuda = torch.from_numpy(x).cuda()
        h_cuda = torch.from_numpy(h).cuda()
        M_cuda = torch.from_numpy(M)
        cuda_results = [
            atrousc_cuda(x_cuda, h_cuda, M_cuda).cpu().numpy()
            for _ in range(3)
        ]
        
        # Check CUDA determinism
        for i in range(1, len(cuda_results)):
            np.testing.assert_array_equal(
                cuda_results[0], cuda_results[i],
                err_msg="CUDA implementation is not deterministic"
            )
    
    def test_output_dtype(self, precision_data):
        """Test that both implementations return the same data type."""
        x, h, M = precision_data['x'], precision_data['h'], precision_data['M']
        
        # C++ implementation
        cpp_result = atrousc_cpp_module.atrousc(x, h, M)
        
        # CUDA implementation
        x_cuda = torch.from_numpy(x).cuda()
        h_cuda = torch.from_numpy(h).cuda()
        M_cuda = torch.from_numpy(M)
        cuda_result = atrousc_cuda(x_cuda, h_cuda, M_cuda).cpu().numpy()
        
        # Check dtypes
        assert cpp_result.dtype == cuda_result.dtype, \
            f"Data type mismatch: CPP {cpp_result.dtype} vs CUDA {cuda_result.dtype}"
        
        # Check that dtype is float64
        assert cpp_result.dtype == np.float64, \
            f"Expected float64, got {cpp_result.dtype}"


class TestAtrouscEdgeCases:
    """Test edge cases for C++ and CUDA consistency."""
    
    def test_identity_upsampling(self):
        """Test with identity upsampling (M=1)."""
        np.random.seed(456)
        x = np.random.randn(50, 50)
        h = np.random.randn(3, 3)
        M = np.array([[1, 0], [0, 1]], dtype=np.int32)
        
        # C++ implementation
        cpp_result = atrousc_cpp_module.atrousc(x, h, M)
        
        # CUDA implementation
        x_cuda = torch.from_numpy(x).cuda()
        h_cuda = torch.from_numpy(h).cuda()
        M_cuda = torch.from_numpy(M)
        cuda_result = atrousc_cuda(x_cuda, h_cuda, M_cuda).cpu().numpy()
        
        np.testing.assert_array_almost_equal(
            cpp_result, cuda_result, decimal=10,
            err_msg="Mismatch with identity upsampling"
        )
    
    def test_single_pixel_filter(self):
        """Test with 1x1 filter."""
        np.random.seed(789)
        x = np.random.randn(50, 50)
        h = np.array([[2.5]])
        M = np.array([[2, 0], [0, 2]], dtype=np.int32)
        
        # C++ implementation
        cpp_result = atrousc_cpp_module.atrousc(x, h, M)
        
        # CUDA implementation
        x_cuda = torch.from_numpy(x).cuda()
        h_cuda = torch.from_numpy(h).cuda()
        M_cuda = torch.from_numpy(M)
        cuda_result = atrousc_cuda(x_cuda, h_cuda, M_cuda).cpu().numpy()
        
        np.testing.assert_array_almost_equal(
            cpp_result, cuda_result, decimal=10,
            err_msg="Mismatch with 1x1 filter"
        )
    
    def test_zero_input(self):
        """Test with zero input signal."""
        x = np.zeros((50, 50))
        h = np.random.randn(5, 5)
        M = np.array([[2, 0], [0, 2]], dtype=np.int32)
        
        # C++ implementation
        cpp_result = atrousc_cpp_module.atrousc(x, h, M)
        
        # CUDA implementation
        x_cuda = torch.from_numpy(x).cuda()
        h_cuda = torch.from_numpy(h).cuda()
        M_cuda = torch.from_numpy(M)
        cuda_result = atrousc_cuda(x_cuda, h_cuda, M_cuda).cpu().numpy()
        
        # Both should return zero
        np.testing.assert_array_almost_equal(
            cpp_result, np.zeros_like(cpp_result), decimal=10,
            err_msg="C++ result should be zero for zero input"
        )
        np.testing.assert_array_almost_equal(
            cuda_result, np.zeros_like(cuda_result), decimal=10,
            err_msg="CUDA result should be zero for zero input"
        )
        np.testing.assert_array_almost_equal(
            cpp_result, cuda_result, decimal=10,
            err_msg="Mismatch with zero input"
        )
    
    def test_zero_filter(self):
        """Test with zero filter."""
        x = np.random.randn(50, 50)
        h = np.zeros((5, 5))
        M = np.array([[2, 0], [0, 2]], dtype=np.int32)
        
        # C++ implementation
        cpp_result = atrousc_cpp_module.atrousc(x, h, M)
        
        # CUDA implementation
        x_cuda = torch.from_numpy(x).cuda()
        h_cuda = torch.from_numpy(h).cuda()
        M_cuda = torch.from_numpy(M)
        cuda_result = atrousc_cuda(x_cuda, h_cuda, M_cuda).cpu().numpy()
        
        # Both should return zero
        np.testing.assert_array_almost_equal(
            cpp_result, np.zeros_like(cpp_result), decimal=10,
            err_msg="C++ result should be zero for zero filter"
        )
        np.testing.assert_array_almost_equal(
            cuda_result, np.zeros_like(cuda_result), decimal=10,
            err_msg="CUDA result should be zero for zero filter"
        )
        np.testing.assert_array_almost_equal(
            cpp_result, cuda_result, decimal=10,
            err_msg="Mismatch with zero filter"
        )
    
    def test_various_upsampling_factors(self):
        """Test with various upsampling factors."""
        np.random.seed(101112)
        x = np.random.randn(80, 80)
        h = np.random.randn(5, 5)
        
        upsampling_factors = [
            (np.array([[1, 0], [0, 1]], dtype=np.int32), "Identity (1x1)"),
            (np.array([[2, 0], [0, 2]], dtype=np.int32), "Diagonal 2x2"),
            (np.array([[3, 0], [0, 3]], dtype=np.int32), "Diagonal 3x3"),
            (np.array([[4, 0], [0, 4]], dtype=np.int32), "Diagonal 4x4"),
            (np.array([[2, 0], [0, 3]], dtype=np.int32), "Mixed 2x3"),
            (np.array([[3, 0], [0, 2]], dtype=np.int32), "Mixed 3x2"),
        ]
        
        for M, desc in upsampling_factors:
            # C++ implementation
            cpp_result = atrousc_cpp_module.atrousc(x, h, M)
            
            # CUDA implementation
            x_cuda = torch.from_numpy(x).cuda()
            h_cuda = torch.from_numpy(h).cuda()
            M_cuda = torch.from_numpy(M)
            cuda_result = atrousc_cuda(x_cuda, h_cuda, M_cuda).cpu().numpy()
            
            # Check exact match for all cases
            np.testing.assert_array_almost_equal(
                cpp_result, cuda_result, decimal=10,
                err_msg=f"Mismatch with {desc} upsampling"
            )
            
            print(f"✓ {desc} upsampling: shapes match {cpp_result.shape}")
    
    def test_minimum_output_size(self):
        """Test with minimum valid output size."""
        np.random.seed(131415)
        # Minimum size that produces 1x1 output
        M = np.array([[2, 0], [0, 2]], dtype=np.int32)
        h_shape = (3, 3)
        x_shape = (2 * h_shape[0], 2 * h_shape[1])  # Minimum size for valid output
        
        x = np.random.randn(*x_shape)
        h = np.random.randn(*h_shape)
        
        # C++ implementation
        cpp_result = atrousc_cpp_module.atrousc(x, h, M)
        
        # CUDA implementation
        x_cuda = torch.from_numpy(x).cuda()
        h_cuda = torch.from_numpy(h).cuda()
        M_cuda = torch.from_numpy(M)
        cuda_result = atrousc_cuda(x_cuda, h_cuda, M_cuda).cpu().numpy()
        
        # Should have small output
        assert cpp_result.size > 0, "C++ result is empty"
        assert cuda_result.size > 0, "CUDA result is empty"
        
        np.testing.assert_array_almost_equal(
            cpp_result, cuda_result, decimal=10,
            err_msg="Mismatch with minimum output size"
        )
    
    def test_non_square_filters(self):
        """Test with non-square filters."""
        np.random.seed(161718)
        x = np.random.randn(80, 80)
        M = np.array([[2, 0], [0, 2]], dtype=np.int32)
        
        filter_shapes = [
            (3, 5),
            (5, 3),
            (7, 3),
            (3, 7),
        ]
        
        for h_shape in filter_shapes:
            h = np.random.randn(*h_shape)
            
            # C++ implementation
            cpp_result = atrousc_cpp_module.atrousc(x, h, M)
            
            # CUDA implementation
            x_cuda = torch.from_numpy(x).cuda()
            h_cuda = torch.from_numpy(h).cuda()
            M_cuda = torch.from_numpy(M)
            cuda_result = atrousc_cuda(x_cuda, h_cuda, M_cuda).cpu().numpy()
            
            np.testing.assert_array_almost_equal(
                cpp_result, cuda_result, decimal=10,
                err_msg=f"Mismatch with {h_shape} filter"
            )
            
            print(f"✓ Filter shape {h_shape}: output shape {cpp_result.shape}")


class TestAtrouscStatistics:
    """Statistical tests for implementation consistency."""
    
    def test_statistical_properties(self):
        """Test that statistical properties are preserved across implementations."""
        np.random.seed(202122)
        x = np.random.randn(150, 150)
        h = np.random.randn(7, 7)
        # Normalize filter
        h = h / np.sum(np.abs(h))
        M = np.array([[2, 0], [0, 2]], dtype=np.int32)
        
        # C++ implementation
        cpp_result = atrousc_cpp_module.atrousc(x, h, M)
        
        # CUDA implementation
        x_cuda = torch.from_numpy(x).cuda()
        h_cuda = torch.from_numpy(h).cuda()
        M_cuda = torch.from_numpy(M)
        cuda_result = atrousc_cuda(x_cuda, h_cuda, M_cuda).cpu().numpy()
        
        # Compare statistical properties
        cpp_mean = np.mean(cpp_result)
        cuda_mean = np.mean(cuda_result)
        cpp_std = np.std(cpp_result)
        cuda_std = np.std(cuda_result)
        cpp_min = np.min(cpp_result)
        cuda_min = np.min(cuda_result)
        cpp_max = np.max(cpp_result)
        cuda_max = np.max(cuda_result)
        
        print(f"\nStatistical properties comparison:")
        print(f"  Mean: CPP={cpp_mean:.6f}, CUDA={cuda_mean:.6f}, diff={abs(cpp_mean-cuda_mean):.2e}")
        print(f"  Std:  CPP={cpp_std:.6f}, CUDA={cuda_std:.6f}, diff={abs(cpp_std-cuda_std):.2e}")
        print(f"  Min:  CPP={cpp_min:.6f}, CUDA={cuda_min:.6f}, diff={abs(cpp_min-cuda_min):.2e}")
        print(f"  Max:  CPP={cpp_max:.6f}, CUDA={cuda_max:.6f}, diff={abs(cpp_max-cuda_max):.2e}")
        
        # Assert statistical similarity
        np.testing.assert_almost_equal(cpp_mean, cuda_mean, decimal=8)
        np.testing.assert_almost_equal(cpp_std, cuda_std, decimal=8)
        np.testing.assert_almost_equal(cpp_min, cuda_min, decimal=8)
        np.testing.assert_almost_equal(cpp_max, cuda_max, decimal=8)
    
    def test_energy_preservation(self):
        """Test energy preservation (sum of squared values)."""
        np.random.seed(232425)
        x = np.random.randn(100, 100)
        h = np.random.randn(5, 5)
        M = np.array([[2, 0], [0, 2]], dtype=np.int32)
        
        # C++ implementation
        cpp_result = atrousc_cpp_module.atrousc(x, h, M)
        
        # CUDA implementation
        x_cuda = torch.from_numpy(x).cuda()
        h_cuda = torch.from_numpy(h).cuda()
        M_cuda = torch.from_numpy(M)
        cuda_result = atrousc_cuda(x_cuda, h_cuda, M_cuda).cpu().numpy()
        
        # Calculate energy
        cpp_energy = np.sum(cpp_result ** 2)
        cuda_energy = np.sum(cuda_result ** 2)
        
        print(f"\nEnergy comparison:")
        print(f"  CPP energy: {cpp_energy:.6f}")
        print(f"  CUDA energy: {cuda_energy:.6f}")
        print(f"  Relative difference: {abs(cpp_energy - cuda_energy) / (cpp_energy + 1e-10):.2e}")
        
        # Energy should be nearly identical
        np.testing.assert_almost_equal(cpp_energy, cuda_energy, decimal=6)


class TestAtrouscPerformance:
    """Performance comparison tests."""
    
    def test_performance_comparison(self):
        """Compare performance between C++ and CUDA implementations."""
        import time
        
        np.random.seed(262728)
        sizes = [(100, 100), (200, 200), (400, 400)]
        h = np.random.randn(7, 7)
        M = np.array([[2, 0], [0, 2]], dtype=np.int32)
        
        print("\nPerformance comparison:")
        print("-" * 60)
        
        for size in sizes:
            x = np.random.randn(*size)
            
            # Warm up and test C++
            _ = atrousc_cpp_module.atrousc(x, h, M)
            start = time.perf_counter()
            cpp_result = atrousc_cpp_module.atrousc(x, h, M)
            cpp_time = (time.perf_counter() - start) * 1000
            
            # Warm up and test CUDA
            x_cuda = torch.from_numpy(x).cuda()
            h_cuda = torch.from_numpy(h).cuda()
            M_cuda = torch.from_numpy(M)
            _ = atrousc_cuda(x_cuda, h_cuda, M_cuda)
            torch.cuda.synchronize()
            
            start = time.perf_counter()
            cuda_result = atrousc_cuda(x_cuda, h_cuda, M_cuda)
            torch.cuda.synchronize()
            cuda_time = (time.perf_counter() - start) * 1000
            
            speedup = cpp_time / cuda_time
            
            print(f"Size {size}:")
            print(f"  C++:  {cpp_time:>8.2f} ms")
            print(f"  CUDA: {cuda_time:>8.2f} ms")
            print(f"  Speedup: {speedup:>6.2f}×")
            
            # Verify results still match
            cuda_result_np = cuda_result.cpu().numpy()
            max_diff = np.max(np.abs(cpp_result - cuda_result_np))
            print(f"  Max diff: {max_diff:.2e}")
            assert max_diff < 1e-9, f"Results differ after performance test"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
