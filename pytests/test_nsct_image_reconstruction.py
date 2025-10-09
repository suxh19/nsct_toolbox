"""
Integration-style test that runs nsctdec/nsctrec on the project test image.
Ensures that decomposition subbands preserve shape and that reconstruction
matches the original input on an element-wise basis.

Comprehensive test coverage includes:
- Shape consistency tests (input/output dimensions)
- Numerical accuracy tests (element-wise comparison)
- Structure validation (subband counts and hierarchy)
- One-to-one reconstruction tests
- Edge case handling
"""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PIL.Image")
from PIL import Image

from nsct_python.core import nsctdec, nsctrec


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_IMAGE_PATH = PROJECT_ROOT / "test_image.jpg"


class TestNSCTImageReconstruction:
    """Comprehensive test suite for NSCT decomposition and reconstruction."""

    @pytest.fixture(scope="class")
    def test_image(self):
        """Load and prepare test image for all tests."""
        assert TEST_IMAGE_PATH.exists(), f"Missing test image at {TEST_IMAGE_PATH}"
        with Image.open(TEST_IMAGE_PATH) as img:
            if img.mode == "RGB":
                img = img.convert("L")
            return np.asarray(img, dtype=np.float64)

    @pytest.fixture(scope="class")
    def default_params(self):
        """Default NSCT parameters."""
        return {
            "levels": [2, 3],
            "dfilt": "dmaxflat7",
            "pfilt": "maxflat"
        }

    # === Shape Consistency Tests ===

    def test_decomposition_output_structure(self, test_image, default_params):
        """Test that decomposition output has correct structure."""
        decomposition = nsctdec(test_image, **default_params)
        levels = default_params["levels"]
        
        # Check overall structure: lowpass + one entry per pyramid level
        expected_length = len(levels) + 1
        assert len(decomposition) == expected_length, \
            f"Expected {expected_length} subbands, got {len(decomposition)}"
        
        # First element should be lowpass (numpy array)
        assert isinstance(decomposition[0], np.ndarray), \
            "First element should be lowpass band (numpy array)"
        
        # Remaining elements should be lists of directional subbands
        for idx in range(1, len(decomposition)):
            assert isinstance(decomposition[idx], list), \
                f"Element {idx} should be list of directional subbands"

    def test_subband_shape_preservation(self, test_image, default_params):
        """Test that all subbands preserve the input spatial shape."""
        decomposition = nsctdec(test_image, **default_params)
        
        # Check lowpass band
        assert decomposition[0].shape == test_image.shape, \
            f"Lowpass band shape {decomposition[0].shape} != input {test_image.shape}"
        
        # Check all directional subbands
        for level_idx, subband_list in enumerate(decomposition[1:], start=1):
            assert isinstance(subband_list, list), \
                f"Level {level_idx} should be a list"
            assert len(subband_list) > 0, \
                f"Level {level_idx} has empty directional subband list"
            
            for dir_idx, band in enumerate(subband_list):
                assert band.shape == test_image.shape, \
                    f"Level {level_idx}, direction {dir_idx}: " \
                    f"shape {band.shape} != input {test_image.shape}"

    def test_directional_subband_counts(self, test_image, default_params):
        """Test that directional subband counts match expected values."""
        decomposition = nsctdec(test_image, **default_params)
        levels = default_params["levels"]
        
        for level_idx, num_directions in enumerate(levels):
            # Number of directional subbands = 2^n where n is the decomposition level
            expected_count = 2 ** num_directions
            actual_count = len(decomposition[level_idx + 1])
            assert actual_count == expected_count, \
                f"Level {level_idx}: expected {expected_count} subbands, " \
                f"got {actual_count}"

    def test_reconstruction_shape_preservation(self, test_image, default_params):
        """Test that reconstruction preserves input shape."""
        decomposition = nsctdec(test_image, **default_params)
        reconstructed = nsctrec(decomposition, default_params["dfilt"], 
                               default_params["pfilt"])
        
        assert reconstructed.shape == test_image.shape, \
            f"Reconstructed shape {reconstructed.shape} != input {test_image.shape}"

    # === Numerical Accuracy Tests ===

    @pytest.mark.slow
    def test_perfect_reconstruction_numerical_accuracy(self, test_image, default_params):
        """Test numerical accuracy of perfect reconstruction."""
        decomposition = nsctdec(test_image, **default_params)
        reconstructed = nsctrec(decomposition, default_params["dfilt"], 
                               default_params["pfilt"])
        
        # Calculate error metrics
        diff = reconstructed - test_image
        abs_diff = np.abs(diff)
        max_error = float(np.max(abs_diff))
        mean_error = float(np.mean(abs_diff))
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        
        # Test with strict tolerances
        np.testing.assert_allclose(
            reconstructed,
            test_image,
            rtol=1e-12,
            atol=1e-10,
            err_msg=f"Reconstruction failed: max_error={max_error:g}, "
                    f"mean_error={mean_error:g}, rmse={rmse:g}"
        )
        
        # Additional assertions for error metrics
        assert max_error < 1e-9, f"Maximum reconstruction error {max_error:g} too large"
        assert rmse < 1e-10, f"RMSE {rmse:g} exceeds threshold"

    def test_element_wise_reconstruction_accuracy(self, test_image, default_params):
        """Test element-wise accuracy of reconstruction."""
        decomposition = nsctdec(test_image, **default_params)
        reconstructed = nsctrec(decomposition, default_params["dfilt"], 
                               default_params["pfilt"])
        
        # Check that no element differs by more than tolerance
        abs_diff = np.abs(reconstructed - test_image)
        exceeding_elements = np.sum(abs_diff > 1e-9)
        
        assert exceeding_elements == 0, \
            f"{exceeding_elements} elements exceed tolerance"

    def test_subband_dtype_consistency(self, test_image, default_params):
        """Test that all subbands maintain consistent dtype."""
        decomposition = nsctdec(test_image, **default_params)
        
        # Check lowpass
        assert decomposition[0].dtype == test_image.dtype, \
            f"Lowpass dtype {decomposition[0].dtype} != input {test_image.dtype}"
        
        # Check all directional subbands
        for level_idx, subband_list in enumerate(decomposition[1:], start=1):
            for dir_idx, band in enumerate(subband_list):
                assert band.dtype == test_image.dtype, \
                    f"Level {level_idx}, direction {dir_idx}: " \
                    f"dtype {band.dtype} != input {test_image.dtype}"

    # === One-to-One Consistency Tests ===

    def test_decomposition_determinism(self, test_image, default_params):
        """Test that decomposition is deterministic (same input -> same output)."""
        decomp1 = nsctdec(test_image, **default_params)
        decomp2 = nsctdec(test_image, **default_params)
        
        # Check lowpass
        np.testing.assert_array_equal(decomp1[0], decomp2[0],
                                     err_msg="Lowpass bands differ")
        
        # Check all directional subbands
        for level_idx in range(1, len(decomp1)):
            assert len(decomp1[level_idx]) == len(decomp2[level_idx])
            for dir_idx in range(len(decomp1[level_idx])):
                np.testing.assert_array_equal(
                    decomp1[level_idx][dir_idx],
                    decomp2[level_idx][dir_idx],
                    err_msg=f"Level {level_idx}, direction {dir_idx} differs"
                )

    def test_reconstruction_determinism(self, test_image, default_params):
        """Test that reconstruction is deterministic."""
        decomposition = nsctdec(test_image, **default_params)
        recon1 = nsctrec(decomposition, default_params["dfilt"], 
                        default_params["pfilt"])
        recon2 = nsctrec(decomposition, default_params["dfilt"], 
                        default_params["pfilt"])
        
        np.testing.assert_array_equal(recon1, recon2,
                                     err_msg="Reconstruction not deterministic")

    def test_invertibility(self, test_image, default_params):
        """Test that decomposition followed by reconstruction is invertible."""
        # First round: decompose and reconstruct
        decomp1 = nsctdec(test_image, **default_params)
        recon1 = nsctrec(decomp1, default_params["dfilt"], default_params["pfilt"])
        
        # Second round: decompose and reconstruct again
        decomp2 = nsctdec(recon1, **default_params)
        recon2 = nsctrec(decomp2, default_params["dfilt"], default_params["pfilt"])
        
        # Both reconstructions should be close to original
        np.testing.assert_allclose(recon1, test_image, rtol=1e-12, atol=1e-10)
        np.testing.assert_allclose(recon2, test_image, rtol=1e-12, atol=1e-10)
        np.testing.assert_allclose(recon2, recon1, rtol=1e-12, atol=1e-10)

    # === Different Parameter Tests ===

    @pytest.mark.parametrize("levels", [
        [2],
        [2, 3],
        [3, 3],
        [2, 3, 4],
    ])
    def test_different_pyramid_levels(self, test_image, levels):
        """Test reconstruction with different pyramid level configurations."""
        decomposition = nsctdec(test_image, levels, "dmaxflat7", "maxflat")
        reconstructed = nsctrec(decomposition, "dmaxflat7", "maxflat")
        
        assert reconstructed.shape == test_image.shape
        np.testing.assert_allclose(reconstructed, test_image, 
                                   rtol=1e-12, atol=1e-10)

    @pytest.mark.parametrize("dfilt,pfilt", [
        ("pkva", "maxflat"),
        ("dmaxflat7", "maxflat"),
        ("dmaxflat5", "maxflat"),
    ])
    def test_different_filter_types(self, test_image, dfilt, pfilt):
        """Test reconstruction with different filter combinations."""
        levels = [2, 3]
        decomposition = nsctdec(test_image, levels, dfilt, pfilt)
        reconstructed = nsctrec(decomposition, dfilt, pfilt)
        
        assert reconstructed.shape == test_image.shape
        np.testing.assert_allclose(reconstructed, test_image, 
                                   rtol=1e-10, atol=1e-9)

    # === Energy Conservation Tests ===

    def test_energy_conservation(self, test_image, default_params):
        """Test that total energy is conserved through decomposition."""
        decomposition = nsctdec(test_image, **default_params)
        
        # Calculate energy in original
        original_energy = np.sum(test_image ** 2)
        
        # Calculate energy in all subbands
        decomp_energy = np.sum(decomposition[0] ** 2)
        for subband_list in decomposition[1:]:
            for band in subband_list:
                decomp_energy += np.sum(band ** 2)
        
        # Energy should be approximately conserved
        energy_ratio = decomp_energy / original_energy
        assert 0.95 < energy_ratio < 1.05, \
            f"Energy not conserved: ratio = {energy_ratio:.4f}"

    # === Edge Case Tests ===

    def test_small_image_reconstruction(self):
        """Test reconstruction on small images."""
        small_image = np.random.rand(32, 32)
        levels = [2]
        
        decomposition = nsctdec(small_image, levels, "dmaxflat7", "maxflat")
        reconstructed = nsctrec(decomposition, "dmaxflat7", "maxflat")
        
        assert reconstructed.shape == small_image.shape
        np.testing.assert_allclose(reconstructed, small_image, 
                                   rtol=1e-12, atol=1e-10)

    def test_constant_image_reconstruction(self):
        """Test reconstruction of constant image."""
        constant_image = np.ones((64, 64)) * 128.0
        levels = [2, 3]
        
        decomposition = nsctdec(constant_image, levels, "dmaxflat7", "maxflat")
        reconstructed = nsctrec(decomposition, "dmaxflat7", "maxflat")
        
        np.testing.assert_allclose(reconstructed, constant_image, 
                                   rtol=1e-12, atol=1e-10)

    def test_zero_image_reconstruction(self):
        """Test reconstruction of zero image."""
        zero_image = np.zeros((64, 64))
        levels = [2, 3]
        
        decomposition = nsctdec(zero_image, levels, "dmaxflat7", "maxflat")
        reconstructed = nsctrec(decomposition, "dmaxflat7", "maxflat")
        
        np.testing.assert_allclose(reconstructed, zero_image, atol=1e-14)

    # === Legacy Test (kept for backward compatibility) ===

    @pytest.mark.slow
    def test_nsct_decomposition_reconstruction_on_sample_image(
        self, test_image, default_params
    ):
        """Original integration test - kept for backward compatibility."""
        decomposition = nsctdec(test_image, **default_params)
        levels = default_params["levels"]

        # NSCT output should contain one lowpass band plus one entry per pyramid level
        expected_length = len(levels) + 1
        assert (
            len(decomposition) == expected_length
        ), f"Expected {expected_length} subbands, got {len(decomposition)}"

        # Every subband (lowpass or directional) should preserve the input spatial shape
        for index, subband in enumerate(decomposition):
            if isinstance(subband, list):
                assert subband, f"Empty directional subband list at index {index}"
                for direction, band in enumerate(subband):
                    assert (
                        band.shape == test_image.shape
                    ), f"Directional band {direction} at level {index} has shape {band.shape}"
            else:
                assert (
                    subband.shape == test_image.shape
                ), f"Subband {index} has unexpected shape {subband.shape}"

        reconstructed = nsctrec(decomposition, default_params["dfilt"], 
                               default_params["pfilt"])

        assert (
            reconstructed.shape == test_image.shape
        ), "Reconstructed image shape does not match the original"

        diff = np.abs(reconstructed - test_image)
        max_error = float(np.max(diff))

        np.testing.assert_allclose(
            reconstructed,
            test_image,
            rtol=1e-12,
            atol=1e-10,
            err_msg=f"Reconstruction max error {max_error:g} exceeds tolerance",
        )
