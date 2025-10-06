import numpy as np
import torch
import pytest
import sys
import os
from pathlib import Path

# Add the parent directory to sys.path to import nsct_torch and nsct_python
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))
# --- Import original NumPy functions ---
from nsct_python import utils as utils_np
from nsct_python import filters as filters_np
from nsct_python import core as core_np

# --- Import new PyTorch functions ---
from nsct_torch import utils as utils_torch
from nsct_torch import filters as filters_torch
from nsct_torch import core as core_torch

# --- Test Fixtures ---
@pytest.fixture
def sample_image_np():
    # Using an even-sized image as required by some filters for perfect reconstruction
    np.random.seed(42)
    return np.random.rand(32, 32)

@pytest.fixture
def sample_image_torch(sample_image_np):
    return torch.from_numpy(sample_image_np)

@pytest.fixture
def sample_filter_np():
    np.random.seed(42)
    return np.random.rand(5, 7)

@pytest.fixture
def sample_filter_torch(sample_filter_np):
    return torch.from_numpy(sample_filter_np)

@pytest.fixture
def small_image_np():
    """Small test image for edge cases"""
    np.random.seed(42)
    return np.random.rand(8, 8)

@pytest.fixture
def small_image_torch(small_image_np):
    return torch.from_numpy(small_image_np)

@pytest.fixture
def rectangular_image_np():
    """Rectangular image to test non-square inputs (足够大以避免padding限制)"""
    np.random.seed(42)
    return np.random.rand(64, 96)  # 增大尺寸以避免padding限制

@pytest.fixture
def rectangular_image_torch(rectangular_image_np):
    return torch.from_numpy(rectangular_image_np)

# --- Helper for comparison ---
def assert_tensors_close(t1, t2, atol=1e-7, rtol=1e-5):
    """Asserts that two tensors are close, handling both torch and numpy arrays."""
    if isinstance(t1, torch.Tensor):
        t1 = t1.detach().cpu().numpy()
    if isinstance(t2, torch.Tensor):
        t2 = t2.detach().cpu().numpy()
    
    if not np.allclose(t1, t2, atol=atol, rtol=rtol):
        diff = np.abs(t1 - t2)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        raise AssertionError(
            f"Tensors not close.\n"
            f"Max difference: {max_diff}\n"
            f"Mean difference: {mean_diff}\n"
            f"Shape: {t1.shape}\n"
            f"Numpy sample:\n{t1[:3, :3] if t1.ndim >= 2 else t1[:5]}\n"
            f"Torch sample:\n{t2[:3, :3] if t2.ndim >= 2 else t2[:5]}"
        )

# ====================================================================================
# UTILS FUNCTIONS TESTS - 确保所有工具函数输出一致
# ====================================================================================

class TestExtend2:
    """测试 extend2 函数 - 图像边界扩展"""
    
    def test_all_extension_modes(self, sample_image_np, sample_image_torch):
        """测试所有扩展模式"""
        args = (4, 5, 6, 7)
        modes = ['per', 'qper_row', 'qper_col']
        for mode in modes:
            np_res = utils_np.extend2(sample_image_np, *args, extmod=mode)
            torch_res = utils_torch.extend2(sample_image_torch, *args, extmod=mode)
            assert_tensors_close(np_res, torch_res, atol=1e-10)
    
    def test_symmetric_extension(self, sample_image_np, sample_image_torch):
        """测试对称扩展"""
        np_res = utils_np.extend2(sample_image_np, 3, 3, 3, 3, extmod='per')
        torch_res = utils_torch.extend2(sample_image_torch, 3, 3, 3, 3, extmod='per')
        assert_tensors_close(np_res, torch_res, atol=1e-10)
    
    def test_asymmetric_extension(self, sample_image_np, sample_image_torch):
        """测试非对称扩展"""
        np_res = utils_np.extend2(sample_image_np, 2, 8, 3, 5, extmod='per')
        torch_res = utils_torch.extend2(sample_image_torch, 2, 8, 3, 5, extmod='per')
        assert_tensors_close(np_res, torch_res, atol=1e-10)
    
    def test_zero_extension(self, sample_image_np, sample_image_torch):
        """测试零扩展情况"""
        np_res = utils_np.extend2(sample_image_np, 0, 0, 0, 0, extmod='per')
        torch_res = utils_torch.extend2(sample_image_torch, 0, 0, 0, 0, extmod='per')
        assert_tensors_close(np_res, torch_res, atol=1e-10)
    
    def test_rectangular_image(self, rectangular_image_np, rectangular_image_torch):
        """测试矩形图像"""
        np_res = utils_np.extend2(rectangular_image_np, 4, 5, 6, 7, extmod='per')
        torch_res = utils_torch.extend2(rectangular_image_torch, 4, 5, 6, 7, extmod='per')
        assert_tensors_close(np_res, torch_res, atol=1e-10)


class TestUpsample2df:
    """测试 upsample2df 函数 - 2D滤波器上采样"""
    
    def test_power_1(self, sample_filter_np, sample_filter_torch):
        """测试 power=1"""
        np_res = utils_np.upsample2df(sample_filter_np, power=1)
        torch_res = utils_torch.upsample2df(sample_filter_torch, power=1)
        assert_tensors_close(np_res, torch_res, atol=1e-10)
    
    def test_power_2(self, sample_filter_np, sample_filter_torch):
        """测试 power=2"""
        np_res = utils_np.upsample2df(sample_filter_np, power=2)
        torch_res = utils_torch.upsample2df(sample_filter_torch, power=2)
        assert_tensors_close(np_res, torch_res, atol=1e-10)
    
    def test_power_3(self, sample_filter_np, sample_filter_torch):
        """测试 power=3"""
        np_res = utils_np.upsample2df(sample_filter_np, power=3)
        torch_res = utils_torch.upsample2df(sample_filter_torch, power=3)
        assert_tensors_close(np_res, torch_res, atol=1e-10)
    
    def test_small_filter(self):
        """测试小滤波器"""
        np_f = np.array([[1, 2], [3, 4]], dtype=float)
        torch_f = torch.from_numpy(np_f)
        np_res = utils_np.upsample2df(np_f, power=2)
        torch_res = utils_torch.upsample2df(torch_f, power=2)
        assert_tensors_close(np_res, torch_res, atol=1e-10)


class TestModulate2:
    """测试 modulate2 函数 - 2D调制"""
    
    def test_all_modes(self, sample_image_np, sample_image_torch):
        """测试所有模式"""
        modes = ['r', 'c', 'b']
        for mode in modes:
            np_res = utils_np.modulate2(sample_image_np, mode=mode)
            torch_res = utils_torch.modulate2(sample_image_torch, mode=mode)
            assert_tensors_close(np_res, torch_res, atol=1e-10)
    
    def test_with_custom_center(self, sample_image_np, sample_image_torch):
        """测试自定义中心点"""
        center = [10, 15]
        for mode in ['r', 'c', 'b']:
            np_res = utils_np.modulate2(sample_image_np, mode=mode, center=center)
            torch_res = utils_torch.modulate2(sample_image_torch, mode=mode, center=(center[0], center[1]))
            assert_tensors_close(np_res, torch_res, atol=1e-10)
    
    def test_small_image(self, small_image_np, small_image_torch):
        """测试小图像"""
        for mode in ['r', 'c', 'b']:
            np_res = utils_np.modulate2(small_image_np, mode=mode)
            torch_res = utils_torch.modulate2(small_image_torch, mode=mode)
            assert_tensors_close(np_res, torch_res, atol=1e-10)


class TestResampz:
    """测试 resampz 函数 - 重采样"""
    
    def test_all_types(self, sample_filter_np, sample_filter_torch):
        """测试所有重采样类型"""
        for type_val in range(1, 5):
            np_res = utils_np.resampz(sample_filter_np, type=type_val)
            torch_res = utils_torch.resampz(sample_filter_torch, type=type_val)
            assert_tensors_close(np_res, torch_res, atol=1e-10)
    
    def test_with_shift(self, sample_filter_np, sample_filter_torch):
        """测试带偏移的重采样"""
        for type_val in range(1, 5):
            for shift in [0, 1, 2]:
                np_res = utils_np.resampz(sample_filter_np, type=type_val, shift=shift)
                torch_res = utils_torch.resampz(sample_filter_torch, type=type_val, shift=shift)
                assert_tensors_close(np_res, torch_res, atol=1e-10)
    
    def test_small_filter(self):
        """测试小滤波器"""
        np_f = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        torch_f = torch.from_numpy(np_f)
        for type_val in range(1, 5):
            np_res = utils_np.resampz(np_f, type=type_val)
            torch_res = utils_torch.resampz(torch_f, type=type_val)
            assert_tensors_close(np_res, torch_res, atol=1e-10)


class TestQupz:
    """测试 qupz 函数 - Quincunx上采样"""
    
    def test_type_1(self, sample_filter_np, sample_filter_torch):
        """测试 type=1"""
        np_res = utils_np.qupz(sample_filter_np, type=1)
        torch_res = utils_torch.qupz(sample_filter_torch, type=1)
        assert_tensors_close(np_res, torch_res, atol=1e-10)
    
    def test_type_2(self, sample_filter_np, sample_filter_torch):
        """测试 type=2"""
        np_res = utils_np.qupz(sample_filter_np, type=2)
        torch_res = utils_torch.qupz(sample_filter_torch, type=2)
        assert_tensors_close(np_res, torch_res, atol=1e-10)
    
    def test_small_filter(self):
        """测试小滤波器"""
        np_f = np.array([[1, 2], [3, 4]], dtype=float)
        torch_f = torch.from_numpy(np_f)
        for type_val in [1, 2]:
            np_res = utils_np.qupz(np_f, type=type_val)
            torch_res = utils_torch.qupz(torch_f, type=type_val)
            assert_tensors_close(np_res, torch_res, atol=1e-10)

# ====================================================================================
# FILTERS FUNCTIONS TESTS - 确保所有滤波器函数输出一致
# ====================================================================================

class TestLdfilter:
    """测试 ldfilter 函数 - Ladder滤波器"""
    
    def test_all_filter_names(self):
        """测试所有预定义滤波器"""
        names = ['pkva6', 'pkva8', 'pkva', 'pkva12']
        for name in names:
            np_res = filters_np.ldfilter(name)
            torch_res = filters_torch.ldfilter(name)
            assert_tensors_close(np_res, torch_res, atol=1e-10)
    
    def test_output_shape(self):
        """验证输出形状正确"""
        np_res = filters_np.ldfilter('pkva6')
        torch_res = filters_torch.ldfilter('pkva6')
        assert np_res.shape == torch_res.shape


class TestDmaxflat:
    """测试 dmaxflat 函数 - 最大平坦滤波器"""
    
    def test_different_n_values(self):
        """测试不同的N值（只测试已实现的N=1,2,3）"""
        for n in range(1, 4):  # 只测试实现了的N值
            np_res = filters_np.dmaxflat(n, d=0.5)
            torch_res = filters_torch.dmaxflat(n, d=0.5)
            assert_tensors_close(np_res, torch_res, atol=1e-10)
    
    def test_different_d_values(self):
        """测试不同的d值"""
        for d in [0.0, 0.25, 0.5, 0.75]:
            np_res = filters_np.dmaxflat(2, d=d)
            torch_res = filters_torch.dmaxflat(2, d=d)
            assert_tensors_close(np_res, torch_res, atol=1e-10)
    
    def test_edge_cases(self):
        """测试边界情况"""
        np_res = filters_np.dmaxflat(1, d=0.0)
        torch_res = filters_torch.dmaxflat(1, d=0.0)
        assert_tensors_close(np_res, torch_res, atol=1e-10)


class TestAtrousfilters:
    """测试 atrousfilters 函数 - Atrous滤波器"""
    
    def test_pyr_filter(self):
        """测试 pyr 滤波器"""
        np_res = filters_np.atrousfilters('pyr')
        torch_res = filters_torch.atrousfilters('pyr')
        for i in range(4):
            assert_tensors_close(np_res[i], torch_res[i], atol=1e-10)
    
    def test_pyrexc_filter(self):
        """测试 pyrexc 滤波器"""
        np_res = filters_np.atrousfilters('pyrexc')
        torch_res = filters_torch.atrousfilters('pyrexc')
        for i in range(4):
            assert_tensors_close(np_res[i], torch_res[i], atol=1e-10)
    
    def test_output_count(self):
        """验证输出4个滤波器"""
        np_res = filters_np.atrousfilters('pyr')
        torch_res = filters_torch.atrousfilters('pyr')
        assert len(np_res) == 4
        assert len(torch_res) == 4


class TestMctrans:
    """测试 mctrans 函数 - McClellan变换"""
    
    def test_with_ldfilter_and_dmaxflat(self):
        """使用ldfilter和dmaxflat测试"""
        b_np = filters_np.ldfilter('pkva6')
        t_np = filters_np.dmaxflat(2, 0)
        b_torch = torch.from_numpy(b_np)
        t_torch = torch.from_numpy(t_np)
        
        np_res = filters_np.mctrans(b_np, t_np)
        torch_res = filters_torch.mctrans(b_torch, t_torch)
        assert_tensors_close(np_res, torch_res, atol=1e-10)
    
    def test_with_different_filters(self):
        """使用不同滤波器测试"""
        b_np = filters_np.ldfilter('pkva8')
        t_np = filters_np.dmaxflat(3, 0.5)
        b_torch = torch.from_numpy(b_np)
        t_torch = torch.from_numpy(t_np)
        
        np_res = filters_np.mctrans(b_np, t_np)
        torch_res = filters_torch.mctrans(b_torch, t_torch)
        assert_tensors_close(np_res, torch_res, atol=1e-10)
    
    def test_small_filters(self):
        """测试小滤波器"""
        b_np = np.array([1, 2, 1], dtype=float) / 4
        t_np = filters_np.dmaxflat(1, 0)
        b_torch = torch.from_numpy(b_np)
        t_torch = torch.from_numpy(t_np)
        
        np_res = filters_np.mctrans(b_np, t_np)
        torch_res = filters_torch.mctrans(b_torch, t_torch)
        assert_tensors_close(np_res, torch_res, atol=1e-10)


class TestDfilters:
    """测试 dfilters 函数 - 方向滤波器"""
    
    def test_all_filter_types(self):
        """测试所有滤波器类型（只测试已实现的）"""
        names = ['pkva', 'db2', 'dmaxflat3']  # dmaxflat4和dmaxflat5未实现
        types = ['d', 'r']
        for name in names:
            for ftype in types:
                np_h0, np_h1 = filters_np.dfilters(name, ftype)
                torch_h0, torch_h1 = filters_torch.dfilters(name, ftype)
                assert_tensors_close(np_h0, torch_h0, atol=1e-10)
                assert_tensors_close(np_h1, torch_h1, atol=1e-10)
    
    def test_decomposition_filters(self):
        """测试分解滤波器"""
        np_h0, np_h1 = filters_np.dfilters('pkva', 'd')
        torch_h0, torch_h1 = filters_torch.dfilters('pkva', 'd')
        assert_tensors_close(np_h0, torch_h0, atol=1e-10)
        assert_tensors_close(np_h1, torch_h1, atol=1e-10)
    
    def test_reconstruction_filters(self):
        """测试重建滤波器"""
        np_g0, np_g1 = filters_np.dfilters('pkva', 'r')
        torch_g0, torch_g1 = filters_torch.dfilters('pkva', 'r')
        assert_tensors_close(np_g0, torch_g0, atol=1e-10)
        assert_tensors_close(np_g1, torch_g1, atol=1e-10)


class TestLd2quin:
    """测试 ld2quin 函数 - Ladder到Quincunx转换"""
    
    def test_with_different_ldfilters(self):
        """使用不同的ldfilter测试"""
        names = ['pkva6', 'pkva8', 'pkva']
        for name in names:
            beta_np = filters_np.ldfilter(name)
            beta_torch = torch.from_numpy(beta_np)
            np_h0, np_h1 = filters_np.ld2quin(beta_np)
            torch_h0, torch_h1 = filters_torch.ld2quin(beta_torch)
            assert_tensors_close(np_h0, torch_h0, atol=1e-10)
            assert_tensors_close(np_h1, torch_h1, atol=1e-10)
    
    def test_output_shapes(self):
        """验证输出形状"""
        beta_np = filters_np.ldfilter('pkva6')
        beta_torch = torch.from_numpy(beta_np)
        np_h0, np_h1 = filters_np.ld2quin(beta_np)
        torch_h0, torch_h1 = filters_torch.ld2quin(beta_torch)
        assert np_h0.shape == torch_h0.shape
        assert np_h1.shape == torch_h1.shape


class TestParafilters:
    """测试 parafilters 函数 - 平行滤波器"""
    
    def test_with_random_filters(self, sample_filter_np, sample_filter_torch):
        """使用随机滤波器测试"""
        np.random.seed(42)
        f1_np, f2_np = sample_filter_np, np.random.rand(4, 6)
        f1_torch, f2_torch = torch.from_numpy(f1_np), torch.from_numpy(f2_np)
        
        np_y1, np_y2 = filters_np.parafilters(f1_np, f2_np)
        torch_y1, torch_y2 = filters_torch.parafilters(f1_torch, f2_torch)
        
        for i in range(4):
            assert_tensors_close(np_y1[i], torch_y1[i], atol=1e-10)
            assert_tensors_close(np_y2[i], torch_y2[i], atol=1e-10)
    
    def test_output_count(self, sample_filter_np, sample_filter_torch):
        """验证输出4个滤波器"""
        np.random.seed(42)
        f2_np = np.random.rand(4, 6)
        f2_torch = torch.from_numpy(f2_np)
        
        np_y1, np_y2 = filters_np.parafilters(sample_filter_np, f2_np)
        torch_y1, torch_y2 = filters_torch.parafilters(sample_filter_torch, f2_torch)
        
        assert len(np_y1) == 4
        assert len(torch_y1) == 4
        assert len(np_y2) == 4
        assert len(torch_y2) == 4


class TestEfilter2:
    """测试 efilter2 函数 - 扩展滤波"""
    
    def test_basic_filtering(self, sample_image_np, sample_image_torch, sample_filter_np, sample_filter_torch):
        """基本滤波测试"""
        np_res = filters_np.efilter2(sample_image_np, sample_filter_np)
        torch_res = filters_torch.efilter2(sample_image_torch, sample_filter_torch)
        assert_tensors_close(np_res, torch_res, atol=1e-9)
    
    def test_different_extension_modes(self, sample_image_np, sample_image_torch, sample_filter_np, sample_filter_torch):
        """测试不同扩展模式"""
        modes = ['per', 'qper_row', 'qper_col']
        for mode in modes:
            np_res = filters_np.efilter2(sample_image_np, sample_filter_np, extmod=mode)
            torch_res = filters_torch.efilter2(sample_image_torch, sample_filter_torch, extmod=mode)
            assert_tensors_close(np_res, torch_res, atol=1e-9)
    
    def test_with_shift(self, sample_image_np, sample_image_torch, sample_filter_np, sample_filter_torch):
        """测试带偏移的滤波"""
        shifts = [[0, 0], [1, 1], [2, 3]]
        for shift in shifts:
            np_res = filters_np.efilter2(sample_image_np, sample_filter_np, shift=shift)
            torch_res = filters_torch.efilter2(sample_image_torch, sample_filter_torch, shift=shift)
            assert_tensors_close(np_res, torch_res, atol=1e-9)
    
    def test_small_filter(self, sample_image_np, sample_image_torch):
        """测试小滤波器"""
        np_f = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=float) / 8
        torch_f = torch.from_numpy(np_f)
        np_res = filters_np.efilter2(sample_image_np, np_f)
        torch_res = filters_torch.efilter2(sample_image_torch, torch_f)
        assert_tensors_close(np_res, torch_res, atol=1e-9)
    
    def test_rectangular_filter(self, sample_image_np, sample_image_torch):
        """测试矩形滤波器"""
        np.random.seed(42)
        np_f = np.random.rand(3, 5)
        torch_f = torch.from_numpy(np_f)
        np_res = filters_np.efilter2(sample_image_np, np_f)
        torch_res = filters_torch.efilter2(sample_image_torch, torch_f)
        assert_tensors_close(np_res, torch_res, atol=1e-9)


# ====================================================================================
# CORE FUNCTIONS TESTS - 确保核心分解和重建函数输出一致
# ====================================================================================

class TestUpsampleAndFindOrigin:
    """测试 _upsample_and_find_origin 内部函数"""
    
    def test_identity_upsampling(self):
        """测试恒等上采样 (mup=1)"""
        np.random.seed(42)
        f_np = np.random.rand(5, 7)
        f_torch = torch.from_numpy(f_np)
        
        np_f_up, np_origin = core_np._upsample_and_find_origin(f_np, 1)
        torch_f_up, torch_origin = core_torch._upsample_and_find_origin(f_torch, 1)
        
        assert_tensors_close(np_f_up, torch_f_up, atol=1e-10)
        assert_tensors_close(np_origin, torch_origin, atol=1e-10)
    
    def test_scalar_upsampling(self):
        """测试标量上采样"""
        np.random.seed(42)
        f_np = np.random.rand(3, 3)
        f_torch = torch.from_numpy(f_np)
        
        for mup in [2, 3]:
            np_f_up, np_origin = core_np._upsample_and_find_origin(f_np, mup)
            torch_f_up, torch_origin = core_torch._upsample_and_find_origin(f_torch, mup)
            
            assert_tensors_close(np_f_up, torch_f_up, atol=1e-10)
            assert_tensors_close(np_origin, torch_origin, atol=1e-10)
    
    def test_matrix_upsampling(self):
        """测试矩阵上采样"""
        np.random.seed(42)
        f_np = np.random.rand(3, 3)
        f_torch = torch.from_numpy(f_np)
        
        mup_np = np.array([[1, 1], [-1, 1]])
        mup_torch = torch.from_numpy(mup_np)
        
        np_f_up, np_origin = core_np._upsample_and_find_origin(f_np, mup_np)
        torch_f_up, torch_origin = core_torch._upsample_and_find_origin(f_torch, mup_torch)
        
        assert_tensors_close(np_f_up, torch_f_up, atol=1e-10)
        assert_tensors_close(np_origin, torch_origin, atol=1e-10)


class TestNssfbdec:
    """测试 nssfbdec 函数 - 非下采样滤波器组分解"""
    
    def test_without_upsampling(self, sample_image_np, sample_image_torch):
        """测试无上采样矩阵的分解"""
        h0_np, h1_np = filters_np.dfilters('pkva', 'd')
        h0_torch = torch.from_numpy(h0_np)
        h1_torch = torch.from_numpy(h1_np)
        
        np_y1, np_y2 = core_np.nssfbdec(sample_image_np, h0_np, h1_np)
        torch_y1, torch_y2 = core_torch.nssfbdec(sample_image_torch, h0_torch, h1_torch)
        
        assert_tensors_close(np_y1, torch_y1, atol=1e-9)
        assert_tensors_close(np_y2, torch_y2, atol=1e-9)
    
    def test_with_quincunx_upsampling(self, sample_image_np, sample_image_torch):
        """测试Quincunx上采样矩阵的分解"""
        h0_np, h1_np = filters_np.dfilters('pkva', 'd')
        h0_torch = torch.from_numpy(h0_np)
        h1_torch = torch.from_numpy(h1_np)
        
        mup_np = np.array([[1, 1], [-1, 1]])
        mup_torch = torch.from_numpy(mup_np)
        
        np_y1, np_y2 = core_np.nssfbdec(sample_image_np, h0_np, h1_np, mup_np)
        torch_y1, torch_y2 = core_torch.nssfbdec(sample_image_torch, h0_torch, h1_torch, mup_torch)
        
        assert_tensors_close(np_y1, torch_y1, atol=1e-9)
        assert_tensors_close(np_y2, torch_y2, atol=1e-9)
    
    def test_with_different_filters(self, sample_image_np, sample_image_torch):
        """使用不同滤波器测试（只测试2D滤波器与mup）"""
        # 注意：db2是1D滤波器，不能与mup一起使用，所以只测试pkva
        filter_names = ['pkva']  # db2是1D，会导致维度问题
        for fname in filter_names:
            h0_np, h1_np = filters_np.dfilters(fname, 'd')
            h0_torch = torch.from_numpy(h0_np)
            h1_torch = torch.from_numpy(h1_np)
            
            mup_np = np.array([[1, 1], [-1, 1]])
            mup_torch = torch.from_numpy(mup_np)
            
            np_y1, np_y2 = core_np.nssfbdec(sample_image_np, h0_np, h1_np, mup_np)
            torch_y1, torch_y2 = core_torch.nssfbdec(sample_image_torch, h0_torch, h1_torch, mup_torch)
            
            assert_tensors_close(np_y1, torch_y1, atol=1e-9)
            assert_tensors_close(np_y2, torch_y2, atol=1e-9)
    

    

    
    def test_rectangular_image(self, rectangular_image_np, rectangular_image_torch):
        """测试矩形图像（使用mup）"""
        h0_np, h1_np = filters_np.dfilters('pkva', 'd')
        h0_torch = torch.from_numpy(h0_np)
        h1_torch = torch.from_numpy(h1_np)
        
        mup_np = np.array([[1, 1], [-1, 1]])
        mup_torch = torch.from_numpy(mup_np)
        
        np_y1, np_y2 = core_np.nssfbdec(rectangular_image_np, h0_np, h1_np, mup_np)
        torch_y1, torch_y2 = core_torch.nssfbdec(rectangular_image_torch, h0_torch, h1_torch, mup_torch)
        
        assert_tensors_close(np_y1, torch_y1, atol=1e-9)
        assert_tensors_close(np_y2, torch_y2, atol=1e-9)


class TestNssfbrec:
    """测试 nssfbrec 函数 - 非下采样滤波器组重建"""
    
    def test_without_upsampling(self, sample_image_np, sample_image_torch):
        """测试无上采样矩阵的重建"""
        h0_np, h1_np = filters_np.dfilters('pkva', 'd')
        g0_np, g1_np = filters_np.dfilters('pkva', 'r')
        h0_torch = torch.from_numpy(h0_np)
        h1_torch = torch.from_numpy(h1_np)
        g0_torch = torch.from_numpy(g0_np)
        g1_torch = torch.from_numpy(g1_np)
        
        # 分解
        np_y1, np_y2 = core_np.nssfbdec(sample_image_np, h0_np, h1_np)
        torch_y1, torch_y2 = core_torch.nssfbdec(sample_image_torch, h0_torch, h1_torch)
        
        # 重建
        np_recon = core_np.nssfbrec(np_y1, np_y2, g0_np, g1_np)
        torch_recon = core_torch.nssfbrec(torch_y1, torch_y2, g0_torch, g1_torch)
        
        assert_tensors_close(np_recon, torch_recon, atol=1e-9)
    
    def test_with_quincunx_upsampling(self, sample_image_np, sample_image_torch):
        """测试Quincunx上采样矩阵的重建"""
        h0_np, h1_np = filters_np.dfilters('pkva', 'd')
        g0_np, g1_np = filters_np.dfilters('pkva', 'r')
        h0_torch = torch.from_numpy(h0_np)
        h1_torch = torch.from_numpy(h1_np)
        g0_torch = torch.from_numpy(g0_np)
        g1_torch = torch.from_numpy(g1_np)
        
        mup_np = np.array([[1, 1], [-1, 1]])
        mup_torch = torch.from_numpy(mup_np)
        
        # 分解
        np_y1, np_y2 = core_np.nssfbdec(sample_image_np, h0_np, h1_np, mup_np)
        torch_y1, torch_y2 = core_torch.nssfbdec(sample_image_torch, h0_torch, h1_torch, mup_torch)
        
        # 重建
        np_recon = core_np.nssfbrec(np_y1, np_y2, g0_np, g1_np, mup_np)
        torch_recon = core_torch.nssfbrec(torch_y1, torch_y2, g0_torch, g1_torch, mup_torch)
        
        assert_tensors_close(np_recon, torch_recon, atol=1e-9)
    
    def test_perfect_reconstruction(self, sample_image_np, sample_image_torch):
        """测试完美重建特性"""
        h0_np, h1_np = filters_np.dfilters('pkva', 'd')
        g0_np, g1_np = filters_np.dfilters('pkva', 'r')
        h0_torch = torch.from_numpy(h0_np)
        h1_torch = torch.from_numpy(h1_np)
        g0_torch = torch.from_numpy(g0_np)
        g1_torch = torch.from_numpy(g1_np)
        
        mup_np = np.array([[1, 1], [-1, 1]])
        mup_torch = torch.from_numpy(mup_np)
        
        # NumPy版本
        np_y1, np_y2 = core_np.nssfbdec(sample_image_np, h0_np, h1_np, mup_np)
        np_recon = core_np.nssfbrec(np_y1, np_y2, g0_np, g1_np, mup_np)
        
        # PyTorch版本
        torch_y1, torch_y2 = core_torch.nssfbdec(sample_image_torch, h0_torch, h1_torch, mup_torch)
        torch_recon = core_torch.nssfbrec(torch_y1, torch_y2, g0_torch, g1_torch, mup_torch)
        
        # 验证两个版本的重建结果一致
        assert_tensors_close(np_recon, torch_recon, atol=1e-9)
        
        # 注意: pkva滤波器有增益为2, 所以重建后应该是原图的2倍
        assert_tensors_close(sample_image_np * 2, np_recon, atol=1e-6)
        assert_tensors_close(sample_image_torch * 2, torch_recon, atol=1e-6)
    
    def test_with_different_filters(self, sample_image_np, sample_image_torch):
        """使用不同滤波器测试重建（只测试2D滤波器）"""
        filter_names = ['pkva']  # db2是1D滤波器，不能与mup一起使用
        mup_np = np.array([[1, 1], [-1, 1]])
        mup_torch = torch.from_numpy(mup_np)
        
        for fname in filter_names:
            h0_np, h1_np = filters_np.dfilters(fname, 'd')
            g0_np, g1_np = filters_np.dfilters(fname, 'r')
            h0_torch = torch.from_numpy(h0_np)
            h1_torch = torch.from_numpy(h1_np)
            g0_torch = torch.from_numpy(g0_np)
            g1_torch = torch.from_numpy(g1_np)
            
            # 分解
            np_y1, np_y2 = core_np.nssfbdec(sample_image_np, h0_np, h1_np, mup_np)
            torch_y1, torch_y2 = core_torch.nssfbdec(sample_image_torch, h0_torch, h1_torch, mup_torch)
            
            # 重建
            np_recon = core_np.nssfbrec(np_y1, np_y2, g0_np, g1_np, mup_np)
            torch_recon = core_torch.nssfbrec(torch_y1, torch_y2, g0_torch, g1_torch, mup_torch)
            
            assert_tensors_close(np_recon, torch_recon, atol=1e-9)
    

    
    def test_shape_mismatch_error(self, sample_image_np, sample_image_torch):
        """测试输入形状不匹配时的错误"""
        g0_np, g1_np = filters_np.dfilters('pkva', 'r')
        g0_torch = torch.from_numpy(g0_np)
        g1_torch = torch.from_numpy(g1_np)
        
        # 创建不同形状的输入
        np_y1 = np.random.rand(32, 32)
        np_y2 = np.random.rand(16, 16)
        torch_y1 = torch.from_numpy(np_y1)
        torch_y2 = torch.from_numpy(np_y2)
        
        # 应该抛出ValueError
        with pytest.raises(ValueError):
            core_np.nssfbrec(np_y1, np_y2, g0_np, g1_np)
        
        with pytest.raises(ValueError):
            core_torch.nssfbrec(torch_y1, torch_y2, g0_torch, g1_torch)


# ====================================================================================
# GPU TESTS - 测试CUDA版本与CPU版本的一致性
# ====================================================================================

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestGPUEquivalence:
    """测试GPU版本与CPU版本的一致性"""
    
    def test_extend2_gpu(self, sample_image_np):
        """测试 extend2 在GPU上的一致性"""
        sample_image_cpu = torch.from_numpy(sample_image_np)
        sample_image_gpu = sample_image_cpu.cuda()
        
        args = (4, 5, 6, 7)
        for mode in ['per', 'qper_row', 'qper_col']:
            cpu_res = utils_torch.extend2(sample_image_cpu, *args, extmod=mode)
            gpu_res = utils_torch.extend2(sample_image_gpu, *args, extmod=mode)
            assert_tensors_close(cpu_res, gpu_res, atol=1e-10)
    
    def test_upsample2df_gpu(self, sample_filter_np):
        """测试 upsample2df 在GPU上的一致性"""
        sample_filter_cpu = torch.from_numpy(sample_filter_np)
        sample_filter_gpu = sample_filter_cpu.cuda()
        
        for power in [1, 2, 3]:
            cpu_res = utils_torch.upsample2df(sample_filter_cpu, power=power)
            gpu_res = utils_torch.upsample2df(sample_filter_gpu, power=power)
            assert_tensors_close(cpu_res, gpu_res, atol=1e-10)
    
    def test_modulate2_gpu(self, sample_image_np):
        """测试 modulate2 在GPU上的一致性"""
        sample_image_cpu = torch.from_numpy(sample_image_np)
        sample_image_gpu = sample_image_cpu.cuda()
        
        for mode in ['r', 'c', 'b']:
            cpu_res = utils_torch.modulate2(sample_image_cpu, mode=mode)
            gpu_res = utils_torch.modulate2(sample_image_gpu, mode=mode)
            assert_tensors_close(cpu_res, gpu_res, atol=1e-10)
    
    def test_resampz_gpu(self, sample_filter_np):
        """测试 resampz 在GPU上的一致性"""
        sample_filter_cpu = torch.from_numpy(sample_filter_np)
        sample_filter_gpu = sample_filter_cpu.cuda()
        
        for type_val in range(1, 5):
            cpu_res = utils_torch.resampz(sample_filter_cpu, type=type_val)
            gpu_res = utils_torch.resampz(sample_filter_gpu, type=type_val)
            assert_tensors_close(cpu_res, gpu_res, atol=1e-10)
    
    def test_qupz_gpu(self, sample_filter_np):
        """测试 qupz 在GPU上的一致性"""
        sample_filter_cpu = torch.from_numpy(sample_filter_np)
        sample_filter_gpu = sample_filter_cpu.cuda()
        
        for type_val in [1, 2]:
            cpu_res = utils_torch.qupz(sample_filter_cpu, type=type_val)
            gpu_res = utils_torch.qupz(sample_filter_gpu, type=type_val)
            assert_tensors_close(cpu_res, gpu_res, atol=1e-10)
    
    def test_ldfilter_gpu(self):
        """测试 ldfilter 在GPU上的一致性"""
        for name in ['pkva6', 'pkva8', 'pkva']:
            cpu_res = filters_torch.ldfilter(name, device='cpu')
            gpu_res = filters_torch.ldfilter(name, device='cuda')
            assert_tensors_close(cpu_res, gpu_res, atol=1e-10)
    
    def test_dmaxflat_gpu(self):
        """测试 dmaxflat 在GPU上的一致性"""
        for n in range(1, 4):
            cpu_res = filters_torch.dmaxflat(n, d=0.5, device='cpu')
            gpu_res = filters_torch.dmaxflat(n, d=0.5, device='cuda')
            assert_tensors_close(cpu_res, gpu_res, atol=1e-10)
    
    def test_atrousfilters_gpu(self):
        """测试 atrousfilters 在GPU上的一致性"""
        for name in ['pyr', 'pyrexc']:
            cpu_res = filters_torch.atrousfilters(name, device='cpu')
            gpu_res = filters_torch.atrousfilters(name, device='cuda')
            for i in range(4):
                assert_tensors_close(cpu_res[i], gpu_res[i], atol=1e-10)
    
    def test_mctrans_gpu(self):
        """测试 mctrans 在GPU上的一致性"""
        b_cpu = filters_torch.ldfilter('pkva6', device='cpu')
        t_cpu = filters_torch.dmaxflat(2, 0, device='cpu')
        b_gpu = b_cpu.cuda()
        t_gpu = t_cpu.cuda()
        
        cpu_res = filters_torch.mctrans(b_cpu, t_cpu)
        gpu_res = filters_torch.mctrans(b_gpu, t_gpu)
        assert_tensors_close(cpu_res, gpu_res, atol=1e-10)
    
    def test_dfilters_gpu(self):
        """测试 dfilters 在GPU上的一致性"""
        for name in ['pkva', 'db2']:
            for ftype in ['d', 'r']:
                cpu_h0, cpu_h1 = filters_torch.dfilters(name, ftype, device='cpu')
                gpu_h0, gpu_h1 = filters_torch.dfilters(name, ftype, device='cuda')
                assert_tensors_close(cpu_h0, gpu_h0, atol=1e-10)
                assert_tensors_close(cpu_h1, gpu_h1, atol=1e-10)
    
    def test_ld2quin_gpu(self):
        """测试 ld2quin 在GPU上的一致性"""
        beta_cpu = filters_torch.ldfilter('pkva6', device='cpu')
        beta_gpu = beta_cpu.cuda()
        
        cpu_h0, cpu_h1 = filters_torch.ld2quin(beta_cpu)
        gpu_h0, gpu_h1 = filters_torch.ld2quin(beta_gpu)
        assert_tensors_close(cpu_h0, gpu_h0, atol=1e-10)
        assert_tensors_close(cpu_h1, gpu_h1, atol=1e-10)
    
    def test_parafilters_gpu(self, sample_filter_np):
        """测试 parafilters 在GPU上的一致性"""
        np.random.seed(42)
        f1_np = sample_filter_np
        f2_np = np.random.rand(4, 6)
        
        f1_cpu = torch.from_numpy(f1_np)
        f2_cpu = torch.from_numpy(f2_np)
        f1_gpu = f1_cpu.cuda()
        f2_gpu = f2_cpu.cuda()
        
        cpu_y1, cpu_y2 = filters_torch.parafilters(f1_cpu, f2_cpu)
        gpu_y1, gpu_y2 = filters_torch.parafilters(f1_gpu, f2_gpu)
        
        for i in range(4):
            assert_tensors_close(cpu_y1[i], gpu_y1[i], atol=1e-10)
            assert_tensors_close(cpu_y2[i], gpu_y2[i], atol=1e-10)
    
    def test_efilter2_gpu(self, sample_image_np, sample_filter_np):
        """测试 efilter2 在GPU上的一致性"""
        sample_image_cpu = torch.from_numpy(sample_image_np)
        sample_filter_cpu = torch.from_numpy(sample_filter_np)
        sample_image_gpu = sample_image_cpu.cuda()
        sample_filter_gpu = sample_filter_cpu.cuda()
        
        cpu_res = filters_torch.efilter2(sample_image_cpu, sample_filter_cpu)
        gpu_res = filters_torch.efilter2(sample_image_gpu, sample_filter_gpu)
        assert_tensors_close(cpu_res, gpu_res, atol=1e-9)
    
    def test_nssfbdec_gpu(self, sample_image_np):
        """测试 nssfbdec 在GPU上的一致性"""
        h0_cpu, h1_cpu = filters_torch.dfilters('pkva', 'd', device='cpu')
        h0_gpu, h1_gpu = filters_torch.dfilters('pkva', 'd', device='cuda')
        
        sample_image_cpu = torch.from_numpy(sample_image_np)
        sample_image_gpu = sample_image_cpu.cuda()
        
        mup_cpu = torch.tensor([[1, 1], [-1, 1]], dtype=torch.long)
        mup_gpu = mup_cpu.cuda()
        
        cpu_y1, cpu_y2 = core_torch.nssfbdec(sample_image_cpu, h0_cpu, h1_cpu, mup_cpu)
        gpu_y1, gpu_y2 = core_torch.nssfbdec(sample_image_gpu, h0_gpu, h1_gpu, mup_gpu)
        
        assert_tensors_close(cpu_y1, gpu_y1, atol=1e-9)
        assert_tensors_close(cpu_y2, gpu_y2, atol=1e-9)
    
    def test_nssfbrec_gpu(self, sample_image_np):
        """测试 nssfbrec 在GPU上的一致性"""
        h0_cpu, h1_cpu = filters_torch.dfilters('pkva', 'd', device='cpu')
        g0_cpu, g1_cpu = filters_torch.dfilters('pkva', 'r', device='cpu')
        h0_gpu, h1_gpu = filters_torch.dfilters('pkva', 'd', device='cuda')
        g0_gpu, g1_gpu = filters_torch.dfilters('pkva', 'r', device='cuda')
        
        sample_image_cpu = torch.from_numpy(sample_image_np)
        sample_image_gpu = sample_image_cpu.cuda()
        
        mup_cpu = torch.tensor([[1, 1], [-1, 1]], dtype=torch.long)
        mup_gpu = mup_cpu.cuda()
        
        # 分解
        cpu_y1, cpu_y2 = core_torch.nssfbdec(sample_image_cpu, h0_cpu, h1_cpu, mup_cpu)
        gpu_y1, gpu_y2 = core_torch.nssfbdec(sample_image_gpu, h0_gpu, h1_gpu, mup_gpu)
        
        # 重建
        cpu_recon = core_torch.nssfbrec(cpu_y1, cpu_y2, g0_cpu, g1_cpu, mup_cpu)
        gpu_recon = core_torch.nssfbrec(gpu_y1, gpu_y2, g0_gpu, g1_gpu, mup_gpu)
        
        assert_tensors_close(cpu_recon, gpu_recon, atol=1e-9)
    
    def test_full_pipeline_gpu(self, sample_image_np):
        """测试完整管道在GPU上的一致性"""
        # CPU版本
        h0_cpu, h1_cpu = filters_torch.dfilters('pkva', 'd', device='cpu')
        g0_cpu, g1_cpu = filters_torch.dfilters('pkva', 'r', device='cpu')
        sample_image_cpu = torch.from_numpy(sample_image_np)
        mup_cpu = torch.tensor([[1, 1], [-1, 1]], dtype=torch.long)
        
        cpu_y1, cpu_y2 = core_torch.nssfbdec(sample_image_cpu, h0_cpu, h1_cpu, mup_cpu)
        cpu_recon = core_torch.nssfbrec(cpu_y1, cpu_y2, g0_cpu, g1_cpu, mup_cpu)
        
        # GPU版本
        h0_gpu, h1_gpu = filters_torch.dfilters('pkva', 'd', device='cuda')
        g0_gpu, g1_gpu = filters_torch.dfilters('pkva', 'r', device='cuda')
        sample_image_gpu = sample_image_cpu.cuda()
        mup_gpu = mup_cpu.cuda()
        
        gpu_y1, gpu_y2 = core_torch.nssfbdec(sample_image_gpu, h0_gpu, h1_gpu, mup_gpu)
        gpu_recon = core_torch.nssfbrec(gpu_y1, gpu_y2, g0_gpu, g1_gpu, mup_gpu)
        
        # 比较结果
        assert_tensors_close(cpu_recon, gpu_recon, atol=1e-9)
        assert_tensors_close(sample_image_cpu * 2, cpu_recon, atol=1e-6)
        assert_tensors_close(sample_image_gpu * 2, gpu_recon, atol=1e-6)