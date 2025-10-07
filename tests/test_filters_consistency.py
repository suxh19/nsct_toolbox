"""
测试滤波器函数 - nsct_python.filters vs nsct_torch.filters
"""
import numpy as np
import torch
import pytest

from nsct_python import filters as np_filters
from nsct_torch.filters import (
    ld2quin as torch_ld2quin,
    efilter2 as torch_efilter2,
    dmaxflat as torch_dmaxflat,
    atrousfilters as torch_atrousfilters,
    mctrans as torch_mctrans,
    ldfilter as torch_ldfilter,
    dfilters as torch_dfilters,
    parafilters as torch_parafilters,
)

from tests.test_helpers import (
    assert_shape_equal, assert_values_close, 
    assert_list_values_close,
    print_comparison_report
)


class TestLd2quin:
    """测试 ld2quin 函数"""
    
    @pytest.mark.parametrize("beta_length", [6, 8, 10])
    def test_ld2quin_basic(self, beta_length, random_seed):
        """测试基本 quincunx 滤波器生成"""
        # 创建测试输入 - 偶数长度的 1D 滤波器
        np_beta = np.random.randn(beta_length).astype(np.float64)
        torch_beta = torch.from_numpy(np_beta)
        
        # NumPy 版本
        np_h0, np_h1 = np_filters.ld2quin(np_beta)
        
        # PyTorch 版本
        from nsct_torch.filters.ld2quin import ld2quin
        torch_h0, torch_h1 = ld2quin(torch_beta)
        
        # 验证 h0
        assert_shape_equal(np_h0, torch_h0, "ld2quin h0 形状不匹配")
        assert_values_close(np_h0, torch_h0, rtol=1e-10, atol=1e-12,
                          message=f"ld2quin h0 数值不匹配 (beta_length={beta_length})")
        
        # 验证 h1
        assert_shape_equal(np_h1, torch_h1, "ld2quin h1 形状不匹配")
        assert_values_close(np_h1, torch_h1, rtol=1e-10, atol=1e-12,
                          message=f"ld2quin h1 数值不匹配 (beta_length={beta_length})")


class TestEfilter2:
    """测试 efilter2 函数"""
    
    @pytest.mark.parametrize("x_shape", [(32, 32), (64, 64)])
    @pytest.mark.parametrize("f_shape", [(3, 3), (5, 5), (7, 7)])
    @pytest.mark.parametrize("extmod", ['per'])
    def test_efilter2_basic(self, x_shape, f_shape, extmod, random_seed):
        """测试基本 2D 滤波"""
        # 创建测试输入
        np_x = np.random.randn(*x_shape).astype(np.float64)
        np_f = np.random.randn(*f_shape).astype(np.float64)
        torch_x = torch.from_numpy(np_x)
        torch_f = torch.from_numpy(np_f)
        
        # NumPy 版本
        np_output = np_filters.efilter2(np_x, np_f, extmod)
        
        # PyTorch 版本
        from nsct_torch.filters.efilter2 import efilter2
        torch_output = efilter2(torch_x, torch_f, extmod)
        
        # 验证形状（输出应与输入相同）
        assert np_output.shape == x_shape, "NumPy 输出形状错误"
        assert_shape_equal(np_output, torch_output, "efilter2 形状不匹配")
        
        # 验证数值
        assert_values_close(np_output, torch_output, rtol=1e-8, atol=1e-10,
                          message=f"efilter2 数值不匹配 (x_shape={x_shape}, f_shape={f_shape})")
    
    @pytest.mark.parametrize("shift", [[0, 0], [1, 0], [0, 1], [1, 1]])
    def test_efilter2_with_shift(self, shift, random_seed):
        """测试带偏移的滤波"""
        np_x = np.random.randn(32, 32).astype(np.float64)
        np_f = np.random.randn(5, 5).astype(np.float64)
        torch_x = torch.from_numpy(np_x)
        torch_f = torch.from_numpy(np_f)
        
        np_output = np_filters.efilter2(np_x, np_f, 'per', shift)
        
        from nsct_torch.filters.efilter2 import efilter2
        torch_output = efilter2(torch_x, torch_f, 'per', shift)
        
        assert_values_close(np_output, torch_output, rtol=1e-8, atol=1e-10,
                          message=f"efilter2 with shift 数值不匹配 (shift={shift})")


class TestDmaxflat:
    """测试 dmaxflat 函数"""
    
    @pytest.mark.parametrize("N", [3, 5, 7, 9])
    @pytest.mark.parametrize("d", [0.0, 0.5])
    def test_dmaxflat_basic(self, N, d, random_seed):
        """测试 diamond maxflat 滤波器生成"""
        # NumPy 版本
        np_filter = np_filters.dmaxflat(N, d)
        
        # PyTorch 版本
        from nsct_torch.filters.dmaxflat import dmaxflat
        torch_filter = dmaxflat(N, d, device='cpu')
        
        # 验证形状
        assert_shape_equal(np_filter, torch_filter, f"dmaxflat 形状不匹配 (N={N}, d={d})")
        
        # 验证数值
        assert_values_close(np_filter, torch_filter, rtol=1e-10, atol=1e-12,
                          message=f"dmaxflat 数值不匹配 (N={N}, d={d})")


class TestAtrousfilters:
    """测试 atrousfilters 函数"""
    
    @pytest.mark.parametrize("fname", ['maxflat', '9-7', 'pkva'])
    def test_atrousfilters_basic(self, fname):
        """测试 à trous 滤波器生成"""
        # NumPy 版本
        np_h0, np_g0, np_h1, np_g1 = np_filters.atrousfilters(fname)
        
        # PyTorch 版本
        from nsct_torch.filters.atrousfilters import atrousfilters
        torch_h0, torch_g0, torch_h1, torch_g1 = atrousfilters(fname, device='cpu')
        
        # 验证所有四个滤波器
        assert_values_close(np_h0, torch_h0, rtol=1e-10, atol=1e-12,
                          message=f"atrousfilters h0 数值不匹配 (fname={fname})")
        assert_values_close(np_g0, torch_g0, rtol=1e-10, atol=1e-12,
                          message=f"atrousfilters g0 数值不匹配 (fname={fname})")
        assert_values_close(np_h1, torch_h1, rtol=1e-10, atol=1e-12,
                          message=f"atrousfilters h1 数值不匹配 (fname={fname})")
        assert_values_close(np_g1, torch_g1, rtol=1e-10, atol=1e-12,
                          message=f"atrousfilters g1 数值不匹配 (fname={fname})")


class TestMctrans:
    """测试 mctrans 函数"""
    
    @pytest.mark.parametrize("b_length", [5, 7, 9])
    @pytest.mark.parametrize("t_length", [3, 5])
    def test_mctrans_basic(self, b_length, t_length, random_seed):
        """测试 McClellan 变换"""
        # 创建测试输入
        np_b = np.random.randn(b_length).astype(np.float64)
        np_t = np.random.randn(t_length, t_length).astype(np.float64)
        torch_b = torch.from_numpy(np_b)
        torch_t = torch.from_numpy(np_t)
        
        # NumPy 版本
        np_output = np_filters.mctrans(np_b, np_t)
        
        # PyTorch 版本
        from nsct_torch.filters.mctrans import mctrans
        torch_output = mctrans(torch_b, torch_t)
        
        # 验证形状
        assert_shape_equal(np_output, torch_output, "mctrans 形状不匹配")
        
        # 验证数值
        assert_values_close(np_output, torch_output, rtol=1e-8, atol=1e-10,
                          message=f"mctrans 数值不匹配 (b_length={b_length}, t_length={t_length})")


class TestLdfilter:
    """测试 ldfilter 函数"""
    
    @pytest.mark.parametrize("fname", ['9-7', '5-3', 'maxflat', 'pkva'])
    def test_ldfilter_basic(self, fname):
        """测试 ladder 滤波器加载"""
        # NumPy 版本
        np_filter = np_filters.ldfilter(fname)
        
        # PyTorch 版本
        from nsct_torch.filters.ldfilter import ldfilter
        torch_filter = ldfilter(fname, device='cpu')
        
        # 验证形状
        assert_shape_equal(np_filter, torch_filter, f"ldfilter 形状不匹配 (fname={fname})")
        
        # 验证数值
        assert_values_close(np_filter, torch_filter, rtol=1e-10, atol=1e-12,
                          message=f"ldfilter 数值不匹配 (fname={fname})")


class TestDfilters:
    """测试 dfilters 函数"""
    
    @pytest.mark.parametrize("fname", ['dmaxflat7', 'dmaxflat5', 'dmaxflat4'])
    @pytest.mark.parametrize("type", ['d', 'r'])
    def test_dfilters_basic(self, fname, type):
        """测试方向滤波器生成"""
        # NumPy 版本
        np_h0, np_h1 = np_filters.dfilters(fname, type)
        
        # PyTorch 版本
        from nsct_torch.filters.dfilters import dfilters
        torch_h0, torch_h1 = dfilters(fname, type, device='cpu')
        
        # 验证 h0
        assert_shape_equal(np_h0, torch_h0, f"dfilters h0 形状不匹配 (fname={fname}, type={type})")
        assert_values_close(np_h0, torch_h0, rtol=1e-10, atol=1e-12,
                          message=f"dfilters h0 数值不匹配 (fname={fname}, type={type})")
        
        # 验证 h1
        assert_shape_equal(np_h1, torch_h1, f"dfilters h1 形状不匹配 (fname={fname}, type={type})")
        assert_values_close(np_h1, torch_h1, rtol=1e-10, atol=1e-12,
                          message=f"dfilters h1 数值不匹配 (fname={fname}, type={type})")


class TestParafilters:
    """测试 parafilters 函数"""
    
    def test_parafilters_basic(self, random_seed):
        """测试 parallelogram 滤波器生成"""
        # 创建测试输入 - 两个随机滤波器
        np_f1 = np.random.randn(5, 5).astype(np.float64)
        np_f2 = np.random.randn(5, 5).astype(np.float64)
        torch_f1 = torch.from_numpy(np_f1)
        torch_f2 = torch.from_numpy(np_f2)
        
        # NumPy 版本
        np_f1_list, np_f2_list = np_filters.parafilters(np_f1, np_f2)
        
        # PyTorch 版本
        from nsct_torch.filters.parafilters import parafilters
        torch_f1_list, torch_f2_list = parafilters(torch_f1, torch_f2)
        
        # 验证列表长度
        assert len(np_f1_list) == len(torch_f1_list), "parafilters f1 列表长度不匹配"
        assert len(np_f2_list) == len(torch_f2_list), "parafilters f2 列表长度不匹配"
        
        # 验证每个滤波器
        assert_list_values_close(np_f1_list, torch_f1_list, rtol=1e-10, atol=1e-12,
                               message="parafilters f1 列表数值不匹配")
        assert_list_values_close(np_f2_list, torch_f2_list, rtol=1e-10, atol=1e-12,
                               message="parafilters f2 列表数值不匹配")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
