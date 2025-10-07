"""
测试核心滤波器组函数 - nsct_python.core vs nsct_torch.core
"""
import numpy as np
import torch
import pytest

from nsct_python import core as np_core
from nsct_python import filters as np_filters

from tests.test_helpers import (
    assert_shape_equal, assert_values_close,
    print_comparison_report
)


class TestNssfbdec:
    """测试 nssfbdec 函数"""
    
    @pytest.mark.parametrize("x_shape", [(32, 32), (64, 64)])
    def test_nssfbdec_no_upsample(self, x_shape, random_seed):
        """测试不带上采样的双通道分解"""
        # 创建测试输入
        np_x = np.random.randn(*x_shape).astype(np.float64)
        np_f1 = np.random.randn(7, 7).astype(np.float64)
        np_f2 = np.random.randn(7, 7).astype(np.float64)
        
        torch_x = torch.from_numpy(np_x)
        torch_f1 = torch.from_numpy(np_f1)
        torch_f2 = torch.from_numpy(np_f2)
        
        # NumPy 版本
        np_y1, np_y2 = np_core.nssfbdec(np_x, np_f1, np_f2)
        
        # PyTorch 版本
        from nsct_torch.core.filterbank import nssfbdec
        torch_y1, torch_y2 = nssfbdec(torch_x, torch_f1, torch_f2)
        
        # 验证输出
        assert_shape_equal(np_y1, torch_y1, "nssfbdec y1 形状不匹配")
        assert_shape_equal(np_y2, torch_y2, "nssfbdec y2 形状不匹配")
        
        assert_values_close(np_y1, torch_y1, rtol=1e-8, atol=1e-10,
                          message=f"nssfbdec y1 数值不匹配 (x_shape={x_shape})")
        assert_values_close(np_y2, torch_y2, rtol=1e-8, atol=1e-10,
                          message=f"nssfbdec y2 数值不匹配 (x_shape={x_shape})")
    
    @pytest.mark.parametrize("mup", [2, 4])
    def test_nssfbdec_with_upsample(self, mup, random_seed):
        """测试带上采样矩阵的双通道分解"""
        np_x = np.random.randn(32, 32).astype(np.float64)
        np_f1 = np.random.randn(5, 5).astype(np.float64)
        np_f2 = np.random.randn(5, 5).astype(np.float64)
        
        torch_x = torch.from_numpy(np_x)
        torch_f1 = torch.from_numpy(np_f1)
        torch_f2 = torch.from_numpy(np_f2)
        
        # NumPy 版本
        np_y1, np_y2 = np_core.nssfbdec(np_x, np_f1, np_f2, mup)
        
        # PyTorch 版本
        from nsct_torch.core.filterbank import nssfbdec
        torch_y1, torch_y2 = nssfbdec(torch_x, torch_f1, torch_f2, mup)
        
        assert_values_close(np_y1, torch_y1, rtol=1e-8, atol=1e-10,
                          message=f"nssfbdec y1 数值不匹配 (mup={mup})")
        assert_values_close(np_y2, torch_y2, rtol=1e-8, atol=1e-10,
                          message=f"nssfbdec y2 数值不匹配 (mup={mup})")


class TestNssfbrec:
    """测试 nssfbrec 函数"""
    
    @pytest.mark.parametrize("x_shape", [(32, 32), (64, 64)])
    def test_nssfbrec_basic(self, x_shape, random_seed):
        """测试双通道重构"""
        # 创建测试输入
        np_x1 = np.random.randn(*x_shape).astype(np.float64)
        np_x2 = np.random.randn(*x_shape).astype(np.float64)
        np_f1 = np.random.randn(7, 7).astype(np.float64)
        np_f2 = np.random.randn(7, 7).astype(np.float64)
        
        torch_x1 = torch.from_numpy(np_x1)
        torch_x2 = torch.from_numpy(np_x2)
        torch_f1 = torch.from_numpy(np_f1)
        torch_f2 = torch.from_numpy(np_f2)
        
        # NumPy 版本
        np_output = np_core.nssfbrec(np_x1, np_x2, np_f1, np_f2)
        
        # PyTorch 版本
        from nsct_torch.core.filterbank import nssfbrec
        torch_output = nssfbrec(torch_x1, torch_x2, torch_f1, torch_f2)
        
        # 验证输出
        assert_shape_equal(np_output, torch_output, "nssfbrec 形状不匹配")
        assert_values_close(np_output, torch_output, rtol=1e-8, atol=1e-10,
                          message=f"nssfbrec 数值不匹配 (x_shape={x_shape})")


class TestNsfbdec:
    """测试 nsfbdec 函数"""
    
    @pytest.mark.parametrize("x_shape", [(64, 64), (128, 128)])
    @pytest.mark.parametrize("lev", [0, 1, 2])
    def test_nsfbdec_basic(self, x_shape, lev, random_seed):
        """测试非下采样滤波器组分解"""
        # 创建测试输入
        np_x = np.random.randn(*x_shape).astype(np.float64)
        
        # 生成 à trous 滤波器
        np_h0, np_g0, np_h1, np_g1 = np_filters.atrousfilters('maxflat')
        
        torch_x = torch.from_numpy(np_x)
        torch_h0 = torch.from_numpy(np_h0)
        torch_h1 = torch.from_numpy(np_h1)
        
        # NumPy 版本
        np_y0, np_y1 = np_core.nsfbdec(np_x, np_h0, np_h1, lev)
        
        # PyTorch 版本
        from nsct_torch.core.filterbank import nsfbdec
        torch_y0, torch_y1 = nsfbdec(torch_x, torch_h0, torch_h1, lev)
        
        # 验证输出
        assert_shape_equal(np_y0, torch_y0, f"nsfbdec y0 形状不匹配 (lev={lev})")
        assert_shape_equal(np_y1, torch_y1, f"nsfbdec y1 形状不匹配 (lev={lev})")
        
        assert_values_close(np_y0, torch_y0, rtol=1e-8, atol=1e-10,
                          message=f"nsfbdec y0 数值不匹配 (x_shape={x_shape}, lev={lev})")
        assert_values_close(np_y1, torch_y1, rtol=1e-8, atol=1e-10,
                          message=f"nsfbdec y1 数值不匹配 (x_shape={x_shape}, lev={lev})")


class TestNsfbrec:
    """测试 nsfbrec 函数"""
    
    @pytest.mark.parametrize("x_shape", [(64, 64), (128, 128)])
    @pytest.mark.parametrize("lev", [0, 1, 2])
    def test_nsfbrec_basic(self, x_shape, lev, random_seed):
        """测试非下采样滤波器组重构"""
        # 创建测试输入
        np_y0 = np.random.randn(*x_shape).astype(np.float64)
        np_y1 = np.random.randn(*x_shape).astype(np.float64)
        
        # 生成 à trous 滤波器
        np_h0, np_g0, np_h1, np_g1 = np_filters.atrousfilters('maxflat')
        
        torch_y0 = torch.from_numpy(np_y0)
        torch_y1 = torch.from_numpy(np_y1)
        torch_g0 = torch.from_numpy(np_g0)
        torch_g1 = torch.from_numpy(np_g1)
        
        # NumPy 版本
        np_output = np_core.nsfbrec(np_y0, np_y1, np_g0, np_g1, lev)
        
        # PyTorch 版本
        from nsct_torch.core.filterbank import nsfbrec
        torch_output = nsfbrec(torch_y0, torch_y1, torch_g0, torch_g1, lev)
        
        # 验证输出
        assert_shape_equal(np_output, torch_output, f"nsfbrec 形状不匹配 (lev={lev})")
        assert_values_close(np_output, torch_output, rtol=1e-8, atol=1e-10,
                          message=f"nsfbrec 数值不匹配 (x_shape={x_shape}, lev={lev})")


class TestNsfbDecRecRoundtrip:
    """测试 nsfbdec 和 nsfbrec 的往返一致性"""
    
    @pytest.mark.parametrize("x_shape", [(64, 64)])
    @pytest.mark.parametrize("lev", [0, 1])
    @pytest.mark.parametrize("pfilt", ['maxflat', '9-7'])
    def test_roundtrip(self, x_shape, lev, pfilt, random_seed):
        """测试分解-重构往返"""
        # 创建测试输入
        np_x = np.random.randn(*x_shape).astype(np.float64)
        
        # 生成滤波器
        np_h0, np_g0, np_h1, np_g1 = np_filters.atrousfilters(pfilt)
        
        torch_x = torch.from_numpy(np_x)
        torch_h0 = torch.from_numpy(np_h0)
        torch_h1 = torch.from_numpy(np_h1)
        torch_g0 = torch.from_numpy(np_g0)
        torch_g1 = torch.from_numpy(np_g1)
        
        # NumPy 版本往返
        np_y0, np_y1 = np_core.nsfbdec(np_x, np_h0, np_h1, lev)
        np_rec = np_core.nsfbrec(np_y0, np_y1, np_g0, np_g1, lev)
        
        # PyTorch 版本往返
        from nsct_torch.core.filterbank import nsfbdec, nsfbrec
        torch_y0, torch_y1 = nsfbdec(torch_x, torch_h0, torch_h1, lev)
        torch_rec = nsfbrec(torch_y0, torch_y1, torch_g0, torch_g1, lev)
        
        # 验证重构结果一致
        assert_values_close(np_rec, torch_rec, rtol=1e-7, atol=1e-9,
                          message=f"nsfb 往返数值不匹配 (lev={lev}, pfilt={pfilt})")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
