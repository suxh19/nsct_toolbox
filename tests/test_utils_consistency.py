"""
测试工具函数 - nsct_python.utils vs nsct_torch.utils
"""
import numpy as np
import torch
import pytest
import os
import sys

# 将 nsct_python 和 nsct_torch 目录添加到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
nsct_python_dir = os.path.join(parent_dir, 'nsct_python')
nsct_torch_dir = os.path.join(parent_dir, 'nsct_torch')
sys.path.insert(0, nsct_python_dir)
sys.path.insert(0, nsct_torch_dir)

from nsct_python import utils as np_utils
from nsct_torch.utils import extension as torch_extension
from nsct_torch.utils import sampling as torch_sampling
from nsct_torch.utils import modulation as torch_modulation

from tests.test_helpers import (
    assert_shape_equal, assert_values_close, assert_elementwise_equal,
    print_comparison_report, compute_statistics
)


class TestExtend2:
    """测试 extend2 函数"""
    
    @pytest.mark.parametrize("shape", [(16, 16), (32, 32), (15, 17)])
    @pytest.mark.parametrize("ru,rd,cl,cr", [
        (2, 2, 2, 2),
        (3, 5, 4, 6),
        (0, 2, 3, 0),
    ])
    @pytest.mark.parametrize("extmod", ['per'])
    def test_extend2_periodic(self, shape, ru, rd, cl, cr, extmod, random_seed):
        """测试周期扩展模式"""
        # 创建测试输入
        np_input = np.random.randn(*shape).astype(np.float64)
        torch_input = torch.from_numpy(np_input)
        
        # NumPy 版本
        np_output = np_utils.extend2(np_input, ru, rd, cl, cr, extmod)
        
        # PyTorch 版本
        torch_output = torch_extension.extend2(torch_input, ru, rd, cl, cr, extmod)
        
        # 验证形状
        expected_shape = (shape[0] + ru + rd, shape[1] + cl + cr)
        assert np_output.shape == expected_shape, f"NumPy 输出形状错误"
        assert_shape_equal(np_output, torch_output, "extend2 形状不匹配")
        
        # 验证数值
        assert_values_close(np_output, torch_output, rtol=1e-10, atol=1e-12,
                          message=f"extend2 数值不匹配 (shape={shape}, extmod={extmod})")
    
    @pytest.mark.parametrize("shape", [(16, 16), (32, 32)])
    @pytest.mark.parametrize("ru,rd,cl,cr", [(2, 2, 2, 2), (3, 3, 3, 3)])
    @pytest.mark.parametrize("extmod", ['qper_row', 'qper_col'])
    def test_extend2_quincunx(self, shape, ru, rd, cl, cr, extmod, random_seed):
        """测试 quincunx 扩展模式"""
        # 创建测试输入
        np_input = np.random.randn(*shape).astype(np.float64)
        torch_input = torch.from_numpy(np_input)
        
        # NumPy 版本
        np_output = np_utils.extend2(np_input, ru, rd, cl, cr, extmod)
        
        # PyTorch 版本
        torch_output = torch_extension.extend2(torch_input, ru, rd, cl, cr, extmod)
        
        # 验证形状
        expected_shape = (shape[0] + ru + rd, shape[1] + cl + cr)
        assert np_output.shape == expected_shape, f"NumPy 输出形状错误"
        assert_shape_equal(np_output, torch_output, "extend2 形状不匹配")
        
        # 验证数值
        assert_values_close(np_output, torch_output, rtol=1e-10, atol=1e-12,
                          message=f"extend2 数值不匹配 (shape={shape}, extmod={extmod})")


class TestSymext:
    """测试 symext 函数"""
    
    @pytest.mark.parametrize("x_shape", [(16, 16), (32, 32), (15, 17)])
    @pytest.mark.parametrize("h_shape", [(3, 3), (5, 5), (7, 7)])
    @pytest.mark.parametrize("shift", [[0, 0], [1, 1], [0, 1]])
    def test_symext_basic(self, x_shape, h_shape, shift, random_seed):
        """测试基本对称扩展"""
        # 创建测试输入
        np_x = np.random.randn(*x_shape).astype(np.float64)
        np_h = np.random.randn(*h_shape).astype(np.float64)
        torch_x = torch.from_numpy(np_x)
        torch_h = torch.from_numpy(np_h)
        
        # NumPy 版本
        np_output = np_utils.symext(np_x, np_h, shift)
        
        # PyTorch 版本
        torch_output = torch_extension.symext(torch_x, torch_h, shift)
        
        # 验证形状
        expected_shape = (x_shape[0] + h_shape[0] - 1, x_shape[1] + h_shape[1] - 1)
        assert np_output.shape == expected_shape, f"NumPy 输出形状错误"
        assert_shape_equal(np_output, torch_output, "symext 形状不匹配")
        
        # 验证数值
        assert_values_close(np_output, torch_output, rtol=1e-10, atol=1e-12,
                          message=f"symext 数值不匹配 (x_shape={x_shape}, h_shape={h_shape}, shift={shift})")


class TestUpsample2df:
    """测试 upsample2df 函数"""
    
    @pytest.mark.parametrize("h_shape", [(3, 3), (5, 5), (7, 7)])
    @pytest.mark.parametrize("power", [1, 2, 3])
    def test_upsample2df_basic(self, h_shape, power, random_seed):
        """测试基本上采样功能"""
        # 创建测试输入
        np_h = np.random.randn(*h_shape).astype(np.float64)
        torch_h = torch.from_numpy(np_h)
        
        # NumPy 版本
        np_output = np_utils.upsample2df(np_h, power)
        
        # PyTorch 版本
        torch_output = torch_sampling.upsample2df(torch_h, power)
        
        # 验证形状
        # upsample2df 通过零插入上采样,输出大小为 factor * 输入大小
        factor = 2 ** power
        expected_shape = tuple(np.array(h_shape) * factor)
        assert np_output.shape == expected_shape, f"NumPy 输出形状错误"
        assert_shape_equal(np_output, torch_output, "upsample2df 形状不匹配")
        
        # 验证数值
        assert_values_close(np_output, torch_output, rtol=1e-10, atol=1e-12,
                          message=f"upsample2df 数值不匹配 (h_shape={h_shape}, power={power})")
        
        # 验证非零位置的值完全相等(零插入操作应该产生完全相同的结果)
        # 只检查原始滤波器位置的值
        np_nonzero = np_output[::factor, ::factor]
        torch_nonzero = torch_output[::factor, ::factor]
        assert_elementwise_equal(np_nonzero, torch_nonzero, 
                                f"upsample2df 非零位置不完全相等 (h_shape={h_shape}, power={power})")
    
    def test_upsample2df_zero_power(self, random_seed):
        """测试 power=0 的情况（应返回原滤波器）"""
        np_h = np.random.randn(5, 5).astype(np.float64)
        torch_h = torch.from_numpy(np_h)
        
        np_output = np_utils.upsample2df(np_h, 0)
        torch_output = torch_sampling.upsample2df(torch_h, 0)
        
        # 验证返回原滤波器
        assert_values_close(np_output, torch_output, rtol=1e-10, atol=1e-12,
                          message="upsample2df power=0 数值不匹配")
        
        # power=0 时应该完全相等
        assert_elementwise_equal(np_output, torch_output, 
                                "upsample2df power=0 不完全相等")


class TestModulate2:
    """测试 modulate2 函数"""
    
    @pytest.mark.parametrize("shape", [(16, 16), (32, 32), (15, 17)])
    @pytest.mark.parametrize("mode", ['b', 'c', 'r'])
    def test_modulate2_basic(self, shape, mode, random_seed):
        """测试基本调制功能"""
        # 创建测试输入
        np_x = np.random.randn(*shape).astype(np.float64)
        torch_x = torch.from_numpy(np_x)
        
        # NumPy 版本
        np_output = np_utils.modulate2(np_x, mode)
        
        # PyTorch 版本
        torch_output = torch_modulation.modulate2(torch_x, mode)
        
        # 验证形状
        assert_shape_equal(np_output, torch_output, f"modulate2 形状不匹配 (mode={mode})")
        
        # 验证数值
        assert_values_close(np_output, torch_output, rtol=1e-10, atol=1e-12,
                          message=f"modulate2 数值不匹配 (shape={shape}, mode={mode})")
        
        # 调制操作是确定性的,应该产生完全相同的结果
        assert_elementwise_equal(np_output, torch_output, 
                                f"modulate2 不完全相等 (shape={shape}, mode={mode})")
    
    @pytest.mark.parametrize("shape,center", [
        ((16, 16), [8, 8]),
        ((32, 32), [16, 16]),
        ((15, 17), [7, 8]),
    ])
    def test_modulate2_with_center(self, shape, center, random_seed):
        """测试带有自定义中心的调制"""
        np_x = np.random.randn(*shape).astype(np.float64)
        torch_x = torch.from_numpy(np_x)
        
        np_output = np_utils.modulate2(np_x, 'c', center)
        torch_output = torch_modulation.modulate2(torch_x, 'c', center)
        
        assert_values_close(np_output, torch_output, rtol=1e-10, atol=1e-12,
                          message=f"modulate2 with center 数值不匹配 (shape={shape}, center={center})")
        
        # 带中心的调制也应该完全相等
        assert_elementwise_equal(np_output, torch_output, 
                                f"modulate2 with center 不完全相等 (shape={shape}, center={center})")


class TestResampz:
    """测试 resampz 函数"""
    
    @pytest.mark.parametrize("shape", [(16, 16), (32, 32)])
    @pytest.mark.parametrize("type", [1, 2])
    @pytest.mark.parametrize("shift", [1, 2])
    def test_resampz_basic(self, shape, type, shift, random_seed):
        """测试基本重采样功能"""
        # 创建测试输入
        np_x = np.random.randn(*shape).astype(np.float64)
        torch_x = torch.from_numpy(np_x)
        
        # NumPy 版本
        np_output = np_utils.resampz(np_x, type, shift)
        
        # PyTorch 版本
        torch_output = torch_sampling.resampz(torch_x, type, shift)
        
        # 验证形状
        assert_shape_equal(np_output, torch_output, f"resampz 形状不匹配 (type={type}, shift={shift})")
        
        # 验证数值
        assert_values_close(np_output, torch_output, rtol=1e-10, atol=1e-12,
                          message=f"resampz 数值不匹配 (shape={shape}, type={type}, shift={shift})")


class TestQupz:
    """测试 qupz 函数"""
    
    @pytest.mark.parametrize("shape", [(8, 8), (16, 16)])
    @pytest.mark.parametrize("type", [1, 2])
    def test_qupz_basic(self, shape, type, random_seed):
        """测试基本 quincunx 上采样功能"""
        # 创建测试输入
        np_x = np.random.randn(*shape).astype(np.float64)
        torch_x = torch.from_numpy(np_x)
        
        # NumPy 版本
        np_output = np_utils.qupz(np_x, type)
        
        # PyTorch 版本
        torch_output = torch_sampling.qupz(torch_x, type)
        
        # 验证形状
        expected_shape = (2 * shape[0] - 1, 2 * shape[1] - 1)
        assert np_output.shape == expected_shape, f"NumPy 输出形状错误"
        assert_shape_equal(np_output, torch_output, f"qupz 形状不匹配 (type={type})")
        
        # 验证数值
        assert_values_close(np_output, torch_output, rtol=1e-10, atol=1e-12,
                          message=f"qupz 数值不匹配 (shape={shape}, type={type})")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
