"""
测试方向滤波器组和完整NSCT - nsct_python vs nsct_torch
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

from nsct_python import core as np_core, filters as np_filters
from tests.test_helpers import (
    assert_shape_equal, assert_values_close,
    assert_list_values_close, print_comparison_report
)


class TestNsdfbdec:
    """测试 nsdfbdec 函数"""
    
    @pytest.mark.parametrize("x_shape", [(64, 64), (128, 128)])
    @pytest.mark.parametrize("clevels", [0, 1, 2])
    def test_nsdfbdec_dmaxflat(self, x_shape, clevels, random_seed):
        """测试方向滤波器组分解 - dmaxflat 滤波器"""
        # 创建测试输入
        np_x = np.random.randn(*x_shape).astype(np.float64)
        torch_x = torch.from_numpy(np_x)
        
        # NumPy 版本
        np_output = np_core.nsdfbdec(np_x, 'dmaxflat7', clevels)
        
        # PyTorch 版本
        from nsct_torch.core.directional import nsdfbdec
        torch_output = nsdfbdec(torch_x, 'dmaxflat7', clevels)
        
        # 验证输出是列表
        assert isinstance(np_output, list), "NumPy 输出应该是列表"
        assert isinstance(torch_output, list), "Torch 输出应该是列表"
        
        # 验证列表长度
        expected_len = 2 ** clevels if clevels > 0 else 1
        assert len(np_output) == expected_len, f"NumPy 输出长度错误"
        assert len(torch_output) == expected_len, f"Torch 输出长度错误"
        
        # 验证每个子带
        assert_list_values_close(np_output, torch_output, rtol=1e-7, atol=1e-9,
                               message=f"nsdfbdec 数值不匹配 (x_shape={x_shape}, clevels={clevels})")
    
    @pytest.mark.parametrize("clevels", [1, 2])
    def test_nsdfbdec_different_filters(self, clevels, random_seed):
        """测试不同滤波器"""
        np_x = np.random.randn(64, 64).astype(np.float64)
        torch_x = torch.from_numpy(np_x)
        
        for dfilt in ['dmaxflat7', 'dmaxflat5', 'pkva']:
            # NumPy 版本
            np_output = np_core.nsdfbdec(np_x, dfilt, clevels)
            
            # PyTorch 版本
            from nsct_torch.core.directional import nsdfbdec
            torch_output = nsdfbdec(torch_x, dfilt, clevels)
            
            # 验证
            assert_list_values_close(np_output, torch_output, rtol=1e-7, atol=1e-9,
                                   message=f"nsdfbdec 数值不匹配 (dfilt={dfilt}, clevels={clevels})")


class TestNsdfbrec:
    """测试 nsdfbrec 函数"""
    
    @pytest.mark.parametrize("clevels", [0, 1, 2])
    def test_nsdfbrec_basic(self, clevels, random_seed):
        """测试方向滤波器组重构"""
        # 创建测试输入 - 子带列表
        expected_len = 2 ** clevels if clevels > 0 else 1
        np_y = [np.random.randn(64, 64).astype(np.float64) for _ in range(expected_len)]
        torch_y = [torch.from_numpy(y) for y in np_y]
        
        # NumPy 版本
        np_output = np_core.nsdfbrec(np_y, 'dmaxflat7')
        
        # PyTorch 版本
        from nsct_torch.core.directional import nsdfbrec
        torch_output = nsdfbrec(torch_y, 'dmaxflat7')
        
        # 验证输出
        assert_shape_equal(np_output, torch_output, f"nsdfbrec 形状不匹配 (clevels={clevels})")
        assert_values_close(np_output, torch_output, rtol=1e-7, atol=1e-9,
                          message=f"nsdfbrec 数值不匹配 (clevels={clevels})")


class TestNsdfbDecRecRoundtrip:
    """测试 nsdfbdec 和 nsdfbrec 的往返一致性"""
    
    @pytest.mark.parametrize("x_shape", [(64, 64)])
    @pytest.mark.parametrize("clevels", [1, 2])
    @pytest.mark.parametrize("dfilt", ['dmaxflat7', 'pkva'])
    def test_roundtrip(self, x_shape, clevels, dfilt, random_seed):
        """测试分解-重构往返"""
        # 创建测试输入
        np_x = np.random.randn(*x_shape).astype(np.float64)
        torch_x = torch.from_numpy(np_x)
        
        # NumPy 版本往返
        np_y = np_core.nsdfbdec(np_x, dfilt, clevels)
        np_rec = np_core.nsdfbrec(np_y, dfilt)
        
        # PyTorch 版本往返
        from nsct_torch.core.directional import nsdfbdec, nsdfbrec
        torch_y = nsdfbdec(torch_x, dfilt, clevels)
        torch_rec = nsdfbrec(torch_y, dfilt)
        
        # 验证重构结果一致
        assert_values_close(np_rec, torch_rec, rtol=1e-6, atol=1e-8,
                          message=f"nsdfb 往返数值不匹配 (clevels={clevels}, dfilt={dfilt})")


class TestNsctdec:
    """测试完整的 NSCT 分解"""
    
    @pytest.mark.parametrize("x_shape", [(128, 128), (256, 256)])
    @pytest.mark.parametrize("levels", [[2, 3], [3, 3, 4]])
    def test_nsctdec_basic(self, x_shape, levels, random_seed):
        """测试完整 NSCT 分解"""
        # 创建测试输入
        np_x = np.random.randn(*x_shape).astype(np.float64)
        torch_x = torch.from_numpy(np_x)
        
        # NumPy 版本
        np_y = np_core.nsctdec(np_x, levels)
        
        # PyTorch 版本
        from nsct_torch.core.nsct import nsctdec
        torch_y = nsctdec(torch_x, levels)
        
        # 验证输出结构
        assert isinstance(np_y, list), "NumPy 输出应该是列表"
        assert isinstance(torch_y, list), "Torch 输出应该是列表"
        assert len(np_y) == len(torch_y), "输出列表长度不匹配"
        
        # 验证每一级
        for i, (np_level, torch_level) in enumerate(zip(np_y, torch_y)):
            if isinstance(np_level, list):
                # 带通子带（列表）
                assert isinstance(torch_level, list), f"级别 {i} 应该是列表"
                assert_list_values_close(np_level, torch_level, rtol=1e-6, atol=1e-8,
                                       message=f"nsctdec 级别 {i} 数值不匹配 (levels={levels})")
            else:
                # 低频子带（数组）
                assert_values_close(np_level, torch_level, rtol=1e-6, atol=1e-8,
                                  message=f"nsctdec 低频子带数值不匹配 (levels={levels})")
    
    @pytest.mark.parametrize("dfilt", ['dmaxflat7', 'dmaxflat5'])
    @pytest.mark.parametrize("pfilt", ['maxflat', '9-7'])
    def test_nsctdec_different_filters(self, dfilt, pfilt, random_seed):
        """测试不同滤波器组合"""
        np_x = np.random.randn(128, 128).astype(np.float64)
        torch_x = torch.from_numpy(np_x)
        levels = [2, 3]
        
        # NumPy 版本
        np_y = np_core.nsctdec(np_x, levels, dfilt, pfilt)
        
        # PyTorch 版本
        from nsct_torch.core.nsct import nsctdec
        torch_y = nsctdec(torch_x, levels, dfilt, pfilt)
        
        # 验证每一级
        for i, (np_level, torch_level) in enumerate(zip(np_y, torch_y)):
            if isinstance(np_level, list):
                assert_list_values_close(np_level, torch_level, rtol=1e-6, atol=1e-8,
                                       message=f"nsctdec 数值不匹配 (dfilt={dfilt}, pfilt={pfilt}, 级别={i})")
            else:
                assert_values_close(np_level, torch_level, rtol=1e-6, atol=1e-8,
                                  message=f"nsctdec 低频数值不匹配 (dfilt={dfilt}, pfilt={pfilt})")


class TestNsctrec:
    """测试完整的 NSCT 重构"""
    
    @pytest.mark.parametrize("levels", [[2, 3], [3, 3, 4]])
    def test_nsctrec_basic(self, levels, random_seed):
        """测试完整 NSCT 重构"""
        # 先进行分解获取测试数据
        np_x = np.random.randn(128, 128).astype(np.float64)
        torch_x = torch.from_numpy(np_x)
        
        # 分解
        np_y = np_core.nsctdec(np_x, levels)
        
        from nsct_torch.core.nsct import nsctdec
        torch_y = nsctdec(torch_x, levels)
        
        # 重构 - NumPy 版本
        np_rec = np_core.nsctrec(np_y)
        
        # 重构 - PyTorch 版本
        from nsct_torch.core.nsct import nsctrec
        torch_rec = nsctrec(torch_y)
        
        # 验证重构结果
        assert_shape_equal(np_rec, torch_rec, "nsctrec 形状不匹配")
        assert_values_close(np_rec, torch_rec, rtol=1e-6, atol=1e-8,
                          message=f"nsctrec 数值不匹配 (levels={levels})")


class TestNsctDecRecRoundtrip:
    """测试完整 NSCT 分解-重构往返"""
    
    @pytest.mark.parametrize("x_shape", [(128, 128)])
    @pytest.mark.parametrize("levels", [[2, 3], [2, 2, 3]])
    @pytest.mark.parametrize("dfilt", ['dmaxflat7'])
    @pytest.mark.parametrize("pfilt", ['maxflat', '9-7'])
    def test_full_roundtrip(self, x_shape, levels, dfilt, pfilt, random_seed):
        """测试完整的分解-重构往返"""
        # 创建测试输入
        np_x = np.random.randn(*x_shape).astype(np.float64)
        torch_x = torch.from_numpy(np_x)
        
        # NumPy 版本往返
        np_y = np_core.nsctdec(np_x, levels, dfilt, pfilt)
        np_rec = np_core.nsctrec(np_y, dfilt, pfilt)
        
        # PyTorch 版本往返
        from nsct_torch.core.nsct import nsctdec, nsctrec
        torch_y = nsctdec(torch_x, levels, dfilt, pfilt)
        torch_rec = nsctrec(torch_y, dfilt, pfilt)
        
        # 验证重构结果一致
        assert_values_close(np_rec, torch_rec, rtol=1e-5, atol=1e-7,
                          message=f"NSCT 往返数值不匹配 (levels={levels}, dfilt={dfilt}, pfilt={pfilt})")
        
        # 验证重构误差
        np_error = np.max(np.abs(np_x - np_rec))
        torch_error = torch.max(torch.abs(torch_x - torch_rec)).item()
        
        print(f"\nNSCT 往返误差 (levels={levels}, dfilt={dfilt}, pfilt={pfilt}):")
        print(f"  NumPy 误差: {np_error:.2e}")
        print(f"  Torch 误差: {torch_error:.2e}")
        
        # 两者的重构误差应该接近
        assert abs(np_error - torch_error) < 1e-6, "往返误差不一致"


class TestNsctPerfectReconstruction:
    """测试 NSCT 的完美重构特性"""
    
    @pytest.mark.parametrize("levels", [[2, 3], [3, 3, 4]])
    def test_perfect_reconstruction_property(self, levels, random_seed):
        """测试完美重构特性 - Python 和 Torch 应该有相似的重构误差"""
        # 创建测试输入
        np_x = np.random.randn(128, 128).astype(np.float64)
        torch_x = torch.from_numpy(np_x)
        
        # NumPy 往返
        np_y = np_core.nsctdec(np_x, levels)
        np_rec = np_core.nsctrec(np_y)
        np_error = np.max(np.abs(np_x - np_rec))
        
        # PyTorch 往返
        from nsct_torch.core.nsct import nsctdec, nsctrec
        torch_y = nsctdec(torch_x, levels)
        torch_rec = nsctrec(torch_y)
        torch_error = torch.max(torch.abs(torch_x - torch_rec)).item()
        
        print(f"\n完美重构测试 (levels={levels}):")
        print(f"  NumPy 重构误差: {np_error:.2e}")
        print(f"  Torch 重构误差: {torch_error:.2e}")
        print(f"  误差差异: {abs(np_error - torch_error):.2e}")
        
        # 两者的重构误差应该非常接近
        assert abs(np_error - torch_error) < 1e-6, \
            f"重构误差差异过大: NumPy={np_error:.2e}, Torch={torch_error:.2e}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
