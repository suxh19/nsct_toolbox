"""
测试 nsct_python/utils.py 和 nsct_torch/utils.py 之间的一致性
验证：数值一致性、精度一致性、形状一致性
"""

import pytest
import numpy as np
import torch
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from nsct_python import utils as np_utils
from nsct_torch import utils as torch_utils


class TestUtilsConsistency:
    """测试 numpy 和 torch 版本 utils 的一致性"""
    
    @pytest.fixture
    def tolerance(self):
        """定义数值容差"""
        return {
            'rtol': 1e-6,  # 相对容差
            'atol': 1e-8   # 绝对容差
        }
    
    def numpy_to_torch(self, arr):
        """将 numpy 数组转换为 torch tensor"""
        return torch.from_numpy(arr).float()
    
    def torch_to_numpy(self, tensor):
        """将 torch tensor 转换为 numpy 数组"""
        return tensor.cpu().numpy()
    
    def assert_arrays_equal(self, np_result, torch_result, tolerance, show_values=True):
        """断言两个数组相等（numpy vs torch）"""
        # 形状一致性
        assert np_result.shape == torch_result.shape, \
            f"形状不一致: numpy {np_result.shape} vs torch {torch_result.shape}"
        
        # 数值一致性（转换为 numpy 进行比较）
        torch_np = self.torch_to_numpy(torch_result)
        np.testing.assert_allclose(
            np_result, torch_np,
            rtol=tolerance['rtol'],
            atol=tolerance['atol'],
            err_msg="数值不一致"
        )
        
        # 计算差异
        diff = np.abs(np_result - torch_np)
        max_diff = np.max(diff)
        
        # 数据类型兼容性检查
        print(f"  ✓ 形状一致性: {np_result.shape}")
        print(f"  ✓ numpy dtype: {np_result.dtype}, torch dtype: {torch_result.dtype}")
        print(f"  ✓ 最大数值差异: {max_diff}")
        
        # 显示数组内容对比（小数组完整显示，大数组显示部分）
        if show_values:
            total_elements = np_result.size
            if total_elements <= 100:  # 小数组完整显示
                print("\n  📊 NumPy 输出:")
                print(f"  {np_result}")
                print("\n  📊 PyTorch 输出:")
                print(f"  {torch_np}")
                if max_diff > 0:
                    print("\n  ⚠️  差异矩阵:")
                    print(f"  {diff}")
            else:  # 大数组显示统计信息和部分元素
                print(f"\n  📊 数组统计 (共 {total_elements} 个元素):")
                print(f"     NumPy  - min: {np_result.min():.6f}, max: {np_result.max():.6f}, mean: {np_result.mean():.6f}")
                print(f"     PyTorch - min: {torch_np.min():.6f}, max: {torch_np.max():.6f}, mean: {torch_np.mean():.6f}")
                
                # 显示左上角 3x3 子矩阵
                if len(np_result.shape) == 2:
                    rows, cols = np_result.shape
                    show_rows = min(3, rows)
                    show_cols = min(3, cols)
                    print(f"\n  📊 左上角 {show_rows}x{show_cols} 子矩阵对比:")
                    print("     NumPy:")
                    print(f"     {np_result[:show_rows, :show_cols]}")
                    print("     PyTorch:")
                    print(f"     {torch_np[:show_rows, :show_cols]}")
        
        # 逐元素验证（随机抽样检查）
        if total_elements > 0:
            sample_size = min(10, total_elements)
            sample_indices = np.random.choice(total_elements, sample_size, replace=False)
            flat_np = np_result.flatten()
            flat_torch = torch_np.flatten()
            print(f"\n  🔍 随机抽样验证 ({sample_size} 个位置):")
            all_match = True
            for idx in sample_indices[:5]:  # 只显示前5个
                np_val = flat_np[idx]
                torch_val = flat_torch[idx]
                match = np.isclose(np_val, torch_val, rtol=tolerance['rtol'], atol=tolerance['atol'])
                status = "✓" if match else "✗"
                print(f"     位置 [{idx}]: numpy={np_val:.8f}, torch={torch_val:.8f} {status}")
                if not match:
                    all_match = False
            if all_match and sample_size > 5:
                print(f"     ... (其余 {sample_size - 5} 个位置均匹配)")
    
    # ==================== extend2 测试 ====================
    
    def test_extend2_per_mode(self, tolerance):
        """测试 extend2 函数 - 周期扩展模式"""
        print("\n测试 extend2 (per 模式):")
        x_np = np.arange(16).reshape((4, 4)).astype(np.float32)
        x_torch = self.numpy_to_torch(x_np)
        
        # 执行扩展
        result_np = np_utils.extend2(x_np, 1, 1, 1, 1, 'per')
        result_torch = torch_utils.extend2(x_torch, 1, 1, 1, 1, 'per')
        
        self.assert_arrays_equal(result_np, result_torch, tolerance)
    
    def test_extend2_qper_row_mode(self, tolerance):
        """测试 extend2 函数 - 行方向 quincunx 扩展模式"""
        print("\n测试 extend2 (qper_row 模式):")
        x_np = np.arange(16).reshape((4, 4)).astype(np.float32)
        x_torch = self.numpy_to_torch(x_np)
        
        result_np = np_utils.extend2(x_np, 1, 1, 1, 1, 'qper_row')
        result_torch = torch_utils.extend2(x_torch, 1, 1, 1, 1, 'qper_row')
        
        self.assert_arrays_equal(result_np, result_torch, tolerance)
    
    def test_extend2_qper_col_mode(self, tolerance):
        """测试 extend2 函数 - 列方向 quincunx 扩展模式"""
        print("\n测试 extend2 (qper_col 模式):")
        x_np = np.arange(16).reshape((4, 4)).astype(np.float32)
        x_torch = self.numpy_to_torch(x_np)
        
        result_np = np_utils.extend2(x_np, 1, 1, 1, 1, 'qper_col')
        result_torch = torch_utils.extend2(x_torch, 1, 1, 1, 1, 'qper_col')
        
        self.assert_arrays_equal(result_np, result_torch, tolerance)
    
    # ==================== symext 测试 ====================
    
    def test_symext_basic(self, tolerance):
        """测试 symext 函数 - 基本对称扩展"""
        print("\n测试 symext (基本情况):")
        x_np = np.arange(16).reshape(4, 4).astype(np.float32)
        h_np = np.ones((3, 3)).astype(np.float32)
        shift = [1, 1]
        
        x_torch = self.numpy_to_torch(x_np)
        h_torch = self.numpy_to_torch(h_np)
        
        result_np = np_utils.symext(x_np, h_np, shift)
        result_torch = torch_utils.symext(x_torch, h_torch, shift)
        
        self.assert_arrays_equal(result_np, result_torch, tolerance)
    
    def test_symext_different_shifts(self, tolerance):
        """测试 symext 函数 - 不同的 shift 参数"""
        print("\n测试 symext (不同 shift):")
        x_np = np.arange(20).reshape(4, 5).astype(np.float32)
        h_np = np.ones((5, 3)).astype(np.float32)
        shift = [2, 1]
        
        x_torch = self.numpy_to_torch(x_np)
        h_torch = self.numpy_to_torch(h_np)
        
        result_np = np_utils.symext(x_np, h_np, shift)
        result_torch = torch_utils.symext(x_torch, h_torch, shift)
        
        self.assert_arrays_equal(result_np, result_torch, tolerance)
    
    # ==================== upsample2df 测试 ====================
    
    def test_upsample2df_power1(self, tolerance):
        """测试 upsample2df 函数 - power=1"""
        print("\n测试 upsample2df (power=1):")
        h_np = np.array([[1, 2], [3, 4]]).astype(np.float32)
        h_torch = self.numpy_to_torch(h_np)
        
        result_np = np_utils.upsample2df(h_np, power=1)
        result_torch = torch_utils.upsample2df(h_torch, power=1)
        
        self.assert_arrays_equal(result_np, result_torch, tolerance)
    
    def test_upsample2df_power2(self, tolerance):
        """测试 upsample2df 函数 - power=2"""
        print("\n测试 upsample2df (power=2):")
        h_np = np.array([[1, 2], [3, 4]]).astype(np.float32)
        h_torch = self.numpy_to_torch(h_np)
        
        result_np = np_utils.upsample2df(h_np, power=2)
        result_torch = torch_utils.upsample2df(h_torch, power=2)
        
        self.assert_arrays_equal(result_np, result_torch, tolerance)
    
    # ==================== modulate2 测试 ====================
    
    def test_modulate2_both(self, tolerance):
        """测试 modulate2 函数 - both 模式"""
        print("\n测试 modulate2 (both 模式):")
        x_np = np.ones((3, 4)).astype(np.float32)
        x_torch = self.numpy_to_torch(x_np)
        
        result_np = np_utils.modulate2(x_np, 'b')
        result_torch = torch_utils.modulate2(x_torch, 'b')
        
        self.assert_arrays_equal(result_np.astype(np.float64), 
                                  result_torch, tolerance)
    
    def test_modulate2_row(self, tolerance):
        """测试 modulate2 函数 - row 模式"""
        print("\n测试 modulate2 (row 模式):")
        x_np = np.arange(12).reshape((3, 4)).astype(np.float32)
        x_torch = self.numpy_to_torch(x_np)
        
        result_np = np_utils.modulate2(x_np, 'r')
        result_torch = torch_utils.modulate2(x_torch, 'r')
        
        self.assert_arrays_equal(result_np.astype(np.float64), 
                                  result_torch, tolerance)
    
    def test_modulate2_col(self, tolerance):
        """测试 modulate2 函数 - col 模式"""
        print("\n测试 modulate2 (col 模式):")
        x_np = np.arange(12).reshape((3, 4)).astype(np.float32)
        x_torch = self.numpy_to_torch(x_np)
        
        result_np = np_utils.modulate2(x_np, 'c')
        result_torch = torch_utils.modulate2(x_torch, 'c')
        
        self.assert_arrays_equal(result_np.astype(np.float64), 
                                  result_torch, tolerance)
    
    # ==================== resampz 测试 ====================
    
    def test_resampz_type1(self, tolerance):
        """测试 resampz 函数 - type 1 (垂直剪切)"""
        print("\n测试 resampz (type=1):")
        x_np = np.arange(1, 7).reshape(2, 3).astype(np.float32)
        x_torch = self.numpy_to_torch(x_np)
        
        result_np = np_utils.resampz(x_np, 1, shift=1)
        result_torch = torch_utils.resampz(x_torch, 1, shift=1)
        
        self.assert_arrays_equal(result_np, result_torch, tolerance)
    
    def test_resampz_type2(self, tolerance):
        """测试 resampz 函数 - type 2 (垂直剪切)"""
        print("\n测试 resampz (type=2):")
        x_np = np.arange(1, 7).reshape(2, 3).astype(np.float32)
        x_torch = self.numpy_to_torch(x_np)
        
        result_np = np_utils.resampz(x_np, 2, shift=1)
        result_torch = torch_utils.resampz(x_torch, 2, shift=1)
        
        self.assert_arrays_equal(result_np, result_torch, tolerance)
    
    def test_resampz_type3(self, tolerance):
        """测试 resampz 函数 - type 3 (水平剪切)"""
        print("\n测试 resampz (type=3):")
        x_np = np.arange(1, 7).reshape(2, 3).astype(np.float32)
        x_torch = self.numpy_to_torch(x_np)
        
        result_np = np_utils.resampz(x_np, 3, shift=1)
        result_torch = torch_utils.resampz(x_torch, 3, shift=1)
        
        self.assert_arrays_equal(result_np, result_torch, tolerance)
    
    def test_resampz_type4(self, tolerance):
        """测试 resampz 函数 - type 4 (水平剪切)"""
        print("\n测试 resampz (type=4):")
        x_np = np.arange(1, 7).reshape(2, 3).astype(np.float32)
        x_torch = self.numpy_to_torch(x_np)
        
        result_np = np_utils.resampz(x_np, 4, shift=1)
        result_torch = torch_utils.resampz(x_torch, 4, shift=1)
        
        self.assert_arrays_equal(result_np, result_torch, tolerance)
    
    # ==================== qupz 测试 ====================
    
    def test_qupz_type1(self, tolerance):
        """测试 qupz 函数 - type 1"""
        print("\n测试 qupz (type=1):")
        x_np = np.array([[1, 2], [3, 4]]).astype(np.float32)
        x_torch = self.numpy_to_torch(x_np)
        
        result_np = np_utils.qupz(x_np, 1)
        result_torch = torch_utils.qupz(x_torch, 1)
        
        self.assert_arrays_equal(result_np, result_torch, tolerance)
    
    def test_qupz_type2(self, tolerance):
        """测试 qupz 函数 - type 2"""
        print("\n测试 qupz (type=2):")
        x_np = np.array([[1, 2], [3, 4]]).astype(np.float32)
        x_torch = self.numpy_to_torch(x_np)
        
        result_np = np_utils.qupz(x_np, 2)
        result_torch = torch_utils.qupz(x_torch, 2)
        
        self.assert_arrays_equal(result_np, result_torch, tolerance)
    
    def test_qupz_larger_matrix(self, tolerance):
        """测试 qupz 函数 - 更大的矩阵"""
        print("\n测试 qupz (更大矩阵):")
        x_np = np.arange(1, 13).reshape(3, 4).astype(np.float32)
        x_torch = self.numpy_to_torch(x_np)
        
        result_np = np_utils.qupz(x_np, 1)
        result_torch = torch_utils.qupz(x_torch, 1)
        
        self.assert_arrays_equal(result_np, result_torch, tolerance)


if __name__ == '__main__':
    # 可以直接运行此文件进行快速测试
    pytest.main([__file__, '-v', '-s'])
