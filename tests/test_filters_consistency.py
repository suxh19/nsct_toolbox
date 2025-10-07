"""
测试 nsct_python/filters.py 和 nsct_torch/filters.py 之间的一致性
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

from nsct_python import filters as np_filters
from nsct_torch import filters as torch_filters


class TestFiltersConsistency:
    """测试 numpy 和 torch 版本 filters 的一致性"""
    
    @pytest.fixture
    def tolerance(self):
        """定义数值容差"""
        return {
            'rtol': 1e-5,  # 相对容差（filters 计算更复杂，放宽一点）
            'atol': 1e-7   # 绝对容差
        }
    
    def numpy_to_torch(self, arr):
        """将 numpy 数组转换为 torch tensor"""
        return torch.from_numpy(arr)
    
    def torch_to_numpy(self, tensor):
        """将 torch tensor 转换为 numpy 数组"""
        return tensor.cpu().numpy()
    
    def assert_arrays_equal(self, np_result, torch_result, tolerance, show_values=False):
        """断言两个数组相等（numpy vs torch），包含逐元素一致性检查"""
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
        
        # 逐元素一致性分析
        diff = np.abs(np_result - torch_np)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        median_diff = np.median(diff)
        
        # 计算相对误差
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_diff = np.abs(np_result - torch_np) / (np.abs(np_result) + 1e-10)
            max_rel_diff = np.max(rel_diff)
            mean_rel_diff = np.mean(rel_diff)
        
        # 统计超出容差的元素
        atol_violations = np.sum(diff > tolerance['atol'])
        rtol_violations = np.sum(rel_diff > tolerance['rtol'])
        total_elements = np_result.size
        
        print(f"  ✓ 形状一致性: {np_result.shape}")
        print(f"  ✓ numpy dtype: {np_result.dtype}, torch dtype: {torch_result.dtype}")
        print(f"  ✓ 逐元素一致性分析:")
        print(f"    - 总元素数: {total_elements}")
        print(f"    - 最大绝对差异: {max_diff:.2e}")
        print(f"    - 平均绝对差异: {mean_diff:.2e}")
        print(f"    - 中位数绝对差异: {median_diff:.2e}")
        print(f"    - 最大相对差异: {max_rel_diff:.2e}")
        print(f"    - 平均相对差异: {mean_rel_diff:.2e}")
        print(f"    - 超出绝对容差 ({tolerance['atol']:.2e}) 的元素: {atol_violations} ({100*atol_violations/total_elements:.2f}%)")
        print(f"    - 超出相对容差 ({tolerance['rtol']:.2e}) 的元素: {rtol_violations} ({100*rtol_violations/total_elements:.2f}%)")
        
        # 如果有不一致的元素，显示详细信息
        if atol_violations > 0 and np_result.size <= 100:
            print(f"  ⚠️  不一致元素位置:")
            inconsistent_indices = np.where(diff > tolerance['atol'])
            for idx in zip(*inconsistent_indices):
                if len(idx) == 1:
                    i = idx[0]
                    print(f"    位置 [{i}]: numpy={np_result.flat[i]:.6e}, torch={torch_np.flat[i]:.6e}, diff={diff.flat[i]:.6e}")
                elif len(idx) == 2:
                    i, j = idx
                    print(f"    位置 [{i},{j}]: numpy={np_result[i,j]:.6e}, torch={torch_np[i,j]:.6e}, diff={diff[i,j]:.6e}")
        
        if show_values and np_result.size <= 50:
            print(f"\n  📊 NumPy 输出:\n  {np_result}")
            print(f"\n  📊 PyTorch 输出:\n  {torch_np}")
            print(f"\n  📊 逐元素差异:\n  {diff}")
    
    # ==================== ldfilter 测试 ====================
    
    def test_ldfilter_pkva6(self, tolerance):
        """测试 ldfilter 函数 - pkva6"""
        print("\n测试 ldfilter (pkva6):")
        
        result_np = np_filters.ldfilter('pkva6')
        result_torch = torch_filters.ldfilter('pkva6')
        
        self.assert_arrays_equal(result_np, result_torch, tolerance, show_values=True)
    
    def test_ldfilter_pkva8(self, tolerance):
        """测试 ldfilter 函数 - pkva8"""
        print("\n测试 ldfilter (pkva8):")
        
        result_np = np_filters.ldfilter('pkva8')
        result_torch = torch_filters.ldfilter('pkva8')
        
        self.assert_arrays_equal(result_np, result_torch, tolerance)
    
    # ==================== dmaxflat 测试 ====================
    
    def test_dmaxflat_N2(self, tolerance):
        """测试 dmaxflat 函数 - N=2"""
        print("\n测试 dmaxflat (N=2, d=0):")
        
        result_np = np_filters.dmaxflat(2, 0)
        result_torch = torch_filters.dmaxflat(2, 0)
        
        self.assert_arrays_equal(result_np, result_torch, tolerance)
    
    def test_dmaxflat_N3(self, tolerance):
        """测试 dmaxflat 函数 - N=3"""
        print("\n测试 dmaxflat (N=3, d=1):")
        
        result_np = np_filters.dmaxflat(3, 1)
        result_torch = torch_filters.dmaxflat(3, 1)
        
        self.assert_arrays_equal(result_np, result_torch, tolerance)
    
    def test_dmaxflat_N5(self, tolerance):
        """测试 dmaxflat 函数 - N=5"""
        print("\n测试 dmaxflat (N=5, d=0):")
        
        result_np = np_filters.dmaxflat(5, 0)
        result_torch = torch_filters.dmaxflat(5, 0)
        
        self.assert_arrays_equal(result_np, result_torch, tolerance)
    
    # ==================== atrousfilters 测试 ====================
    
    def test_atrousfilters_pyr(self, tolerance):
        """测试 atrousfilters 函数 - pyr"""
        print("\n测试 atrousfilters (pyr):")
        
        h0_np, h1_np, g0_np, g1_np = np_filters.atrousfilters('pyr')
        h0_torch, h1_torch, g0_torch, g1_torch = torch_filters.atrousfilters('pyr')
        
        print("  h0:")
        self.assert_arrays_equal(h0_np, h0_torch, tolerance)
        print("  h1:")
        self.assert_arrays_equal(h1_np, h1_torch, tolerance)
        print("  g0:")
        self.assert_arrays_equal(g0_np, g0_torch, tolerance)
        print("  g1:")
        self.assert_arrays_equal(g1_np, g1_torch, tolerance)
    
    def test_atrousfilters_pyrexc(self, tolerance):
        """测试 atrousfilters 函数 - pyrexc"""
        print("\n测试 atrousfilters (pyrexc):")
        
        h0_np, h1_np, g0_np, g1_np = np_filters.atrousfilters('pyrexc')
        h0_torch, h1_torch, g0_torch, g1_torch = torch_filters.atrousfilters('pyrexc')
        
        print("  h0:")
        self.assert_arrays_equal(h0_np, h0_torch, tolerance)
        print("  h1:")
        self.assert_arrays_equal(h1_np, h1_torch, tolerance)
        print("  g0:")
        self.assert_arrays_equal(g0_np, g0_torch, tolerance)
        print("  g1:")
        self.assert_arrays_equal(g1_np, g1_torch, tolerance)
    
    # ==================== mctrans 测试 ====================
    
    def test_mctrans_basic(self, tolerance):
        """测试 mctrans 函数 - 基本情况"""
        print("\n测试 mctrans (基本情况):")
        
        b_np = np.array([1, 2, 1]) / 4.0
        t_np = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / 4.0
        
        b_torch = self.numpy_to_torch(b_np)
        t_torch = self.numpy_to_torch(t_np)
        
        result_np = np_filters.mctrans(b_np, t_np)
        result_torch = torch_filters.mctrans(b_torch, t_torch)
        
        self.assert_arrays_equal(result_np, result_torch, tolerance, show_values=True)
    
    # ==================== ld2quin 测试 ====================
    
    def test_ld2quin_pkva6(self, tolerance):
        """测试 ld2quin 函数 - pkva6"""
        print("\n测试 ld2quin (pkva6):")
        
        beta_np = np_filters.ldfilter('pkva6')
        beta_torch = torch_filters.ldfilter('pkva6')
        
        h0_np, h1_np = np_filters.ld2quin(beta_np)
        h0_torch, h1_torch = torch_filters.ld2quin(beta_torch)
        
        print("  h0:")
        self.assert_arrays_equal(h0_np, h0_torch, tolerance)
        print("  h1:")
        self.assert_arrays_equal(h1_np, h1_torch, tolerance)
    
    # ==================== efilter2 测试 ====================
    
    def test_efilter2_basic(self, tolerance):
        """测试 efilter2 函数 - 基本情况"""
        print("\n测试 efilter2 (基本情况):")
        
        x_np = np.arange(9).reshape(3, 3).astype(np.float32)
        f_np = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]).astype(np.float32)
        
        x_torch = self.numpy_to_torch(x_np)
        f_torch = self.numpy_to_torch(f_np)
        
        result_np = np_filters.efilter2(x_np, f_np, 'per')
        result_torch = torch_filters.efilter2(x_torch, f_torch, 'per')
        
        self.assert_arrays_equal(result_np, result_torch, tolerance, show_values=True)
    
    def test_efilter2_larger(self, tolerance):
        """测试 efilter2 函数 - 更大矩阵"""
        print("\n测试 efilter2 (更大矩阵):")
        
        x_np = np.random.randn(16, 16).astype(np.float32)
        f_np = np.ones((5, 5)).astype(np.float32) / 25.0
        
        x_torch = self.numpy_to_torch(x_np)
        f_torch = self.numpy_to_torch(f_np)
        
        result_np = np_filters.efilter2(x_np, f_np, 'per')
        result_torch = torch_filters.efilter2(x_torch, f_torch, 'per')
        
        self.assert_arrays_equal(result_np, result_torch, tolerance)
    
    # ==================== dfilters 测试 ====================
    
    def test_dfilters_pkva(self, tolerance):
        """测试 dfilters 函数 - pkva"""
        print("\n测试 dfilters (pkva):")
        
        h0_np, h1_np = np_filters.dfilters('pkva', 'd')
        h0_torch, h1_torch = torch_filters.dfilters('pkva', 'd')
        
        print("  h0:")
        self.assert_arrays_equal(h0_np, h0_torch, tolerance)
        print("  h1:")
        self.assert_arrays_equal(h1_np, h1_torch, tolerance)
    
    def test_dfilters_dmaxflat3(self, tolerance):
        """测试 dfilters 函数 - dmaxflat3"""
        print("\n测试 dfilters (dmaxflat3):")
        
        h0_np, h1_np = np_filters.dfilters('dmaxflat3', 'd')
        h0_torch, h1_torch = torch_filters.dfilters('dmaxflat3', 'd')
        
        print("  h0:")
        self.assert_arrays_equal(h0_np, h0_torch, tolerance)
        print("  h1:")
        self.assert_arrays_equal(h1_np, h1_torch, tolerance)
    
    # ==================== parafilters 测试 ====================
    
    def test_parafilters_basic(self, tolerance):
        """测试 parafilters 函数 - 基本情况"""
        print("\n测试 parafilters (基本情况):")
        
        f1_np = np.ones((3, 3)).astype(np.float64)
        f2_np = np.ones((3, 3)).astype(np.float64) * 2
        
        f1_torch = self.numpy_to_torch(f1_np)
        f2_torch = self.numpy_to_torch(f2_np)
        
        y1_np, y2_np = np_filters.parafilters(f1_np, f2_np)
        y1_torch, y2_torch = torch_filters.parafilters(f1_torch, f2_torch)
        
        # 检查列表长度
        assert len(y1_np) == len(y1_torch) == 4
        assert len(y2_np) == len(y2_torch) == 4
        
        # 检查每个滤波器
        for i in range(4):
            print(f"  y1[{i}]:")
            self.assert_arrays_equal(y1_np[i], y1_torch[i], tolerance)
            print(f"  y2[{i}]:")
            self.assert_arrays_equal(y2_np[i], y2_torch[i], tolerance)


if __name__ == '__main__':
    # 可以直接运行此文件进行快速测试
    pytest.main([__file__, '-v', '-s'])
