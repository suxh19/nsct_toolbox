"""
测试辅助函数 - 用于比较 NumPy 和 PyTorch 输出
"""
import numpy as np
import torch
from typing import Union, Tuple, List


def assert_shape_equal(np_array: np.ndarray, torch_tensor: torch.Tensor, 
                       message: str = "形状不匹配"):
    """
    断言 NumPy 数组和 PyTorch 张量的形状相同
    
    Args:
        np_array: NumPy 数组
        torch_tensor: PyTorch 张量
        message: 错误消息
    """
    np_shape = np_array.shape
    torch_shape = tuple(torch_tensor.cpu().numpy().shape)
    assert np_shape == torch_shape, \
        f"{message}: NumPy shape {np_shape} != Torch shape {torch_shape}"


def assert_values_close(np_array: np.ndarray, torch_tensor: torch.Tensor,
                       rtol: float = 1e-5, atol: float = 1e-8,
                       message: str = "数值不匹配"):
    """
    断言 NumPy 数组和 PyTorch 张量的数值接近
    
    Args:
        np_array: NumPy 数组
        torch_tensor: PyTorch 张量
        rtol: 相对容差
        atol: 绝对容差
        message: 错误消息
    """
    torch_array = torch_tensor.cpu().numpy()
    
    # 检查形状
    assert_shape_equal(np_array, torch_tensor, message)
    
    # 检查数值
    max_diff = np.max(np.abs(np_array - torch_array))
    relative_diff = np.max(np.abs((np_array - torch_array) / (np.abs(np_array) + 1e-10)))
    
    try:
        np.testing.assert_allclose(np_array, torch_array, rtol=rtol, atol=atol,
                                   err_msg=f"{message}\n最大绝对差异: {max_diff:.2e}\n最大相对差异: {relative_diff:.2e}")
    except AssertionError as e:
        # 提供更详细的错误信息
        print(f"\n{message}")
        print(f"NumPy 统计: min={np_array.min():.6f}, max={np_array.max():.6f}, mean={np_array.mean():.6f}")
        print(f"Torch 统计: min={torch_array.min():.6f}, max={torch_array.max():.6f}, mean={torch_array.mean():.6f}")
        print(f"最大绝对差异: {max_diff:.2e}")
        print(f"最大相对差异: {relative_diff:.2e}")
        
        # 找出差异最大的位置
        diff = np.abs(np_array - torch_array)
        max_diff_idx = np.unravel_index(np.argmax(diff), diff.shape)
        print(f"最大差异位置: {max_diff_idx}")
        print(f"  NumPy 值: {np_array[max_diff_idx]:.10f}")
        print(f"  Torch 值: {torch_array[max_diff_idx]:.10f}")
        print(f"  差异: {diff[max_diff_idx]:.10e}")
        raise


def assert_elementwise_equal(np_array: np.ndarray, torch_tensor: torch.Tensor,
                             message: str = "逐元素比较失败"):
    """
    断言 NumPy 数组和 PyTorch 张量逐元素完全相等
    
    Args:
        np_array: NumPy 数组
        torch_tensor: PyTorch 张量
        message: 错误消息
    """
    torch_array = torch_tensor.cpu().numpy()
    
    # 检查形状
    assert_shape_equal(np_array, torch_tensor, message)
    
    # 逐元素比较
    diff = np_array != torch_array
    if np.any(diff):
        num_diff = np.sum(diff)
        total = np_array.size
        print(f"\n{message}")
        print(f"不同元素数量: {num_diff}/{total} ({100*num_diff/total:.2f}%)")
        
        # 显示前几个不同的元素
        diff_indices = np.argwhere(diff)[:5]
        for idx in diff_indices:
            idx_tuple = tuple(idx)
            print(f"位置 {idx_tuple}: NumPy={np_array[idx_tuple]}, Torch={torch_array[idx_tuple]}")
        
        raise AssertionError(f"{message}: {num_diff} 个元素不同")


def compute_statistics(np_array: np.ndarray, torch_tensor: torch.Tensor) -> dict:
    """
    计算 NumPy 和 PyTorch 输出的统计信息
    
    Args:
        np_array: NumPy 数组
        torch_tensor: PyTorch 张量
        
    Returns:
        包含统计信息的字典
    """
    torch_array = torch_tensor.cpu().numpy()
    
    diff = np.abs(np_array - torch_array)
    relative_diff = np.abs((np_array - torch_array) / (np.abs(np_array) + 1e-10))
    
    stats = {
        'shape_match': np_array.shape == torch_array.shape,
        'numpy_shape': np_array.shape,
        'torch_shape': torch_array.shape,
        'numpy_min': float(np_array.min()),
        'numpy_max': float(np_array.max()),
        'numpy_mean': float(np_array.mean()),
        'numpy_std': float(np_array.std()),
        'torch_min': float(torch_array.min()),
        'torch_max': float(torch_array.max()),
        'torch_mean': float(torch_array.mean()),
        'torch_std': float(torch_array.std()),
        'max_abs_diff': float(diff.max()),
        'mean_abs_diff': float(diff.mean()),
        'max_rel_diff': float(relative_diff.max()),
        'mean_rel_diff': float(relative_diff.mean()),
    }
    
    return stats


def print_comparison_report(name: str, np_array: np.ndarray, torch_tensor: torch.Tensor):
    """
    打印详细的比较报告
    
    Args:
        name: 测试名称
        np_array: NumPy 数组
        torch_tensor: PyTorch 张量
    """
    stats = compute_statistics(np_array, torch_tensor)
    
    print(f"\n{'='*60}")
    print(f"比较报告: {name}")
    print(f"{'='*60}")
    print(f"形状匹配: {stats['shape_match']}")
    print(f"NumPy 形状: {stats['numpy_shape']}")
    print(f"Torch 形状: {stats['torch_shape']}")
    print(f"\nNumPy 统计:")
    print(f"  min={stats['numpy_min']:.6f}, max={stats['numpy_max']:.6f}")
    print(f"  mean={stats['numpy_mean']:.6f}, std={stats['numpy_std']:.6f}")
    print(f"\nTorch 统计:")
    print(f"  min={stats['torch_min']:.6f}, max={stats['torch_max']:.6f}")
    print(f"  mean={stats['torch_mean']:.6f}, std={stats['torch_std']:.6f}")
    print(f"\n差异统计:")
    print(f"  最大绝对差异: {stats['max_abs_diff']:.2e}")
    print(f"  平均绝对差异: {stats['mean_abs_diff']:.2e}")
    print(f"  最大相对差异: {stats['max_rel_diff']:.2e}")
    print(f"  平均相对差异: {stats['mean_rel_diff']:.2e}")
    print(f"{'='*60}\n")


def assert_list_values_close(np_list: List[np.ndarray], 
                             torch_list: List[torch.Tensor],
                             rtol: float = 1e-5, atol: float = 1e-8,
                             message: str = "列表数值不匹配"):
    """
    断言 NumPy 数组列表和 PyTorch 张量列表的数值接近
    
    Args:
        np_list: NumPy 数组列表
        torch_list: PyTorch 张量列表
        rtol: 相对容差
        atol: 绝对容差
        message: 错误消息
    """
    assert len(np_list) == len(torch_list), \
        f"{message}: 列表长度不匹配 (NumPy: {len(np_list)}, Torch: {len(torch_list)})"
    
    for i, (np_arr, torch_tensor) in enumerate(zip(np_list, torch_list)):
        assert_values_close(np_arr, torch_tensor, rtol, atol, 
                          f"{message} [索引 {i}]")
