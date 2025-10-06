"""
基准测试脚本 - C++ vs Python 实现性能对比测试

这个脚本会:
1. 验证 C++ 和 Python 实现的正确性
2. 对比 C++ 和 Python 实现的数值精度
3. 测试不同图像大小下的性能
4. 测试不同滤波器大小的性能
5. 详细分析输出大小、数值范围等
"""

import numpy as np
import time
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from __init__ import atrousc, is_cpp_available, get_backend_info
except ImportError:
    print("错误: 无法导入 atrousc_cpp 模块")
    print("请先编译 C++ 扩展:")
    print("  cd nsct_python/atrousc_cpp")
    print("  python setup.py build_ext --inplace")
    sys.exit(1)

# 导入纯 Python 实现
try:
    from atrousc import _atrousc_equivalent as atrousc_python
    PYTHON_AVAILABLE = True
except ImportError:
    print("警告: 无法导入纯 Python 实现")
    PYTHON_AVAILABLE = False


def compare_arrays_detailed(arr1, arr2, name1="Array 1", name2="Array 2"):
    """
    详细对比两个数组的数值一致性
    
    Args:
        arr1: 第一个数组
        arr2: 第二个数组
        name1: 第一个数组的名称
        name2: 第二个数组的名称
    
    Returns:
        dict: 包含详细对比信息的字典
    """
    comparison = {
        'shape_match': arr1.shape == arr2.shape,
        'exact_match': False,
        'allclose_default': False,
        'allclose_strict': False,
        'max_abs_diff': None,
        'mean_abs_diff': None,
        'median_abs_diff': None,
        'std_abs_diff': None,
        'relative_error': None,
        'nonzero_diff_count': 0,
        'nonzero_diff_ratio': 0.0,
    }
    
    if not comparison['shape_match']:
        print(f"  警告: 形状不匹配! {name1}: {arr1.shape}, {name2}: {arr2.shape}")
        return comparison
    
    # 基本相等性检查
    comparison['exact_match'] = np.array_equal(arr1, arr2)
    comparison['allclose_default'] = np.allclose(arr1, arr2)
    comparison['allclose_strict'] = np.allclose(arr1, arr2, rtol=1e-15, atol=1e-15)
    
    # 计算差异
    diff = arr1 - arr2
    abs_diff = np.abs(diff)
    
    comparison['max_abs_diff'] = np.max(abs_diff)
    comparison['mean_abs_diff'] = np.mean(abs_diff)
    comparison['median_abs_diff'] = np.median(abs_diff)
    comparison['std_abs_diff'] = np.std(abs_diff)
    
    # 相对误差
    max_val = max(np.max(np.abs(arr1)), np.max(np.abs(arr2)))
    if max_val > 0:
        comparison['relative_error'] = comparison['max_abs_diff'] / max_val
    else:
        comparison['relative_error'] = 0.0
    
    # 非零差异统计
    nonzero_mask = abs_diff > 0
    comparison['nonzero_diff_count'] = np.sum(nonzero_mask)
    comparison['nonzero_diff_ratio'] = comparison['nonzero_diff_count'] / arr1.size
    
    return comparison


def print_array_comparison(comparison):
    """打印数组对比结果"""
    print(f"    形状匹配: {'✓' if comparison['shape_match'] else '✗'}")
    print(f"    逐元素完全相等: {'✓' if comparison['exact_match'] else '✗'}")
    print(f"    allclose(默认 rtol=1e-5, atol=1e-8): {'✓' if comparison['allclose_default'] else '✗'}")
    print(f"    allclose(严格 rtol=1e-15, atol=1e-15): {'✓' if comparison['allclose_strict'] else '✗'}")
    
    if comparison['max_abs_diff'] is not None:
        print(f"    最大绝对误差: {comparison['max_abs_diff']:.6e}")
        print(f"    平均绝对误差: {comparison['mean_abs_diff']:.6e}")
        print(f"    中位数绝对误差: {comparison['median_abs_diff']:.6e}")
        print(f"    标准差绝对误差: {comparison['std_abs_diff']:.6e}")
        print(f"    相对误差: {comparison['relative_error']:.6e}")
        
        if comparison['nonzero_diff_count'] > 0:
            print(f"    非零误差点数: {comparison['nonzero_diff_count']}")
            print(f"    非零误差比例: {comparison['nonzero_diff_ratio'] * 100:.4f}%")


def test_correctness():
    """测试 C++ 实现的基本功能"""
    print("=" * 70)
    print("功能测试")
    print("=" * 70)
    
    # 测试用例
    test_cases = [
        {"size": (64, 64), "filter": (3, 3), "M": 2, "name": "小图像+小滤波器"},
        {"size": (128, 128), "filter": (5, 5), "M": 4, "name": "中等图像+中等滤波器"},
        {"size": (256, 256), "filter": (7, 7), "M": 8, "name": "大图像+大滤波器"},
    ]
    
    all_passed = True
    
    for case in test_cases:
        x = np.random.rand(*case["size"])
        h = np.random.rand(*case["filter"])
        M = np.array([[case["M"], 0], [0, case["M"]]])
        
        try:
            # C++ 实现
            result = atrousc(x, h, M)
            
            # 验证输出形状是否正确
            expected_rows = x.shape[0] - case["M"] * h.shape[0] + 1
            expected_cols = x.shape[1] - case["M"] * h.shape[1] + 1
            
            passed = (result.shape == (expected_rows, expected_cols))
            status = "✓ 通过" if passed else "✗ 失败"
            
            print(f"\n{case['name']}: {status}")
            print(f"  图像大小: {case['size']}")
            print(f"  滤波器: {case['filter']}, M={case['M']}")
            print(f"  输出形状: {result.shape}")
            print(f"  预期形状: ({expected_rows}, {expected_cols})")
            
            if not passed:
                all_passed = False
                
        except Exception as e:
            print(f"\n{case['name']}: ✗ 失败")
            print(f"  错误: {e}")
            all_passed = False
    
    return all_passed


def test_numerical_accuracy():
    """测试 C++ 和 Python 实现的数值精度对比"""
    if not PYTHON_AVAILABLE:
        print("\n纯 Python 实现不可用，跳过数值精度测试")
        return True
    
    print("\n" + "=" * 70)
    print("数值精度对比测试 (C++ vs Python)")
    print("=" * 70)
    
    test_cases = [
        {"size": (32, 32), "filter": (3, 3), "M": 2, "name": "小测试"},
        {"size": (64, 64), "filter": (5, 5), "M": 4, "name": "中等测试"},
        {"size": (128, 128), "filter": (7, 7), "M": 8, "name": "大测试"},
    ]
    
    all_passed = True
    
    for case in test_cases:
        # 使用固定种子以确保可重复性
        np.random.seed(42)
        x = np.random.rand(*case["size"])
        h = np.random.rand(*case["filter"])
        M = np.array([[case["M"], 0], [0, case["M"]]])
        
        try:
            # C++ 实现
            result_cpp = atrousc(x, h, M)
            
            # Python 实现
            result_py = atrousc_python(x, h, M)
            
            # 计算差异
            diff_array = result_cpp - result_py
            abs_diff_array = np.abs(diff_array)
            max_diff = np.max(abs_diff_array)
            mean_diff = np.mean(abs_diff_array)
            rel_diff = max_diff / (np.max(np.abs(result_cpp)) + 1e-10)
            
            # 数值统计
            print(f"\n{case['name']}:")
            print(f"  输入大小: {case['size']}, 滤波器: {case['filter']}, M={case['M']}")
            print(f"  输出大小: {result_cpp.shape}")
            print(f"\n  C++ 结果统计:")
            print(f"    最小值: {np.min(result_cpp):.6e}")
            print(f"    最大值: {np.max(result_cpp):.6e}")
            print(f"    均值:   {np.mean(result_cpp):.6e}")
            print(f"    标准差: {np.std(result_cpp):.6e}")
            print(f"\n  Python 结果统计:")
            print(f"    最小值: {np.min(result_py):.6e}")
            print(f"    最大值: {np.max(result_py):.6e}")
            print(f"    均值:   {np.mean(result_py):.6e}")
            print(f"    标准差: {np.std(result_py):.6e}")
            print(f"\n  误差分析:")
            print(f"    最大绝对误差: {max_diff:.6e}")
            print(f"    平均绝对误差: {mean_diff:.6e}")
            print(f"    相对误差:     {rel_diff:.6e}")
            
            # 额外的数值一致性检查
            print(f"\n  数值一致性验证:")
            
            # 1. 逐元素相等检查
            exact_match = np.array_equal(result_cpp, result_py)
            print(f"    逐元素完全相等: {'✓ 是' if exact_match else '✗ 否'}")
            
            # 2. 使用 allclose 检查
            allclose_default = np.allclose(result_cpp, result_py)
            allclose_strict = np.allclose(result_cpp, result_py, rtol=1e-15, atol=1e-15)
            print(f"    allclose(默认): {'✓ 是' if allclose_default else '✗ 否'}")
            print(f"    allclose(严格): {'✓ 是' if allclose_strict else '✗ 否'}")
            
            # 3. 差异分布统计
            if max_diff > 0:
                nonzero_diff = abs_diff_array[abs_diff_array > 0]
                if len(nonzero_diff) > 0:
                    print(f"    非零误差点数: {len(nonzero_diff)} / {result_cpp.size}")
                    print(f"    非零误差比例: {len(nonzero_diff) / result_cpp.size * 100:.2f}%")
                    print(f"    非零误差均值: {np.mean(nonzero_diff):.6e}")
            
            # 4. 极值位置对比
            cpp_min_idx = np.unravel_index(np.argmin(result_cpp), result_cpp.shape)
            cpp_max_idx = np.unravel_index(np.argmax(result_cpp), result_cpp.shape)
            py_min_idx = np.unravel_index(np.argmin(result_py), result_py.shape)
            py_max_idx = np.unravel_index(np.argmax(result_py), result_py.shape)
            
            print(f"\n  极值位置对比:")
            print(f"    C++ 最小值位置: {cpp_min_idx}, 值: {result_cpp[cpp_min_idx]:.6e}")
            print(f"    Python 最小值位置: {py_min_idx}, 值: {result_py[py_min_idx]:.6e}")
            print(f"    最小值位置一致: {'✓ 是' if cpp_min_idx == py_min_idx else '✗ 否'}")
            print(f"    C++ 最大值位置: {cpp_max_idx}, 值: {result_cpp[cpp_max_idx]:.6e}")
            print(f"    Python 最大值位置: {py_max_idx}, 值: {result_py[py_max_idx]:.6e}")
            print(f"    最大值位置一致: {'✓ 是' if cpp_max_idx == py_max_idx else '✗ 否'}")
            
            # 判断是否通过（相对误差小于 1e-10）
            passed = rel_diff < 1e-10 and allclose_strict
            status = "✓ 通过" if passed else "✗ 失败"
            print(f"\n  最终结果: {status}")
            
            if not passed:
                all_passed = False
                print(f"  警告: 数值精度不符合预期！")
                
        except Exception as e:
            print(f"\n{case['name']}: ✗ 失败")
            print(f"  错误: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    return all_passed


def test_array_values_consistency():
    """详细验证 C++ 和 Python 输出数组的数值一致性"""
    if not PYTHON_AVAILABLE:
        print("\n纯 Python 实现不可用，跳过数组数值一致性测试")
        return True
    
    print("\n" + "=" * 70)
    print("数组数值一致性详细验证")
    print("=" * 70)
    
    # 使用多组不同的随机种子测试
    test_configs = [
        {"size": (50, 50), "filter": (3, 3), "M": 2, "seed": 12345, "name": "测试 1 (seed=12345)"},
        {"size": (80, 80), "filter": (5, 5), "M": 3, "seed": 67890, "name": "测试 2 (seed=67890)"},
        {"size": (100, 100), "filter": (7, 7), "M": 4, "seed": 54321, "name": "测试 3 (seed=54321)"},
    ]
    
    all_passed = True
    
    for config in test_configs:
        print(f"\n{'=' * 70}")
        print(f"{config['name']}")
        print(f"{'=' * 70}")
        
        # 设置随机种子
        np.random.seed(config['seed'])
        x = np.random.rand(*config["size"])
        h = np.random.rand(*config["filter"])
        M = np.array([[config["M"], 0], [0, config["M"]]])
        
        print(f"配置: 输入={config['size']}, 滤波器={config['filter']}, M={config['M']}")
        
        try:
            # 执行 C++ 实现
            result_cpp = atrousc(x, h, M)
            
            # 执行 Python 实现
            result_py = atrousc_python(x, h, M)
            
            print(f"输出大小: {result_cpp.shape}")
            
            # 详细对比
            print("\n  详细数值对比:")
            comparison = compare_arrays_detailed(result_cpp, result_py, "C++", "Python")
            print_array_comparison(comparison)
            
            # 采样对比 - 显示几个具体位置的数值
            print("\n  采样点数值对比 (前5个位置):")
            sample_positions = [
                (0, 0),
                (result_cpp.shape[0]//4, result_cpp.shape[1]//4),
                (result_cpp.shape[0]//2, result_cpp.shape[1]//2),
                (result_cpp.shape[0]*3//4, result_cpp.shape[1]*3//4),
                (result_cpp.shape[0]-1, result_cpp.shape[1]-1),
            ]
            
            print(f"    {'位置':<20} {'C++ 值':<20} {'Python 值':<20} {'差异':<15}")
            print(f"    {'-'*75}")
            for pos in sample_positions:
                cpp_val = result_cpp[pos]
                py_val = result_py[pos]
                diff = abs(cpp_val - py_val)
                print(f"    {str(pos):<20} {cpp_val:<20.10e} {py_val:<20.10e} {diff:<15.6e}")
            
            # 统计分布对比
            print("\n  统计分布对比:")
            percentiles = [0, 25, 50, 75, 100]
            cpp_percentiles = np.percentile(result_cpp, percentiles)
            py_percentiles = np.percentile(result_py, percentiles)
            
            print(f"    {'百分位':<15} {'C++':<20} {'Python':<20} {'差异':<15}")
            print(f"    {'-'*70}")
            for i, p in enumerate(percentiles):
                diff = abs(cpp_percentiles[i] - py_percentiles[i])
                print(f"    {p}%{'':<12} {cpp_percentiles[i]:<20.10e} {py_percentiles[i]:<20.10e} {diff:<15.6e}")
            
            # 判断是否通过
            passed = comparison['exact_match'] or (
                comparison['allclose_strict'] and 
                comparison['relative_error'] < 1e-10
            )
            
            print(f"\n  测试结果: {'✓ 通过' if passed else '✗ 失败'}")
            
            if not passed:
                all_passed = False
                print("  警告: 数组数值不一致！")
                
                # 如果失败，显示最大差异位置
                diff_array = np.abs(result_cpp - result_py)
                max_diff_pos = np.unravel_index(np.argmax(diff_array), diff_array.shape)
                print(f"  最大差异位置: {max_diff_pos}")
                print(f"    C++ 值: {result_cpp[max_diff_pos]:.15e}")
                print(f"    Python 值: {result_py[max_diff_pos]:.15e}")
                print(f"    差异: {diff_array[max_diff_pos]:.15e}")
            
        except Exception as e:
            print(f"\n  ✗ 测试失败")
            print(f"  错误: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    return all_passed


def test_output_size_analysis():
    """详细测试输出大小计算"""
    print("\n" + "=" * 70)
    print("输出大小分析")
    print("=" * 70)
    
    test_cases = [
        {"size": (100, 100), "filter": (3, 3), "M": 2},
        {"size": (100, 100), "filter": (5, 5), "M": 3},
        {"size": (200, 150), "filter": (7, 7), "M": 4},
        {"size": (256, 256), "filter": (9, 9), "M": 8},
    ]
    
    print(f"\n{'输入大小':<15} {'滤波器':<12} {'M':<6} {'输出大小':<15} {'公式验证'}")
    print("-" * 70)
    
    for case in test_cases:
        x = np.random.rand(*case["size"])
        h = np.random.rand(*case["filter"])
        M = np.array([[case["M"], 0], [0, case["M"]]])
        
        result = atrousc(x, h, M)
        
        # 验证公式: out_size = in_size - M * filter_size + 1
        expected_rows = x.shape[0] - case["M"] * h.shape[0] + 1
        expected_cols = x.shape[1] - case["M"] * h.shape[1] + 1
        
        match = (result.shape[0] == expected_rows and result.shape[1] == expected_cols)
        status = "✓" if match else "✗"
        
        print(f"{str(case['size']):<15} {str(case['filter']):<12} {case['M']:<6} "
              f"{str(result.shape):<15} {status}")
    
    print("\n公式: output_size = input_size - M × filter_size + 1")


def benchmark_cpp_vs_python():
    """性能对比：C++ vs Python"""
    if not PYTHON_AVAILABLE:
        print("\n纯 Python 实现不可用，跳过性能对比测试")
        return
    
    print("\n" + "=" * 70)
    print("性能对比测试 (C++ vs Python)")
    print("=" * 70)
    
    test_sizes = [
        (64, 64),
        (128, 128),
        (256, 256),
        (512, 512),
    ]
    
    h = np.random.rand(7, 7)
    M = np.array([[4, 0], [0, 4]])
    
    print(f"\n{'图像大小':<15} {'C++ (ms)':<15} {'Python (ms)':<15} {'加速比':<10}")
    print("-" * 60)
    
    for size in test_sizes:
        x = np.random.rand(*size)
        
        # C++ 实现（多次运行取平均）
        times_cpp = []
        for _ in range(5):
            start = time.time()
            result_cpp = atrousc(x, h, M)
            times_cpp.append((time.time() - start) * 1000)
        time_cpp = np.mean(times_cpp)
        
        # Python 实现（多次运行取平均）
        times_py = []
        for _ in range(5):
            start = time.time()
            result_py = atrousc_python(x, h, M)
            times_py.append((time.time() - start) * 1000)
        time_py = np.mean(times_py)
        
        speedup = time_py / time_cpp if time_cpp > 0 else float('inf')
        
        print(f"{str(size):<15} {time_cpp:<15.2f} {time_py:<15.2f} {speedup:<10.1f}x")


def benchmark_image_sizes():
    """测试不同图像大小的性能"""
    print("\n" + "=" * 70)
    print("图像大小性能基准测试 (C++ 实现)")
    print("=" * 70)
    
    sizes = [(128, 128), (256, 256), (512, 512), (1024, 1024)]
    h = np.random.rand(7, 7)
    M = np.array([[4, 0], [0, 4]])
    
    print(f"\n{'图像大小':<15} {'输出大小':<15} {'时间 (ms)':<15} {'内存 (MB)':<15}")
    print("-" * 65)
    
    for size in sizes:
        x = np.random.rand(*size)
        
        # C++ 实现
        start = time.time()
        result = atrousc(x, h, M)
        time_cpp = (time.time() - start) * 1000
        
        # 计算内存使用
        memory_mb = (x.nbytes + h.nbytes + result.nbytes) / (1024 * 1024)
        
        print(f"{str(size):<15} {str(result.shape):<15} {time_cpp:<15.2f} {memory_mb:<15.2f}")


def benchmark_filter_sizes():
    """测试不同滤波器大小的性能"""
    print("\n" + "=" * 70)
    print("滤波器大小性能基准测试 (C++ 实现)")
    print("=" * 70)
    
    x = np.random.rand(512, 512)
    filter_sizes = [(3, 3), (5, 5), (7, 7), (9, 9), (11, 11)]
    M = np.array([[4, 0], [0, 4]])
    
    print(f"\n{'滤波器大小':<15} {'输出大小':<15} {'时间 (ms)':<15}")
    print("-" * 50)
    
    for fsize in filter_sizes:
        h = np.random.rand(*fsize)
        
        # C++ 实现
        start = time.time()
        result = atrousc(x, h, M)
        time_cpp = (time.time() - start) * 1000
        
        print(f"{str(fsize):<15} {str(result.shape):<15} {time_cpp:<15.2f}")




def main():
    """主函数"""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "À Trous 卷积性能基准测试" + " " * 29 + "║")
    print("╚" + "═" * 68 + "╝")
    
    # 显示后端信息
    backend_info = get_backend_info()
    print(f"\n后端信息:")
    print(f"  C++ 可用: {backend_info['cpp_available']}")
    print(f"  Python 可用: {PYTHON_AVAILABLE}")
    print(f"  当前后端: {backend_info['backend']}")
    if backend_info['import_error']:
        print(f"  导入错误: {backend_info['import_error']}")
    
    if not is_cpp_available():
        print("\n警告: C++ 扩展不可用，无法进行性能比较")
        print("请先编译 C++ 扩展:")
        print("  cd nsct_python/atrousc_cpp")
        print("  python setup.py build_ext --inplace")
        return
    
    # 运行测试
    print("\n")
    
    # 1. 基本功能测试
    if not test_correctness():
        print("\n警告: 功能测试未全部通过！")
        print("建议检查 C++ 扩展是否正确编译。")
        return
    
    # 2. 数值精度对比测试
    test_numerical_accuracy()
    
    # 3. 数组数值一致性详细验证（新增）
    test_array_values_consistency()
    
    # 4. 输出大小分析
    test_output_size_analysis()
    
    # 5. 性能对比测试 (C++ vs Python)
    benchmark_cpp_vs_python()
    
    # 6. 图像大小性能测试
    benchmark_image_sizes()
    
    # 7. 滤波器大小性能测试
    benchmark_filter_sizes()
    
    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)
    
    # 总结
    print("\n总结:")
    print("  ✓ 基本功能测试完成")
    if PYTHON_AVAILABLE:
        print("  ✓ 数值精度对比完成 (C++ vs Python)")
        print("  ✓ 数组数值一致性详细验证完成")
        print("  ✓ 性能对比完成 (C++ vs Python)")
    else:
        print("  - 纯 Python 实现不可用，跳过部分测试")
    print("  ✓ 输出大小分析完成")
    print("  ✓ 性能基准测试完成")
    print()


if __name__ == "__main__":
    main()
