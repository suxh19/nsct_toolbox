"""
基准测试脚本 - C++ 实现性能测试

这个脚本会:
1. 验证 C++ 实现的正确性
2. 测试不同图像大小下的性能
3. 测试不同滤波器大小的性能

注意: 此版本仅测试 C++ 实现，不再包含 Python 备份。
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


def benchmark_image_sizes():
    """测试不同图像大小的性能"""
    print("\n" + "=" * 70)
    print("图像大小性能基准测试")
    print("=" * 70)
    
    sizes = [(128, 128), (256, 256), (512, 512), (1024, 1024)]
    h = np.random.rand(7, 7)
    M = np.array([[4, 0], [0, 4]])
    
    print(f"\n{'图像大小':<15} {'C++ 时间 (ms)':<20} {'内存 (MB)':<15}")
    print("-" * 50)
    
    for size in sizes:
        x = np.random.rand(*size)
        
        # C++ 实现
        start = time.time()
        result = atrousc(x, h, M)
        time_cpp = (time.time() - start) * 1000
        
        # 计算内存使用
        memory_mb = (x.nbytes + h.nbytes + result.nbytes) / (1024 * 1024)
        
        print(f"{str(size):<15} {time_cpp:<20.2f} {memory_mb:<15.2f}")


def benchmark_filter_sizes():
    """测试不同滤波器大小的性能"""
    print("\n" + "=" * 70)
    print("滤波器大小性能基准测试")
    print("=" * 70)
    
    x = np.random.rand(512, 512)
    filter_sizes = [(3, 3), (5, 5), (7, 7), (9, 9), (11, 11)]
    M = np.array([[4, 0], [0, 4]])
    
    print(f"\n{'滤波器大小':<15} {'C++ 时间 (ms)':<20}")
    print("-" * 35)
    
    for fsize in filter_sizes:
        h = np.random.rand(*fsize)
        
        # C++ 实现
        start = time.time()
        result = atrousc(x, h, M)
        time_cpp = (time.time() - start) * 1000
        
        print(f"{str(fsize):<15} {time_cpp:<20.2f}")




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
    if not test_correctness():
        print("\n警告: 功能测试未全部通过！")
        print("建议检查 C++ 扩展是否正确编译。")
        return
    
    benchmark_image_sizes()
    benchmark_filter_sizes()
    
    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
