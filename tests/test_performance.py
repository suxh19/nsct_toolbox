"""
性能测试 - 对比 nsct_python (NumPy CPU) 和 nsct_torch (PyTorch CPU/GPU) 的性能
"""

import pytest
import numpy as np
import torch
import time
import sys
from pathlib import Path
from typing import Dict, List, Callable

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from nsct_python import utils as np_utils
from nsct_torch import utils as torch_utils


class TestPerformance:
    """性能测试类"""
    
    @pytest.fixture
    def devices(self):
        """获取可用的设备列表"""
        devices = ['cpu']
        if torch.cuda.is_available():
            devices.append('cuda')
        return devices
    
    @pytest.fixture
    def test_sizes(self):
        """定义不同的测试尺寸"""
        return {
            'small': (32, 32),
            'medium': (128, 128),
            'large': (256, 256),
            'xlarge': (512, 512)
        }
    
    def time_function(self, func: Callable, *args, warmup: int = 3, iterations: int = 10, **kwargs) -> Dict:
        """
        测量函数执行时间
        
        Args:
            func: 要测试的函数
            *args: 函数参数
            warmup: 预热次数
            iterations: 测试迭代次数
            **kwargs: 函数关键字参数
            
        Returns:
            包含时间统计的字典
        """
        # 预热
        for _ in range(warmup):
            _ = func(*args, **kwargs)
        
        # 如果是 CUDA，同步
        if len(args) > 0 and isinstance(args[0], torch.Tensor) and args[0].is_cuda:
            torch.cuda.synchronize()
        
        # 测量时间
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            
            # 如果是 CUDA，同步
            if isinstance(result, torch.Tensor) and result.is_cuda:
                torch.cuda.synchronize()
            
            end = time.perf_counter()
            times.append((end - start) * 1000)  # 转换为毫秒
        
        return {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times),
            'times': times
        }
    
    def print_performance_comparison(self, name: str, size: tuple, results: Dict):
        """打印性能对比结果"""
        print(f"\n{'='*80}")
        print(f"📊 {name} 性能测试 - 输入大小: {size}")
        print(f"{'='*80}")
        
        # 收集所有结果
        all_results = []
        for device, stats in results.items():
            all_results.append({
                'device': device,
                'mean': stats['mean'],
                'std': stats['std']
            })
        
        # 按平均时间排序
        all_results.sort(key=lambda x: x['mean'])
        
        # 找到最快的
        fastest = all_results[0]['mean']
        
        print(f"\n{'设备':<15} {'平均时间 (ms)':<20} {'标准差 (ms)':<20} {'加速比':<15}")
        print(f"{'-'*80}")
        
        for result in all_results:
            speedup = fastest / result['mean']
            device_name = result['device'].upper() if result['device'] != 'numpy_cpu' else 'NumPy (CPU)'
            if result['device'] == 'cpu':
                device_name = 'PyTorch (CPU)'
            elif result['device'] == 'cuda':
                device_name = 'PyTorch (CUDA)'
            
            print(f"{device_name:<15} {result['mean']:<20.4f} {result['std']:<20.4f} {speedup:<15.2f}x")
        
        print(f"\n🏆 最快: {all_results[0]['device'].upper()}")
        if len(all_results) > 1:
            slowest_speedup = all_results[-1]['mean'] / fastest
            print(f"⚡ 相比最慢加速: {slowest_speedup:.2f}x")
    
    # ==================== extend2 性能测试 ====================
    
    def test_performance_extend2(self, devices, test_sizes):
        """测试 extend2 函数的性能"""
        for size_name, size in test_sizes.items():
            print(f"\n\n{'#'*80}")
            print(f"测试尺寸: {size_name.upper()} {size}")
            print(f"{'#'*80}")
            
            # 准备数据
            x_np = np.random.randn(*size).astype(np.float32)
            
            results = {}
            
            # NumPy CPU 版本
            print("\n🔄 测试 NumPy (CPU) 版本...")
            results['numpy_cpu'] = self.time_function(
                np_utils.extend2, x_np, 2, 2, 2, 2, 'per'
            )
            
            # PyTorch 版本
            for device in devices:
                print(f"\n🔄 测试 PyTorch ({device.upper()}) 版本...")
                x_torch = torch.from_numpy(x_np).to(device)
                results[device] = self.time_function(
                    torch_utils.extend2, x_torch, 2, 2, 2, 2, 'per'
                )
            
            # 打印对比结果
            self.print_performance_comparison('extend2', size, results)
    
    # ==================== symext 性能测试 ====================
    
    def test_performance_symext(self, devices, test_sizes):
        """测试 symext 函数的性能"""
        for size_name, size in test_sizes.items():
            print(f"\n\n{'#'*80}")
            print(f"测试尺寸: {size_name.upper()} {size}")
            print(f"{'#'*80}")
            
            # 准备数据
            x_np = np.random.randn(*size).astype(np.float32)
            h_np = np.ones((5, 5), dtype=np.float32)
            shift = [2, 2]
            
            results = {}
            
            # NumPy CPU 版本
            print("\n🔄 测试 NumPy (CPU) 版本...")
            results['numpy_cpu'] = self.time_function(
                np_utils.symext, x_np, h_np, shift
            )
            
            # PyTorch 版本
            for device in devices:
                print(f"\n🔄 测试 PyTorch ({device.upper()}) 版本...")
                x_torch = torch.from_numpy(x_np).to(device)
                h_torch = torch.from_numpy(h_np).to(device)
                results[device] = self.time_function(
                    torch_utils.symext, x_torch, h_torch, shift
                )
            
            # 打印对比结果
            self.print_performance_comparison('symext', size, results)
    
    # ==================== upsample2df 性能测试 ====================
    
    def test_performance_upsample2df(self, devices):
        """测试 upsample2df 函数的性能"""
        test_sizes = {
            'small': (8, 8),
            'medium': (32, 32),
            'large': (64, 64),
            'xlarge': (128, 128)
        }
        
        for size_name, size in test_sizes.items():
            print(f"\n\n{'#'*80}")
            print(f"测试尺寸: {size_name.upper()} {size}")
            print(f"{'#'*80}")
            
            # 准备数据
            h_np = np.random.randn(*size).astype(np.float32)
            power = 2
            
            results = {}
            
            # NumPy CPU 版本
            print("\n🔄 测试 NumPy (CPU) 版本...")
            results['numpy_cpu'] = self.time_function(
                np_utils.upsample2df, h_np, power
            )
            
            # PyTorch 版本
            for device in devices:
                print(f"\n🔄 测试 PyTorch ({device.upper()}) 版本...")
                h_torch = torch.from_numpy(h_np).to(device)
                results[device] = self.time_function(
                    torch_utils.upsample2df, h_torch, power
                )
            
            # 打印对比结果
            self.print_performance_comparison('upsample2df', size, results)
    
    # ==================== modulate2 性能测试 ====================
    
    def test_performance_modulate2(self, devices, test_sizes):
        """测试 modulate2 函数的性能"""
        for size_name, size in test_sizes.items():
            print(f"\n\n{'#'*80}")
            print(f"测试尺寸: {size_name.upper()} {size}")
            print(f"{'#'*80}")
            
            # 准备数据
            x_np = np.random.randn(*size).astype(np.float32)
            
            results = {}
            
            # NumPy CPU 版本
            print("\n🔄 测试 NumPy (CPU) 版本...")
            results['numpy_cpu'] = self.time_function(
                np_utils.modulate2, x_np, 'b'
            )
            
            # PyTorch 版本
            for device in devices:
                print(f"\n🔄 测试 PyTorch ({device.upper()}) 版本...")
                x_torch = torch.from_numpy(x_np).to(device)
                results[device] = self.time_function(
                    torch_utils.modulate2, x_torch, 'b'
                )
            
            # 打印对比结果
            self.print_performance_comparison('modulate2', size, results)
    
    # ==================== resampz 性能测试 ====================
    
    def test_performance_resampz(self, devices, test_sizes):
        """测试 resampz 函数的性能"""
        for size_name, size in test_sizes.items():
            print(f"\n\n{'#'*80}")
            print(f"测试尺寸: {size_name.upper()} {size}")
            print(f"{'#'*80}")
            
            # 准备数据
            x_np = np.random.randn(*size).astype(np.float32)
            
            results = {}
            
            # NumPy CPU 版本
            print("\n🔄 测试 NumPy (CPU) 版本...")
            results['numpy_cpu'] = self.time_function(
                np_utils.resampz, x_np, 1, 1
            )
            
            # PyTorch 版本
            for device in devices:
                print(f"\n🔄 测试 PyTorch ({device.upper()}) 版本...")
                x_torch = torch.from_numpy(x_np).to(device)
                results[device] = self.time_function(
                    torch_utils.resampz, x_torch, 1, 1
                )
            
            # 打印对比结果
            self.print_performance_comparison('resampz', size, results)
    
    # ==================== qupz 性能测试 ====================
    
    def test_performance_qupz(self, devices, test_sizes):
        """测试 qupz 函数的性能"""
        for size_name, size in test_sizes.items():
            print(f"\n\n{'#'*80}")
            print(f"测试尺寸: {size_name.upper()} {size}")
            print(f"{'#'*80}")
            
            # 准备数据
            x_np = np.random.randn(*size).astype(np.float32)
            
            results = {}
            
            # NumPy CPU 版本
            print("\n🔄 测试 NumPy (CPU) 版本...")
            results['numpy_cpu'] = self.time_function(
                np_utils.qupz, x_np, 1
            )
            
            # PyTorch 版本
            for device in devices:
                print(f"\n🔄 测试 PyTorch ({device.upper()}) 版本...")
                x_torch = torch.from_numpy(x_np).to(device)
                results[device] = self.time_function(
                    torch_utils.qupz, x_torch, 1
                )
            
            # 打印对比结果
            self.print_performance_comparison('qupz', size, results)
    
    # ==================== 综合性能报告 ====================
    
    def test_performance_summary(self, devices):
        """生成综合性能报告"""
        print(f"\n\n{'='*80}")
        print("📊 综合性能测试总结")
        print(f"{'='*80}")
        
        print(f"\n可用设备: {', '.join([d.upper() for d in devices])}")
        
        if 'cuda' in devices:
            print(f"CUDA 设备: {torch.cuda.get_device_name(0)}")
            print(f"CUDA 版本: {torch.version.cuda}")
        
        print(f"PyTorch 版本: {torch.__version__}")
        print(f"NumPy 版本: {np.__version__}")
        
        print("\n" + "="*80)
        print("✅ 性能测试完成！")
        print("="*80)
        
        print("\n💡 提示:")
        print("  - CPU 版本适合小规模数据和调试")
        print("  - GPU 版本在大规模数据上有显著优势")
        print("  - 数据传输开销在小数据上可能影响 GPU 性能")


if __name__ == '__main__':
    # 可以直接运行此文件进行性能测试
    # 运行所有性能测试
    pytest.main([__file__, '-v', '-s'])
    
    # 或运行特定测试
    # pytest.main([__file__, '-v', '-s', '-k', 'extend2'])
