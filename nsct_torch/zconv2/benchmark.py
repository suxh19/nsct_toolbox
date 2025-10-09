"""
benchmark.py - Comprehensive benchmarking for zconv2

This script provides detailed performance analysis and comparison between
CPU and CUDA implementations of zconv2.
"""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional, TYPE_CHECKING, Any

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. CUDA benchmarks will be skipped.")

if TYPE_CHECKING:
    from typing import Callable

# Import CUDA extension
try:
    from zconv2 import zconv2 as zconv2_impl  # type: ignore
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    zconv2_impl = None  # type: ignore
    print("CUDA extension not available.")

# Import CPU extension
try:
    # Add parent directory to path to import zconv2_cpp module
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    import zconv2_cpp
    CPU_AVAILABLE = zconv2_cpp.CPP_AVAILABLE
    if not CPU_AVAILABLE:
        print(f"CPU extension not available: {zconv2_cpp._cpp_import_error}")
        zconv2_cpp = None  # type: ignore
except ImportError as e:
    CPU_AVAILABLE = False
    zconv2_cpp = None  # type: ignore
    print(f"CPU extension import error: {e}")


class BenchmarkRunner:
    """Benchmark runner for zconv2 implementations"""
    
    def __init__(self, warmup_iterations=3, benchmark_iterations=10):
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations
        self.results = []
    
    def benchmark_cuda(
        self, 
        image_size: Tuple[int, int], 
        filter_size: Tuple[int, int],
        mup: np.ndarray
    ) -> Optional[Dict[str, Any]]:
        """Benchmark CUDA implementation"""
        
        if not CUDA_AVAILABLE or not TORCH_AVAILABLE or zconv2_impl is None:
            return None
        
        # Prepare data on GPU
        x = torch.rand(*image_size, dtype=torch.float64, device='cuda')
        h = torch.rand(*filter_size, dtype=torch.float64, device='cuda')
        mup_tensor = torch.from_numpy(mup).to('cuda', dtype=torch.int32)
        
        # Warmup
        for _ in range(self.warmup_iterations):
            _ = zconv2_impl(x, h, mup_tensor)  # type: ignore
        torch.cuda.synchronize()
        
        # Benchmark
        times = []
        for _ in range(self.benchmark_iterations):
            start = time.perf_counter()
            result = zconv2_impl(x, h, mup_tensor)  # type: ignore
            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
        
        # Calculate statistics
        times = np.array(times)
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        
        # Calculate throughput
        num_pixels = image_size[0] * image_size[1]
        throughput = num_pixels / (avg_time / 1000) / 1e6  # MPixels/s
        
        return {
            'implementation': 'CUDA',
            'avg_time': avg_time,
            'std_time': std_time,
            'min_time': min_time,
            'max_time': max_time,
            'throughput': throughput,
            'times': times
        }
    
    def benchmark_cpu(
        self, 
        image_size: Tuple[int, int], 
        filter_size: Tuple[int, int],
        mup: np.ndarray
    ) -> Optional[Dict[str, Any]]:
        """Benchmark CPU implementation"""
        
        if not CPU_AVAILABLE or zconv2_cpp is None:
            return None
        
        # Prepare data on CPU
        x = np.random.rand(*image_size).astype(np.float64)
        h = np.random.rand(*filter_size).astype(np.float64)
        
        # Benchmark (no warmup needed for CPU)
        times = []
        for _ in range(self.benchmark_iterations):
            start = time.perf_counter()
            result = zconv2_cpp.zconv2(x, h, mup)  # type: ignore
            end = time.perf_counter()
            times.append((end - start) * 1000)
        
        # Calculate statistics
        times = np.array(times)
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        
        # Calculate throughput
        num_pixels = image_size[0] * image_size[1]
        throughput = num_pixels / (avg_time / 1000) / 1e6
        
        return {
            'implementation': 'CPU',
            'avg_time': avg_time,
            'std_time': std_time,
            'min_time': min_time,
            'max_time': max_time,
            'throughput': throughput,
            'times': times
        }
    
    def run_benchmark_suite(self):
        """Run comprehensive benchmark suite"""
        
        print("="*70)
        print("zconv2 Performance Benchmark Suite")
        print("="*70)
        
        # Print system info
        self._print_system_info()
        
        # Test configurations
        configs = [
            ((256, 256), (5, 5), "Small"),
            ((512, 512), (7, 7), "Medium-Small"),
            ((1024, 1024), (9, 9), "Medium"),
            ((2048, 2048), (9, 9), "Large"),
            ((4096, 4096), (7, 7), "Very Large"),
        ]
        
        mup = np.array([[1, -1], [1, 1]], dtype=np.int32)
        
        print(f"\nRunning benchmarks ({self.benchmark_iterations} iterations each)...")
        print("-"*70)
        
        results = []
        
        for image_size, filter_size, desc in configs:
            print(f"\n{desc}: Image {image_size}, Filter {filter_size}")
            
            result = {
                'description': desc,
                'image_size': image_size,
                'filter_size': filter_size,
            }
            
            # CPU benchmark
            if CPU_AVAILABLE:
                cpu_result = self.benchmark_cpu(image_size, filter_size, mup)
                if cpu_result is not None:
                    result['cpu'] = cpu_result
                    print(f"  CPU:  {cpu_result['avg_time']:>8.2f} ms "
                          f"(±{cpu_result['std_time']:.2f}) "
                          f"[{cpu_result['throughput']:.1f} MP/s]")
            
            # CUDA benchmark
            if CUDA_AVAILABLE:
                cuda_result = self.benchmark_cuda(image_size, filter_size, mup)
                if cuda_result is not None:
                    result['cuda'] = cuda_result
                    print(f"  CUDA: {cuda_result['avg_time']:>8.2f} ms "
                          f"(±{cuda_result['std_time']:.2f}) "
                          f"[{cuda_result['throughput']:.1f} MP/s]")
                    
                    # Speedup
                    if CPU_AVAILABLE and 'cpu' in result and result['cpu'] is not None:
                        cpu_result_data = result['cpu']
                        speedup = cpu_result_data['avg_time'] / cuda_result['avg_time']
                        result['speedup'] = speedup
                        print(f"  Speedup: {speedup:.1f}×")
            
            results.append(result)
        
        self.results = results
        return results
    
    def _print_system_info(self):
        """Print system information"""
        print("\nSystem Information:")
        print("-"*70)
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_props = torch.cuda.get_device_properties(0)
            print(f"  GPU: {gpu_name}")
            print(f"  GPU Memory: {gpu_props.total_memory / 1e9:.1f} GB")
            print(f"  Compute Capability: {gpu_props.major}.{gpu_props.minor}")
            # Safely access cuda version
            try:
                cuda_version = torch.version.cuda  # type: ignore
            except AttributeError:
                cuda_version = 'Unknown'
            print(f"  CUDA Version: {cuda_version}")
        
        print(f"  CPU Implementation: {'Available' if CPU_AVAILABLE else 'Not Available'}")
        print(f"  CUDA Implementation: {'Available' if CUDA_AVAILABLE else 'Not Available'}")
    
    def plot_results(self, save_path='benchmark_results.png'):
        """Plot benchmark results"""
        
        if not self.results:
            print("No results to plot. Run benchmarks first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('zconv2 Performance Benchmark Results', fontsize=16, fontweight='bold')
        
        # Extract data
        labels = [r['description'] for r in self.results]
        cpu_times = [r['cpu']['avg_time'] if 'cpu' in r else 0 for r in self.results]
        cuda_times = [r['cuda']['avg_time'] if 'cuda' in r else 0 for r in self.results]
        speedups = [r.get('speedup', 0) for r in self.results]
        
        x = np.arange(len(labels))
        width = 0.35
        
        # Plot 1: Execution Time Comparison
        ax1 = axes[0, 0]
        if CPU_AVAILABLE and cpu_times:
            ax1.bar(x - width/2, cpu_times, width, label='CPU', alpha=0.8)
        if CUDA_AVAILABLE and cuda_times:
            ax1.bar(x + width/2, cuda_times, width, label='CUDA', alpha=0.8)
        ax1.set_ylabel('Time (ms)')
        ax1.set_title('Execution Time Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Plot 2: Speedup
        ax2 = axes[0, 1]
        if speedups and any(speedups):
            bars = ax2.bar(x, speedups, width*2, alpha=0.8, color='green')
            ax2.axhline(y=1, color='r', linestyle='--', alpha=0.5)
            ax2.set_ylabel('Speedup (×)')
            ax2.set_title('CUDA Speedup over CPU')
            ax2.set_xticks(x)
            ax2.set_xticklabels(labels, rotation=45, ha='right')
            ax2.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}×', ha='center', va='bottom')
        
        # Plot 3: Throughput
        ax3 = axes[1, 0]
        cpu_throughput = [r['cpu']['throughput'] if 'cpu' in r else 0 for r in self.results]
        cuda_throughput = [r['cuda']['throughput'] if 'cuda' in r else 0 for r in self.results]
        
        if CPU_AVAILABLE and cpu_throughput:
            ax3.plot(labels, cpu_throughput, 'o-', label='CPU', linewidth=2, markersize=8)
        if CUDA_AVAILABLE and cuda_throughput:
            ax3.plot(labels, cuda_throughput, 's-', label='CUDA', linewidth=2, markersize=8)
        ax3.set_ylabel('Throughput (MPixels/s)')
        ax3.set_title('Processing Throughput')
        ax3.set_xticklabels(labels, rotation=45, ha='right')
        ax3.legend()
        ax3.grid(alpha=0.3)
        
        # Plot 4: Time Distribution (Box Plot)
        ax4 = axes[1, 1]
        if CUDA_AVAILABLE and any('cuda' in r for r in self.results):
            cuda_time_data = [r['cuda']['times'] for r in self.results if 'cuda' in r]
            bp = ax4.boxplot(cuda_time_data, labels=labels, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
            ax4.set_ylabel('Time (ms)')
            ax4.set_title('CUDA Execution Time Distribution')
            ax4.set_xticklabels(labels, rotation=45, ha='right')
            ax4.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nResults plot saved to: {save_path}")
        plt.show()


def main():
    """Main benchmark entry point"""
    
    if not CUDA_AVAILABLE:
        print("ERROR: CUDA extension not available.")
        print("Please compile it first: python setup.py build_ext --inplace")
        return 1
    
    # Create benchmark runner
    runner = BenchmarkRunner(warmup_iterations=3, benchmark_iterations=10)
    
    # Run benchmarks
    results = runner.run_benchmark_suite()
    
    # Plot results
    try:
        runner.plot_results()
    except ImportError:
        print("\nMatplotlib not available. Skipping plots.")
    except Exception as e:
        print(f"\nFailed to generate plots: {e}")
    
    print("\n" + "="*70)
    print("Benchmark completed successfully!")
    print("="*70)
    
    return 0


if __name__ == '__main__':
    exit(main())
