"""
详细对比 Python 和 MATLAB 的 NSCT 实现结果
包括：形状对比、数值对比、精度对比
"""

import numpy as np
import pickle
import scipy.io as sio
import h5py
from tabulate import tabulate
import matplotlib.pyplot as plt
from matplotlib import font_manager
import os
import sys

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 输出重定向类
class TeeOutput:
    """同时输出到控制台和文件"""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()

print("=" * 80)
print("Python vs MATLAB NSCT 实现详细对比")
print("=" * 80)

# 读取参数（从命令行或使用默认值）
if len(sys.argv) > 1:
    param_folder = sys.argv[1]
else:
    # 尝试从 nsct_params.json 读取参数
    params_file = os.path.join('scripts', 'nsct_params.json')
    if os.path.exists(params_file):
        import json
        try:
            with open(params_file, 'r', encoding='utf-8') as f:
                params = json.load(f)
            levels = params.get('levels', [2, 3])
            dfilt = params.get('dfilt', 'dmaxflat7')
            pfilt = params.get('pfilt', 'maxflat')
            print(f"从配置文件读取参数: {params_file}")
        except Exception as e:
            print(f"读取配置文件失败，使用默认参数: {e}")
            levels = [2, 3]
            dfilt = 'dmaxflat7'
            pfilt = 'maxflat'
    else:
        # 默认参数
        levels = [2, 3]
        dfilt = 'dmaxflat7'
        pfilt = 'maxflat'
        print("使用默认参数")
    
    levels_str = '_'.join(map(str, levels))
    param_folder = f'levels_{levels_str}_{dfilt}_{pfilt}'

# 构建输入路径
base_output_dir = os.path.join('output', param_folder)
python_dir = os.path.join(base_output_dir, 'python')
matlab_dir = os.path.join(base_output_dir, 'matlab')

# 创建对比结果输出目录
comparison_dir = os.path.join(base_output_dir, 'comparison')
os.makedirs(comparison_dir, exist_ok=True)

# 设置输出重定向到文件
output_file = os.path.join(comparison_dir, 'comparison_report.txt')
tee = TeeOutput(output_file)
sys.stdout = tee

print(f"\n参数文件夹: {param_folder}")
print(f"Python 结果目录: {python_dir}")
print(f"MATLAB 结果目录: {matlab_dir}")
print(f"对比结果输出目录: {comparison_dir}")
print(f"报告文件: {output_file}")

# 1. 加载 Python 结果
print("\n1. 加载 Python 结果...")
try:
    py_results_file = os.path.join(python_dir, 'python_nsct_results.pkl')
    py_subband_file = os.path.join(python_dir, 'python_subband_details.pkl')
    with open(py_results_file, 'rb') as f:
        py_results = pickle.load(f)
    with open(py_subband_file, 'rb') as f:
        py_subband = pickle.load(f)
    print("   ✓ Python 结果加载成功")
except FileNotFoundError as e:
    print(f"   ✗ 错误: {e}")
    print("   请先运行 run_python_nsct.py")
    exit(1)

# 2. 加载 MATLAB 结果
print("\n2. 加载 MATLAB 结果...")
try:
    mat_results_file = os.path.join(matlab_dir, 'matlab_nsct_results.mat')
    mat_subband_file = os.path.join(matlab_dir, 'matlab_subband_details.mat')
    
    # 使用 h5py 读取 MATLAB v7.3 格式文件
    with h5py.File(mat_results_file, 'r') as f:
        mat_results = {}
        mat_results['results'] = {}
        
        # 读取主要数据
        mat_results['results']['original_image'] = np.array(f['results']['original_image']).T  # type: ignore
        mat_results['results']['reconstructed_image'] = np.array(f['results']['reconstructed_image']).T  # type: ignore
        
        # 读取参数
        mat_results['results']['parameters'] = {}
        # 处理 levels 数组 - 可能是多维的
        levels_data = np.array(f['results']['parameters']['levels'])  # type: ignore
        if levels_data.ndim == 1:
            mat_results['results']['parameters']['levels'] = [int(x) for x in levels_data]
        else:
            mat_results['results']['parameters']['levels'] = [int(x) for x in levels_data.flatten()]
        
        # 读取字符串参数
        dfilt_data = np.array(f['results']['parameters']['dfilt'][:])  # type: ignore
        mat_results['results']['parameters']['dfilt'] = ''.join(chr(int(c)) for c in dfilt_data.flatten())
        
        pfilt_data = np.array(f['results']['parameters']['pfilt'][:])  # type: ignore
        mat_results['results']['parameters']['pfilt'] = ''.join(chr(int(c)) for c in pfilt_data.flatten())
        
        # 读取时间信息
        mat_results['results']['timing'] = {}
        mat_results['results']['timing']['decomposition_time'] = f['results']['timing']['decomposition_time'][()].item()  # type: ignore
        mat_results['results']['timing']['reconstruction_time'] = f['results']['timing']['reconstruction_time'][()].item()  # type: ignore
        
        # 读取度量信息
        mat_results['results']['metrics'] = {}
        mat_results['results']['metrics']['mse'] = f['results']['metrics']['mse'][()].item()  # type: ignore
        mat_results['results']['metrics']['psnr'] = f['results']['metrics']['psnr'][()].item()  # type: ignore
        mat_results['results']['metrics']['max_error'] = f['results']['metrics']['max_error'][()].item()  # type: ignore
        mat_results['results']['metrics']['relative_error'] = f['results']['metrics']['relative_error'][()].item()  # type: ignore
    
    # 读取子带详细信息
    with h5py.File(mat_subband_file, 'r') as f:
        mat_subband = {}
        mat_subband['subband_info'] = {}
        
        # 读取低频子带
        mat_subband['subband_info']['lowpass'] = {}
        mat_subband['subband_info']['lowpass']['data'] = np.array(f['subband_info']['lowpass']['data']).T  # type: ignore
        mat_subband['subband_info']['lowpass']['mean'] = f['subband_info']['lowpass']['mean'][()].item()  # type: ignore
        mat_subband['subband_info']['lowpass']['std'] = f['subband_info']['lowpass']['std'][()].item()  # type: ignore
        mat_subband['subband_info']['lowpass']['min'] = f['subband_info']['lowpass']['min'][()].item()  # type: ignore
        mat_subband['subband_info']['lowpass']['max'] = f['subband_info']['lowpass']['max'][()].item()  # type: ignore
        mat_subband['subband_info']['lowpass']['shape'] = mat_subband['subband_info']['lowpass']['data'].shape
        
        # 读取方向子带
        mat_subband['subband_info']['bandpass'] = []
        bandpass_refs = f['subband_info']['bandpass'][:].flatten()  # type: ignore
        for scale_ref in bandpass_refs:
            scale_obj = f[scale_ref]
            scale_info = {}
            scale_info['scale'] = scale_obj['scale'][()].item()  # type: ignore
            scale_info['num_directions'] = scale_obj['num_directions'][()].item()  # type: ignore
            
            if scale_info['num_directions'] > 0:
                scale_info['directions'] = []
                dir_refs = scale_obj['directions'][:].flatten()  # type: ignore
                for dir_ref in dir_refs:
                    dir_obj = f[dir_ref]
                    dir_info = {}
                    dir_info['data'] = np.array(dir_obj['data']).T  # type: ignore
                    dir_info['mean'] = dir_obj['mean'][()].item()  # type: ignore
                    dir_info['std'] = dir_obj['std'][()].item()  # type: ignore
                    dir_info['min'] = dir_obj['min'][()].item()  # type: ignore
                    dir_info['max'] = dir_obj['max'][()].item()  # type: ignore
                    dir_info['shape'] = dir_info['data'].shape
                    dir_info['direction'] = dir_obj['direction'][()].item()  # type: ignore
                    scale_info['directions'].append(dir_info)
            
            mat_subband['subband_info']['bandpass'].append(scale_info)
    
    print("   ✓ MATLAB 结果加载成功")
except FileNotFoundError as e:
    print(f"   ✗ 错误: {e}")
    print("   请先运行 run_matlab_nsct.m")
    exit(1)
except Exception as e:
    print(f"   ✗ 加载 MATLAB 结果时出错: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 3. 提取数据
py_img = py_results['original_image']
py_rec = py_results['reconstructed_image']
py_dec = py_results['decomposition']
py_metrics = py_results['metrics']
py_timing = py_results['timing']

mat_res = mat_results['results']
mat_img = mat_res['original_image']
mat_rec = mat_res['reconstructed_image']
mat_metrics = mat_res['metrics']
mat_timing = mat_res['timing']

print("\n" + "=" * 80)
print("一、基本信息对比")
print("=" * 80)

# 4. 参数对比
print("\n【参数设置】")
param_table = [
    ["参数", "Python", "MATLAB", "一致性"],
    ["-" * 20, "-" * 20, "-" * 20, "-" * 10],
    ["金字塔层级", len(py_results['parameters']['levels']), 
     len(mat_res['parameters']['levels']), "✓"],
    ["方向分解层级", str(py_results['parameters']['levels']), 
     str(mat_res['parameters']['levels']), "✓"],
    ["方向滤波器", py_results['parameters']['dfilt'], 
     mat_res['parameters']['dfilt'], "✓"],
    ["金字塔滤波器", py_results['parameters']['pfilt'], 
     mat_res['parameters']['pfilt'], "✓"]
]
print(tabulate(param_table, tablefmt='grid'))

# 5. 时间对比
print("\n【运行时间对比】")
time_table = [
    ["操作", "Python (秒)", "MATLAB (秒)", "速度比 (M/P)"],
    ["-" * 20, "-" * 15, "-" * 15, "-" * 15],
    ["分解", f"{py_timing['decomposition_time']:.3f}", 
     f"{mat_timing['decomposition_time']:.3f}",
     f"{mat_timing['decomposition_time'] / py_timing['decomposition_time']:.2f}x"],
    ["重建", f"{py_timing['reconstruction_time']:.3f}", 
     f"{mat_timing['reconstruction_time']:.3f}",
     f"{mat_timing['reconstruction_time'] / py_timing['reconstruction_time']:.2f}x"]
]
print(tabulate(time_table, tablefmt='grid'))

print("\n" + "=" * 80)
print("二、形状对比")
print("=" * 80)

# 6. 原始图像和重建图像形状对比
print("\n【图像形状对比】")
img_shape_table = [
    ["图像", "Python 形状", "MATLAB 形状", "一致性"],
    ["-" * 20, "-" * 20, "-" * 20, "-" * 10],
    ["原始图像", str(py_img.shape), str(mat_img.shape), 
     "✓" if py_img.shape == mat_img.shape else "✗"],
    ["重建图像", str(py_rec.shape), str(mat_rec.shape), 
     "✓" if py_rec.shape == mat_rec.shape else "✗"]
]
print(tabulate(img_shape_table, tablefmt='grid'))

# 7. 子带形状对比
print("\n【分解子带形状对比】")
print("\n低频子带:")
lowpass_table = [
    ["属性", "Python", "MATLAB", "一致性"],
    ["-" * 20, "-" * 20, "-" * 20, "-" * 10],
    ["形状", str(py_subband['lowpass']['shape']), 
     str(tuple(py_subband['lowpass']['shape'])), "✓"]
]
print(tabulate(lowpass_table, tablefmt='grid'))

print("\n方向子带:")
for scale_idx in range(len(py_subband['bandpass'])):
    py_scale = py_subband['bandpass'][scale_idx]
    mat_scale = mat_subband['subband_info']['bandpass'][scale_idx]
    
    print(f"\n  尺度 {scale_idx + 1}:")
    scale_table = [
        ["属性", "Python", "MATLAB", "一致性"],
        ["-" * 20, "-" * 20, "-" * 20, "-" * 10],
        ["方向数", py_scale['num_directions'], 
         mat_scale['num_directions'], 
         "✓" if py_scale['num_directions'] == mat_scale['num_directions'] else "✗"]
    ]
    
    if py_scale['num_directions'] > 0:
        for dir_idx in range(py_scale['num_directions']):
            py_dir = py_scale['directions'][dir_idx]
            mat_dir = mat_scale['directions'][dir_idx]
            scale_table.append([
                f"方向 {dir_idx} 形状",
                str(py_dir['shape']),
                str(mat_dir['shape']),
                "✓" if py_dir['shape'] == mat_dir['shape'] else "✗"
            ])
    
    print(tabulate(scale_table, tablefmt='grid'))

print("\n" + "=" * 80)
print("三、数值对比")
print("=" * 80)

# 8. 原始图像数值对比
print("\n【原始图像数值对比】")
img_diff = np.abs(py_img - mat_img)
img_table = [
    ["统计量", "Python", "MATLAB", "绝对差"],
    ["-" * 20, "-" * 20, "-" * 20, "-" * 20],
    ["均值", f"{np.mean(py_img):.6f}", f"{np.mean(mat_img):.6f}", f"{np.mean(img_diff):.6e}"],
    ["标准差", f"{np.std(py_img):.6f}", f"{np.std(mat_img):.6f}", f"{np.std(img_diff):.6e}"],
    ["最小值", f"{np.min(py_img):.6f}", f"{np.min(mat_img):.6f}", f"{np.min(img_diff):.6e}"],
    ["最大值", f"{np.max(py_img):.6f}", f"{np.max(mat_img):.6f}", f"{np.max(img_diff):.6e}"],
    ["最大绝对差", "-", "-", f"{np.max(img_diff):.6e}"],
    ["相对误差 (%)", "-", "-", f"{np.linalg.norm(img_diff) / np.linalg.norm(py_img) * 100:.6e}"]
]
print(tabulate(img_table, tablefmt='grid'))

# 9. 重建图像数值对比
print("\n【重建图像数值对比】")
rec_diff = np.abs(py_rec - mat_rec)
rec_table = [
    ["统计量", "Python", "MATLAB", "绝对差"],
    ["-" * 20, "-" * 20, "-" * 20, "-" * 20],
    ["均值", f"{np.mean(py_rec):.6f}", f"{np.mean(mat_rec):.6f}", f"{np.mean(rec_diff):.6e}"],
    ["标准差", f"{np.std(py_rec):.6f}", f"{np.std(mat_rec):.6f}", f"{np.std(rec_diff):.6e}"],
    ["最小值", f"{np.min(py_rec):.6f}", f"{np.min(mat_rec):.6f}", f"{np.min(rec_diff):.6e}"],
    ["最大值", f"{np.max(py_rec):.6f}", f"{np.max(mat_rec):.6f}", f"{np.max(rec_diff):.6e}"],
    ["最大绝对差", "-", "-", f"{np.max(rec_diff):.6e}"],
    ["相对误差 (%)", "-", "-", f"{np.linalg.norm(rec_diff) / np.linalg.norm(py_rec) * 100:.6e}"]
]
print(tabulate(rec_table, tablefmt='grid'))

# 10. 低频子带数值对比
print("\n【低频子带数值对比】")
py_lowpass = py_subband['lowpass']['data']
mat_lowpass = mat_subband['subband_info']['lowpass']['data']
lowpass_diff = np.abs(py_lowpass - mat_lowpass)

lowpass_val_table = [
    ["统计量", "Python", "MATLAB", "绝对差"],
    ["-" * 20, "-" * 20, "-" * 20, "-" * 20],
    ["均值", f"{py_subband['lowpass']['mean']:.6f}", 
     f"{mat_subband['subband_info']['lowpass']['mean']:.6f}", f"{np.mean(lowpass_diff):.6e}"],
    ["标准差", f"{py_subband['lowpass']['std']:.6f}", 
     f"{mat_subband['subband_info']['lowpass']['std']:.6f}", f"{np.std(lowpass_diff):.6e}"],
    ["最小值", f"{py_subband['lowpass']['min']:.6f}", 
     f"{mat_subband['subband_info']['lowpass']['min']:.6f}", f"{np.min(lowpass_diff):.6e}"],
    ["最大值", f"{py_subband['lowpass']['max']:.6f}", 
     f"{mat_subband['subband_info']['lowpass']['max']:.6f}", f"{np.max(lowpass_diff):.6e}"],
    ["最大绝对差", "-", "-", f"{np.max(lowpass_diff):.6e}"],
    ["相对误差 (%)", "-", "-", 
     f"{np.linalg.norm(lowpass_diff) / np.linalg.norm(py_lowpass) * 100:.6e}"]
]
print(tabulate(lowpass_val_table, tablefmt='grid'))

# 11. 方向子带数值对比
print("\n【方向子带数值对比】")
for scale_idx in range(len(py_subband['bandpass'])):
    py_scale = py_subband['bandpass'][scale_idx]
    mat_scale = mat_subband['subband_info']['bandpass'][scale_idx]
    
    print(f"\n  尺度 {scale_idx + 1}:")
    
    if py_scale['num_directions'] > 0:
        for dir_idx in range(py_scale['num_directions']):
            py_dir = py_scale['directions'][dir_idx]
            mat_dir = mat_scale['directions'][dir_idx]
            
            py_data = py_dir['data']
            mat_data = mat_dir['data']
            dir_diff = np.abs(py_data - mat_data)
            
            dir_table = [
                ["统计量", "Python", "MATLAB", "绝对差"],
                ["-" * 20, "-" * 20, "-" * 20, "-" * 20],
                [f"方向 {dir_idx} - 均值", f"{py_dir['mean']:.6f}", 
                 f"{mat_dir['mean']:.6f}", f"{np.mean(dir_diff):.6e}"],
                [f"方向 {dir_idx} - 标准差", f"{py_dir['std']:.6f}", 
                 f"{mat_dir['std']:.6f}", f"{np.std(dir_diff):.6e}"],
                [f"方向 {dir_idx} - 最大绝对差", "-", "-", f"{np.max(dir_diff):.6e}"],
                [f"方向 {dir_idx} - 相对误差 (%)", "-", "-", 
                 f"{np.linalg.norm(dir_diff) / np.linalg.norm(py_data) * 100:.6e}"]
            ]
            print(tabulate(dir_table, tablefmt='grid'))

print("\n" + "=" * 80)
print("四、重建质量对比")
print("=" * 80)

# 12. 重建质量指标对比
print("\n【重建质量指标】")
quality_table = [
    ["指标", "Python", "MATLAB", "差异"],
    ["-" * 25, "-" * 20, "-" * 20, "-" * 20],
    ["MSE", f"{py_metrics['mse']:.6e}", f"{mat_metrics['mse']:.6e}", 
     f"{abs(py_metrics['mse'] - mat_metrics['mse']):.6e}"],
    ["PSNR (dB)", f"{py_metrics['psnr']:.2f}" if py_metrics['psnr'] != np.inf else "∞", 
     f"{mat_metrics['psnr']:.2f}" if not np.isinf(mat_metrics['psnr']) else "∞", "-"],
    ["最大绝对误差", f"{py_metrics['max_error']:.6e}", f"{mat_metrics['max_error']:.6e}", 
     f"{abs(py_metrics['max_error'] - mat_metrics['max_error']):.6e}"],
    ["相对误差 (%)", f"{py_metrics['relative_error']:.6f}", f"{mat_metrics['relative_error']:.6f}", 
     f"{abs(py_metrics['relative_error'] - mat_metrics['relative_error']):.6e}"]
]
print(tabulate(quality_table, tablefmt='grid'))

print("\n" + "=" * 80)
print("五、可视化对比")
print("=" * 80)

# 13. 创建对比图
print("\n生成对比图...")

fig = plt.figure(figsize=(16, 12))

# 子图1: 原始图像对比
ax1 = plt.subplot(3, 3, 1)
plt.imshow(py_img, cmap='gray')
plt.title('Python - 原始图像')
plt.axis('off')

ax2 = plt.subplot(3, 3, 2)
plt.imshow(mat_img, cmap='gray')
plt.title('MATLAB - 原始图像')
plt.axis('off')

ax3 = plt.subplot(3, 3, 3)
plt.imshow(img_diff, cmap='hot')
plt.colorbar()
plt.title(f'原始图像差异\n最大差: {np.max(img_diff):.2e}')
plt.axis('off')

# 子图2: 重建图像对比
ax4 = plt.subplot(3, 3, 4)
plt.imshow(py_rec, cmap='gray')
plt.title(f'Python - 重建图像\nPSNR: {py_metrics["psnr"]:.2f} dB')
plt.axis('off')

ax5 = plt.subplot(3, 3, 5)
plt.imshow(mat_rec, cmap='gray')
plt.title(f'MATLAB - 重建图像\nPSNR: {mat_metrics["psnr"]:.2f} dB')
plt.axis('off')

ax6 = plt.subplot(3, 3, 6)
plt.imshow(rec_diff, cmap='hot')
plt.colorbar()
plt.title(f'重建图像差异\n最大差: {np.max(rec_diff):.2e}')
plt.axis('off')

# 子图3: 低频子带对比
ax7 = plt.subplot(3, 3, 7)
plt.imshow(py_lowpass, cmap='gray')
plt.title('Python - 低频子带')
plt.axis('off')

ax8 = plt.subplot(3, 3, 8)
plt.imshow(mat_lowpass, cmap='gray')
plt.title('MATLAB - 低频子带')
plt.axis('off')

ax9 = plt.subplot(3, 3, 9)
plt.imshow(lowpass_diff, cmap='hot')
plt.colorbar()
plt.title(f'低频子带差异\n最大差: {np.max(lowpass_diff):.2e}')
plt.axis('off')

plt.tight_layout()
summary_file = os.path.join(comparison_dir, 'nsct_comparison_summary.png')
plt.savefig(summary_file, dpi=150, bbox_inches='tight')
print(f"   对比图已保存: {summary_file}")

# 14. 方向子带对比图
print("   生成方向子带对比图...")
for scale_idx in range(len(py_subband['bandpass'])):
    py_scale = py_subband['bandpass'][scale_idx]
    mat_scale = mat_subband['subband_info']['bandpass'][scale_idx]
    
    if py_scale['num_directions'] > 0:
        n_dirs = py_scale['num_directions']
        fig, axes = plt.subplots(3, n_dirs, figsize=(4*n_dirs, 10))
        
        for dir_idx in range(n_dirs):
            py_data = py_scale['directions'][dir_idx]['data']
            mat_data = mat_scale['directions'][dir_idx]['data']
            diff = np.abs(py_data - mat_data)
            
            # Python
            axes[0, dir_idx].imshow(py_data, cmap='gray')
            axes[0, dir_idx].set_title(f'Python - 方向 {dir_idx}')
            axes[0, dir_idx].axis('off')
            
            # MATLAB
            axes[1, dir_idx].imshow(mat_data, cmap='gray')
            axes[1, dir_idx].set_title(f'MATLAB - 方向 {dir_idx}')
            axes[1, dir_idx].axis('off')
            
            # 差异
            im = axes[2, dir_idx].imshow(diff, cmap='hot')
            axes[2, dir_idx].set_title(f'差异 - 方向 {dir_idx}\n最大: {np.max(diff):.2e}')
            axes[2, dir_idx].axis('off')
            plt.colorbar(im, ax=axes[2, dir_idx])
        
        plt.suptitle(f'尺度 {scale_idx + 1} 方向子带对比', fontsize=16)
        plt.tight_layout()
        scale_file = os.path.join(comparison_dir, f'nsct_comparison_scale_{scale_idx + 1}.png')
        plt.savefig(scale_file, dpi=150, bbox_inches='tight')
        print(f"   尺度 {scale_idx + 1} 对比图已保存: {scale_file}")

print("\n" + "=" * 80)
print("六、总结")
print("=" * 80)

# 15. 生成总结报告
print("\n【一致性检查】")
checks = []

# 形状一致性
shape_match = (py_img.shape == mat_img.shape and 
               py_rec.shape == mat_rec.shape)
checks.append(["形状一致性", "✓" if shape_match else "✗"])

# 数值精度
img_relative_error = np.linalg.norm(img_diff) / np.linalg.norm(py_img) * 100
rec_relative_error = np.linalg.norm(rec_diff) / np.linalg.norm(py_rec) * 100
lowpass_relative_error = np.linalg.norm(lowpass_diff) / np.linalg.norm(py_lowpass) * 100

checks.append(["原始图像相对误差", f"{img_relative_error:.2e}%"])
checks.append(["重建图像相对误差", f"{rec_relative_error:.2e}%"])
checks.append(["低频子带相对误差", f"{lowpass_relative_error:.2e}%"])

# 重建质量
perfect_reconstruction_py = py_metrics['max_error'] < 1e-10
perfect_reconstruction_mat = mat_metrics['max_error'] < 1e-10
checks.append(["Python 完美重建", "✓" if perfect_reconstruction_py else "✗"])
checks.append(["MATLAB 完美重建", "✓" if perfect_reconstruction_mat else "✗"])

print(tabulate(checks, headers=["检查项", "结果"], tablefmt='grid'))

print("\n【结论】")
if img_relative_error < 1e-10 and rec_relative_error < 1e-10:
    print("✓ Python 和 MATLAB 实现在数值上完全一致（误差 < 1e-10）")
elif img_relative_error < 1e-6 and rec_relative_error < 1e-6:
    print("✓ Python 和 MATLAB 实现在数值上高度一致（误差 < 1e-6）")
elif img_relative_error < 1e-3 and rec_relative_error < 1e-3:
    print("⚠ Python 和 MATLAB 实现存在轻微数值差异（误差 < 1e-3）")
else:
    print("✗ Python 和 MATLAB 实现存在显著数值差异")

print("\n" + "=" * 80)
print("对比完成！所有结果已保存。")
print("=" * 80)

# 关闭输出重定向
tee.close()
sys.stdout = tee.terminal
