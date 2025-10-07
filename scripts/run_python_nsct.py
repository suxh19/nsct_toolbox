"""
Python 版本的 NSCT 分解和重建
独立运行并保存结果供后续对比
"""

import numpy as np
from PIL import Image
import pickle
import time
import os
import sys

# 添加项目根目录到 Python 路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from nsct_python.core import nsctdec, nsctrec


print("=" * 70)
print("Python 版本 - NSCT 分解和重建")
print("=" * 70)

# 1. 加载图像
print("\n1. 加载图像...")
img_path = os.path.join(project_root, 'test_image.jpg')
img = Image.open(img_path)
if img.mode == 'RGB':
    img = img.convert('L')
img_array = np.array(img, dtype=np.float64)
print(f"   图像尺寸: {img_array.shape}")
print(f"   像素值范围: [{img_array.min():.2f}, {img_array.max():.2f}]")

# 2. 设置参数
# 尝试从 nsct_params.json 读取参数
import json
params_file = os.path.join(script_dir, 'nsct_params.json')
if os.path.exists(params_file):
    try:
        with open(params_file, 'r', encoding='utf-8') as f:
            params = json.load(f)
        levels = params.get('levels', [2, 3])
        dfilt = params.get('dfilt', 'dmaxflat7')
        pfilt = params.get('pfilt', 'maxflat')
        print(f"\n从配置文件读取参数: {params_file}")
    except Exception as e:
        print(f"\n读取配置文件失败，使用默认参数: {e}")
        levels = [2, 3]
        dfilt = 'dmaxflat7'
        pfilt = 'maxflat'
else:
    # 默认参数
    levels = [2, 3]  # 2个金字塔层级，每层分别进行2和3级方向分解
    dfilt = 'dmaxflat7'  # 方向滤波器
    pfilt = 'maxflat'  # 金字塔滤波器
    print("\n使用默认参数")

# 创建输出文件夹结构: output/levels_X_Y_dfilt_pfilt/python/
levels_str = '_'.join(map(str, levels))
output_dir = os.path.join('output', f'levels_{levels_str}_{dfilt}_{pfilt}', 'python')
os.makedirs(output_dir, exist_ok=True)
print(f"\n输出目录: {output_dir}")

print("\n2. NSCT 分解参数:")
print(f"   金字塔层级: {len(levels)}")
print(f"   方向分解层级: {levels}")
print(f"   方向滤波器: {dfilt}")
print(f"   金字塔滤波器: {pfilt}")

# 动态构建预计子带数描述
subband_desc = "1个低频"
for i, level in enumerate(levels):
    subband_desc += f" + {2**level}个方向(尺度{i})"
print(f"   预计子带数: {subband_desc}")

# 3. NSCT 分解
print("\n3. 执行 NSCT 分解...")
start_time = time.time()
y = nsctdec(img_array, levels, dfilt, pfilt)
dec_time = time.time() - start_time
print(f"   分解完成，耗时: {dec_time:.3f} 秒")

# 4. 显示分解结果信息
print("\n4. 分解结果:")
print(f"   总子带数: {len(y)}")
print(f"   - y[0] (低频): {y[0].shape}")
for i in range(1, len(y)):
    if isinstance(y[i], list):
        print(f"   - y[{i}] (尺度{i}方向子带): {len(y[i])}个子带，每个尺寸 {y[i][0].shape}")
    else:
        print(f"   - y[{i}] (尺度{i}): {y[i].shape}")

# 5. NSCT 重建
print("\n5. 执行 NSCT 重建...")
start_time = time.time()
img_rec = nsctrec(y, dfilt, pfilt)
rec_time = time.time() - start_time
print(f"   重建完成，耗时: {rec_time:.3f} 秒")
print(f"   重建图像尺寸: {img_rec.shape}")

# 6. 重建质量评估
print("\n6. 重建质量评估:")
mse = np.mean((img_array - img_rec) ** 2)
if mse > 0:
    psnr = 10 * np.log10(255**2 / mse)
else:
    psnr = np.inf
max_error = np.max(np.abs(img_array - img_rec))
relative_error = np.linalg.norm(img_array - img_rec) / np.linalg.norm(img_array) * 100

print(f"   均方误差 (MSE): {mse:.6e}")
print(f"   峰值信噪比 (PSNR): {psnr:.2f} dB")
print(f"   最大绝对误差: {max_error:.6e}")
print(f"   相对误差: {relative_error:.6f}%")

# 7. 保存结果到文件
print("\n7. 保存结果到文件...")
results = {
    'original_image': img_array,
    'reconstructed_image': img_rec,
    'decomposition': y,
    'parameters': {
        'levels': levels,
        'dfilt': dfilt,
        'pfilt': pfilt
    },
    'timing': {
        'decomposition_time': dec_time,
        'reconstruction_time': rec_time
    },
    'metrics': {
        'mse': mse,
        'psnr': psnr,
        'max_error': max_error,
        'relative_error': relative_error
    }
}

results_file = os.path.join(output_dir, 'python_nsct_results.pkl')
with open(results_file, 'wb') as f:
    pickle.dump(results, f)
print(f"   结果已保存到: {results_file}")

# 8. 保存详细的子带信息
print("\n8. 保存详细的子带信息...")
subband_info = {
    'lowpass': {
        'shape': y[0].shape,
        'data': y[0],
        'mean': np.mean(y[0]),
        'std': np.std(y[0]),
        'min': np.min(y[0]),
        'max': np.max(y[0])
    },
    'bandpass': []
}

for i in range(1, len(y)):
    if isinstance(y[i], list):
        scale_info = {
            'scale': i,
            'num_directions': len(y[i]),
            'directions': []
        }
        for j, band in enumerate(y[i]):
            dir_info = {
                'direction': j,
                'shape': band.shape,
                'data': band,
                'mean': np.mean(band),
                'std': np.std(band),
                'min': np.min(band),
                'max': np.max(band)
            }
            scale_info['directions'].append(dir_info)
        subband_info['bandpass'].append(scale_info)
    else:
        scale_info = {
            'scale': i,
            'num_directions': 0,
            'shape': y[i].shape,
            'data': y[i],
            'mean': np.mean(y[i]),
            'std': np.std(y[i]),
            'min': np.min(y[i]),
            'max': np.max(y[i])
        }
        subband_info['bandpass'].append(scale_info)

subband_file = os.path.join(output_dir, 'python_subband_details.pkl')
with open(subband_file, 'wb') as f:
    pickle.dump(subband_info, f)
print(f"   子带详细信息已保存到: {subband_file}")

print("\n" + "=" * 70)
print("Python 版本完成！")
print("=" * 70)
