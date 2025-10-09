"""
对比 NumPy 和 PyTorch 实现的 NSCT 分解结果，并进行可视化。

此脚本执行以下任务:
1. 加载测试图像
2. 使用 NumPy 实现进行 NSCT 分解
3. 使用 PyTorch 实现进行 NSCT 分解
4. 对比分解结果的数值一致性
5. 可视化分解子带
"""

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，直接保存图片
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import sys

# 添加项目根目录到 Python 路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from nsct_python.core import nsctdec as nsctdec_np, nsctrec as nsctrec_np
from nsct_torch.core import nsctdec as nsctdec_torch, nsctrec as nsctrec_torch


def load_test_image(image_path: Path) -> np.ndarray:
    """加载测试图像并转换为灰度图像。"""
    with Image.open(image_path) as img:
        if img.mode == "RGB":
            img = img.convert("L")
        return np.asarray(img, dtype=np.float64)


def numpy_to_torch(arr: np.ndarray) -> torch.Tensor:
    """将 NumPy 数组转换为 PyTorch 张量（使用 float64 精度），并移至 CUDA 设备。"""
    return torch.from_numpy(arr.copy()).cuda()


def torch_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """将 PyTorch 张量转换为 NumPy 数组。"""
    return tensor.detach().cpu().numpy()


def compare_decompositions(decomp_np, decomp_torch, levels):
    """
    对比 NumPy 和 PyTorch 分解结果的数值一致性。
    
    Args:
        decomp_np: NumPy 分解结果
        decomp_torch: PyTorch 分解结果
        levels: 金字塔层级配置
    
    Returns:
        dict: 包含对比统计信息的字典
    """
    stats = {
        'lowpass': {},
        'directional': []
    }
    
    # 对比低通带
    lowpass_np = decomp_np[0]
    lowpass_torch = torch_to_numpy(decomp_torch[0])
    
    diff = np.abs(lowpass_np - lowpass_torch)
    stats['lowpass'] = {
        'max_error': float(np.max(diff)),
        'mean_error': float(np.mean(diff)),
        'rmse': float(np.sqrt(np.mean((lowpass_np - lowpass_torch) ** 2))),
        'shape_match': lowpass_np.shape == lowpass_torch.shape,
        'allclose': np.allclose(lowpass_np, lowpass_torch, rtol=1e-10, atol=1e-12)
    }
    
    # 对比方向子带
    for level_idx, num_directions in enumerate(levels):
        level_stats = {
            'level': level_idx,
            'num_directions': num_directions,
            'expected_subbands': 2 ** num_directions,
            'bands': []
        }
        
        bands_np = decomp_np[level_idx + 1]
        bands_torch = decomp_torch[level_idx + 1]
        
        for dir_idx in range(len(bands_np)):
            band_np = bands_np[dir_idx]
            band_torch = torch_to_numpy(bands_torch[dir_idx])
            
            diff = np.abs(band_np - band_torch)
            band_stats = {
                'direction': dir_idx,
                'max_error': float(np.max(diff)),
                'mean_error': float(np.mean(diff)),
                'rmse': float(np.sqrt(np.mean((band_np - band_torch) ** 2))),
                'shape_match': band_np.shape == band_torch.shape,
                'allclose': np.allclose(band_np, band_torch, rtol=1e-10, atol=1e-12)
            }
            level_stats['bands'].append(band_stats)
        
        stats['directional'].append(level_stats)
    
    return stats


def print_comparison_results(stats):
    """打印对比结果统计信息。"""
    print("\n" + "=" * 80)
    print("NSCT 分解结果对比: NumPy vs PyTorch")
    print("=" * 80)
    
    # 低通带统计
    print("\n【低通带对比】")
    print(f"  形状匹配: {stats['lowpass']['shape_match']}")
    print(f"  数值接近: {stats['lowpass']['allclose']}")
    print(f"  最大误差: {stats['lowpass']['max_error']:.2e}")
    print(f"  平均误差: {stats['lowpass']['mean_error']:.2e}")
    print(f"  RMSE:     {stats['lowpass']['rmse']:.2e}")
    
    # 方向子带统计
    print("\n【方向子带对比】")
    for level_stats in stats['directional']:
        level_idx = level_stats['level']
        num_bands = len(level_stats['bands'])
        print(f"\n  层级 {level_idx} (配置: {level_stats['num_directions']}, 子带数: {num_bands}):")
        
        all_match = all(b['shape_match'] for b in level_stats['bands'])
        all_close = all(b['allclose'] for b in level_stats['bands'])
        max_errors = [b['max_error'] for b in level_stats['bands']]
        mean_errors = [b['mean_error'] for b in level_stats['bands']]
        
        print(f"    所有形状匹配: {all_match}")
        print(f"    所有数值接近: {all_close}")
        print(f"    最大误差范围: [{min(max_errors):.2e}, {max(max_errors):.2e}]")
        print(f"    平均误差范围: [{min(mean_errors):.2e}, {max(mean_errors):.2e}]")
        
        # 显示前3个子带的详细信息
        for band_stats in level_stats['bands'][:3]:
            dir_idx = band_stats['direction']
            print(f"      方向 {dir_idx}: max_err={band_stats['max_error']:.2e}, "
                  f"mean_err={band_stats['mean_error']:.2e}, "
                  f"match={band_stats['allclose']}")
        
        if len(level_stats['bands']) > 3:
            print(f"      ... (共 {len(level_stats['bands'])} 个方向子带)")
    
    print("\n" + "=" * 80)


def visualize_decomposition(decomp, title_prefix, levels, save_path=None):
    """
    可视化 NSCT 分解结果。
    
    Args:
        decomp: 分解结果（NumPy 或 PyTorch）
        title_prefix: 图表标题前缀
        levels: 金字塔层级配置
        save_path: 可选的保存路径
    """
    # 转换 PyTorch 张量为 NumPy
    if torch.is_tensor(decomp[0]):
        decomp = [[torch_to_numpy(decomp[0])]] + \
                 [[torch_to_numpy(b) for b in level] for level in decomp[1:]]
    else:
        decomp = [[decomp[0]]] + [list(level) for level in decomp[1:]]
    
    # 计算总的子图数量
    total_subbands = 1  # 低通带
    for level in levels:
        total_subbands += 2 ** level
    
    # 计算网格布局
    n_cols = 4
    n_rows = (total_subbands + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    subplot_idx = 0
    
    # 绘制低通带
    ax = axes[subplot_idx]
    lowpass = decomp[0][0]
    im = ax.imshow(lowpass, cmap='gray')
    ax.set_title(f'{title_prefix} - 低通带\n{lowpass.shape}')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    subplot_idx += 1
    
    # 绘制方向子带
    for level_idx, level_bands in enumerate(decomp[1:]):
        for dir_idx, band in enumerate(level_bands):
            if subplot_idx >= len(axes):
                break
            
            ax = axes[subplot_idx]
            # 对于可视化，显示幅度
            band_abs = np.abs(band)
            im = ax.imshow(band_abs, cmap='hot')
            ax.set_title(f'{title_prefix} - L{level_idx} D{dir_idx}\n{band.shape}')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            subplot_idx += 1
    
    # 隐藏未使用的子图
    for idx in range(subplot_idx, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"可视化已保存至: {save_path}")
    
    return fig


def visualize_difference(decomp_np, decomp_torch, levels, save_path=None):
    """
    可视化 NumPy 和 PyTorch 分解结果的差异。
    
    Args:
        decomp_np: NumPy 分解结果
        decomp_torch: PyTorch 分解结果
        levels: 金字塔层级配置
        save_path: 可选的保存路径
    """
    # 计算总的子图数量
    total_subbands = 1  # 低通带
    for level in levels:
        total_subbands += 2 ** level
    
    # 计算网格布局
    n_cols = 4
    n_rows = (total_subbands + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    subplot_idx = 0
    
    # 低通带差异
    ax = axes[subplot_idx]
    lowpass_diff = np.abs(decomp_np[0] - torch_to_numpy(decomp_torch[0]))
    im = ax.imshow(lowpass_diff, cmap='viridis')
    max_diff = np.max(lowpass_diff)
    ax.set_title(f'差异 - 低通带\nmax: {max_diff:.2e}')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    subplot_idx += 1
    
    # 方向子带差异
    for level_idx, _ in enumerate(levels):
        bands_np = decomp_np[level_idx + 1]
        bands_torch = decomp_torch[level_idx + 1]
        
        for dir_idx in range(len(bands_np)):
            if subplot_idx >= len(axes):
                break
            
            ax = axes[subplot_idx]
            band_diff = np.abs(bands_np[dir_idx] - torch_to_numpy(bands_torch[dir_idx]))
            im = ax.imshow(band_diff, cmap='viridis')
            max_diff = np.max(band_diff)
            ax.set_title(f'差异 - L{level_idx} D{dir_idx}\nmax: {max_diff:.2e}')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            subplot_idx += 1
    
    # 隐藏未使用的子图
    for idx in range(subplot_idx, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"差异可视化已保存至: {save_path}")
    
    return fig


def test_reconstruction(image, decomp_np, decomp_torch, dfilt, pfilt):
    """测试重建质量。"""
    print("\n" + "=" * 80)
    print("重建质量测试")
    print("=" * 80)
    
    # NumPy 重建
    recon_np = nsctrec_np(decomp_np, dfilt, pfilt)
    diff_np = np.abs(recon_np - image)
    
    print("\n【NumPy 重建】")
    print(f"  最大误差: {np.max(diff_np):.2e}")
    print(f"  平均误差: {np.mean(diff_np):.2e}")
    print(f"  RMSE:     {np.sqrt(np.mean(diff_np ** 2)):.2e}")
    print(f"  完美重建: {np.allclose(recon_np, image, rtol=1e-12, atol=1e-10)}")
    
    # PyTorch 重建（使用 float64 精度）
    image_torch = numpy_to_torch(image)
    recon_torch = nsctrec_torch(decomp_torch, dfilt, pfilt, dtype=torch.float64)
    recon_torch_np = torch_to_numpy(recon_torch)
    diff_torch = np.abs(recon_torch_np - image)
    
    print("\n【PyTorch 重建】")
    print(f"  最大误差: {np.max(diff_torch):.2e}")
    print(f"  平均误差: {np.mean(diff_torch):.2e}")
    print(f"  RMSE:     {np.sqrt(np.mean(diff_torch ** 2)):.2e}")
    print(f"  完美重建: {np.allclose(recon_torch_np, image, rtol=1e-12, atol=1e-10)}")
    
    # 对比两种重建结果
    diff_recons = np.abs(recon_np - recon_torch_np)
    print("\n【NumPy vs PyTorch 重建对比】")
    print(f"  最大差异: {np.max(diff_recons):.2e}")
    print(f"  平均差异: {np.mean(diff_recons):.2e}")
    print(f"  一致性:   {np.allclose(recon_np, recon_torch_np, rtol=1e-10, atol=1e-12)}")
    
    print("=" * 80)
    
    return recon_np, recon_torch_np


def main():
    """主函数。"""
    # 设置路径
    script_root = Path(__file__).resolve().parent
    project_root = script_root.parent
    test_image_path = project_root / "test_image.jpg"
    output_dir = script_root / "results"
    output_dir.mkdir(exist_ok=True)
    
    # NSCT 参数
    levels = [2, 3]
    dfilt = "dmaxflat7"
    pfilt = "maxflat"
    
    print("=" * 80)
    print("NSCT 分解对比: NumPy vs PyTorch")
    print("=" * 80)
    print(f"图像路径: {test_image_path}")
    print(f"金字塔层级: {levels}")
    print(f"方向滤波器: {dfilt}")
    print(f"金字塔滤波器: {pfilt}")
    
    # 加载图像
    print("\n正在加载图像...")
    image = load_test_image(test_image_path)
    print(f"图像形状: {image.shape}")
    print(f"图像范围: [{np.min(image):.2f}, {np.max(image):.2f}]")
    
    # NumPy 分解
    print("\n正在执行 NumPy NSCT 分解...")
    decomp_np = nsctdec_np(image, levels, dfilt, pfilt)
    print(f"完成! 低通带形状: {decomp_np[0].shape}")
    for i, level in enumerate(decomp_np[1:]):
        print(f"  层级 {i}: {len(level)} 个方向子带")
    
    # PyTorch 分解（使用 float64 精度）
    print("\n正在执行 PyTorch NSCT 分解...")
    image_torch = numpy_to_torch(image)
    print(f"PyTorch 张量 dtype: {image_torch.dtype}")
    decomp_torch = nsctdec_torch(image_torch, levels, dfilt, pfilt, dtype=torch.float32)
    print(f"完成! 低通带形状: {decomp_torch[0].shape}, dtype: {decomp_torch[0].dtype}")
    for i, level in enumerate(decomp_torch[1:]):
        print(f"  层级 {i}: {len(level)} 个方向子带")
    
    # 对比分解结果
    print("\n正在对比分解结果...")
    stats = compare_decompositions(decomp_np, decomp_torch, levels)
    print_comparison_results(stats)
    
    # 测试重建
    recon_np, recon_torch = test_reconstruction(
        image, decomp_np, decomp_torch, dfilt, pfilt
    )
    
    # 可视化
    print("\n正在生成可视化...")
    
    # 原始图像
    plt.figure(figsize=(8, 6))
    plt.imshow(image, cmap='gray')
    plt.title(f'原始图像\n{image.shape}')
    plt.colorbar()
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_dir / "original_image.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"原始图像已保存: {output_dir / 'original_image.png'}")
    
    # NumPy 分解可视化
    visualize_decomposition(
        decomp_np, "NumPy", levels,
        save_path=output_dir / "decomposition_numpy.png"
    )
    plt.close()
    
    # PyTorch 分解可视化
    visualize_decomposition(
        decomp_torch, "PyTorch", levels,
        save_path=output_dir / "decomposition_torch.png"
    )
    plt.close()
    
    # 差异可视化
    visualize_difference(
        decomp_np, decomp_torch, levels,
        save_path=output_dir / "decomposition_difference.png"
    )
    plt.close()
    
    # 重建对比
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('原始图像')
    axes[0].axis('off')
    
    axes[1].imshow(recon_np, cmap='gray')
    axes[1].set_title('NumPy 重建')
    axes[1].axis('off')
    
    axes[2].imshow(recon_torch, cmap='gray')
    axes[2].set_title('PyTorch 重建')
    axes[2].axis('off')
    
    diff_display = np.abs(recon_np - recon_torch)
    im = axes[3].imshow(diff_display, cmap='viridis')
    axes[3].set_title(f'重建差异\nmax: {np.max(diff_display):.2e}')
    axes[3].axis('off')
    plt.colorbar(im, ax=axes[3])
    
    plt.tight_layout()
    plt.savefig(output_dir / "reconstruction_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"重建对比已保存: {output_dir / 'reconstruction_comparison.png'}")
    
    # 保存统计信息到文件
    stats_file = output_dir / "comparison_stats.txt"
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("NSCT 分解结果对比: NumPy vs PyTorch\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"图像路径: {test_image_path}\n")
        f.write(f"图像形状: {image.shape}\n")
        f.write(f"金字塔层级: {levels}\n")
        f.write(f"方向滤波器: {dfilt}\n")
        f.write(f"金字塔滤波器: {pfilt}\n\n")
        
        f.write("【低通带对比】\n")
        f.write(f"  形状匹配: {stats['lowpass']['shape_match']}\n")
        f.write(f"  数值接近: {stats['lowpass']['allclose']}\n")
        f.write(f"  最大误差: {stats['lowpass']['max_error']:.2e}\n")
        f.write(f"  平均误差: {stats['lowpass']['mean_error']:.2e}\n")
        f.write(f"  RMSE:     {stats['lowpass']['rmse']:.2e}\n\n")
        
        f.write("【方向子带对比】\n")
        for level_stats in stats['directional']:
            level_idx = level_stats['level']
            f.write(f"\n层级 {level_idx}:\n")
            for band_stats in level_stats['bands']:
                dir_idx = band_stats['direction']
                f.write(f"  方向 {dir_idx}: ")
                f.write(f"max_err={band_stats['max_error']:.2e}, ")
                f.write(f"mean_err={band_stats['mean_error']:.2e}, ")
                f.write(f"match={band_stats['allclose']}\n")
    
    print(f"统计信息已保存: {stats_file}")
    
    print("\n" + "=" * 80)
    print("对比完成!")
    print(f"所有结果已保存至: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
