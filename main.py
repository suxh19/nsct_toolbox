from nsct_torch.core import nsctdec, nsctrec
import torch
from PIL import Image
import numpy as np
import os


def main():
    # 创建 results 文件夹
    os.makedirs("results", exist_ok=True)
    
    # 读取输入图像
    image = Image.open("test.png").convert("L")
    input_image = torch.from_numpy(np.array(image)).float().cuda()
    
    # NSCT 分解
    coeffs = nsctdec(input_image, [2, 2])
    
    # 保存分解系数
    torch.save(coeffs, "results/nsct_coeffs.pt")
    print("分解系数已保存到: results/nsct_coeffs.pt")
    
    # 保存低频子带
    lowpass = coeffs[0].cpu().numpy()
    lowpass_normalized = ((lowpass - lowpass.min()) / (lowpass.max() - lowpass.min()) * 255).astype(np.uint8)
    Image.fromarray(lowpass_normalized).save("results/lowpass.png")
    print("低频子带已保存到: results/lowpass.png")
    
    # 保存各个方向的高频子带
    for level, bandpass_level in enumerate(coeffs[1:], 1):
        for direction, bandpass in enumerate(bandpass_level):
            bandpass_np = bandpass.cpu().numpy()
            # 归一化到 0-255
            bandpass_normalized = ((bandpass_np - bandpass_np.min()) / (bandpass_np.max() - bandpass_np.min()) * 255).astype(np.uint8)
            filename = f"results/level{level}_dir{direction}.png"
            Image.fromarray(bandpass_normalized).save(filename)
            print(f"高频子带已保存到: {filename}")
    
    # 重构图像
    reconstructed_image = nsctrec(coeffs)
    
    # 保存重构图像
    reconstructed_np = reconstructed_image.cpu().numpy().astype(np.uint8)
    Image.fromarray(reconstructed_np).save("results/reconstructed.png")
    print("重构图像已保存到: results/reconstructed.png")
    
    # 计算重构误差
    error = torch.abs(input_image - reconstructed_image).mean().item()
    print(f"\n平均重构误差: {error:.6f}")

if __name__ == "__main__":
    main()
