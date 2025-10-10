from nsct_torch.core import nsctdec, nsctrec
import torch
import torchvision.transforms as transforms
from PIL import Image


def main():
    # 方法1: 使用 torchvision transforms (推荐)
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ])
    
    image = Image.open("test.png")
    input_image = transform(image).cuda()  # 直接读取为 torch.Tensor 并移到 CUDA
    
    # 或者方法2: 使用 PIL + torch
    # image = Image.open("test.png").convert("L")
    # import numpy as np
    # input_image = torch.from_numpy(np.array(image)).float().cuda()
    
    coeffs = nsctdec(input_image, [2, 2])
    reconstructed_image = nsctrec(coeffs)
    # Compare input_image and reconstructed_image

if __name__ == "__main__":
    main()
