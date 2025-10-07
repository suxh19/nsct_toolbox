# NSCT PyTorch Implementation

这是 NSCT (Nonsubsampled Contourlet Transform) 的 PyTorch 实现版本，从原始的 NumPy 实现转换而来。

## 特性

- ✅ 完全基于 PyTorch 实现
- ✅ 支持 GPU 加速
- ✅ 支持自动微分
- ✅ 与原始 NumPy 版本输出完全一致
- ✅ 所有函数都经过测试验证

## 文件结构

```
nsct_torch/
├── __init__.py      # 包初始化文件
├── utils.py         # 工具函数（PyTorch 版本）
└── README.md        # 本文件
```

## 已实现的函数

### `extend2(x, ru, rd, cl, cr, extmod='per')`
2D 图像扩展函数，支持：
- `'per'`: 周期性扩展
- `'qper_row'`: 行方向的 quincunx 周期性扩展
- `'qper_col'`: 列方向的 quincunx 周期性扩展

### `symext(x, h, shift)`
对称扩展函数，用于滤波器处理。

### `upsample2df(h, power=1)`
通过插入零点对 2D 滤波器进行上采样。

### `modulate2(x, mode='b', center=None)`
2D 调制函数，支持行、列或双向调制。

### `resampz(x, type, shift=1)`
矩阵重采样（剪切变换）。

### `qupz(x, type=1)`
Quincunx 上采样。

## 使用示例

```python
import torch
from nsct_torch import extend2, upsample2df, modulate2

# 创建测试图像
img = torch.arange(16).reshape((4, 4)).float()

# 周期性扩展
ext_img = extend2(img, 1, 1, 1, 1, 'per')
print(f"Extended shape: {ext_img.shape}")  # (6, 6)

# 在 GPU 上运行
if torch.cuda.is_available():
    img_gpu = img.cuda()
    ext_img_gpu = extend2(img_gpu, 1, 1, 1, 1, 'per')
    print(f"GPU result shape: {ext_img_gpu.shape}")

# 滤波器上采样
h = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
h_up = upsample2df(h, power=1)
print(f"Upsampled filter:\n{h_up}")

# 2D 调制
m = torch.ones((3, 4))
m_mod = modulate2(m, 'b')
print(f"Modulated matrix:\n{m_mod}")
```

## 运行测试

```bash
# 激活虚拟环境
.venv\Scripts\Activate.ps1

# 运行测试
python nsct_torch/utils.py
```

## 与 NumPy 版本的对比

| 特性 | NumPy 版本 | PyTorch 版本 |
|------|-----------|-------------|
| GPU 支持 | ❌ | ✅ |
| 自动微分 | ❌ | ✅ |
| 批处理 | 需要手动实现 | 原生支持 |
| 性能 | 一般 | 更快（尤其是 GPU） |
| API 兼容性 | NumPy | PyTorch |

## 主要转换说明

1. **数组创建**：`np.array()` → `torch.tensor()`
2. **填充**：`np.pad()` → `F.pad()`，注意 circular 模式需要添加 batch 维度
3. **翻转**：`np.fliplr()`/`np.flipud()` → `torch.fliplr()`/`torch.flipud()`
4. **拼接**：`np.concatenate()` → `torch.cat()`
5. **范数计算**：`np.linalg.norm()` → `torch.linalg.norm()`
6. **设备支持**：所有函数自动支持 CUDA 设备

## 注意事项

- PyTorch 的 `F.pad` 在使用 `circular` 模式时需要至少 3D 张量，因此在 2D 情况下需要临时添加 batch 维度
- 所有函数都保持了与原始 MATLAB/NumPy 版本相同的行为和输出
- 索引操作已经从 NumPy 风格转换为 PyTorch 风格

## 后续开发

接下来需要转换的文件：
- [ ] `core.py` - 核心 NSCT 变换函数
- [ ] `filters.py` - 滤波器设计函数
- [ ] C++ 扩展的 PyTorch 版本（如果需要）

## 依赖

- Python >= 3.8
- PyTorch >= 1.9.0

## 许可

与原项目保持一致。
