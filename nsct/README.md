# NSCT 批处理数据流指南

本文档聚焦 `nsct` 包中批处理接口（`nsct.api`）的实际数据流。从磁盘载入图像张量开始，依次梳理分解、方向滤波、金字塔处理以及最终重建阶段的中间形状变化，便于开发者在调试或扩展 CUDA 核心算子时定位问题。

## 环境准备

1. 安装依赖并确保有 CUDA 可用的 PyTorch。
2. 在仓库根目录构建扩展：
   ```bash
   python setup.py build_ext --inplace
   ```
3. 所有批处理入口都假定输入张量在 GPU 上（`torch.device("cuda")`）。

## 数据加载与规范化


```python
from pathlib import Path
import numpy as np
import torch
from PIL import Image

def load_dataset(data_dir: Path, device: torch.device) -> torch.Tensor:
    tensors = []
    for path in sorted(data_dir.glob("*.png")):
        with Image.open(path) as img:
            image = img.convert("F")               # 灰度 float32
        tensor = torch.from_numpy(np.array(image, dtype=np.float32))
        tensor = tensor.unsqueeze(0)               # C=1
        tensors.append(tensor)

    batch = torch.stack(tensors, dim=0)            # [B, C, H, W]
    return (batch / 255.0).to(device=device)
```

- **输入文件**：`H×W` 或 `H×W×C` 图像。
- **规范化后形状**：`[B, C, H, W]`（批次、通道、空间尺寸）。
- **张量所在设备**：GPU。

## 分解流程总览

`nsct.api.nsct_batch_dec` 是批处理入口，内部步骤如下：

| 步骤 | 位置 | 输入形状 | 输出形状 | 要点 |
| --- | --- | --- | --- | --- |
| 1 | `_flatten_batch` | `[B, C, H, W]` | `[B·C, H, W]` | 将批次与通道合并，形成“伪批次”以复用 3D CUDA 核心。 |
| 2 | `core.nsctdec` | `[N, H, W]` (`N=B·C`) | 系数树（列表） | 逐层执行非下采样金字塔 + 方向滤波，保持空间尺度不变。 |
| 3 | `_reshape_coeff_tree` | 系数树（每项 `[N, H, W]`） | 系数树（每项 `[B, C, H, W]`） | 将阶段 2 的结果还原批次和通道语义。 |

### 金字塔分解（`core.nsctdec`）

对每个“图像”（`[N, H, W]`）逐层执行：

1. **`nsfbdec`（无下采样滤波金字塔）**  
   - 输入：上一层低频 `x_current`（`[N, H, W]`）  
   - 输出：`x_low`、`x_high`，形状均为 `[N, H, W]`  
   - 说明：调用 `extend2`/`symext` 进行边界扩展，并使用 CUDA 算子 `atrousc`、`zconv2` 进行非下采样卷积。

2. **`nsdfbdec`（方向滤波）**  
   - 输入：`x_high`，形状 `[N, H, W]`  
   - 输出：长度 `2^d` 的列表，每个方向子带的张量仍为 `[N, H, W]`  
   - 说明：首层调用 `nssfbdec`（无上采样），更高层使用不同的 quincunx 采样矩阵来模拟方向分解。

3. 将方向子带列表写入 `coeffs[level]`，`x_low` 作为下一层输入，直至完成所有金字塔层级。最终 `coeffs[0]` 为最底层低频 `[N, H, W]`。

### 系数树结构（批处理重排后）

```
coeffs = [
    lowpass,                         # torch.Tensor, [B, C, H, W]
    [dir_0, dir_1, ..., dir_{k-1}],  # List[Tensor], 每个 [B, C, H, W]
    ...                              # 逐层对应 nlevs，从粗到细
]
```

列表索引从 1 开始对应最粗的方向层，`nlevs` 中的最后一个值指向最顶层。

## 重建流程总览

`nsct.api.nsct_batch_rec` 反向执行上述过程：

| 步骤 | 位置 | 输入形状 | 输出形状 | 要点 |
| --- | --- | --- | --- | --- |
| 1 | `_extract_batch_channel` | 系数树（第一项 `[B, C, H, W]`） | `B, C` | 读取批次/通道大小以验证后续重排。 |
| 2 | `_flatten_coeff_tree` | 系数树（`[B, C, H, W]`） | 系数树（`[B·C, H, W]`） | 与分解阶段对称地合并维度。 |
| 3 | `core.nsctrec` | 系数树（`[N, H, W]`） | `[N, H, W]` | 逐层执行方向重建（`nsdfbrec`）与金字塔重建（`nsfbrec`）。 |
| 4 | `reshape` | `[N, H, W]` | `[B, C, H, W]` | 恢复原始批次和通道结构。 |

### 金字塔与方向重建细节

1. `nsdfbrec`：补偿方向子带，先将 `[N, H, W]` 张量列表合成为单一高频张量。
2. `nsfbrec`：与 `nsfbdec` 对偶的两通道非下采样滤波器，逐层将高频补偿回低频。
3. 最终输出 `recon`（`[B, C, H, W]`），与输入形状完全一致。

## 典型调用示例

```python
import torch
from nsct.api import nsct_batch_dec, nsct_batch_rec

device = torch.device("cuda")
x = torch.randn(8, 1, 256, 256, device=device)  # 批大小 8
nlevs = [3, 3, 2]

coeffs = nsct_batch_dec(x, nlevs)
recon = nsct_batch_rec(coeffs)

print(f"最大重建误差: {(recon - x).abs().max().item():.3e}")
```

- `coeffs[0]`：最低频张量 `[8, 1, 256, 256]`。
- `coeffs[1]`：方向子带列表长度 `2^2 = 4`，每个张量 `[8, 1, 256, 256]`。
- `coeffs[2]`：列表长度 `2^3 = 8`，形状同上。
- `coeffs[3]`：列表长度 `2^3 = 8`。

## 中间结果与调试

- `scripts/test_batch.py` 会同时运行批处理与旧版逐样本流程，并将系数树、重建结果以及计时信息保存到 `results/batched/` 与 `results/sequential/`。
- 若需要验证特定方向子带，可以直接从 `coeffs[level][direction]` 中取出 `[B, C, H, W]` 张量进行可视化或进一步运算。
- 若引入新的 CUDA 核心，请重点关注 `[B·C, H, W]` 张量在进入 `core.nsctdec/nsctrec` 前后的形状保持。

## 常见问题

- **输入维度错误**：若传入的张量不是四维，`_flatten_batch` 会抛出异常提示期望 `[B, C, H, W]`。
- **特定子带缺失**：方向层级设置为 0 时，`coeffs[level]` 直接返回单个 `[B, C, H, W]` 张量而不是列表。
- **CPU 张量**：CUDA 核心不支持 CPU 输入，请提前调用 `.to("cuda")`。

通过以上形状追踪，可快速定位数据在批处理 NSCT 管线中的流动位置，方便调试和扩展。
