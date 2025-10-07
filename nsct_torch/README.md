# NSCT Torch

PyTorch GPU-accelerated implementation of Nonsubsampled Contourlet Transform (NSCT).

## 项目状态

本项目正在从 `nsct_python` (基于 NumPy 和 C++) 迁移到 `nsct_torch` (基于 PyTorch 和 CUDA)。

### 已完成

✅ **项目初始化**
- 创建了基本的目录结构
- 设置了 `setup.py` 用于编译 CUDA 扩展
- 配置了依赖项 (`requirements.txt`)

✅ **工具函数迁移** (`nsct_torch/utils.py`)
- `extend2` - 2D 图像扩展
- `symext` - 对称扩展
- `upsample2df` - 2D 滤波器上采样
- `modulate2` - 2D 调制
- `resampz` - 矩阵重采样（剪切）
- `qupz` - 准棋盘格上采样

✅ **滤波器函数迁移** (`nsct_torch/filters/`)
- `ld2quin` - 从梯形网络结构全通滤波器构造准棋盘格滤波器
- `efilter2` - 带边缘处理的 2D 滤波
- `dmaxflat` - 2D 菱形最平坦滤波器
- `atrousfilters` - 金字塔 2D 滤波器生成
- `mctrans` - McClellan 变换
- `ldfilter` - 梯形结构网络滤波器
- `dfilters` - 方向 2D 滤波器（菱形滤波器对）
- `parafilters` - 从菱形滤波器生成平行四边形滤波器

### 待完成

⏳ **CUDA 内核开发**
- [ ] `atrousc_kernel.cu` - à trous 卷积的 CUDA 实现
- [ ] `zconv2_kernel.cu` - 周期性边界卷积的 CUDA 实现
- [ ] `nsct_cuda.cpp` - PyTorch C++ 扩展绑定

⏳ **核心算法迁移** (`nsct_torch/core.py`)
- [ ] `nsctdec` - NSCT 分解
- [ ] `nsctrec` - NSCT 重构
- [ ] 其他辅助函数

⏳ **测试与验证**
- [ ] 单元测试
- [ ] 与 `nsct_python` 的数值一致性测试
- [ ] 端到端测试

⏳ **性能优化与文档**
- [ ] 性能基准测试
- [ ] API 文档
- [ ] 使用示例

## 项目结构

```
nsct_torch/
├── nsct_torch/              # Python 包
│   ├── __init__.py         # 包初始化
│   ├── utils.py            # 工具函数（已完成）
│   ├── filters/            # 滤波器模块（已完成）
│   │   ├── __init__.py
│   │   ├── ld2quin.py
│   │   ├── efilter2.py
│   │   ├── dmaxflat.py
│   │   ├── atrousfilters.py
│   │   ├── mctrans.py
│   │   ├── ldfilter.py
│   │   ├── dfilters.py
│   │   └── parafilters.py
│   └── core.py             # 核心算法（待完成）
├── csrc/                    # C++/CUDA 源码（待完成）
│   ├── nsct_cuda.cpp       # PyTorch 扩展绑定
│   ├── atrousc_kernel.cu   # à trous 卷积 CUDA 内核
│   └── zconv2_kernel.cu    # 周期性卷积 CUDA 内核
├── tests/                   # 测试（待完成）
├── setup.py                # 安装脚本
├── requirements.txt        # 依赖项
└── README.md              # 本文件
```

## 安装

### 依赖项

```bash
pip install -r requirements.txt
```

主要依赖：
- PyTorch >= 2.0.0
- NumPy >= 1.20.0
- SciPy >= 1.7.0
- PyWavelets (用于 1D 小波滤波器)

### 编译 CUDA 扩展（待实现）

```bash
python setup.py install
```

## 使用示例（待完成）

```python
import torch
import nsct_torch

# 加载图像
image = torch.randn(256, 256)

# NSCT 分解（待实现）
# coeffs = nsct_torch.nsctdec(image, ...)

# NSCT 重构（待实现）
# reconstructed = nsct_torch.nsctrec(coeffs, ...)
```

## 设计原则

1. **KISS 原则** - 保持简单直接，避免过度工程
2. **DRY 原则** - 避免代码重复，提取公共逻辑
3. **模块化** - 将大型文件拆分为独立的功能模块
4. **GPU 加速** - 利用 CUDA 实现性能关键部分
5. **数值一致性** - 确保与原 `nsct_python` 结果一致

## 开发计划

详细的迁移计划请参考项目根目录的 `plan.md` 文件。

## 许可证

待定

## 贡献

本项目正在积极开发中。欢迎提出建议和反馈！
