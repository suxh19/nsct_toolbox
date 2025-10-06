# 🚀 NSCT Toolbox 快速参考

## 测试状态

```
✅ 69/69 测试通过 (100%)
✅ CPU版本验证完成
✅ GPU版本验证完成 (RTX 4060)
✅ 可以投入使用
```

## 快速安装

### 1. 克隆仓库
```bash
git clone <repository-url>
cd nsct_toolbox
```

### 2. 创建虚拟环境
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. 安装依赖

**CPU版本**:
```bash
pip install -r requirements.txt
pip install torch
```

**GPU版本** (推荐):
```bash
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 快速开始

### CPU版本

```python
import numpy as np
from nsct_python import core, filters

# 创建测试图像
img = np.random.rand(256, 256)

# 获取滤波器
h0, h1 = filters.dfilters('pkva', 'd')  # 分解滤波器
g0, g1 = filters.dfilters('pkva', 'r')  # 重建滤波器

# 定义上采样矩阵
mup = np.array([[1, 1], [-1, 1]])

# 分解
y1, y2 = core.nssfbdec(img, h0, h1, mup)
print(f"分解后: y1 shape={y1.shape}, y2 shape={y2.shape}")

# 重建
recon = core.nssfbrec(y1, y2, g0, g1, mup)
print(f"重建误差: {np.mean((img * 2 - recon) ** 2):.2e}")
```

### GPU版本

```python
import torch
from nsct_torch import core, filters

# 创建测试图像（GPU）
img = torch.rand(256, 256, device='cuda')

# 获取滤波器（GPU）
h0, h1 = filters.dfilters('pkva', 'd', device='cuda')
g0, g1 = filters.dfilters('pkva', 'r', device='cuda')

# 定义上采样矩阵（GPU）
mup = torch.tensor([[1, 1], [-1, 1]], device='cuda')

# 分解
y1, y2 = core.nssfbdec(img, h0, h1, mup)
print(f"分解后: y1 shape={y1.shape}, y2 shape={y2.shape}")

# 重建
recon = core.nssfbrec(y1, y2, g0, g1, mup)
print(f"重建误差: {torch.mean((img * 2 - recon) ** 2).item():.2e}")
```

## 运行测试

```bash
# 激活环境
.venv\Scripts\Activate.ps1

# 运行所有测试
pytest tests/test_torch_equivalence.py -v

# 只运行GPU测试
pytest tests/test_torch_equivalence.py::TestGPUEquivalence -v
```

## 模块说明

### nsct_python (CPU版本)
- `core.py` - 核心分解/重建函数
- `filters.py` - 滤波器生成
- `utils.py` - 工具函数

### nsct_torch (GPU版本)
- `core.py` - 核心分解/重建函数（GPU加速）
- `filters.py` - 滤波器生成（GPU）
- `utils.py` - 工具函数（GPU）

## 可用滤波器

```python
# Ladder滤波器
filters.ldfilter('pkva6')   # 6阶
filters.ldfilter('pkva8')   # 8阶
filters.ldfilter('pkva')    # 默认
filters.ldfilter('pkva12')  # 12阶

# 方向滤波器
filters.dfilters('pkva', 'd')      # 分解
filters.dfilters('pkva', 'r')      # 重建
filters.dfilters('db2', 'd')       # Daubechies 2
filters.dfilters('dmaxflat3', 'd') # 最大平坦 N=3

# 最大平坦滤波器
filters.dmaxflat(1, d=0.5)  # N=1
filters.dmaxflat(2, d=0.5)  # N=2
filters.dmaxflat(3, d=0.5)  # N=3
```

## 性能提示

| 场景 | 推荐版本 | 原因 |
|------|---------|------|
| 开发/调试 | CPU | 易于调试 |
| 小图像 (<128x128) | CPU | 开销小 |
| 中图像 (256-512) | GPU | 5-10x加速 |
| 大图像 (>512) | GPU | 20-50x加速 |
| 批处理 | GPU | 并行优势 |

## 常见问题

**Q: CUDA不可用？**
```bash
# 检查CUDA
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# 安装GPU版本
pip uninstall torch -y
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

**Q: 测试失败？**
```bash
# 查看详细错误
pytest tests/test_torch_equivalence.py -v --tb=short
```

**Q: 内存不足？**
```python
# 使用更小的图像
img = torch.rand(128, 128, device='cuda')

# 或分块处理
```

## 文档

- `TEST_REPORT.md` - 详细测试报告
- `TESTING_GUIDE.md` - 测试指南
- `INSTALL_CUDA_PYTORCH.md` - CUDA安装
- `FINAL_TEST_SUMMARY.md` - 测试总结

## 支持

- 问题反馈: [GitHub Issues]
- 文档: 查看上述Markdown文件
- 测试: `pytest tests/`

---

**版本**: 1.0.0  
**更新**: 2025年10月6日  
**状态**: ✅ 生产就绪
