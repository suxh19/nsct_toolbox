# NSCT Torch 代码重构总结

## 重构目的
将 `core.py` 和 `utils.py` 两个大文件拆分为模块化的文件夹结构，提高代码的可维护性和可读性。

## 新的文件结构

### 1. `core/` 文件夹
将原 `core.py` 拆分为以下模块：

```
core/
├── __init__.py          # 模块导出
├── convolution.py       # 卷积操作
│   ├── _zconv2_torch()         # 2D上采样滤波器卷积
│   ├── _atrousc_torch()        # À trous卷积
│   └── _convolve_upsampled()   # 上采样卷积辅助函数
├── filterbank.py        # 滤波器组操作
│   ├── nssfbdec()              # 双通道非下采样滤波器组分解
│   ├── nssfbrec()              # 双通道非下采样滤波器组重构
│   ├── nsfbdec()               # 非下采样滤波器组分解（多级）
│   └── nsfbrec()               # 非下采样滤波器组重构（多级）
├── directional.py       # 方向滤波器组
│   ├── nsdfbdec()              # 非下采样方向滤波器组分解
│   └── nsdfbrec()              # 非下采样方向滤波器组重构
└── nsct.py              # NSCT主函数
    ├── nsctdec()               # NSCT分解
    └── nsctrec()               # NSCT重构
```

### 2. `utils/` 文件夹
将原 `utils.py` 拆分为以下模块：

```
utils/
├── __init__.py          # 模块导出
├── extension.py         # 扩展操作
│   ├── extend2()               # 2D图像扩展
│   └── symext()                # 对称扩展
├── sampling.py          # 采样操作
│   ├── upsample2df()           # 2D滤波器上采样
│   ├── resampz()               # 矩阵重采样（剪切）
│   └── qupz()                  # Quincunx上采样
└── modulation.py        # 调制操作
    └── modulate2()             # 2D调制
```

### 3. `filters/` 文件夹
保持原有结构不变：

```
filters/
├── __init__.py
├── atrousfilters.py     # À trous滤波器
├── dfilters.py          # 方向滤波器
├── dmaxflat.py          # Maxflat滤波器
├── efilter2.py          # 扩展滤波器
├── ld2quin.py           # Ladder quincunx滤波器
├── ldfilter.py          # Ladder滤波器
├── mctrans.py           # McClellan变换
└── parafilters.py       # 平行四边形滤波器
```

## 模块功能说明

### Core 模块
- **convolution.py**: 底层卷积实现，包括周期边界卷积和à trous卷积
- **filterbank.py**: 非下采样滤波器组的基本分解和重构操作
- **directional.py**: 实现方向性分解，捕获图像的方向特征
- **nsct.py**: NSCT的主要入口函数，组合各个组件完成完整的变换

### Utils 模块
- **extension.py**: 处理图像边界扩展，支持周期、quincunx等模式
- **sampling.py**: 上采样和重采样操作，用于多尺度处理
- **modulation.py**: 信号调制操作，用于频域处理

## 导入方式

### 旧方式（仍然兼容）
```python
from nsct_torch import nsctdec, nsctrec
from nsct_torch.utils import symext, modulate2
```

### 新方式（更明确）
```python
from nsct_torch.core import nsctdec, nsctrec
from nsct_torch.core.filterbank import nsfbdec, nsfbrec
from nsct_torch.utils.extension import extend2, symext
from nsct_torch.utils.sampling import upsample2df, qupz
```

## 优势
1. **模块化**: 每个文件职责单一，易于理解和维护
2. **可扩展性**: 新增功能只需在对应模块中添加
3. **可读性**: 文件名清晰表明功能用途
4. **符合Python最佳实践**: 与 `filters/` 文件夹结构保持一致

## 测试状态
✅ 所有模块导入测试通过
✅ 保持向后兼容性
✅ 原有功能不受影响

## 文件对照表

| 原文件 | 新位置 | 说明 |
|--------|--------|------|
| `core.py` | `core/` 文件夹 | 拆分为4个子模块 |
| `utils.py` | `utils/` 文件夹 | 拆分为3个子模块 |
| - | `core/__init__.py` | 导出核心函数 |
| - | `utils/__init__.py` | 导出工具函数 |

---

重构完成时间: 2025年10月7日
