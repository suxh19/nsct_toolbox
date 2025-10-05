# NSCT MATLAB 到 Python 翻译指南

## 快速开始

### 1. 生成测试数据（MATLAB）

```matlab
cd nsct_toolbox
run_all_tests
```

### 2. Python 翻译顺序

按以下顺序翻译，每完成一步都用对应的 `.mat` 文件验证：

## 翻译步骤详解

### 第 1 步：底层操作 (5 个函数)

| 函数 | 功能 | Python 实现建议 | 测试文件 |
|------|------|-----------------|----------|
| `extend2.m` | 边界扩展 | `numpy.pad()` | `step1_extend2.mat` |
| `symext.m` | 对称扩展 | `numpy.pad(mode='symmetric')` | `step1_symext.mat` |
| `upsample2df.m` | 滤波器上采样 | NumPy 切片赋值 | `step1_upsample2df.mat` |
| `modulate2.m` | 矩阵调制 | `numpy.meshgrid()` | `step1_modulate2.mat` |
| `resampz.m` | 剪切重采样 | NumPy 索引操作 | `step1_resampz.mat` |

**预计时间**：2-3 天

### 第 2 步：卷积和滤波 (2 个功能)

| 函数/功能 | Python 实现 | 测试文件 |
|-----------|-------------|----------|
| `efilter2.m` | `numpy.pad()` + `scipy.signal.convolve2d()` | `step2_efilter2.mat` |
| MEX 膨胀卷积 | `scipy.signal.convolve2d(dilation=...)` | `step2_dilated_conv.mat` |

**关键点**：`scipy.signal.convolve2d` 的 `dilation` 参数完全替代 MEX 文件功能

**预计时间**：1-2 天

### 第 3 步：滤波器生成 (4 个主函数 + 辅助函数)

| 函数 | 复杂度 | 测试文件 |
|------|--------|----------|
| `dmaxflat.m` | 中 | `step3_dmaxflat.mat` |
| `dfilters.m` | 高 | `step3_dfilters.mat` |
| `atrousfilters.m` | 低 | `step3_atrousfilters.mat` |
| `parafilters.m` | 中 | `step3_parafilters.mat` |

**辅助函数**（需要先翻译）：
- `mctrans.m` - McClellan 变换
- `ld2quin.m` - Ladder to Quincunx
- `qupz.m` - Quincunx 上采样
- `wfilters.m` - 小波滤波器（可选）
- `ldfilter.m` - Ladder 滤波器（可选）

**预计时间**：3-5 天

### 第 4 步：核心分解重构 (6 个函数)

| 函数对 | 功能 | 测试文件 |
|--------|------|----------|
| `nssfbdec.m` / `nssfbrec.m` | 双通道滤波器组 | `step4_core_decomposition.mat` |
| `nsdfbdec.m` / `nsdfbrec.m` | 方向滤波器组 | 同上 |
| `atrousdec.m` / `atrousrec.m` | 金字塔分解 | 同上 |

**预计时间**：3-4 天

### 第 5 步：顶层接口 (2 个函数)

| 函数 | 功能 | 测试文件 |
|------|------|----------|
| `nsctdec.m` | NSCT 分解 | `step5_nsct_full.mat` |
| `nsctrec.m` | NSCT 重构 | 同上 |

**预计时间**：1-2 天

## Python 验证模板

```python
import numpy as np
import scipy.io as sio

# 加载测试数据
data = sio.loadmat('test_data/step1_extend2.mat')

# 获取测试输入和期望输出
input_data = data['test_matrix']
expected = data['result1']

# 运行你的实现
result = your_python_function(input_data, ...)

# 验证结果
error = np.max(np.abs(result - expected))
print(f'最大误差: {error:.2e}')

# 检查是否通过
assert error < 1e-10, f"误差太大: {error}"
print('✓ 测试通过!')
```

## 关键的 Python 库

```python
import numpy as np
import scipy.signal
import scipy.io
from scipy.ndimage import convolve
```

## MATLAB vs Python 对照表

| MATLAB | Python (NumPy/SciPy) |
|--------|----------------------|
| `size(A)` | `A.shape` |
| `length(A)` | `len(A)` or `A.size` |
| `A'` (转置) | `A.T` |
| `fliplr(A)` | `np.fliplr(A)` |
| `flipud(A)` | `np.flipud(A)` |
| `conv2(A, B, 'valid')` | `scipy.signal.convolve2d(A, B, mode='valid')` |
| `padarray(A, [m,n], 'symmetric')` | `np.pad(A, ((m,m),(n,n)), mode='symmetric')` |
| `zeros(m, n)` | `np.zeros((m, n))` |
| `ones(m, n)` | `np.ones((m, n))` |
| `A(i, j)` (索引从1开始) | `A[i-1, j-1]` (索引从0开始) |
| `A(i, :)` | `A[i-1, :]` |
| `A(:, j)` | `A[:, j-1]` |
| `cell(n)` | `[]` (Python list) |
| `A{i}` | `A[i-1]` |

## 常见问题

### Q1: MATLAB 的 cell 数组如何翻译？
A: 使用 Python 的 list：
```python
# MATLAB: y = cell(3, 1)
y = [None] * 3  # 或者 y = []
```

### Q2: MATLAB 的 struct 如何翻译？
A: 使用 Python 的 dict 或自定义类：
```python
# MATLAB: s.field1 = value1
s = {'field1': value1}
# 或者使用 namedtuple/dataclass
```

### Q3: 索引差异如何处理？
A: MATLAB 索引从 1 开始，Python 从 0 开始。翻译时要格外注意：
```python
# MATLAB: A(1, 1)
A[0, 0]  # Python
```

### Q4: 如何处理 MATLAB 的 end 关键字？
A: 
```python
# MATLAB: A(1:end, 2:end)
A[0:, 1:]  # Python
```

### Q5: 卷积的 'same', 'valid', 'full' 模式？
A: `scipy.signal.convolve2d` 直接支持这些模式：
```python
from scipy.signal import convolve2d
result = convolve2d(image, filter, mode='valid')
```

## 性能优化建议

1. **避免循环**：尽量使用向量化操作
2. **使用 NumPy 的广播**：避免显式复制数据
3. **预分配数组**：使用 `np.zeros()` 预先分配
4. **使用 SciPy 函数**：它们是 C 实现的，很快
5. **考虑使用 Numba**：如果有unavoidable的循环

## 项目结构建议

```
nsct_python/
├── nsct/
│   ├── __init__.py
│   ├── basic_ops.py      # 第 1 步函数
│   ├── filtering.py      # 第 2 步函数
│   ├── filters.py        # 第 3 步函数
│   ├── decomposition.py  # 第 4 步函数
│   └── nsct.py          # 第 5 步函数
├── tests/
│   ├── test_step1.py
│   ├── test_step2.py
│   ├── test_step3.py
│   ├── test_step4.py
│   └── test_step5.py
├── test_data/           # 从 MATLAB 复制过来
└── README.md
```

## 总预计时间

- **最快**：10-12 天（全职工作）
- **正常**：2-3 周（每天 4-6 小时）
- **保守**：3-4 周（考虑调试和优化）

## 成功标准

✓ 所有测试通过（误差 < 1e-10）  
✓ 完整 NSCT 重构误差 < 1e-10  
✓ 代码清晰可读  
✓ 有完整的文档字符串  
✓ 通过 pytest 测试  

---

**祝翻译成功！** 🚀
