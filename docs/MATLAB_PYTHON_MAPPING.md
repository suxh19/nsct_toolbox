# MATLAB到Python函数映射详细分析

本文档详细记录了NSCT工具箱中每个Python函数与对应MATLAB函数的映射关系、实现细节对比及已知差异。

## 目录
- [核心函数 (core.py)](#核心函数-corepy)
- [滤波器函数 (filters.py)](#滤波器函数-filterspy)
- [工具函数 (utils.py)](#工具函数-utilspy)
- [数值精度分析](#数值精度分析)
- [已知差异和注意事项](#已知差异和注意事项)

---

## 核心函数 (core.py)

### 1. nssfbdec - 双通道非下采样滤波器组分解

#### MATLAB实现
**文件**: `nsct_matlab/nssfbdec.m`  
**函数签名**: 
```matlab
function [y1, y2] = nssfbdec(x, f1, f2, mup)
```

**参数**:
- `x`: 输入图像矩阵
- `f1`: 第一分支滤波器
- `f2`: 第二分支滤波器
- `mup`: 可选，上采样矩阵（整数或2x2矩阵）

**返回值**:
- `y1`: 第一分支输出
- `y2`: 第二分支输出

**关键步骤**:
1. 如果`mup`不存在：使用`efilter2`进行周期扩展卷积
2. 如果`mup == 1`或`mup == eye(2)`：同样使用`efilter2`
3. 如果`size(mup) == [2, 2]`：使用`zconv2`进行非可分离上采样卷积
4. 如果`size(mup) == [1, 1]`：使用`zconv2S`进行可分离上采样卷积

#### Python实现
**文件**: `nsct_python/core.py`  
**函数签名**:
```python
def nssfbdec(x, f1, f2, mup=None) -> Tuple[np.ndarray, np.ndarray]
```

**参数**:
- `x`: 输入图像（numpy数组）
- `f1`: 第一分支滤波器（numpy数组）
- `f2`: 第二分支滤波器（numpy数组）
- `mup`: 可选，上采样矩阵（None、int、float或numpy数组）

**返回值**:
- `y1`: 第一分支输出（numpy数组）
- `y2`: 第二分支输出（numpy数组）

**关键步骤**:
1. 如果`mup is None`：使用`efilter2`
2. 否则：使用`_convolve_upsampled`辅助函数
3. `_convolve_upsampled`内部：
   - 调用`_upsample_and_find_origin`进行滤波器上采样
   - 使用`extend2`进行边界扩展
   - 使用`scipy.signal.convolve2d`进行卷积

**映射关系**:
- MATLAB的`efilter2` ↔ Python的`efilter2`
- MATLAB的`zconv2`/`zconv2S` ↔ Python的`_convolve_upsampled`
- MATLAB依赖MEX文件，Python使用纯Python实现

**实现差异**:
1. **上采样逻辑**: MATLAB使用编译的C代码（`zconv2.c`），Python使用纯NumPy实现
2. **矩阵判断**: MATLAB使用`size(mup) == [2,2]`，Python使用`isinstance`和`np.array_equal`
3. **卷积方向**: Python中需要旋转卷积核180度以模拟MATLAB的`conv2`行为

**测试覆盖**:
- ✅ 基本测试（无mup）
- ✅ 单位矩阵mup（mup=1）
- ✅ 可分离mup（mup=2）
- ✅ Quincunx mup（2x2矩阵）
- ✅ 不同滤波器类型（pkva, pyr）

---

### 2. nssfbrec - 双通道非下采样滤波器组重建

#### MATLAB实现
**文件**: `nsct_matlab/nssfbrec.m`  
**函数签名**:
```matlab
function y = nssfbrec(x1, x2, f1, f2, mup)
```

**参数**:
- `x1`: 第一分支输入
- `x2`: 第二分支输入
- `f1`: 第一分支滤波器
- `f2`: 第二分支滤波器
- `mup`: 可选，上采样矩阵

**返回值**:
- `y`: 重建图像

**关键步骤**:
1. 检查`x1`和`x2`尺寸是否相同
2. 对两个分支分别进行卷积（逻辑与nssfbdec相同）
3. 将两个结果相加：`y = y1 + y2`

#### Python实现
**文件**: `nsct_python/core.py`  
**函数签名**:
```python
def nssfbrec(x1, x2, f1, f2, mup=None) -> np.ndarray
```

**关键步骤**:
1. 检查输入尺寸：`if x1.shape != x2.shape: raise ValueError`
2. 根据mup选择卷积方法（与nssfbdec一致）
3. 返回`y1 + y2`

**映射关系**:
- 完全对应MATLAB实现
- 使用`is_rec=True`参数来处理重建时的滤波器时间反转

**实现差异**:
1. **滤波器处理**: Python中通过`is_rec`参数在`_convolve_upsampled`中旋转滤波器
2. **错误信息**: Python使用英文错误信息

**测试覆盖**:
- ✅ 基本重建测试
- ✅ 各种mup参数的重建
- ✅ 完美重建测试（MSE < 1e-10）

---

## 滤波器函数 (filters.py)

### 3. efilter2 - 2D边界处理滤波

#### MATLAB实现
**文件**: `nsct_matlab/efilter2.m`  
**函数签名**:
```matlab
function y = efilter2(x, f, extmod, shift)
```

**参数**:
- `x`: 输入图像
- `f`: 2D滤波器
- `extmod`: 可选，扩展模式（默认'per'）
- `shift`: 可选，卷积窗口偏移（默认[0; 0]）

**返回值**:
- `y`: 滤波后的图像

**关键步骤**:
1. 计算滤波器半径：`sf = (size(f) - 1) / 2`
2. 使用`extend2`扩展图像
3. 使用`conv2(xext, f, 'valid')`进行卷积

#### Python实现
**文件**: `nsct_python/filters.py`  
**函数签名**:
```python
def efilter2(x, f, extmod='per', shift=None) -> np.ndarray
```

**关键步骤**:
1. 将输入转换为float64：`x_float = x.astype(np.float64)`
2. 计算扩展量：`sf = (np.array(f.shape) - 1) / 2`
3. 调用`extend2`进行扩展
4. 使用`convolve2d(xext, f, 'valid')`

**映射关系**:
- MATLAB的`conv2` ↔ Python的`scipy.signal.convolve2d`
- MATLAB的`size(f)` ↔ Python的`f.shape`
- MATLAB的列向量shift ↔ Python的列表shift

**实现差异**:
1. **默认参数**: Python使用`None`而非MATLAB的`~exist('var')`
2. **类型转换**: Python显式转换为float64
3. **数组索引**: Python使用`floor()`和`ceil()`的`int()`转换

**测试覆盖**:
- ✅ 基本滤波测试
- ✅ 带偏移的滤波
- ✅ 不同扩展模式

---

### 4. ldfilter - 梯形网络滤波器生成

#### MATLAB实现
**文件**: `nsct_matlab/ldfilter.m`  
**函数签名**:
```matlab
function beta = ldfilter(fname)
```

**参数**:
- `fname`: 滤波器名称（'pkva', 'pkva12', 'pkva8', 'pkva6'）

**返回值**:
- `beta`: 1D滤波器

**关键步骤**:
1. 根据名称选择预定义系数向量`v`
2. 创建对称脉冲响应：`beta = [v(end:-1:1), v]`

#### Python实现
**文件**: `nsct_python/filters.py`  
**函数签名**:
```python
def ldfilter(fname: str) -> np.ndarray
```

**关键步骤**:
1. 根据名称选择系数：使用字典或if-elif
2. 返回对称向量：`np.concatenate((v[::-1], v))`

**映射关系**:
- MATLAB的`[v(end:-1:1), v]` ↔ Python的`np.concatenate((v[::-1], v))`
- 完全数值一致

**实现差异**:
- 无实质差异，仅语法不同

**测试覆盖**:
- ✅ pkva12
- ✅ pkva8
- ✅ pkva6

---

### 5. ld2quin - 梯形网络到Quincunx滤波器转换

#### MATLAB实现
**文件**: `nsct_matlab/ld2quin.m`  
**函数签名**:
```matlab
function [h0, h1] = ld2quin(beta)
```

**参数**:
- `beta`: 1D全通滤波器（偶数长度）

**返回值**:
- `h0`: Quincunx低通滤波器
- `h1`: Quincunx高通滤波器

**关键步骤**:
1. 验证输入：偶数长度、1D
2. 计算`n = lf / 2`
3. 计算外积：`sp = beta' * beta`
4. Quincunx上采样：`h = qupz(sp, 1)`
5. 低通滤波器：`h0 = h; h0(2*n, 2*n) = h0(2*n, 2*n) + 1; h0 = h0 / 2`
6. 高通滤波器：`h1 = -conv2(h, rot90(h0, 2), 'full'); h1(4*n-1, 4*n-1) = h1(4*n-1, 4*n-1) + 1`

#### Python实现
**文件**: `nsct_python/filters.py`  
**函数签名**:
```python
def ld2quin(beta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]
```

**关键步骤**:
1. 验证维度：`if beta.ndim != 1: raise ValueError`
2. 验证长度：`if n * 2 != lf: raise ValueError`
3. 外积：`sp = np.outer(beta, beta)`
4. 上采样：`h = qupz(sp, 1)`
5. 低通：`h0 = h.copy(); h0[lf-1, lf-1] += 1; h0 = h0 / 2.0`
6. 高通：`h1 = -convolve2d(h, np.rot90(h0, 2), 'full'); h1[4*n-2, 4*n-2] += 1`

**映射关系**:
- MATLAB的`beta' * beta` ↔ Python的`np.outer(beta, beta)`
- MATLAB的`rot90(h0, 2)` ↔ Python的`np.rot90(h0, 2)`
- MATLAB的索引从1开始 ↔ Python的索引从0开始

**实现差异**:
1. **索引转换**: MATLAB的`2*n` → Python的`lf-1` (即`2*n-1`)
2. **索引转换**: MATLAB的`4*n-1` → Python的`4*n-2`
3. **数组复制**: Python显式使用`.copy()`避免修改原数组

**测试覆盖**:
- ✅ pkva6 (lf=6, n=3)
- ✅ pkva12 (lf=12, n=6)
- ✅ 验证h0中心值计算
- ✅ 验证输出尺寸

---

### 6. dmaxflat - 2D菱形最大平坦滤波器

#### MATLAB实现
**文件**: `nsct_matlab/dmaxflat.m`  
**函数签名**:
```matlab
function h = dmaxflat(N, d)
```

**参数**:
- `N`: 滤波器阶数（1-7）
- `d`: (0,0)系数值（0或1）

**返回值**:
- `h`: 2D滤波器

**关键步骤**:
1. 对每个N值，定义预计算的系数矩阵
2. 使用对称扩展：`[h, fliplr(h(:, 1:end-1))]`
3. 行方向扩展：`[h; flipud(h(1:end-1, :))]`
4. 设置中心值为`d`

#### Python实现
**文件**: `nsct_python/filters.py`  
**函数签名**:
```python
def dmaxflat(N: int, d: float = 0.0) -> np.ndarray
```

**关键步骤**:
1. 验证`1 <= N <= 7`
2. 对N=1, 2, 3定义系数
3. 对称扩展：`np.concatenate([h, np.fliplr(h[:, :-1])], axis=1)`
4. 行扩展：`np.concatenate([h, np.flipud(h[:-1, :])], axis=0)`
5. 设置中心：根据N计算中心索引

**映射关系**:
- MATLAB的`fliplr` ↔ Python的`np.fliplr`
- MATLAB的`flipud` ↔ Python的`np.flipud`
- MATLAB的`:` ↔ Python的`axis`参数

**实现差异**:
1. **未完全实现**: Python当前仅实现N=1,2,3，N=4-7抛出NotImplementedError
2. **中心索引**: 需要根据N计算正确的中心位置

**测试覆盖**:
- ✅ N=1, d=0
- ✅ N=2, d=1
- ✅ N=3, d=0
- ⚠️ N=4-7未测试（未实现）

---

### 7. atrousfilters - 金字塔2D滤波器生成

#### MATLAB实现
**文件**: `nsct_matlab/atrousfilters.m`  
**函数签名**:
```matlab
function [h0, h1, g0, g1] = atrousfilters(fname)
```

**参数**:
- `fname`: 滤波器名称（'pyr', 'pyrexc'）

**返回值**:
- `h0, h1`: 分析滤波器
- `g0, g1`: 综合滤波器

**关键步骤**:
1. 定义预计算的滤波器系数
2. 根据fname选择h1和g1的对应关系
3. 对所有滤波器进行对称扩展

#### Python实现
**文件**: `nsct_python/filters.py`  
**函数签名**:
```python
def atrousfilters(fname: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
```

**映射关系**:
- 完全对应，系数完全相同
- 使用相同的对称扩展逻辑

**实现差异**:
- 无实质差异

**测试覆盖**:
- ✅ 'pyr'
- ✅ 'pyrexc'

---

### 8. mctrans - McClellan变换

#### MATLAB实现
**文件**: `nsct_matlab/mctrans.m`  
**函数签名**:
```matlab
function h = mctrans(b, t)
```

**参数**:
- `b`: 1D FIR滤波器
- `t`: 2D变换滤波器

**返回值**:
- `h`: 2D FIR滤波器

**关键步骤**:
1. `n = (length(b) - 1) / 2`
2. `b = ifftshift(b)`
3. `a = [b(1), 2*b(2:n+1)]`
4. 使用Chebyshev多项式迭代计算
5. 旋转结果：`h = rot90(h, 2)`

#### Python实现
**文件**: `nsct_python/filters.py`  
**函数签名**:
```python
def mctrans(b: np.ndarray, t: np.ndarray) -> np.ndarray
```

**关键步骤**:
1. `n = (b.shape[0] - 1) // 2`
2. `b = np.fft.ifftshift(b)`
3. `a = np.concatenate(([b[0]], 2 * b[1:n + 1]))`
4. Chebyshev多项式迭代（与MATLAB相同逻辑）
5. `return np.rot90(h, 2)`

**映射关系**:
- MATLAB的`ifftshift` ↔ Python的`np.fft.ifftshift`
- MATLAB的`conv2` ↔ Python的`convolve2d`
- 算法逻辑完全一致

**实现差异**:
- 无实质差异，仅语法不同

**测试覆盖**:
- ✅ 简单情况（3x3滤波器）
- ✅ 更大滤波器（4系数）

---

### 9. parafilters - 平行四边形滤波器生成

#### MATLAB实现
**文件**: `nsct_matlab/parafilters.m`  
**函数签名**:
```matlab
function [y1, y2] = parafilters(f1, f2)
```

**参数**:
- `f1, f2`: 一对菱形滤波器

**返回值**:
- `y1, y2`: 每个包含4个平行四边形滤波器的cell数组

**关键步骤**:
1. 调制：`y1{1} = modulate2(f1, 'r')` 等
2. 转置：`y1{3} = y1{1}.'` 等
3. 重采样：`y1{k} = resampz(y1{k}, k)`

#### Python实现
**文件**: `nsct_python/filters.py`  
**函数签名**:
```python
def parafilters(f1, f2) -> Tuple[List[np.ndarray], List[np.ndarray]]
```

**关键步骤**:
1. 调制：`y1.append(modulate2(f1, 'r'))` 等
2. 转置：`y1.append(y1[0].T)` 等
3. 重采样：`y1[i] = resampz(y1[i], i + 1)`

**映射关系**:
- MATLAB的cell数组 ↔ Python的list
- MATLAB的`.T` ↔ Python的`.T`
- MATLAB索引从1开始 ↔ Python索引从0开始

**实现差异**:
1. **数据结构**: cell数组 vs list
2. **索引**: MATLAB的`k` ↔ Python的`i+1`

**测试覆盖**:
- ✅ 基本情况（ones矩阵）
- ✅ dmaxflat滤波器

---

### 10. dfilters - 方向2D滤波器生成

#### MATLAB实现
**文件**: `nsct_matlab/dfilters.m`  
**函数签名**:
```matlab
function [h0, h1] = dfilters(fname, type)
```

**参数**:
- `fname`: 滤波器名称
- `type`: 'd'（分解）或'r'（重建）

**返回值**:
- `h0, h1`: 菱形滤波器对

**关键步骤**:
1. 对'pkva'：调用`ldfilter`和`ld2quin`
2. 对'dmaxflat'：调用`dmaxflat`和`mctrans`
3. 归一化并根据type选择滤波器

#### Python实现
**文件**: `nsct_python/filters.py`  
**函数签名**:
```python
def dfilters(fname: str, type: str = 'd') -> Tuple[np.ndarray, np.ndarray]
```

**映射关系**:
- 逻辑完全对应
- 对未识别滤波器，回退到PyWavelets

**实现差异**:
1. **PyWavelets集成**: Python添加了对标准小波滤波器的支持
2. **dmaxflat限制**: Python当前仅支持dmaxflat N<=3

**测试覆盖**:
- ✅ 'pkva'
- ✅ 'dmaxflat7'（部分）
- ✅ 'db2'（PyWavelets）

---

## 工具函数 (utils.py)

### 11. extend2 - 2D图像扩展

#### MATLAB实现
**文件**: `nsct_matlab/extend2.m`  
**函数签名**:
```matlab
function y = extend2(x, ru, rd, cl, cr, extmod)
```

**参数**:
- `x`: 输入图像
- `ru, rd`: 行方向上、下扩展量
- `cl, cr`: 列方向左、右扩展量
- `extmod`: 扩展模式（'per', 'qper_row', 'qper_col'）

**返回值**:
- `y`: 扩展后的图像

**关键步骤**:
1. 'per'模式：使用`getPerIndices`获取周期索引
2. 'qper_row'模式：行方向圆形移位rx/2
3. 'qper_col'模式：列方向圆形移位cx/2

#### Python实现
**文件**: `nsct_python/utils.py`  
**函数签名**:
```python
def extend2(x, ru, rd, cl, cr, extmod='per') -> np.ndarray
```

**关键步骤**:
1. 'per'：使用`np.pad(x, ..., 'wrap')`
2. 'qper_row'：
   - 左右扩展带移位：`np.roll(x[:, -cl:], rx//2, axis=0)`
   - 上下周期扩展：`np.pad(y, ((ru, rd), (0, 0)), 'wrap')`
3. 'qper_col'：类似，但方向相反

**映射关系**:
- MATLAB的索引操作 ↔ Python的`np.pad`和`np.roll`
- MATLAB的`circshift` ↔ Python的`np.roll`

**实现差异**:
1. **实现方式**: MATLAB手动索引，Python使用NumPy函数
2. **移位方向**: 需要仔细验证移位方向的正确性

**测试覆盖**:
- ✅ 'per'模式
- ✅ 'qper_row'模式
- ✅ 'qper_col'模式
- ✅ 各种扩展量组合

---

### 12. upsample2df - 2D滤波器上采样

#### MATLAB实现
**文件**: `nsct_matlab/upsample2df.m`  
**函数签名**:
```matlab
function ho = upsample2df(h, power)
```

**参数**:
- `h`: 输入滤波器
- `power`: 上采样幂次（2^power）

**返回值**:
- `ho`: 上采样后的滤波器

**关键步骤**:
1. `factor = 2^power`
2. 创建零矩阵：`ho = zeros(factor*m, factor*n)`
3. 填充值：`ho(1:factor:end, 1:factor:end) = h`

#### Python实现
**文件**: `nsct_python/utils.py`  
**函数签名**:
```python
def upsample2df(h, power=1) -> np.ndarray
```

**关键步骤**:
1. `factor = 2**power`
2. `ho = np.zeros((factor*m, factor*n), dtype=h.dtype)`
3. `ho[::factor, ::factor] = h`

**映射关系**:
- 完全对应，逻辑相同

**实现差异**:
- 无实质差异

**测试覆盖**:
- ✅ power=1
- ✅ power=2

---

### 13. modulate2 - 2D调制

#### MATLAB实现
**文件**: `nsct_matlab/modulate2.m`  
**函数签名**:
```matlab
function y = modulate2(x, type, center)
```

**参数**:
- `x`: 输入矩阵
- `type`: 'r'（行）, 'c'（列）, 'b'（双向）
- `center`: 调制中心偏移

**返回值**:
- `y`: 调制后的矩阵

**关键步骤**:
1. 计算中心：`o = floor(s/2) + 1 + center`
2. 计算索引：`n1 = [1:s(1)] - o(1)`
3. 计算调制：`m1 = (-1).^n1`
4. 应用调制：`y = y .* m1'` 和/或 `y = y .* m2`

#### Python实现
**文件**: `nsct_python/utils.py`  
**函数签名**:
```python
def modulate2(x, mode='b', center=None) -> np.ndarray
```

**关键步骤**:
1. `o = np.floor(np.array(s) / 2) + 1 + np.array(center)`
2. `n1 = np.arange(1, s[0] + 1) - o[0]`
3. `m1 = (-1) ** n1`
4. `y *= m1[:, np.newaxis]` 和/或 `y *= m2[np.newaxis, :]`

**映射关系**:
- MATLAB的`.*` ↔ Python的`*=`（广播）
- MATLAB的`'`（转置用于广播）↔ Python的`[:, np.newaxis]`

**实现差异**:
1. **参数名**: MATLAB用`type`，Python用`mode`
2. **广播**: Python使用NumPy广播而非显式转置

**测试覆盖**:
- ✅ 行调制（'r'）
- ✅ 列调制（'c'）
- ✅ 双向调制（'b'）
- ✅ 带中心偏移

---

### 14. resampz - 矩阵重采样（剪切变换）

#### MATLAB实现
**文件**: `nsct_matlab/resampz.m`  
**函数签名**:
```matlab
function y = resampz(x, type, shift)
```

**参数**:
- `x`: 输入矩阵
- `type`: 1-4（对应不同重采样矩阵）
- `shift`: 移位量（默认1）

**返回值**:
- `y`: 重采样后的矩阵

**关键步骤**:
1. type 1/2：垂直剪切
   - 为每列计算移位量
   - 填充到新矩阵
   - 裁剪零行
2. type 3/4：水平剪切
   - 为每行计算移位量
   - 填充到新矩阵
   - 裁剪零列

#### Python实现
**文件**: `nsct_python/utils.py`  
**函数签名**:
```python
def resampz(x, type, shift=1) -> np.ndarray
```

**关键步骤**:
1. type in [1, 2]：垂直剪切
   - `shift1 = np.arange(sx[1]) * (-shift)` 或 `* shift`
   - 归一化为非负：`shift1 = shift1 - shift1.min()`
   - 填充：`y[shift1[n]:shift1[n]+sx[0], n] = x[:, n]`
   - 裁剪：使用`np.linalg.norm`找非零行
2. type in [3, 4]：水平剪切（类似）

**映射关系**:
- 逻辑完全对应
- MATLAB的`find` ↔ Python的`np.where`

**实现差异**:
1. **裁剪方法**: Python使用范数判断非零行/列
2. **负索引处理**: Python显式归一化索引

**测试覆盖**:
- ✅ type 1-4
- ✅ 不同shift值

---

### 15. qupz - Quincunx上采样

#### MATLAB实现
**文件**: `nsct_matlab/qupz.m`  
**函数签名**:
```matlab
function y = qupz(x, type)
```

**参数**:
- `x`: 输入图像
- `type`: 1或2（选择Q1或Q2矩阵）

**返回值**:
- `y`: Quincunx上采样图像

**关键步骤**:
1. type 1：`Q1 = [1, -1; 1, 1]`
   - `x1 = resampz(x, 4)`
   - 垂直零填充：`x2(1:2:end, :) = x1`
   - `y = resampz(x2, 1)`
2. type 2：`Q2 = [1, 1; -1, 1]`
   - `x1 = resampz(x, 3)`
   - 垂直零填充
   - `y = resampz(x2, 2)`

#### Python实现
**文件**: `nsct_python/utils.py`  
**函数签名**:
```python
def qupz(x, type=1) -> np.ndarray
```

**关键步骤**:
1. 输出尺寸：`(r + c - 1, r + c - 1)`
2. type 1：
   - 计算偏移：`offset_r = c - 1`
   - 直接映射：`y[n1 + offset_r, n2] = x[r_idx, c_idx]`
   - 其中`n1 = r_idx - c_idx`, `n2 = r_idx + c_idx`
3. type 2：类似，但`n1 = r_idx + c_idx`, `n2 = -r_idx + c_idx`

**映射关系**:
- MATLAB使用Smith分解和resampz组合
- Python直接基于数学定义实现

**实现差异**:
1. **实现方法**: 完全不同！
   - MATLAB：基于Smith分解，使用resampz链
   - Python：基于Quincunx矩阵的数学定义直接实现
2. **性能**: Python方法可能更直接但需要验证正确性

**测试覆盖**:
- ✅ type 1
- ✅ type 2
- ✅ 不同尺寸矩阵
- ⚠️ 需要详细验证与MATLAB结果的数值一致性

---

## 数值精度分析

### 浮点精度考虑
1. **MATLAB默认**: double（64位）
2. **Python默认**: 取决于输入，建议使用`np.float64`
3. **测试阈值**: 
   - 大多数测试使用`decimal=10`（~1e-10）
   - 完美重建测试可能需要`decimal=12`或更高

### 累积误差来源
1. **卷积操作**: `conv2` vs `convolve2d`
2. **FFT操作**: `fft/ifft` vs `np.fft.fft/ifft`
3. **除法**: 整数除法 vs 浮点除法
4. **索引舍入**: `floor/ceil` vs `int()`

### 建议的测试策略
1. **绝对误差**: `np.abs(result - expected).max() < 1e-10`
2. **相对误差**: `np.abs((result - expected) / expected).max() < 1e-6`
3. **MSE**: 对图像重建，MSE < 1e-20
4. **PSNR**: 对图像重建，PSNR > 100 dB

---

## 已知差异和注意事项

### 1. 关键差异

#### qupz实现
- **严重性**: 高
- **描述**: Python和MATLAB实现方法完全不同
- **影响**: 可能导致后续所有依赖qupz的函数出现差异
- **状态**: 需要详细验证数值一致性
- **建议**: 考虑实现MATLAB的resampz链式方法，或彻底验证数学等价性

#### dmaxflat未完全实现
- **严重性**: 中
- **描述**: Python仅实现N=1,2,3
- **影响**: 'dmaxflat4'-'dmaxflat7'无法使用
- **状态**: 需要补充实现
- **建议**: 从MATLAB代码复制N=4-7的系数

#### zconv2/zconv2S的纯Python实现
- **严重性**: 中
- **描述**: MATLAB使用MEX（编译C代码），Python使用纯NumPy
- **影响**: 性能差异，可能的数值微小差异
- **状态**: 功能正常，需要性能测试
- **建议**: 如需性能，考虑使用Cython或Numba

### 2. 索引转换
- **0-based vs 1-based**: 所有索引需要-1
- **end vs -1**: MATLAB的`end`对应Python的`-1`或`len-1`
- **包含性**: MATLAB的`1:n`对应Python的`0:n`或`range(n)`

### 3. 数组操作
- **转置**: MATLAB的`'`对应Python的`.T`
- **元素乘法**: MATLAB的`.*`对应Python的`*`
- **矩阵乘法**: MATLAB的`*`对应Python的`@`

### 4. 数据类型
- **cell数组**: MATLAB的cell对应Python的list
- **结构体**: MATLAB的struct对应Python的dict或dataclass
- **逻辑索引**: MATLAB的逻辑数组对应Python的布尔数组和`np.where`

### 5. 边界情况
- **空矩阵**: 需要特别处理
- **单元素**: 可能导致维度问题
- **非方阵**: 确保所有函数支持
- **NaN/Inf**: 需要测试传播行为

### 6. 测试建议

#### 基本测试
- ✅ 正常尺寸输入（4x4, 8x8, 16x16）
- ⚠️ 边界情况（1x1, 2x2, 奇数尺寸）
- ⚠️ 大尺寸（256x256, 512x512）
- ⚠️ 非方阵（4x8, 7x13）

#### 数值测试
- ✅ 零矩阵
- ⚠️ 单位矩阵
- ⚠️ 随机矩阵（多个随机种子）
- ⚠️ 特殊值（全1, 全-1, 交替符号）

#### 异常测试
- ⚠️ 错误参数类型
- ⚠️ 错误参数范围
- ⚠️ 不匹配的尺寸
- ⚠️ NaN/Inf输入

#### 性能测试
- ⚠️ 时间比较（MATLAB vs Python）
- ⚠️ 内存使用
- ⚠️ 大规模数据处理

### 7. 下一步行动

#### 高优先级
1. **验证qupz**: 创建详细的单元测试，比对MATLAB和Python输出
2. **完成dmaxflat**: 实现N=4-7
3. **边界测试**: 为所有函数添加边界情况测试

#### 中优先级
4. **性能优化**: 分析瓶颈，考虑使用Numba/Cython
5. **文档完善**: 为每个函数添加详细的docstring
6. **示例代码**: 创建完整的使用示例

#### 低优先级
7. **可视化工具**: 创建滤波器可视化函数
8. **GUI界面**: 考虑创建简单的图形界面
9. **更多滤波器**: 添加其他标准滤波器支持

---

## 测试覆盖率总结

### 已测试 (✅)
- **core.py**: nssfbdec, nssfbrec (12个测试)
- **filters.py**: ldfilter, ld2quin, atrousfilters, efilter2, mctrans (部分)
- **utils.py**: extend2, upsample2df, modulate2, resampz (部分)

### 需要更多测试 (⚠️)
- **filters.py**: dmaxflat (N>3), dfilters (更多组合), parafilters (更多输入)
- **utils.py**: qupz (详细验证), resampz (边界情况)
- **所有函数**: 边界情况、异常处理、性能测试

### 未测试
- 大规模数据
- 多线程/并行处理
- 与其他库的集成

---

## 版本历史
- **v1.0** (2025-10-05): 初始版本，基于现有代码分析
- **v1.1** (待定): 补充qupz验证结果
- **v1.2** (待定): 补充性能测试数据

---

## 参考资料
1. NSCT工具箱MATLAB原始实现
2. "The Nonsubsampled Contourlet Transform: Theory, Design, and Applications"
3. NumPy文档
4. SciPy信号处理文档
5. pytest文档

