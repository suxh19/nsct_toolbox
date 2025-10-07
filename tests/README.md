# NSCT Toolbox 测试文档

## 概述

本测试套件用于验证 `nsct_python/utils.py` 和 `nsct_torch/utils.py` 之间的一致性。

## 测试内容

测试文件 `test_utils_consistency.py` 验证以下函数在 NumPy 和 PyTorch 实现之间的一致性：

### 测试的函数

1. **extend2** - 2D 图像扩展
   - 周期扩展模式 (per)
   - 行方向 quincunx 扩展 (qper_row)
   - 列方向 quincunx 扩展 (qper_col)

2. **symext** - 对称扩展
   - 基本对称扩展
   - 不同 shift 参数的扩展

3. **upsample2df** - 2D 滤波器上采样
   - power=1 上采样
   - power=2 上采样

4. **modulate2** - 2D 调制
   - both 模式
   - row 模式
   - col 模式

5. **resampz** - 矩阵重采样（剪切）
   - type 1: 垂直剪切
   - type 2: 垂直剪切
   - type 3: 水平剪切
   - type 4: 水平剪切

6. **qupz** - Quincunx 上采样
   - type 1 上采样
   - type 2 上采样
   - 更大矩阵测试

## 验证指标

每个测试用例验证以下四个方面的一致性：

1. **形状一致性** - 输出数组的形状必须完全相同
2. **数值一致性** - 数值误差必须在容差范围内
   - 相对容差 (rtol): 1e-6
   - 绝对容差 (atol): 1e-8
3. **精度一致性** - 数据类型兼容性检查
4. **逐元素一致性** - 对应位置的数值必须完全一致

## 运行测试

### 一致性测试

激活虚拟环境并运行所有一致性测试：

```powershell
.venv\Scripts\Activate.ps1
pytest tests/test_utils_consistency.py -v -s
```

### 性能测试

运行性能对比测试（CPU vs GPU）：

```powershell
.venv\Scripts\Activate.ps1
pytest tests/test_performance.py -v -s
```

📝 **详细性能测试文档**: 请查看 [PERFORMANCE_TESTING.md](PERFORMANCE_TESTING.md)

### 运行特定测试

```powershell
# 只测试 extend2 函数
pytest tests/test_utils_consistency.py::TestUtilsConsistency::test_extend2_per_mode -v -s

# 只测试 qupz 函数
pytest tests/test_utils_consistency.py -k "qupz" -v -s
```

### 生成测试报告

```powershell
# 生成 HTML 报告
pytest tests/test_utils_consistency.py --html=report.html --self-contained-html

# 生成覆盖率报告
pytest tests/test_utils_consistency.py --cov=nsct_python --cov=nsct_torch
```

## 测试结果

所有 17 个测试用例均已通过，验证了：
- ✅ 形状完全一致
- ✅ 数值误差为 0（完美一致）
- ✅ 数据类型兼容
- ✅ 对应位置的数值完全一致

## 测试输出示例

### 简洁模式输出
```
测试 extend2 (per 模式):
  ✓ 形状一致性: (6, 6)
  ✓ numpy dtype: float32, torch dtype: torch.float32
  ✓ 最大数值差异: 0.0

  📊 NumPy 输出:
  [[15. 12. 13. 14. 15. 12.]
   [ 3.  0.  1.  2.  3.  0.]
   [ 7.  4.  5.  6.  7.  4.]
   [11.  8.  9. 10. 11.  8.]
   [15. 12. 13. 14. 15. 12.]
   [ 3.  0.  1.  2.  3.  0.]]

  📊 PyTorch 输出:
  [[15. 12. 13. 14. 15. 12.]
   [ 3.  0.  1.  2.  3.  0.]
   [ 7.  4.  5.  6.  7.  4.]
   [11.  8.  9. 10. 11.  8.]
   [15. 12. 13. 14. 15. 12.]
   [ 3.  0.  1.  2.  3.  0.]]

  🔍 随机抽样验证 (10 个位置):
     位置 [29]: numpy=12.00000000, torch=12.00000000 ✓
     位置 [22]: numpy=11.00000000, torch=11.00000000 ✓
     位置 [28]: numpy=15.00000000, torch=15.00000000 ✓
     ... (其余 5 个位置均匹配)
PASSED
```

### 详细输出特性

1. **小数组 (≤100 元素)**: 完整显示 NumPy 和 PyTorch 的输出数组
2. **大数组 (>100 元素)**: 显示统计信息和左上角 3x3 子矩阵
3. **随机抽样**: 随机选择 10 个位置，逐个验证数值一致性
4. **差异矩阵**: 如果有差异，会显示差异矩阵

## 项目结构

```
nsct_toolbox/
├── nsct_python/
│   └── utils.py          # NumPy 实现
├── nsct_torch/
│   └── utils.py          # PyTorch 实现
└── tests/
    ├── __init__.py
    ├── test_utils_consistency.py  # 一致性测试
    └── README.md         # 本文档
```

## 依赖项

- pytest
- numpy
- torch

## 注意事项

1. 运行测试前请确保已激活虚拟环境
2. 测试使用 float32 精度进行比较
3. modulate2 函数会自动转换为 float64 进行计算
4. 所有测试都在 CPU 上运行

## 扩展测试

如需添加新的测试用例，请遵循以下模式：

```python
def test_function_name(self, tolerance):
    """测试描述"""
    print("\n测试信息:")
    # 创建 numpy 数据
    x_np = ...
    # 转换为 torch
    x_torch = self.numpy_to_torch(x_np)
    
    # 执行函数
    result_np = np_utils.function_name(x_np, ...)
    result_torch = torch_utils.function_name(x_torch, ...)
    
    # 验证一致性
    self.assert_arrays_equal(result_np, result_torch, tolerance)
```

## 维护者

如有问题或建议，请提交 issue 或 pull request。
