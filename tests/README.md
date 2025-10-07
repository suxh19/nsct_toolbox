# NSCT 测试套件

本目录包含 `nsct_python` 和 `nsct_torch` 实现的详细一致性测试。

## 测试结构

```
tests/
├── __init__.py                    # 测试包初始化
├── conftest.py                    # Pytest 配置和 fixtures
├── test_helpers.py                # 测试辅助函数
├── test_utils_consistency.py      # 工具函数测试
├── test_filters_consistency.py    # 滤波器函数测试
├── test_core_filterbank.py        # 核心滤波器组测试
└── test_nsct_full.py              # 完整 NSCT 变换测试
```

## 测试内容

### 1. 工具函数测试 (`test_utils_consistency.py`)

测试 `nsct_python.utils` vs `nsct_torch.utils`：
- `extend2`: 周期扩展和 quincunx 扩展
- `symext`: 对称扩展
- `upsample2df`: 滤波器上采样
- `modulate2`: 二维调制
- `resampz`: 重采样
- `qupz`: Quincunx 上采样

**验证内容**：
- 输出形状一致性
- 数值精度（rtol=1e-10, atol=1e-12）
- 逐元素比较

### 2. 滤波器函数测试 (`test_filters_consistency.py`)

测试 `nsct_python.filters` vs `nsct_torch.filters`：
- `ld2quin`: Quincunx 滤波器生成
- `efilter2`: 2D 边界处理滤波
- `dmaxflat`: Diamond maxflat 滤波器
- `atrousfilters`: À trous 滤波器
- `mctrans`: McClellan 变换
- `ldfilter`: Ladder 滤波器
- `dfilters`: 方向滤波器
- `parafilters`: Parallelogram 滤波器

**验证内容**：
- 滤波器系数精确匹配
- 多种参数组合测试
- 不同滤波器类型测试

### 3. 核心滤波器组测试 (`test_core_filterbank.py`)

测试核心滤波器组函数：
- `nssfbdec` / `nssfbrec`: 双通道非下采样滤波器组
- `nsfbdec` / `nsfbrec`: 非下采样滤波器组（多级）
- 分解-重构往返测试

**验证内容**：
- 分解输出一致性
- 重构输出一致性
- 完美重构特性

### 4. 完整 NSCT 测试 (`test_nsct_full.py`)

测试完整的 NSCT 变换：
- `nsdfbdec` / `nsdfbrec`: 方向滤波器组
- `nsctdec` / `nsctrec`: 完整 NSCT 分解和重构
- 多级、多方向测试
- 不同滤波器组合

**验证内容**：
- 完整变换输出一致性
- 多级分解结构验证
- 完美重构误差分析
- 不同参数组合测试

## 运行测试

### 运行所有测试

```powershell
# 激活虚拟环境
.\.venv\Scripts\Activate.ps1

# 运行所有测试
pytest tests/

# 运行详细输出
pytest tests/ -v

# 运行并显示打印输出
pytest tests/ -v -s
```

### 运行特定测试文件

```powershell
# 测试工具函数
pytest tests/test_utils_consistency.py -v

# 测试滤波器函数
pytest tests/test_filters_consistency.py -v

# 测试核心滤波器组
pytest tests/test_core_filterbank.py -v

# 测试完整 NSCT
pytest tests/test_nsct_full.py -v
```

### 运行特定测试类或函数

```powershell
# 运行特定测试类
pytest tests/test_utils_consistency.py::TestExtend2 -v

# 运行特定测试函数
pytest tests/test_utils_consistency.py::TestExtend2::test_extend2_periodic -v

# 运行特定参数组合
pytest tests/test_utils_consistency.py::TestExtend2::test_extend2_periodic[16-16-2-2-2-2-per] -v
```

### 生成测试报告

```powershell
# 生成 HTML 报告
pytest tests/ --html=test_report.html --self-contained-html

# 生成覆盖率报告
pytest tests/ --cov=nsct_python --cov=nsct_torch --cov-report=html
```

## 测试辅助函数

`test_helpers.py` 提供了以下辅助函数：

### 断言函数

- `assert_shape_equal(np_array, torch_tensor)`: 验证形状相同
- `assert_values_close(np_array, torch_tensor, rtol, atol)`: 验证数值接近
- `assert_elementwise_equal(np_array, torch_tensor)`: 逐元素精确比较
- `assert_list_values_close(np_list, torch_list, rtol, atol)`: 列表数值比较

### 统计和报告函数

- `compute_statistics(np_array, torch_tensor)`: 计算详细统计信息
- `print_comparison_report(name, np_array, torch_tensor)`: 打印比较报告

## 容差设置

根据不同的测试场景，使用不同的数值容差：

- **高精度测试** (滤波器系数、简单运算):
  - `rtol=1e-10, atol=1e-12`

- **中等精度测试** (滤波操作):
  - `rtol=1e-8, atol=1e-10`

- **宽松精度测试** (复杂变换、多级运算):
  - `rtol=1e-6, atol=1e-8`

## Fixtures

`conftest.py` 提供了共享的 fixtures：

- `random_seed`: 固定随机种子（42）
- `test_image_small`: 小型测试图像 (16x16)
- `test_image_medium`: 中型测试图像 (64x64)
- `test_image_large`: 大型测试图像 (256x256)
- `test_filter_small`: 小型测试滤波器 (3x3)
- `test_filter_medium`: 中型测试滤波器 (7x7)

## 测试原则

1. **以 nsct_python 为基准**: 所有测试以 NumPy 实现为参考标准
2. **形状优先**: 首先验证输出形状是否正确
3. **数值精度**: 验证数值在合理容差内匹配
4. **逐元素分析**: 对于不匹配的情况，提供详细的差异分析
5. **参数化测试**: 使用多种参数组合确保全面覆盖
6. **往返测试**: 验证分解-重构的完美重构特性

## 故障排除

### 导入错误

如果遇到导入错误，确保：
1. 已安装所有依赖：`pip install -r requirements.txt`
2. 已安装 nsct_torch：`pip install -e nsct_torch/`
3. Python 路径正确

### 数值不匹配

数值不匹配时，测试会输出：
- 最大绝对差异
- 最大相对差异
- 差异最大的位置和值
- NumPy 和 Torch 的统计信息

根据这些信息可以诊断问题来源。

### 性能问题

大型测试可能较慢，可以：
1. 使用 `-k` 选项运行特定测试
2. 使用 `--maxfail=1` 在第一个失败后停止
3. 并行运行：`pytest -n auto` (需要 pytest-xdist)

## 持续集成

可以将这些测试集成到 CI/CD 流程中：

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - run: pip install -r requirements.txt
      - run: pytest tests/ -v
```

## 贡献指南

添加新测试时：

1. 遵循现有测试的结构和命名约定
2. 使用参数化测试覆盖多种情况
3. 提供清晰的文档字符串
4. 使用合适的容差设置
5. 添加失败时的详细诊断信息

## 许可证

与主项目相同的许可证。
