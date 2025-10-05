# NSCT Toolbox Test Suite

本测试套件用于验证 Python 实现与 MATLAB 原始代码的一致性。

## 📊 测试统计

| 模块 | 测试数 | 状态 |
|------|--------|------|
| utils.py | 19 | ✅ 100% |
| filters.py | 16 | ✅ 100% |
| core.py | 12 | ✅ 100% |
| **总计** | **47** | **✅ 100%** |

## 测试结构

```
tests/
  ├── __init__.py
  ├── test_utils.py          # utils.py 函数的测试 (19 tests)
  ├── test_filters.py        # filters.py 函数的测试 (16 tests)
  ├── test_core.py           # core.py 函数的测试 (12 tests)
  ├── PROJECT_STRUCTURE.md   # 项目结构说明
  └── README.md              # 本文件
```

## 快速开始

### 一键运行所有测试

**Windows**:
```bash
run_tests.bat
```

**Linux/Mac**:
```bash
./run_tests.sh
```

这将自动：
1. 运行所有 MATLAB 测试生成参考数据
2. 运行所有 Python 测试验证一致性
3. 显示测试结果

## 详细步骤

### 1. 安装依赖

首先确保安装了所有必需的依赖：

```bash
pip install -r requirements.txt
pip install pytest scipy
```

### 2. 生成 MATLAB 参考数据

在项目根目录下运行 MATLAB 测试脚本生成参考数据：

```bash
# Utils 模块
matlab -batch "run('test_utils_matlab.m')"

# Filters 模块
matlab -batch "run('test_filters_matlab.m')"

# Core 模块
matlab -batch "run('test_core_matlab.m')"
```

这将生成 `test_utils_results.mat` 文件，包含所有测试用例的 MATLAB 参考输出。

### 3. 运行 Python 测试

运行所有测试：

```bash
pytest
```

运行特定测试文件：

```bash
pytest tests/test_utils.py
```

运行特定测试类：

```bash
pytest tests/test_utils.py::TestExtend2
```

运行特定测试函数：

```bash
pytest tests/test_utils.py::TestExtend2::test_extend2_symmetric
```

运行时显示详细输出：

```bash
pytest -v -s
```

## 测试覆盖的函数

### test_utils.py

测试 `nsct_python/utils.py` 中的以下函数：

1. **extend2** - 2D 图像扩展
   - 对称扩展 (sym)
   - 周期扩展 (per)
   - Quincunx 周期扩展 - 行方向 (qper_row)
   - Quincunx 周期扩展 - 列方向 (qper_col)

2. **upsample2df** - 2D 滤波器上采样
   - power=1 的上采样
   - power=2 的上采样

3. **modulate2** - 2D 调制
   - 行方向调制 (r)
   - 列方向调制 (c)
   - 双向调制 (b)
   - 带中心偏移的双向调制

4. **resampz** - 矩阵重采样（剪切变换）
   - Type 1: R1 = [1, 1; 0, 1]
   - Type 2: R2 = [1, -1; 0, 1]
   - Type 3: R3 = [1, 0; 1, 1]
   - Type 4: R4 = [1, 0; -1, 1]
   - 不同 shift 值的测试

5. **qupz** - Quincunx 上采样
   - Type 1: Q1 = [1, -1; 1, 1]
   - Type 2: Q2 = [1, 1; -1, 1]
   - 不同矩阵大小的测试

## 测试统计

- **总测试数**: 19
- **测试类数**: 5
- **覆盖函数数**: 5

## 测试原理

每个测试遵循以下流程：

1. 从 `test_utils_results.mat` 加载 MATLAB 生成的参考数据
2. 使用相同的输入参数调用 Python 函数
3. 比较 Python 输出与 MATLAB 参考输出
4. 验证形状和数值的完全一致性

## 故障排除

### 找不到 MATLAB 结果文件

如果看到错误 "MATLAB results file not found"，请确保：
1. 已运行 MATLAB 测试脚本
2. `test_utils_results.mat` 文件存在于项目根目录

### 测试失败

如果测试失败，检查：
1. Python 实现是否正确
2. MATLAB 版本是否兼容
3. 数值精度设置是否合适

### 导入错误

如果遇到导入错误，确保：
1. 在项目根目录运行 pytest
2. `nsct_python` 包可被正确导入
3. 所有依赖已安装

## 持续集成

可以将测试集成到 CI/CD 流程中：

```yaml
# .github/workflows/test.yml 示例
name: Test

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest scipy
      - name: Run tests
        run: pytest -v
```

## 贡献指南

添加新测试时：

1. 在 `test_utils_matlab.m` 中添加 MATLAB 测试用例
2. 重新运行 MATLAB 脚本生成新的参考数据
3. 在相应的 Python 测试文件中添加测试函数
4. 确保测试通过后再提交

## 许可证

与主项目相同的许可证。
