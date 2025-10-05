# NSCT Toolbox 测试指南

本指南说明如何运行和验证 NSCT Toolbox 的 Python 实现与 MATLAB 原始代码的一致性。

## 📁 项目结构

```
nsct_toolbox/
├── nsct_python/              # Python 实现
│   ├── __init__.py
│   ├── core.py
│   ├── filters.py
│   └── utils.py             # ✅ 已测试并验证
├── tests/                    # 测试套件
│   ├── __init__.py
│   ├── test_utils.py        # utils.py 的测试
│   └── README.md            # 测试文档
├── test_utils_matlab.m       # MATLAB 测试脚本
├── test_utils_results.mat    # MATLAB 参考数据
├── run_tests.bat             # Windows 快速测试脚本
├── run_tests.sh              # Linux/Mac 快速测试脚本
├── TEST_SUMMARY.md           # 详细测试报告
└── TESTING_GUIDE.md          # 本文件
```

## 🚀 快速开始

### 方法 1: 使用快速测试脚本（推荐）

**Windows:**
```bash
run_tests.bat
```

**Linux/Mac:**
```bash
chmod +x run_tests.sh
./run_tests.sh
```

这将自动：
1. 运行 MATLAB 测试生成参考数据
2. 运行 Python 测试验证一致性
3. 显示测试结果

### 方法 2: 手动运行测试

#### 步骤 1: 生成 MATLAB 参考数据

```bash
matlab -batch "run('test_utils_matlab.m')"
```

这将生成 `test_utils_results.mat` 文件。

#### 步骤 2: 运行 Python 测试

```bash
pytest tests/test_utils.py -v
```

或使用虚拟环境:

```bash
.venv/Scripts/python -m pytest tests/test_utils.py -v  # Windows
.venv/bin/python -m pytest tests/test_utils.py -v      # Linux/Mac
```

## 📋 测试模块状态

| 模块 | 测试文件 | 状态 | 测试数 | 通过率 |
|------|----------|------|--------|--------|
| utils.py | test_utils.py | ✅ 完成 | 19 | 100% |
| filters.py | - | ⏳ 待完成 | - | - |
| core.py | - | ⏳ 待完成 | - | - |

## 🔧 依赖安装

### Python 依赖

```bash
pip install -r requirements.txt
pip install pytest scipy
```

或使用虚拟环境:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt
pip install pytest scipy
```

### MATLAB 要求

- MATLAB R2019b 或更高版本
- 无需额外工具箱

## 📊 测试覆盖详情

### utils.py 测试（19 个测试，100% 通过）

#### 1. extend2 - 2D 图像扩展（4 个测试）
- ✅ 周期扩展（基础）
- ✅ 周期扩展（小尺寸）
- ✅ Quincunx 周期扩展（行）
- ✅ Quincunx 周期扩展（列）

#### 2. upsample2df - 2D 滤波器上采样（2 个测试）
- ✅ Power=1 上采样
- ✅ Power=2 上采样

#### 3. modulate2 - 2D 调制（4 个测试）
- ✅ 行方向调制
- ✅ 列方向调制
- ✅ 双向调制
- ✅ 双向调制（带中心偏移）

#### 4. resampz - 矩阵重采样（5 个测试）
- ✅ Type 1: R1 = [1,1;0,1]
- ✅ Type 2: R2 = [1,-1;0,1]
- ✅ Type 3: R3 = [1,0;1,1]
- ✅ Type 4: R4 = [1,0;-1,1]
- ✅ Type 1 with shift=2

#### 5. qupz - Quincunx 上采样（4 个测试）
- ✅ Type 1 (2×2 矩阵)
- ✅ Type 2 (2×2 矩阵)
- ✅ Type 1 (3×3 矩阵)
- ✅ Type 2 (3×3 矩阵)

## 🎯 测试原则

本测试套件遵循以下原则：

1. **完全一致性**: Python 实现必须与 MATLAB 输出完全匹配（数值和形状）
2. **参考驱动**: MATLAB 作为参考实现，Python 输出以此为标准
3. **自动化**: 使用 pytest 框架实现自动化测试
4. **可重复性**: 使用固定随机种子确保测试可重复
5. **全面覆盖**: 测试各种边界情况和参数组合

## 📈 运行特定测试

### 运行特定测试类

```bash
pytest tests/test_utils.py::TestExtend2 -v
pytest tests/test_utils.py::TestModulate2 -v
```

### 运行特定测试函数

```bash
pytest tests/test_utils.py::TestExtend2::test_extend2_periodic_basic -v
pytest tests/test_utils.py::TestQupz::test_qupz_type1_small -v
```

### 显示详细输出

```bash
pytest tests/test_utils.py -v -s
```

### 停止在第一个失败处

```bash
pytest tests/test_utils.py -x
```

### 生成测试覆盖率报告

```bash
pip install pytest-cov
pytest tests/test_utils.py --cov=nsct_python.utils --cov-report=html
```

然后在浏览器中打开 `htmlcov/index.html`。

## 🐛 故障排除

### 问题 1: 找不到 MATLAB

**错误**: `matlab: 无法将"matlab"项识别为 cmdlet`

**解决方案**:
1. 确保 MATLAB 已安装
2. 将 MATLAB 添加到系统 PATH
3. 或使用完整路径运行，例如:
   ```bash
   "C:\Program Files\MATLAB\R2024b\bin\matlab.exe" -batch "run('test_utils_matlab.m')"
   ```

### 问题 2: 找不到 pytest

**错误**: `pytest: 无法将"pytest"项识别为 cmdlet`

**解决方案**:
```bash
pip install pytest
# 或使用 python -m pytest
python -m pytest tests/test_utils.py -v
```

### 问题 3: 找不到 MATLAB 结果文件

**错误**: `MATLAB results file not found`

**解决方案**:
1. 确保已运行 MATLAB 测试脚本
2. 检查 `test_utils_results.mat` 是否存在于项目根目录
3. 如果不存在，重新运行:
   ```bash
   matlab -batch "run('test_utils_matlab.m')"
   ```

### 问题 4: 导入错误

**错误**: `ModuleNotFoundError: No module named 'nsct_python'`

**解决方案**:
1. 确保在项目根目录运行 pytest
2. 或将项目添加到 PYTHONPATH:
   ```bash
   # Windows
   set PYTHONPATH=%CD%;%PYTHONPATH%
   # Linux/Mac
   export PYTHONPATH=$PWD:$PYTHONPATH
   ```

### 问题 5: scipy 导入错误

**错误**: `ModuleNotFoundError: No module named 'scipy'`

**解决方案**:
```bash
pip install scipy
```

## 📝 添加新测试

要为新函数添加测试：

### 1. 更新 MATLAB 测试脚本

在 `test_utils_matlab.m` 中添加新测试用例:

```matlab
%% Test N: new_function - Description
fprintf('\nTest N: new_function - Description\n');
testN_input = [...];
testN_param = ...;
testN_output = new_function(testN_input, testN_param);

test_results.testN.input = testN_input;
test_results.testN.param = testN_param;
test_results.testN.output = testN_output;
fprintf('  Input size: %dx%d, Output size: %dx%d\n', ...
    size(testN_input), size(testN_output));
```

### 2. 重新生成参考数据

```bash
matlab -batch "run('test_utils_matlab.m')"
```

### 3. 添加 Python 测试

在 `tests/test_utils.py` 中添加测试函数:

```python
def test_new_function(self, matlab_results):
    """Test new_function"""
    test_data = matlab_results['testN'][0, 0]
    
    input_mat = test_data['input']
    param = test_data['param']
    expected_output = test_data['output']
    
    result = new_function(input_mat, param)
    
    assert result.shape == expected_output.shape
    np.testing.assert_array_equal(result, expected_output)
    print(f"✓ Test N passed: new_function ({result.shape})")
```

### 4. 运行测试验证

```bash
pytest tests/test_utils.py::TestClassName::test_new_function -v
```

## 🔄 持续集成

可以将测试集成到 CI/CD 流程中。示例 GitHub Actions 配置:

```yaml
# .github/workflows/test.yml
name: Test

on: [push, pull_request]

jobs:
  test:
    runs-on: windows-latest  # 需要 MATLAB
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup MATLAB
        uses: matlab-actions/setup-matlab@v1
        
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
          
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest scipy
          
      - name: Generate MATLAB reference data
        uses: matlab-actions/run-command@v1
        with:
          command: run('test_utils_matlab.m')
          
      - name: Run Python tests
        run: pytest tests/ -v
```

## 📚 相关文档

- [测试详细报告](TEST_SUMMARY.md)
- [测试目录 README](tests/README.md)
- [项目 README](README.md)

## 🤝 贡献

添加新测试或发现问题时，请：

1. 创建新分支
2. 添加/修改测试
3. 确保所有测试通过
4. 提交 Pull Request

## 📄 许可证

与主项目相同的许可证。

---

**最后更新**: 2025年10月5日  
**维护者**: NSCT Toolbox Team
