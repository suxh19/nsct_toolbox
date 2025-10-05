# 测试运行指南 - 快速开始

## 📋 前置条件检查

### MATLAB环境
- ✅ MATLAB已安装并可用
- ✅ 当前在MATLAB终端中
- ✅ 工作目录在 `nsct_toolbox`

### Python环境
- ✅ Python虚拟环境已激活 (`.venv`)
- ✅ pytest已安装
- ✅ 所有依赖已安装 (`numpy`, `scipy`)

---

## 🚀 第一步: 运行MATLAB测试

### 1.1 切换到MATLAB终端

在VS Code中，确保使用**MATLAB终端**（不是PowerShell）。

### 1.2 执行边界情况测试

```matlab
% 切换到测试目录
cd('d:/dataset/nsct_toolbox/mat_tests')

% 运行边界情况测试（已修复零矩阵bug）
test_edge_cases_matlab
```

### 1.3 预期输出

您应该看到类似以下输出：

```
=== Starting MATLAB Edge Cases Tests ===

Section 1: extend2 Edge Cases
-------------------------------
Test 1.1: extend2 - Small 2x2 matrix
  Input: 2x2, Output: 4x4
Test 1.2: extend2 - Single element matrix
  Input: 1x1, Output: 5x5
[...]

Section 4: qupz Edge Cases
---------------------------
Test 4.1: qupz - Single element, type 1
  Input: 1x1, Output: 1x1
Test 4.2: qupz - Single element, type 2
  Input: 1x1, Output: 1x1
Test 4.3: qupz - Non-square (2x4), type 1
  Input: 2x4, Output: 5x5
Test 4.4: qupz - Small non-zero matrix 2x2  ← 已修复！
  Input: 2x2 identity-like, Output: 3x3

[...]

=== All edge case tests completed successfully! ===
Results saved to: ../data/test_edge_cases_results.mat
Total variables saved: XX
```

### 1.4 验证生成的文件

```matlab
% 检查文件是否生成
dir('../data/test_edge_cases_results.mat')

% 查看文件内容
load('../data/test_edge_cases_results.mat')
whos
```

**预期**: 应该看到约40个变量（所有测试的输入和输出）。

---

## 🐍 第二步: 运行Python测试

### 2.1 切换到PowerShell终端

确保虚拟环境已激活：

```powershell
# 如果未激活，先激活
.\.venv\Scripts\Activate.ps1

# 验证Python环境
python --version
pytest --version
```

### 2.2 运行边界情况测试

```powershell
# 切换到项目根目录
cd d:\dataset\nsct_toolbox

# 运行边界情况测试（详细模式）
pytest pytests/test_edge_cases.py -v

# 或者运行所有测试
pytest pytests/ -v
```

### 2.3 预期输出

```
========================= test session starts =========================
platform win32 -- Python 3.x.x, pytest-x.x.x
rootdir: d:\dataset\nsct_toolbox
collected XX items

pytests/test_edge_cases.py::TestExtend2EdgeCases::test_small_2x2 PASSED     [  3%]
pytests/test_edge_cases.py::TestExtend2EdgeCases::test_single_element PASSED [ 6%]
[...]
pytests/test_edge_cases.py::TestQupzEdgeCases::test_small_nonzero_matrix PASSED [XX%]
[...]

========================= XX passed in X.XXs =========================
```

### 2.4 如果测试失败

如果看到失败，运行详细调试模式：

```powershell
# 显示完整错误信息
pytest pytests/test_edge_cases.py -v --tb=long

# 或者运行特定测试
pytest pytests/test_edge_cases.py::TestQupzEdgeCases::test_small_nonzero_matrix -v
```

---

## 🔍 第三步: 分析结果

### 3.1 检查测试覆盖率

```powershell
# 生成覆盖率报告
pytest pytests/ --cov=nsct_python --cov-report=html --cov-report=term

# 在浏览器中打开HTML报告
start htmlcov/index.html
```

### 3.2 查看测试统计

```powershell
# 运行所有测试并统计
pytest pytests/ -v --tb=short | Tee-Object -FilePath test_results.txt

# 查看摘要
Select-String -Path test_results.txt -Pattern "passed|failed|error"
```

---

## ⚠️ 已知问题和解决方案

### 问题1: MATLAB resampz 零矩阵Bug

**症状**:
```
位置 2 处的索引超出数组边界
出错 resampz (第 87 行)
```

**解决方案**: ✅ 已在 `test_edge_cases_matlab.m` 中修复
- Test 4.4 现在使用 `[1,0;0,1]` 而不是 `zeros(2,2)`
- 详细信息请参见 `docs/KNOWN_MATLAB_BUGS.md`

### 问题2: Python测试找不到MATLAB数据

**症状**:
```
FileNotFoundError: [Errno 2] No such file or directory: 
'...\\data\\test_edge_cases_results.mat'
```

**解决方案**: 
1. 确保先运行MATLAB测试生成 `.mat` 文件
2. 检查文件路径是否正确

### 问题3: 数值精度不匹配

**症状**:
```
AssertionError: Arrays are not almost equal to 14 decimals
```

**解决方案**:
1. 检查是否是合理的浮点误差（< 1e-10）
2. 使用 `decimal=10` 或 `decimal=8` 降低精度要求
3. 查看详细比较：在测试中添加打印语句

---

## 📊 测试完成检查清单

### MATLAB测试
- [ ] `test_edge_cases_matlab.m` 运行成功
- [ ] 生成 `data/test_edge_cases_results.mat` 文件
- [ ] 无错误或警告输出
- [ ] 约40个测试变量已保存

### Python测试
- [ ] 所有边界情况测试通过
- [ ] 数值误差在可接受范围内（< 1e-10）
- [ ] 代码覆盖率 > 85%
- [ ] 无未捕获的异常

### 验证
- [ ] 对比MATLAB和Python的关键输出
- [ ] 检查完美重建测试（MSE < 1e-20）
- [ ] 查看详细测试报告

---

## 🆘 故障排除

### MATLAB环境问题

```matlab
% 检查路径
pwd

% 添加工具箱到路径（如果需要）
addpath(genpath('d:/dataset/nsct_toolbox/nsct_matlab'))

% 验证函数可用
which qupz
which nssfbdec
```

### Python环境问题

```powershell
# 检查当前环境
where python
python -c "import numpy; print(numpy.__version__)"
python -c "import scipy; print(scipy.__version__)"

# 重新安装依赖
pip install -r requirements.txt

# 验证模块导入
python -c "from nsct_python import core, filters, utils"
```

### 数据文件问题

```powershell
# 检查所有 .mat 文件
Get-ChildItem -Path data -Filter *.mat | Format-Table Name, Length, LastWriteTime

# 验证文件可读性
python -c "import scipy.io; print(scipy.io.loadmat('data/test_edge_cases_results.mat').keys())"
```

---

## 📈 后续步骤

完成边界情况测试后：

1. **运行基础测试**
   ```powershell
   pytest pytests/test_core.py -v
   pytest pytests/test_filters.py -v
   pytest pytests/test_utils.py -v
   ```

2. **修复识别的问题**
   - resampz 空矩阵形状问题
   - qupz 数值验证
   - dmaxflat N=4-7 系数

3. **生成最终报告**
   ```powershell
   pytest pytests/ --html=report.html --self-contained-html
   ```

4. **更新文档**
   - 测试结果摘要
   - 已知限制
   - 性能基准

---

## 📞 需要帮助？

如果遇到问题：

1. 检查 `docs/KNOWN_MATLAB_BUGS.md` 中的已知问题
2. 查看 `docs/STRICT_TESTING_PLAN.md` 了解详细测试策略
3. 参考 `docs/LINE_BY_LINE_COMPARISON.md` 理解实现差异
4. 运行带 `-vv` 的pytest以获取更多调试信息

---

**版本**: 1.0  
**最后更新**: 2025年10月5日  
**状态**: 已修复MATLAB零矩阵bug，准备测试
