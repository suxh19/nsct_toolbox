# NSCT工具箱严格测试方案

## 执行总结

本文档描述了NSCT工具箱从MATLAB到Python严格一比一翻译的详细测试策略和执行计划。

### 项目目标
实现MATLAB NSCT工具箱的完整Python翻译，确保：
1. **数值精度**: 与MATLAB结果误差 < 1e-10
2. **功能完整性**: 覆盖所有函数和参数组合
3. **边界条件**: 处理所有特殊情况和边界输入
4. **完美重建**: 所有滤波器组合的MSE < 1e-20

---

## 1. 已完成工作

### 1.1 详细映射文档 ✅
**文件**: `docs/MATLAB_PYTHON_MAPPING.md`

**内容**:
- 15个核心函数的详细对比
- 每个函数的MATLAB和Python实现对比
- 参数映射关系
- 关键算法步骤对应
- 已知差异和注意事项
- 数值精度分析
- 索引转换规则

**关键发现**:
- ⚠️ `qupz`函数实现方法不同（需要验证数值等价性）
- ⚠️ `dmaxflat`未完全实现（仅N=1,2,3）
- ✅ 大部分函数逻辑完全对应

### 1.2 扩展的MATLAB测试套件 ✅
**文件**: `mat_tests/test_edge_cases_matlab.m`

**覆盖范围**:
- **Section 1**: extend2边界情况（5个测试）
  - 2x2小矩阵
  - 单元素矩阵
  - 非方阵
  - 大扩展量
  - 零扩展

- **Section 2**: upsample2df边界情况（3个测试）
  - 零矩阵
  - 单元素
  - 高幂次

- **Section 3**: modulate2边界情况（4个测试）
  - 单行矩阵
  - 单列矩阵
  - 负中心偏移
  - 大中心偏移

- **Section 4**: qupz边界情况（4个测试）
  - 单元素（type 1和2）
  - 非方阵
  - 零矩阵

- **Section 5**: resampz边界情况（4个测试）
  - 单行/单列
  - 大移位量
  - 零移位

- **Section 6**: nssfbdec/nssfbrec边界情况（5个测试）
  - 2x2小输入
  - 零矩阵
  - 常数矩阵
  - 非方阵
  - 小输入完美重建

- **Section 7**: efilter2边界情况（3个测试）
  - 小滤波器
  - 大滤波器在小图像上
  - 最大移位

- **Section 8**: ldfilter和ld2quin边界情况（2个测试）
  - 最小滤波器（pkva6）
  - ld2quin转换

- **Section 9**: 特殊值测试（4个测试）
  - 极小值（1e-10）
  - 极大值（1e10）
  - 负值
  - 混合正负值

- **Section 10**: 不同滤波器类型（3个测试）
  - pyrexc滤波器
  - dmaxflat N=1和N=2

**总计**: 约40个边界情况测试

### 1.3 对应的Python测试套件 ✅
**文件**: `pytests/test_edge_cases.py`

**特点**:
- 使用pytest框架
- 与MATLAB测试一一对应
- 详细的错误信息
- 多种精度级别（decimal=10, 14, 20, 25）
- 相对和绝对误差检查
- 完整的docstring文档

**测试类**:
1. `TestExtend2EdgeCases` - 5个测试
2. `TestUpsample2dfEdgeCases` - 3个测试
3. `TestModulate2EdgeCases` - 4个测试
4. `TestQupzEdgeCases` - 4个测试
5. `TestResampzEdgeCases` - 4个测试
6. `TestNssfbEdgeCases` - 5个测试
7. `TestEfilter2EdgeCases` - 3个测试
8. `TestLdfilterLd2quinEdgeCases` - 2个测试
9. `TestSpecialValuesEdgeCases` - 4个测试
10. `TestDifferentFilterTypes` - 3个测试

---

## 2. 测试执行计划

### 2.1 阶段1: MATLAB基准生成（需要MATLAB环境）

```bash
# 在MATLAB命令窗口中执行
cd('d:\dataset\nsct_toolbox\mat_tests')

# 运行所有MATLAB测试
test_core_matlab
test_filters_matlab
test_utils_matlab
test_edge_cases_matlab

# 验证生成的文件
ls ../data/*.mat
```

**预期输出**:
- `data/test_core_results.mat` (12个测试)
- `data/test_filters_results.mat` (16个测试)
- `data/test_utils_results.mat` (19个测试)
- `data/test_edge_cases_results.mat` (40个测试)

### 2.2 阶段2: Python测试验证

```bash
# 激活虚拟环境
cd d:\dataset\nsct_toolbox
.\.venv\Scripts\Activate.ps1

# 运行所有测试
pytest pytests/ -v --tb=short

# 运行特定测试文件
pytest pytests/test_core.py -v
pytest pytests/test_filters.py -v
pytest pytests/test_utils.py -v
pytest pytests/test_edge_cases.py -v

# 生成覆盖率报告
pytest pytests/ --cov=nsct_python --cov-report=html --cov-report=term
```

### 2.3 阶段3: 详细分析

对于任何失败的测试：

1. **记录差异**:
   ```python
   # 在失败的测试中添加
   diff = result - expected
   print(f"Max abs diff: {np.abs(diff).max()}")
   print(f"Max rel diff: {np.abs(diff / expected).max()}")
   print(f"MSE: {np.mean(diff**2)}")
   ```

2. **可视化差异**（如果适用）:
   ```python
   import matplotlib.pyplot as plt
   plt.figure(figsize=(15, 5))
   plt.subplot(131); plt.imshow(expected); plt.title('MATLAB')
   plt.subplot(132); plt.imshow(result); plt.title('Python')
   plt.subplot(133); plt.imshow(diff); plt.title('Difference')
   plt.colorbar()
   plt.savefig('diff_analysis.png')
   ```

3. **逐行调试**:
   - 使用Python调试器
   - 打印中间结果
   - 与MATLAB中间结果对比

---

## 3. 关键验证点

### 3.1 qupz函数验证（高优先级）

**问题**: Python和MATLAB实现方法完全不同
- MATLAB: 基于Smith分解和resampz链
- Python: 基于数学定义的直接实现

**验证步骤**:
1. 运行所有qupz相关测试
2. 对比以下情况的输出：
   - 单元素矩阵
   - 2x2, 3x3, 4x4方阵
   - 非方阵
   - 零矩阵
3. 验证type 1和type 2的输出
4. 确保与ld2quin的集成正常

**通过标准**: 
- 所有测试的绝对误差 < 1e-14
- 输出尺寸完全匹配
- 非零元素位置完全匹配

### 3.2 完美重建测试（关键指标）

**测试矩阵**:
```
滤波器类型 × 输入尺寸 × mup参数
```

| 滤波器 | 尺寸 | mup | MSE阈值 |
|--------|------|-----|---------|
| pkva | 4x4 | None | 1e-20 |
| pkva | 8x8 | None | 1e-20 |
| pkva | 16x16 | None | 1e-20 |
| pkva | 8x8 | 1 | 1e-20 |
| pkva | 8x8 | 2 | 1e-20 |
| pkva | 8x8 | [[1,1],[-1,1]] | 1e-20 |
| pyr | 8x8 | [[1,1],[-1,1]] | 1e-20 |
| pyrexc | 8x8 | [[1,1],[-1,1]] | 1e-20 |

**验证指标**:
- **MSE**: Mean Squared Error < 1e-20
- **PSNR**: Peak Signal-to-Noise Ratio > 200 dB
- **Max Error**: 最大绝对误差 < 1e-10

### 3.3 数值稳定性测试

**测试场景**:
1. **极小值**: 输入 = 1e-10 * ones(4, 4)
2. **极大值**: 输入 = 1e10 * ones(4, 4)
3. **混合尺度**: 输入 = [1e-10, 1, 1e10] 混合
4. **负值**: 输入 = -rand(4, 4)

**验证**:
- 没有NaN或Inf
- 相对误差在可接受范围内
- 输出符号和尺度正确

---

## 4. 当前测试覆盖率

### 4.1 基础测试覆盖

| 模块 | 函数 | 基础测试 | 边界测试 | 状态 |
|------|------|----------|----------|------|
| core.py | nssfbdec | 5 | 5 | ✅ |
| core.py | nssfbrec | 4 | 1 | ✅ |
| filters.py | ldfilter | 3 | 1 | ✅ |
| filters.py | ld2quin | 2 | 1 | ✅ |
| filters.py | dmaxflat | 3 | 2 | ⚠️ |
| filters.py | atrousfilters | 2 | 1 | ✅ |
| filters.py | mctrans | 2 | 0 | ⚠️ |
| filters.py | efilter2 | 2 | 3 | ✅ |
| filters.py | parafilters | 2 | 0 | ⚠️ |
| filters.py | dfilters | 3 | 0 | ⚠️ |
| utils.py | extend2 | 4 | 5 | ✅ |
| utils.py | upsample2df | 2 | 3 | ✅ |
| utils.py | modulate2 | 4 | 4 | ✅ |
| utils.py | resampz | 5 | 4 | ✅ |
| utils.py | qupz | 2 | 4 | ⚠️ |

**总计**:
- **基础测试**: 47个
- **边界测试**: 34个
- **总测试数**: 81个

**覆盖状态**:
- ✅ 完全覆盖: 11个函数
- ⚠️ 需要更多测试: 4个函数（dmaxflat, mctrans, parafilters, dfilters, qupz）

### 4.2 需要补充的测试

#### 4.2.1 mctrans
- [ ] 更大的滤波器（N > 3）
- [ ] 不同的变换滤波器
- [ ] 边界情况（1x1, 非对称滤波器）

#### 4.2.2 parafilters
- [ ] 不同滤波器尺寸组合
- [ ] 非方阵滤波器
- [ ] 所有4个输出的详细验证

#### 4.2.3 dfilters
- [ ] 所有支持的滤波器名称
- [ ] 分解和重建模式对比
- [ ] 与PyWavelets的集成测试

#### 4.2.4 dmaxflat
- [ ] 实现N=4-7
- [ ] 所有d值（0和1）
- [ ] 验证对称性

#### 4.2.5 qupz（关键）
- [ ] 更大尺寸（5x5, 10x10）
- [ ] 详细的中间步骤验证
- [ ] 与MATLAB resampz链的等价性证明

---

## 5. 执行检查清单

### 5.1 运行MATLAB测试
- [ ] 运行 `test_core_matlab.m`
- [ ] 运行 `test_filters_matlab.m`
- [ ] 运行 `test_utils_matlab.m`
- [ ] 运行 `test_edge_cases_matlab.m`
- [ ] 验证所有`.mat`文件生成
- [ ] 检查MATLAB命令窗口无错误

### 5.2 运行Python测试
- [ ] 激活虚拟环境
- [ ] 运行 `pytest pytests/test_core.py -v`
- [ ] 运行 `pytest pytests/test_filters.py -v`
- [ ] 运行 `pytest pytests/test_utils.py -v`
- [ ] 运行 `pytest pytests/test_edge_cases.py -v`
- [ ] 检查所有测试通过
- [ ] 记录任何失败或警告

### 5.3 分析结果
- [ ] 计算整体通过率
- [ ] 识别失败模式
- [ ] 分析数值误差分布
- [ ] 检查异常值和离群点

### 5.4 问题解决
对于每个失败的测试：
- [ ] 记录输入和预期输出
- [ ] 打印Python实际输出
- [ ] 计算误差指标
- [ ] 调试关键步骤
- [ ] 修复代码或更新测试
- [ ] 重新运行确认

### 5.5 文档更新
- [ ] 更新测试报告
- [ ] 记录已知问题
- [ ] 更新映射文档
- [ ] 添加使用示例

---

## 6. 成功标准

### 6.1 定量指标

| 指标 | 目标 | 当前 |
|------|------|------|
| 测试通过率 | 100% | TBD |
| 平均绝对误差 | < 1e-10 | TBD |
| 最大绝对误差 | < 1e-8 | TBD |
| 完美重建MSE | < 1e-20 | TBD |
| 代码覆盖率 | > 90% | TBD |

### 6.2 定性指标

- [ ] 所有函数均有对应的MATLAB实现
- [ ] 所有参数组合均已测试
- [ ] 边界情况处理正确
- [ ] 错误处理与MATLAB一致
- [ ] 文档完整清晰
- [ ] 代码风格符合规范

---

## 7. 已知限制和未来工作

### 7.1 当前限制

1. **dmaxflat未完全实现**
   - 仅支持N=1,2,3
   - 需要添加N=4-7的系数

2. **性能未优化**
   - 未使用Cython或Numba
   - MEX函数的Python等价物可能较慢

3. **某些滤波器未测试**
   - 'pkva-half'未实现
   - dmaxflat高阶滤波器未测试

### 7.2 未来改进

1. **性能优化**
   - 使用Numba JIT编译关键函数
   - 考虑Cython实现
   - 多线程/GPU加速

2. **功能扩展**
   - 添加更多滤波器类型
   - 支持3D NSCT
   - 添加可视化工具

3. **测试增强**
   - 添加性能基准测试
   - 添加内存使用分析
   - 创建自动化测试流水线

4. **文档改进**
   - 添加完整的API文档
   - 创建教程和示例
   - 添加理论背景说明

---

## 8. 参考资料

### 8.1 论文
1. Cunha, A. L., Zhou, J., & Do, M. N. (2006). "The nonsubsampled contourlet transform: theory, design, and applications." IEEE transactions on image processing, 15(10), 3089-3101.

### 8.2 代码仓库
- MATLAB原始工具箱: [nsct_matlab/]
- Python实现: [nsct_python/]

### 8.3 相关文档
- [MATLAB_PYTHON_MAPPING.md](MATLAB_PYTHON_MAPPING.md): 详细函数映射
- [TEST_SUMMARY.md](TEST_SUMMARY.md): 测试总结（如有）
- [TESTING_GUIDE.md](TESTING_GUIDE.md): 测试指南（如有）

---

## 9. 联系和支持

如有问题或建议，请：
1. 查阅现有文档
2. 检查已知问题列表
3. 运行完整测试套件
4. 提供详细的错误报告（包括输入、输出、错误信息）

---

**版本**: 1.0  
**日期**: 2025-10-05  
**作者**: AI Assistant  
**状态**: 进行中
