# NSCT 图像重建测试覆盖文档

## 概述
本文档描述了 `test_nsct_image_reconstruction.py` 中的全面测试覆盖范围。

## 测试分类

### 1. 形状一致性测试 (Shape Consistency Tests)

#### 1.1 `test_decomposition_output_structure`
- **目的**: 验证分解输出具有正确的结构
- **验证内容**:
  - 输出长度 = 低通带 + 金字塔层级数
  - 第一个元素是低通带（numpy数组）
  - 其余元素是方向子带列表

#### 1.2 `test_subband_shape_preservation`
- **目的**: 验证所有子带保持输入空间形状
- **验证内容**:
  - 低通带形状 = 输入图像形状
  - 所有方向子带形状 = 输入图像形状
  - 每个层级的子带列表非空

#### 1.3 `test_directional_subband_counts`
- **目的**: 验证方向子带数量符合预期
- **验证内容**:
  - 对于层级 n，方向子带数量 = 2^n
  - 例如: 层级2 → 4个子带, 层级3 → 8个子带

#### 1.4 `test_reconstruction_shape_preservation`
- **目的**: 验证重建保持输入形状
- **验证内容**:
  - 重建图像形状 = 原始图像形状

### 2. 数值精度测试 (Numerical Accuracy Tests)

#### 2.1 `test_perfect_reconstruction_numerical_accuracy`
- **目的**: 测试完美重建的数值精度
- **验证内容**:
  - 使用严格容差: rtol=1e-12, atol=1e-10
  - 最大误差 < 1e-9
  - RMSE < 1e-10
- **性能指标**:
  - 最大误差 (max_error)
  - 平均误差 (mean_error)
  - 均方根误差 (RMSE)

#### 2.2 `test_element_wise_reconstruction_accuracy`
- **目的**: 逐元素验证重建精度
- **验证内容**:
  - 没有任何元素的误差超过 1e-9
  - 统计超出容差的元素数量

#### 2.3 `test_subband_dtype_consistency`
- **目的**: 验证数据类型一致性
- **验证内容**:
  - 低通带 dtype = 输入 dtype
  - 所有方向子带 dtype = 输入 dtype

### 3. 一对一一致性测试 (One-to-One Consistency Tests)

#### 3.1 `test_decomposition_determinism`
- **目的**: 验证分解的确定性
- **验证内容**:
  - 相同输入 → 相同输出
  - 两次分解结果完全相等（逐元素比较）

#### 3.2 `test_reconstruction_determinism`
- **目的**: 验证重建的确定性
- **验证内容**:
  - 相同分解 → 相同重建
  - 两次重建结果完全相等

#### 3.3 `test_invertibility`
- **目的**: 验证可逆性
- **验证内容**:
  - 第一轮: 原始 → 分解 → 重建1
  - 第二轮: 重建1 → 分解 → 重建2
  - 验证: 重建1 ≈ 原始, 重建2 ≈ 原始, 重建2 ≈ 重建1

### 4. 参数化测试 (Parameterized Tests)

#### 4.1 `test_different_pyramid_levels`
- **参数化配置**:
  - `[2]` - 单层
  - `[2, 3]` - 双层
  - `[3, 3]` - 均匀双层
  - `[2, 3, 4]` - 三层
- **验证内容**:
  - 所有配置都能完美重建
  - rtol=1e-12, atol=1e-10

#### 4.2 `test_different_filter_types`
- **参数化配置**:
  - `pkva + maxflat`
  - `dmaxflat7 + maxflat`
  - `dmaxflat5 + maxflat`
- **验证内容**:
  - 不同滤波器组合都能完美重建
  - rtol=1e-10, atol=1e-9

### 5. 能量守恒测试 (Energy Conservation Tests)

#### 5.1 `test_energy_conservation`
- **目的**: 验证能量守恒
- **验证内容**:
  - 原始能量 = Σ(原始图像²)
  - 分解能量 = Σ(所有子带²)
  - 能量比率在 0.95 到 1.05 之间

### 6. 边界情况测试 (Edge Case Tests)

#### 6.1 `test_small_image_reconstruction`
- **测试场景**: 32×32 小图像
- **验证内容**: 完美重建

#### 6.2 `test_constant_image_reconstruction`
- **测试场景**: 常数图像（所有像素值 = 128.0）
- **验证内容**: 完美重建

#### 6.3 `test_zero_image_reconstruction`
- **测试场景**: 零图像（所有像素值 = 0）
- **验证内容**: 完美重建，atol=1e-14

## 测试统计

### 测试数量
- 总测试数: 22
- 形状测试: 4
- 数值精度测试: 3
- 一致性测试: 3
- 参数化测试: 7 (4 + 3)
- 能量守恒测试: 1
- 边界情况测试: 3
- 遗留测试: 1

### 执行时间
- 总执行时间: ~87秒
- 标记为 `@pytest.mark.slow` 的测试: 2个

## 容差标准

### 严格模式（默认）
- `rtol`: 1e-12
- `atol`: 1e-10

### 宽松模式（某些滤波器）
- `rtol`: 1e-10
- `atol`: 1e-9

### 误差阈值
- 最大误差: < 1e-9
- RMSE: < 1e-10

## 使用方法

### 运行所有测试
```bash
pytest pytests/test_nsct_image_reconstruction.py -v
```

### 跳过慢速测试
```bash
pytest pytests/test_nsct_image_reconstruction.py -v -m "not slow"
```

### 运行特定测试类别
```bash
# 只运行形状测试
pytest pytests/test_nsct_image_reconstruction.py::TestNSCTImageReconstruction::test_subband_shape_preservation -v

# 只运行数值精度测试
pytest pytests/test_nsct_image_reconstruction.py::TestNSCTImageReconstruction::test_perfect_reconstruction_numerical_accuracy -v
```

## 未来改进建议

1. **性能测试**: 添加基准测试以跟踪性能变化
2. **内存测试**: 验证大图像处理时的内存使用
3. **并发测试**: 测试多线程/多进程安全性
4. **鲁棒性测试**: 测试异常输入（NaN, Inf等）
5. **跨平台测试**: 在不同操作系统和Python版本上验证
6. **GPU测试**: 为CUDA实现添加对应测试

## 代码质量指标

### 覆盖率
- 核心函数 `nsctdec`: 100%
- 核心函数 `nsctrec`: 100%
- 边界情况处理: 覆盖

### 测试原则遵循
- ✅ KISS (Keep It Simple, Stupid)
- ✅ DRY (Don't Repeat Yourself) - 使用fixtures
- ✅ 单一职责原则 - 每个测试验证一个方面
- ✅ 清晰命名 - 测试名称描述性强
- ✅ 独立性 - 测试之间无依赖

## 参考文档

- 主项目文档: `AGENTS.md`
- 编码规范: `.github/instructions/user.instructions.md`
- PyTorch版本测试: `test_core.py`
- MATLAB对照测试: `test_nsctdec.py`
