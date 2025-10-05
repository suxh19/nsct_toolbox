# NSCT 工具箱测试套件

本目录包含了用于验证 MATLAB 到 Python 翻译正确性的测试代码和数据。

## 目录结构

```
nsct_toolbox/
├── matlab_tests/          # 所有 MATLAB 测试脚本
│   ├── test_step1_*.m    # 第 1 步测试：底层操作
│   ├── test_step2_*.m    # 第 2 步测试：卷积和滤波
│   ├── test_step3_*.m    # 第 3 步测试：滤波器生成
│   ├── test_step4_*.m    # 第 4 步测试：核心分解重构
│   └── test_step5_*.m    # 第 5 步测试：完整 NSCT
├── test_data/             # 生成的测试数据（.mat 文件）
├── run_step1_tests.m     # 运行第 1 步所有测试
├── run_step2_tests.m     # 运行第 2 步所有测试
├── run_step3_tests.m     # 运行第 3 步所有测试
├── run_step4_tests.m     # 运行第 4 步所有测试
├── run_step5_tests.m     # 运行第 5 步所有测试
└── run_all_tests.m       # 运行所有测试（主脚本）
```

## 使用方法

### 在 MATLAB 中生成测试数据

1. 打开 MATLAB
2. 切换到 `nsct_toolbox` 目录
3. 运行主测试脚本：

```matlab
run_all_tests
```

这将：
- 依次运行所有 5 个步骤的测试
- 在 `test_data/` 目录下生成 `.mat` 文件
- 显示测试摘要和结果

### 单独运行某个步骤的测试

如果只想测试某个步骤：

```matlab
run_step1_tests  % 底层操作
run_step2_tests  % 卷积和滤波
run_step3_tests  % 滤波器生成
run_step4_tests  % 核心分解重构
run_step5_tests  % 完整 NSCT
```

## 测试步骤说明

### 第 1 步：底层图像和矩阵操作

测试以下函数：
- `extend2.m` - 图像边界扩展
- `symext.m` - 对称边界扩展
- `upsample2df.m` - 滤波器上采样
- `modulate2.m` - 矩阵调制
- `resampz.m` - 矩阵剪切重采样

生成文件：
- `test_data/step1_extend2.mat`
- `test_data/step1_symext.mat`
- `test_data/step1_upsample2df.mat`
- `test_data/step1_modulate2.mat`
- `test_data/step1_resampz.mat`

### 第 2 步：核心卷积和滤波

测试以下功能：
- `efilter2.m` - 带边界扩展的卷积
- MEX 文件功能（带膨胀的卷积）

生成文件：
- `test_data/step2_efilter2.mat`
- `test_data/step2_dilated_conv.mat`

### 第 3 步：滤波器生成

测试以下函数：
- `dmaxflat.m` - 菱形最大平坦滤波器
- `dfilters.m` - 方向滤波器
- `atrousfilters.m` - 金字塔滤波器
- `parafilters.m` - 平行四边形滤波器
- 辅助函数（`mctrans`, `ld2quin`, `qupz` 等）

生成文件：
- `test_data/step3_dmaxflat.mat`
- `test_data/step3_dfilters.mat`
- `test_data/step3_atrousfilters.mat`
- `test_data/step3_parafilters.mat`
- `test_data/step3_auxiliary_filters.mat`

### 第 4 步：核心分解与重构

测试以下函数：
- `nssfbdec.m` / `nssfbrec.m` - 双通道滤波器组
- `nsdfbdec.m` / `nsdfbrec.m` - 方向滤波器组
- `atrousdec.m` / `atrousrec.m` - 金字塔分解

生成文件：
- `test_data/step4_core_decomposition.mat`

### 第 5 步：完整 NSCT

测试以下函数：
- `nsctdec.m` - NSCT 分解
- `nsctrec.m` - NSCT 重构

生成文件：
- `test_data/step5_nsct_full.mat`

## Python 翻译工作流程

1. **运行 MATLAB 测试**：
   ```matlab
   run_all_tests
   ```

2. **查看生成的数据**：
   检查 `test_data/` 目录下的 `.mat` 文件

3. **按顺序翻译**：
   - 从第 1 步开始（底层操作）
   - 翻译完每个函数后，使用对应的 `.mat` 文件验证
   - 确保 Python 实现的输出与 MATLAB 输出一致（误差在可接受范围内）

4. **验证方法**：
   ```python
   import scipy.io as sio
   import numpy as np
   
   # 加载 MATLAB 测试数据
   data = sio.loadmat('test_data/step1_extend2.mat')
   
   # 获取输入和期望输出
   test_matrix = data['test_matrix']
   expected_result = data['result1']
   
   # 运行你的 Python 实现
   result = your_extend2_function(test_matrix, 1, 1, 1, 1, 'per')
   
   # 比较结果
   error = np.max(np.abs(result - expected_result))
   print(f'最大误差: {error}')
   ```

5. **误差标准**：
   - 底层操作：误差应该为 0 或接近机器精度（< 1e-14）
   - 浮点运算：误差应该 < 1e-10
   - 完整 NSCT：重构误差应该 < 1e-10

## 注意事项

- 所有测试都假设你在 `nsct_toolbox` 目录下运行
- 测试数据文件可能会比较大（特别是 step5）
- 如果某些辅助函数（如 `wfilters`, `ldfilter`）不可用，测试会跳过它们
- MEX 文件如果未编译，会使用等效的 MATLAB 代码

## 故障排除

### 问题：MEX 文件未编译

如果看到 "atrousc MEX 文件未编译" 的警告，不用担心。测试会使用等效的 MATLAB 实现。这不会影响测试数据的正确性。

### 问题：内存不足

如果测试 step5 时内存不足，可以：
1. 减小测试图像尺寸
2. 减少分解级数
3. 单独运行较小的测试

### 问题：路径错误

确保：
1. 当前目录是 `nsct_toolbox`
2. `matlab_tests` 和 `test_data` 目录存在
3. 所有 `.m` 文件都在正确的位置

## 联系与支持

如果在使用过程中遇到问题，请检查：
1. MATLAB 版本是否兼容
2. 所有必需的工具箱是否已安装
3. 文件路径是否正确

---

**祝翻译顺利！** 🎉
