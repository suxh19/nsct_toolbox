# 安装支持CUDA的PyTorch

## 当前情况

- **GPU**: NVIDIA GeForce RTX 4060
- **CUDA版本**: 12.9
- **当前PyTorch**: 2.8.0+cpu (不支持CUDA)
- **需要**: 安装CUDA版本的PyTorch

## 安装步骤

### 方法1: 使用pip安装 (推荐)

```powershell
# 激活虚拟环境
D:/dataset/nsct_toolbox/.venv/Scripts/Activate.ps1

# 卸载当前的CPU版本PyTorch
pip uninstall torch torchvision torchaudio -y

# 安装CUDA 12.1版本的PyTorch（推荐，与CUDA 12.9兼容）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 方法2: 如果方法1不行，使用conda (备选)

```powershell
# 使用conda安装
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

### 方法3: 安装最新的CUDA 12.4版本

```powershell
# 激活虚拟环境
D:/dataset/nsct_toolbox/.venv/Scripts/Activate.ps1

# 卸载当前版本
pip uninstall torch torchvision torchaudio -y

# 安装CUDA 12.4版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

## 验证安装

安装完成后，运行以下命令验证：

```powershell
D:/dataset/nsct_toolbox/.venv/Scripts/Activate.ps1
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

**期望输出**:
```
CUDA available: True
CUDA version: 12.1
Device: NVIDIA GeForce RTX 4060
```

## 运行GPU测试

安装CUDA版本后，运行GPU测试：

```powershell
# 激活虚拟环境
D:/dataset/nsct_toolbox/.venv/Scripts/Activate.ps1

# 运行所有GPU测试
pytest tests/test_torch_equivalence.py::TestGPUEquivalence -v

# 或运行完整测试套件
pytest tests/test_torch_equivalence.py -v
```

## 注意事项

1. **CUDA版本兼容性**: 你的系统CUDA是12.9，但PyTorch目前最高支持到12.4。这没问题，向后兼容。

2. **下载大小**: CUDA版本的PyTorch大约2-3GB，下载可能需要一些时间。

3. **requirements.txt更新**: 安装完成后，更新requirements文件：
   ```powershell
   pip freeze > requirements-torch.txt
   ```

4. **如果遇到问题**: 
   - 确保NVIDIA驱动是最新的
   - 检查虚拟环境是否正确激活
   - 尝试清理pip缓存: `pip cache purge`

## 快速安装脚本

复制以下命令到PowerShell一次性执行：

```powershell
# 激活虚拟环境并安装CUDA版本PyTorch
D:/dataset/nsct_toolbox/.venv/Scripts/Activate.ps1
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 验证安装
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# 运行GPU测试
pytest tests/test_torch_equivalence.py::TestGPUEquivalence -v
```

## 预期测试结果

安装CUDA版本后，所有16个GPU测试应该全部通过：

```
tests/test_torch_equivalence.py::TestGPUEquivalence::test_extend2_gpu PASSED
tests/test_torch_equivalence.py::TestGPUEquivalence::test_upsample2df_gpu PASSED
tests/test_torch_equivalence.py::TestGPUEquivalence::test_modulate2_gpu PASSED
... (共16个测试)
================================ 16 passed in X.XXs ================================
```

## 性能提升预期

在RTX 4060上，GPU版本预计会有以下性能提升：

- 小图像 (32x32): 1-2x 加速
- 中等图像 (256x256): 5-10x 加速  
- 大图像 (1024x1024): 20-50x 加速

具体加速比取决于具体操作和数据大小。
