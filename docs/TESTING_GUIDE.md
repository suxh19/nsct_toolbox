# NSCT Toolbox 测试使用指南

## 快速开始

### 运行所有测试
```bash
pytest tests/test_torch_equivalence.py -v
```

### 运行特定模块的测试
```bash
# 测试工具函数
pytest tests/test_torch_equivalence.py -k "Utils" -v

# 测试滤波器函数
pytest tests/test_torch_equivalence.py -k "Filters" -v

# 测试核心函数
pytest tests/test_torch_equivalence.py -k "Core" -v
```

### 运行特定函数的测试
```bash
# 测试 extend2 函数
pytest tests/test_torch_equivalence.py::TestExtend2 -v

# 测试 nssfbdec 函数
pytest tests/test_torch_equivalence.py::TestNssfbdec -v
```

## 测试示例

### 验证 CPU 和 GPU 版本输出一致

```python
import numpy as np
import torch
from nsct_python import core as core_np, filters as filters_np
from nsct_torch import core as core_torch, filters as filters_torch

# 创建测试图像
img_np = np.random.rand(32, 32)
img_torch = torch.from_numpy(img_np)

# 获取滤波器
h0_np, h1_np = filters_np.dfilters('pkva', 'd')
h0_torch = torch.from_numpy(h0_np)
h1_torch = torch.from_numpy(h1_np)

# 定义上采样矩阵
mup_np = np.array([[1, 1], [-1, 1]])
mup_torch = torch.from_numpy(mup_np)

# CPU版本分解
np_y1, np_y2 = core_np.nssfbdec(img_np, h0_np, h1_np, mup_np)

# GPU版本分解
torch_y1, torch_y2 = core_torch.nssfbdec(img_torch, h0_torch, h1_torch, mup_torch)

# 验证结果一致
print(f"Y1 匹配: {np.allclose(np_y1, torch_y1.numpy(), atol=1e-9)}")
print(f"Y2 匹配: {np.allclose(np_y2, torch_y2.numpy(), atol=1e-9)}")
```

### 完美重建验证

```python
# 获取重建滤波器
g0_np, g1_np = filters_np.dfilters('pkva', 'r')
g0_torch = torch.from_numpy(g0_np)
g1_torch = torch.from_numpy(g1_np)

# 重建
np_recon = core_np.nssfbrec(np_y1, np_y2, g0_np, g1_np, mup_np)
torch_recon = core_torch.nssfbrec(torch_y1, torch_y2, g0_torch, g1_torch, mup_torch)

# 验证完美重建（pkva滤波器有增益2）
print(f"CPU 重建误差: {np.mean((img_np * 2 - np_recon) ** 2):.2e}")
print(f"GPU 重建误差: {np.mean((img_torch * 2 - torch_recon).numpy() ** 2):.2e}")
```

## 在GPU上运行测试

如果你有CUDA设备，可以运行GPU等价性测试：

```bash
# 检查CUDA是否可用
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"

# 运行GPU测试
pytest tests/test_torch_equivalence.py::TestGPUEquivalence -v
```

### GPU测试示例

```python
import torch

# 确保CUDA可用
if torch.cuda.is_available():
    # CPU版本
    img_cpu = torch.rand(32, 32)
    h0_cpu, h1_cpu = filters_torch.dfilters('pkva', 'd', device='cpu')
    
    # GPU版本
    img_gpu = img_cpu.cuda()
    h0_gpu, h1_gpu = filters_torch.dfilters('pkva', 'd', device='cuda')
    
    # 分解
    mup = torch.tensor([[1, 1], [-1, 1]], dtype=torch.long)
    cpu_y1, cpu_y2 = core_torch.nssfbdec(img_cpu, h0_cpu, h1_cpu, mup)
    
    mup_gpu = mup.cuda()
    gpu_y1, gpu_y2 = core_torch.nssfbdec(img_gpu, h0_gpu, h1_gpu, mup_gpu)
    
    # 验证一致性
    print(f"GPU匹配: {torch.allclose(cpu_y1, gpu_y1.cpu(), atol=1e-9)}")
else:
    print("CUDA不可用，跳过GPU测试")
```

## 性能对比

```python
import time

# CPU性能
start = time.time()
for _ in range(100):
    np_y1, np_y2 = core_np.nssfbdec(img_np, h0_np, h1_np, mup_np)
cpu_time = time.time() - start

# GPU性能
if torch.cuda.is_available():
    img_gpu = img_torch.cuda()
    h0_gpu = h0_torch.cuda()
    h1_gpu = h1_torch.cuda()
    mup_gpu = mup_torch.cuda()
    
    # 预热
    _ = core_torch.nssfbdec(img_gpu, h0_gpu, h1_gpu, mup_gpu)
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(100):
        torch_y1, torch_y2 = core_torch.nssfbdec(img_gpu, h0_gpu, h1_gpu, mup_gpu)
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    
    print(f"CPU时间: {cpu_time:.3f}s")
    print(f"GPU时间: {gpu_time:.3f}s")
    print(f"加速比: {cpu_time/gpu_time:.2f}x")
```

## 常见问题

### Q: 为什么GPU测试被跳过？
A: GPU测试需要CUDA设备。如果你的机器没有NVIDIA GPU或未安装CUDA，这些测试会自动跳过。

### Q: 测试失败怎么办？
A: 查看详细的错误信息：
```bash
pytest tests/test_torch_equivalence.py -v --tb=short
```

### Q: 如何添加新的测试？
A: 在 `test_torch_equivalence.py` 中添加新的测试类或测试方法，遵循现有的命名约定。

### Q: 精度要求是多少？
A: 
- 工具函数和滤波器: atol=1e-10
- 核心函数: atol=1e-9
- 完美重建: atol=1e-6

## 持续集成

在CI/CD管道中集成测试：

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest
      - name: Run tests
        run: pytest tests/test_torch_equivalence.py -v
```

## 测试覆盖率

查看测试覆盖率：

```bash
pip install pytest-cov
pytest tests/test_torch_equivalence.py --cov=nsct_python --cov=nsct_torch --cov-report=html
```

然后打开 `htmlcov/index.html` 查看详细的覆盖率报告。
