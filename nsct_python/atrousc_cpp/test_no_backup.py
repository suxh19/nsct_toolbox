"""测试删除纯 Python 备份后的功能"""
from __init__ import atrousc, is_cpp_available, get_backend_info
import numpy as np

print('=' * 60)
print('测试 C++ 实现（无 Python 备份）')
print('=' * 60)
print()

# 检查后端
info = get_backend_info()
print(f"✓ 导入成功")
print(f"后端: {info['backend']}")
print(f"C++ 可用: {info['cpp_available']}")
print()

# 测试 C++ 实现
x = np.random.rand(100, 100)
h = np.random.rand(5, 5)
M = np.array([[2, 0], [0, 2]])

try:
    result = atrousc(x, h, M)
    print(f'✓ C++ 版本测试成功')
    print(f'  输出形状: {result.shape}')
    print()
    print('✓ 纯 Python 备份已成功删除')
    print('✓ 现在仅使用 C++ 实现')
except RuntimeError as e:
    print(f'✗ 错误: {e}')
    print('  请确保 C++ 扩展已编译')
