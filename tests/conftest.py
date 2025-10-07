"""
Pytest 配置文件和共享 fixtures
"""
import numpy as np
import torch
import pytest


@pytest.fixture
def random_seed():
    """固定随机种子以确保测试可复现"""
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed


@pytest.fixture
def test_image_small():
    """小型测试图像 (16x16)"""
    np.random.seed(42)
    return np.random.randn(16, 16).astype(np.float64)


@pytest.fixture
def test_image_medium():
    """中型测试图像 (64x64)"""
    np.random.seed(42)
    return np.random.randn(64, 64).astype(np.float64)


@pytest.fixture
def test_image_large():
    """大型测试图像 (256x256)"""
    np.random.seed(42)
    return np.random.randn(256, 256).astype(np.float64)


@pytest.fixture
def test_filter_small():
    """小型测试滤波器 (3x3)"""
    np.random.seed(42)
    return np.random.randn(3, 3).astype(np.float64)


@pytest.fixture
def test_filter_medium():
    """中型测试滤波器 (7x7)"""
    np.random.seed(42)
    return np.random.randn(7, 7).astype(np.float64)
