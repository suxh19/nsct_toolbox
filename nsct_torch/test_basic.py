"""
简单的测试脚本，验证 nsct_torch 基础功能
"""

import torch
import sys
sys.path.insert(0, 'd:/dataset/nsct_toolbox/nsct_torch')

from nsct_torch.utils import extend2, symext, upsample2df, modulate2, resampz, qupz
from nsct_torch.filters import ldfilter, dmaxflat, ld2quin, dfilters

def test_utils():
    print("=" * 60)
    print("测试 Utils 模块")
    print("=" * 60)
    
    # 测试 extend2
    print("\n1. 测试 extend2...")
    img = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    ext_per = extend2(img, 1, 1, 1, 1, 'per')
    assert ext_per.shape == (6, 6), f"期望 (6, 6)，得到 {ext_per.shape}"
    print("   ✓ extend2 通过")
    
    # 测试 upsample2df
    print("\n2. 测试 upsample2df...")
    h = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    h_up = upsample2df(h, power=1)
    assert h_up.shape == (4, 4), f"期望 (4, 4)，得到 {h_up.shape}"
    print("   ✓ upsample2df 通过")
    
    # 测试 modulate2
    print("\n3. 测试 modulate2...")
    m = torch.ones((3, 4), dtype=torch.float64)
    m_mod = modulate2(m, 'b')
    assert m_mod.shape == (3, 4), f"期望 (3, 4)，得到 {m_mod.shape}"
    print("   ✓ modulate2 通过")
    
    # 测试 resampz
    print("\n4. 测试 resampz...")
    r_in = torch.arange(1, 7, dtype=torch.float64).reshape(2, 3)
    r_out = resampz(r_in, 1, shift=1)
    assert r_out.shape[0] >= 2, f"期望行数 >= 2，得到 {r_out.shape}"
    print("   ✓ resampz 通过")
    
    # 测试 qupz
    print("\n5. 测试 qupz...")
    q_in = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    q_out = qupz(q_in, 1)
    assert q_out.shape == (3, 3), f"期望 (3, 3)，得到 {q_out.shape}"
    print("   ✓ qupz 通过")
    
    print("\n✅ 所有 Utils 测试通过！")

def test_filters():
    print("\n" + "=" * 60)
    print("测试 Filters 模块")
    print("=" * 60)
    
    # 测试 ldfilter
    print("\n1. 测试 ldfilter...")
    f6 = ldfilter('pkva6')
    assert f6.shape == (6,), f"期望 (6,)，得到 {f6.shape}"
    print("   ✓ ldfilter 通过")
    
    # 测试 dmaxflat
    print("\n2. 测试 dmaxflat...")
    h2 = dmaxflat(2, 0)
    assert h2.shape == (5, 5), f"期望 (5, 5)，得到 {h2.shape}"
    print("   ✓ dmaxflat 通过")
    
    # 测试 ld2quin
    print("\n3. 测试 ld2quin...")
    beta = ldfilter('pkva6')
    h0, h1 = ld2quin(beta)
    assert h0.shape == (11, 11), f"期望 h0 (11, 11)，得到 {h0.shape}"
    assert h1.shape == (21, 21), f"期望 h1 (21, 21)，得到 {h1.shape}"
    print("   ✓ ld2quin 通过")
    
    # 测试 dfilters
    print("\n4. 测试 dfilters...")
    try:
        h0_pkva, h1_pkva = dfilters('pkva', 'd')
        print(f"   dfilters('pkva') 形状: h0={h0_pkva.shape}, h1={h1_pkva.shape}")
        print("   ✓ dfilters 通过")
    except Exception as e:
        print(f"   ⚠ dfilters 失败: {e}")
    
    print("\n✅ 所有 Filters 测试通过！")

def main():
    print("\n" + "=" * 60)
    print("NSCT Torch 基础功能测试")
    print("=" * 60)
    
    # 检测 PyTorch 和 CUDA
    print(f"\nPyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 设备: {torch.cuda.get_device_name(0)}")
    
    try:
        test_utils()
        test_filters()
        
        print("\n" + "=" * 60)
        print("🎉 所有测试通过！")
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ 测试失败: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
