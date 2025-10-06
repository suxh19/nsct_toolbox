"""
Demo script for nsfbdec (Nonsubsampled Filter Bank Decomposition).

This script demonstrates:
1. Basic decomposition at different levels
2. Energy conservation properties
3. Visualization of lowpass and highpass outputs
"""

import numpy as np
import matplotlib.pyplot as plt
from nsct_python.core import nsfbdec
from nsct_python.filters import atrousfilters


def demo_basic_decomposition():
    """Demonstrate basic nsfbdec usage."""
    print("=" * 60)
    print("Demo 1: Basic Decomposition")
    print("=" * 60)
    
    # Load filters
    print("\n1. Loading 'maxflat' atrous filters...")
    h0, h1, g0, g1 = atrousfilters('maxflat')
    print(f"   h0 shape: {h0.shape}")
    print(f"   h1 shape: {h1.shape}")
    
    # Create test image
    print("\n2. Creating random test image (64x64)...")
    np.random.seed(42)
    x = np.random.rand(64, 64)
    print(f"   Image shape: {x.shape}")
    print(f"   Image range: [{x.min():.3f}, {x.max():.3f}]")
    
    # Decompose at level 0
    print("\n3. Decomposing at level 0...")
    y0, y1 = nsfbdec(x, h0, h1, 0)
    print(f"   y0 (lowpass) shape: {y0.shape}")
    print(f"   y1 (highpass) shape: {y1.shape}")
    print(f"   y0 range: [{y0.min():.3f}, {y0.max():.3f}]")
    print(f"   y1 range: [{y1.min():.3f}, {y1.max():.3f}]")
    
    # Check energy
    energy_in = np.sum(x**2)
    energy_out = np.sum(y0**2) + np.sum(y1**2)
    energy_ratio = energy_out / energy_in
    print(f"\n4. Energy analysis:")
    print(f"   Input energy: {energy_in:.6f}")
    print(f"   Output energy: {energy_out:.6f}")
    print(f"   Energy ratio: {energy_ratio:.6f}")
    
    # Visualize
    print("\n5. Creating visualization...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    im0 = axes[0].imshow(x, cmap='gray')
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046)
    
    im1 = axes[1].imshow(y0, cmap='gray')
    axes[1].set_title('Lowpass Output (y0)')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    
    im2 = axes[2].imshow(y1, cmap='gray', vmin=-0.5, vmax=0.5)
    axes[2].set_title('Highpass Output (y1)')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig('demo_nsfbdec_basic.png', dpi=150, bbox_inches='tight')
    print("   Saved: demo_nsfbdec_basic.png")
    plt.show()


def demo_multi_level():
    """Demonstrate decomposition at multiple levels."""
    print("\n" + "=" * 60)
    print("Demo 2: Multi-Level Decomposition")
    print("=" * 60)
    
    # Load filters
    h0, h1, g0, g1 = atrousfilters('maxflat')
    
    # Create test image (larger for higher levels)
    print("\n1. Creating test image (128x128)...")
    np.random.seed(42)
    x = np.random.rand(128, 128)
    
    # Decompose at multiple levels
    print("\n2. Decomposing at levels 0, 1, 2...")
    levels = [0, 1, 2]
    results = []
    
    for lev in levels:
        print(f"\n   Level {lev}:")
        y0, y1 = nsfbdec(x, h0, h1, lev)
        
        energy_in = np.sum(x**2)
        energy_out = np.sum(y0**2) + np.sum(y1**2)
        energy_y0 = np.sum(y0**2)
        energy_y1 = np.sum(y1**2)
        
        print(f"      Energy ratio: {energy_out/energy_in:.6f}")
        print(f"      Lowpass energy: {energy_y0/energy_in*100:.2f}%")
        print(f"      Highpass energy: {energy_y1/energy_in*100:.2f}%")
        
        results.append((y0, y1))
    
    # Visualize all levels
    print("\n3. Creating multi-level visualization...")
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    # Plot input in first column
    for i in range(3):
        axes[i, 0].imshow(x, cmap='gray')
        axes[i, 0].set_title(f'Input (Level {i})')
        axes[i, 0].axis('off')
    
    # Plot outputs
    for i, (y0, y1) in enumerate(results):
        axes[i, 1].imshow(y0, cmap='gray')
        axes[i, 1].set_title(f'Lowpass (Level {i})')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(y1, cmap='gray', vmin=-0.5, vmax=0.5)
        axes[i, 2].set_title(f'Highpass (Level {i})')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('demo_nsfbdec_multilevel.png', dpi=150, bbox_inches='tight')
    print("   Saved: demo_nsfbdec_multilevel.png")
    plt.show()


def demo_natural_image():
    """Demonstrate decomposition on a natural image pattern."""
    print("\n" + "=" * 60)
    print("Demo 3: Natural Image Pattern")
    print("=" * 60)
    
    # Load filters
    h0, h1, g0, g1 = atrousfilters('maxflat')
    
    # Create a simple pattern (checkerboard + gradient)
    print("\n1. Creating synthetic image pattern...")
    size = 128
    x = np.zeros((size, size))
    
    # Add checkerboard pattern
    block_size = 8
    for i in range(0, size, block_size):
        for j in range(0, size, block_size):
            if ((i // block_size) + (j // block_size)) % 2 == 0:
                x[i:i+block_size, j:j+block_size] = 1.0
    
    # Add smooth gradient
    yy, xx = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
    gradient = 0.5 * (xx + yy) / 2
    x = 0.5 * x + 0.5 * gradient
    
    print(f"   Pattern shape: {x.shape}")
    
    # Decompose
    print("\n2. Decomposing at level 0...")
    y0, y1 = nsfbdec(x, h0, h1, 0)
    
    print(f"   Lowpass captures smooth gradient")
    print(f"   Highpass captures edges/details")
    
    # Analyze frequency content
    energy_in = np.sum(x**2)
    energy_y0 = np.sum(y0**2)
    energy_y1 = np.sum(y1**2)
    
    print(f"\n3. Frequency analysis:")
    print(f"   Lowpass energy: {energy_y0/energy_in*100:.2f}%")
    print(f"   Highpass energy: {energy_y1/energy_in*100:.2f}%")
    
    # Visualize
    print("\n4. Creating visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    axes[0, 0].imshow(x, cmap='gray')
    axes[0, 0].set_title('Input Pattern')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(y0, cmap='gray')
    axes[0, 1].set_title('Lowpass (Smooth Components)')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(y1, cmap='gray', vmin=-0.5, vmax=0.5)
    axes[1, 0].set_title('Highpass (Edge Details)')
    axes[1, 0].axis('off')
    
    # Show absolute value of highpass
    axes[1, 1].imshow(np.abs(y1), cmap='hot')
    axes[1, 1].set_title('Highpass (Magnitude)')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('demo_nsfbdec_pattern.png', dpi=150, bbox_inches='tight')
    print("   Saved: demo_nsfbdec_pattern.png")
    plt.show()


def demo_filter_comparison():
    """Compare different filter types."""
    print("\n" + "=" * 60)
    print("Demo 4: Filter Comparison")
    print("=" * 60)
    
    # Create test image
    print("\n1. Creating test image...")
    np.random.seed(42)
    x = np.random.rand(64, 64)
    
    # Test with different filters
    filter_types = ['maxflat', '9-7']
    print("\n2. Comparing filter types...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for i, fname in enumerate(filter_types):
        print(f"\n   Filter: {fname}")
        h0, h1, g0, g1 = atrousfilters(fname)
        print(f"      h0 shape: {h0.shape}")
        print(f"      h1 shape: {h1.shape}")
        
        y0, y1 = nsfbdec(x, h0, h1, 0)
        
        energy_in = np.sum(x**2)
        energy_out = np.sum(y0**2) + np.sum(y1**2)
        print(f"      Energy ratio: {energy_out/energy_in:.6f}")
        
        axes[i, 0].imshow(x, cmap='gray')
        axes[i, 0].set_title(f'Input ({fname})')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(y0, cmap='gray')
        axes[i, 1].set_title(f'Lowpass ({fname})')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(y1, cmap='gray', vmin=-0.5, vmax=0.5)
        axes[i, 2].set_title(f'Highpass ({fname})')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('demo_nsfbdec_filters.png', dpi=150, bbox_inches='tight')
    print("\n3. Saved: demo_nsfbdec_filters.png")
    plt.show()


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("NSFBDEC Demonstration")
    print("Nonsubsampled Filter Bank Decomposition")
    print("=" * 60)
    
    try:
        demo_basic_decomposition()
        demo_multi_level()
        demo_natural_image()
        demo_filter_comparison()
        
        print("\n" + "=" * 60)
        print("All demos completed successfully!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
