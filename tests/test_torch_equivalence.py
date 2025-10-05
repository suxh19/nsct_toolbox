import numpy as np
import torch
import pytest
import sys
import os
from pathlib import Path

# Add the parent directory to sys.path to import nsct_torch and nsct_python
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))
# --- Import original NumPy functions ---
from nsct_python import utils as utils_np
from nsct_python import filters as filters_np
from nsct_python import core as core_np

# --- Import new PyTorch functions ---
from nsct_torch import utils as utils_torch
from nsct_torch import filters as filters_torch
from nsct_torch import core as core_torch

# --- Test Fixtures ---
@pytest.fixture
def sample_image_np():
    # Using an even-sized image as required by some filters for perfect reconstruction
    return np.random.rand(32, 32)

@pytest.fixture
def sample_image_torch(sample_image_np):
    return torch.from_numpy(sample_image_np)

@pytest.fixture
def sample_filter_np():
    return np.random.rand(5, 7)

@pytest.fixture
def sample_filter_torch(sample_filter_np):
    return torch.from_numpy(sample_filter_np)

# --- Helper for comparison ---
def assert_tensors_close(t1, t2, atol=1e-7):
    """Asserts that two tensors are close, handling both torch and numpy arrays."""
    if isinstance(t1, torch.Tensor):
        t1 = t1.detach().cpu().numpy()
    if isinstance(t2, torch.Tensor):
        t2 = t2.detach().cpu().numpy()
    assert np.allclose(t1, t2, atol=atol), f"Tensors not close.\nNumpy:\n{t1}\nTorch:\n{t2}"

# --- Tests for UTILS functions ---

def test_extend2_equivalence(sample_image_np, sample_image_torch):
    args = (4, 5, 6, 7)
    modes = ['per', 'qper_row', 'qper_col']
    for mode in modes:
        np_res = utils_np.extend2(sample_image_np, *args, extmod=mode)
        torch_res = utils_torch.extend2(sample_image_torch, *args, extmod=mode)
        assert_tensors_close(np_res, torch_res)

def test_upsample2df_equivalence(sample_filter_np, sample_filter_torch):
    np_res = utils_np.upsample2df(sample_filter_np, power=2)
    torch_res = utils_torch.upsample2df(sample_filter_torch, power=2)
    assert_tensors_close(np_res, torch_res)

def test_modulate2_equivalence(sample_image_np, sample_image_torch):
    modes = ['r', 'c', 'b']
    for mode in modes:
        np_res = utils_np.modulate2(sample_image_np, mode=mode)
        torch_res = utils_torch.modulate2(sample_image_torch, mode=mode)
        assert_tensors_close(np_res, torch_res)

def test_resampz_equivalence(sample_filter_np, sample_filter_torch):
    for type in range(1, 5):
        np_res = utils_np.resampz(sample_filter_np, type=type)
        torch_res = utils_torch.resampz(sample_filter_torch, type=type)
        assert_tensors_close(np_res, torch_res)

def test_qupz_equivalence(sample_filter_np, sample_filter_torch):
    for type in range(1, 3):
        np_res = utils_np.qupz(sample_filter_np, type=type)
        torch_res = utils_torch.qupz(sample_filter_torch, type=type)
        assert_tensors_close(np_res, torch_res)

# --- Tests for FILTERS functions ---

def test_ldfilter_equivalence():
    names = ['pkva6', 'pkva8', 'pkva']
    for name in names:
        np_res = filters_np.ldfilter(name)
        torch_res = filters_torch.ldfilter(name)
        assert_tensors_close(np_res, torch_res)

def test_dmaxflat_equivalence():
    for n in range(1, 4): # Only test implemented cases
        np_res = filters_np.dmaxflat(n, d=0.5)
        torch_res = filters_torch.dmaxflat(n, d=0.5)
        assert_tensors_close(np_res, torch_res)

def test_atrousfilters_equivalence():
    names = ['pyr', 'pyrexc']
    for name in names:
        np_res = filters_np.atrousfilters(name)
        torch_res = filters_torch.atrousfilters(name)
        for i in range(4):
            assert_tensors_close(np_res[i], torch_res[i])

def test_mctrans_equivalence():
    b_np = filters_np.ldfilter('pkva6')
    t_np = filters_np.dmaxflat(2, 0)
    b_torch = torch.from_numpy(b_np)
    t_torch = torch.from_numpy(t_np)

    np_res = filters_np.mctrans(b_np, t_np)
    torch_res = filters_torch.mctrans(b_torch, t_torch)
    assert_tensors_close(np_res, torch_res)

def test_dfilters_equivalence():
    names = ['pkva', 'db2', 'dmaxflat3']
    types = ['d', 'r']
    for name in names:
        for type in types:
            np_h0, np_h1 = filters_np.dfilters(name, type)
            torch_h0, torch_h1 = filters_torch.dfilters(name, type)
            assert_tensors_close(np_h0, torch_h0)
            assert_tensors_close(np_h1, torch_h1)

def test_ld2quin_equivalence():
    beta_np = filters_np.ldfilter('pkva6')
    beta_torch = torch.from_numpy(beta_np)
    np_h0, np_h1 = filters_np.ld2quin(beta_np)
    torch_h0, torch_h1 = filters_torch.ld2quin(beta_torch)
    assert_tensors_close(np_h0, torch_h0)
    assert_tensors_close(np_h1, torch_h1)

def test_parafilters_equivalence(sample_filter_np, sample_filter_torch):
    f1_np, f2_np = sample_filter_np, np.random.rand(4, 6)
    f1_torch, f2_torch = torch.from_numpy(f1_np), torch.from_numpy(f2_np)

    np_y1, np_y2 = filters_np.parafilters(f1_np, f2_np)
    torch_y1, torch_y2 = filters_torch.parafilters(f1_torch, f2_torch)

    for i in range(4):
        assert_tensors_close(np_y1[i], torch_y1[i])
        assert_tensors_close(np_y2[i], torch_y2[i])

def test_efilter2_equivalence(sample_image_np, sample_image_torch, sample_filter_np, sample_filter_torch):
    np_res = filters_np.efilter2(sample_image_np, sample_filter_np)
    torch_res = filters_torch.efilter2(sample_image_torch, sample_filter_torch)
    assert_tensors_close(np_res, torch_res)


# --- Tests for CORE functions ---

def test_nssfbdec_rec_equivalence(sample_image_np, sample_image_torch):
    # Get filters
    h0_np, h1_np = filters_np.dfilters('pkva', 'd')
    g0_np, g1_np = filters_np.dfilters('pkva', 'r')

    h0_torch = torch.from_numpy(h0_np)
    h1_torch = torch.from_numpy(h1_np)
    g0_torch = torch.from_numpy(g0_np)
    g1_torch = torch.from_numpy(g1_np)

    # Define upsampling matrix
    mup_np = np.array([[1, 1], [-1, 1]])
    mup_torch = torch.from_numpy(mup_np)

    # --- Test Decomposition ---
    np_y1, np_y2 = core_np.nssfbdec(sample_image_np, h0_np, h1_np, mup_np)
    torch_y1, torch_y2 = core_torch.nssfbdec(sample_image_torch, h0_torch, h1_torch, mup_torch)

    assert_tensors_close(np_y1, torch_y1)
    assert_tensors_close(np_y2, torch_y2)

    # --- Test Reconstruction ---
    np_recon = core_np.nssfbrec(np_y1, np_y2, g0_np, g1_np, mup_np)
    torch_recon = core_torch.nssfbrec(torch_y1, torch_y2, g0_torch, g1_torch, mup_torch)

    assert_tensors_close(np_recon, torch_recon)

    # --- Test "Perfect" Reconstruction ---
    # NOTE: The original numpy implementation has a gain of 2 for the 'pkva'
    # filter. We test against this behavior to ensure our translation is faithful.
    assert_tensors_close(sample_image_np * 2, np_recon, atol=1e-6)
    assert_tensors_close(sample_image_torch * 2, torch_recon, atol=1e-6)