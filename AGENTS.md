# Repository Guidelines

## Project Structure & Module Organization
- `nsct_python/` hosts the Python API for nonsubsampled contourlet transforms. Core numerical routines live in `core.py`, `filters.py`, and `utils.py`; PyTorch equivalents sit in the `*_torch.py` modules. C++/CUDA bindings remain inside `atrousc_cpp/`, `atrousc_cuda/`, `zconv2_cpp/`, and `zconv2_cuda/`—keep their Python interfaces aligned.
- Reusable fixtures and reference data belong in `data/` and `mat_tests/`; MATLAB sources stay in `nsct_matlab/` for parity checks. Public documentation lives under `docs/`.
- Tests belong in `pytests/`, mirroring module names. Integration and comparison helpers reside in `scripts/`, e.g. `run_python_nsct.py`. Avoid modifying `test_image.jpg` unless you regenerate goldens.

## Build, Test, and Development Commands
- `python -m pip install -r requirements.txt` — lightweight development install; extend this file when new Python deps land.
- `pytest` — run the full suite with verbose output and strict marker checks (configured in `pyproject.toml`).
- `pytest -m "not slow"` — iterate quickly by skipping long-running spectral comparisons.
- `python scripts/compare_implementations.py --help` — inspect cross-backend drift; run before shipping changes to compiled paths.

## Coding Style & Naming Conventions
- Enforce Black formatting with 100-character lines and the paired isort profile (`black . && isort .` on touched files).
- Prefer type hints and NumPy-style docstrings as in `nsct_python/core.py`; keep public APIs snake_case, classes PascalCase, modules lowercase.
- Mirror NumPy signatures in Torch code; shared helpers belong in `utils.py` and `utils_torch.py`.

## Testing Guidelines
- Add tests under `pytests/` using files named `test_*.py` and functions `test_*`; reuse existing fixtures for filters and sample images.
- Apply markers `@pytest.mark.slow` or `@pytest.mark.matlab` when invoking MATLAB baselines or long FFT-based loops.
- Store newly generated reference arrays in `data/`, validate with `scripts/compare_implementations.py`, and document any tolerance changes.

## Commit & Pull Request Guidelines
- Write descriptive subject lines (~72 chars) that start with an action, mirroring history such as “Add comprehensive test suite for PyTorch filters”; expand context in the body when compiled assets or tensors change.
- Each PR should summarize motivation, highlight affected modules, list validation commands, and link issues or datasets; attach before/after evidence when reconstruction outputs shift.
- When rebuilding native extensions, note regenerated binaries, build env (compiler, CUDA), and resulting paths in the PR description.

## rules
- 此项目使用uv的 .venv 虚拟环境
