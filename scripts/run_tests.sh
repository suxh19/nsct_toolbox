#!/bin/bash
# Run MATLAB and Python tests for nsct_toolbox
# Usage: ./run_tests.sh

echo "========================================"
echo "NSCT Toolbox Complete Test Suite"
echo "========================================"
echo

# Step 1: Generate MATLAB reference data
echo "[Step 1/2] Generating MATLAB reference data..."
echo

echo "[1/3] Testing utils functions..."
matlab -batch "run('test_utils_matlab.m')"
if [ $? -ne 0 ]; then
    echo "ERROR: MATLAB utils test failed!"
    exit 1
fi

echo "[2/3] Testing filters functions..."
matlab -batch "run('test_filters_matlab.m')"
if [ $? -ne 0 ]; then
    echo "ERROR: MATLAB filters test failed!"
    exit 1
fi

echo "[3/3] Testing core functions..."
matlab -batch "run('test_core_matlab.m')"
if [ $? -ne 0 ]; then
    echo "ERROR: MATLAB core test failed!"
    exit 1
fi

echo
echo "[MATLAB] All reference data generated successfully!"
echo

# Step 2: Run Python tests
echo "[Step 2/2] Running Python tests..."
pytest tests/ -v --tb=short

if [ $? -ne 0 ]; then
    echo "ERROR: Python tests failed!"
    exit 1
fi

echo
echo "========================================"
echo "All 47 tests passed successfully! ✓"
echo "========================================"
echo
echo "Test coverage:"
echo "  - utils.py:    19 tests"
echo "  - filters.py:  16 tests"
echo "  - core.py:     12 tests"
echo "========================================"
