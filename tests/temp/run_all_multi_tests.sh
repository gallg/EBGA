#!/bin/bash

echo "=========================================="
echo "Running all MultiCandidateOptimizer tests"
echo "=========================================="
echo ""

# Set PYTHONPATH
PYTHONPATH=/var/home/ggallitto/Documenti/Projects/EBGA:$PYTHONPATH
PYTHON=/var/home/ggallitto/Documenti/Projects/EBGA/.venv/bin/python

# Run each test
echo "1. Iris Classification Test"
echo "----------------------------------------"
$PYTHON tests/temp/test_iris_multi.py
echo ""

echo "2. Wine Classification Test"
echo "----------------------------------------"
$PYTHON tests/temp/test_wine_multi.py
echo ""

echo "3. Breast Cancer Classification Test"
echo "----------------------------------------"
$PYTHON tests/temp/test_breast_cancer_multi.py
echo ""

echo "4. Diabetes Regression Test"
echo "----------------------------------------"
$PYTHON tests/temp/test_diabetes_multi.py
echo ""

echo "5. California Housing Regression Test"
echo "----------------------------------------"
$PYTHON tests/temp/test_california_housing_multi.py
echo ""

echo "=========================================="
echo "All tests completed!"
echo "=========================================="
