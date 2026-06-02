#!/usr/bin/env python3

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_iris_test():
    from tests.test_iris import run_test
    return run_test()


def run_breast_cancer_test():
    from tests.test_breast_cancer import run_test
    return run_test()


def run_wine_test():
    from tests.test_wine import run_test
    return run_test()


def run_diabetes_test():
    from tests.test_diabetes import run_test
    return run_test()


def run_california_housing_test():
    from tests.test_california_housing import run_test
    return run_test()


def main():
    print("=" * 70)
    print("EBGA FRAMEWORK - TEST SUITE")
    print("Evolutionary-Based Gradient-free Architecture")
    print("=" * 70)
    print("\nAll models use EXPLICIT LAYER SPECIFICATION")
    print("Layer-wise training is enabled by default\n")
    
    results = {}
    
    # Run standard dataset tests
    print("\n" + "=" * 70)
    print("STANDARD DATASET TESTS")
    print("=" * 70)
    
    # Iris
    iris_model, iris_acc, _ = run_iris_test()
    results['Iris'] = {'accuracy': iris_acc, 'model': iris_model}
    
    # Breast Cancer
    cancer_model, cancer_acc, _ = run_breast_cancer_test()
    results['Breast Cancer'] = {'accuracy': cancer_acc, 'model': cancer_model}
    
    # Wine
    wine_model, wine_acc, _ = run_wine_test()
    results['Wine'] = {'accuracy': wine_acc, 'model': wine_model}
    
    # Diabetes (Regression)
    diabetes_model, diabetes_r2, _ = run_diabetes_test()
    results['Diabetes'] = {'r2': diabetes_r2, 'model': diabetes_model}
    
    # California Housing (Regression)
    housing_model, housing_r2, _ = run_california_housing_test()
    results['California Housing'] = {'r2': housing_r2, 'model': housing_model}
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print("\nStandard Classification Datasets:")
    for dataset in ['Iris', 'Breast Cancer', 'Wine']:
        if dataset in results:
            acc = results[dataset]['accuracy']
            print(f"  {dataset:20s}: Accuracy = {acc:.4f}")

    print("\nRegression Datasets:")
    for dataset in ['Diabetes', 'California Housing']:
        if dataset in results:
            r2 = results[dataset]['r2']
            print(f"  {dataset:20s}: R² = {r2:.4f}")
    
    print("\n" + "=" * 70)
    print("All tests completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
