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
    r2 = run_test()
    return r2, None  # baseline not computed in parallel test


def main():
    print("=" * 70)
    print("EBGA FRAMEWORK - TEST SUITE")
    print("Evolutionary-Based Gradient-free Architecture")
    print("=" * 70)
    print("Layer-wise training is enabled by default")
    print("\nComparison: EBGA vs sklearn Ridge/RidgeClassifier\n")
    
    results = {}
    
    # Run standard dataset tests
    print("\n" + "=" * 70)
    print("STANDARD DATASET TESTS")
    print("=" * 70)
    
    # Iris
    iris_acc, iris_baseline = run_iris_test()
    results['Iris'] = {'accuracy': iris_acc, 'baseline': iris_baseline}
    
    # Breast Cancer
    cancer_acc, cancer_baseline = run_breast_cancer_test()
    results['Breast Cancer'] = {'accuracy': cancer_acc, 'baseline': cancer_baseline}
    
    # Wine
    wine_acc, wine_baseline = run_wine_test()
    results['Wine'] = {'accuracy': wine_acc, 'baseline': wine_baseline}
    
    # Diabetes (Regression)
    diabetes_r2, diabetes_baseline = run_diabetes_test()
    results['Diabetes'] = {'r2': diabetes_r2, 'baseline': diabetes_baseline}
    
    # California Housing (Regression)
    housing_r2, housing_baseline = run_california_housing_test()
    results['California Housing'] = {'r2': housing_r2, 'baseline': housing_baseline}
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print("\nClassification Datasets:")
    for dataset in ['Iris', 'Breast Cancer', 'Wine']:
        if dataset in results:
            acc = results[dataset]['accuracy']
            baseline = results[dataset]['baseline']
            print(f"  {dataset:20s}: EBGA Accuracy = {acc:.4f}, sklearn Baseline = {baseline:.4f}")
    
    print("\nRegression Datasets:")
    for dataset in ['Diabetes', 'California Housing']:
        if dataset in results:
            r2 = results[dataset]['r2']
            baseline = results[dataset]['baseline']
            if baseline is not None:
                print(f"  {dataset:20s}: EBGA R² = {r2:.4f}, sklearn Baseline = {baseline:.4f}")
            else:
                print(f"  {dataset:20s}: EBGA R² = {r2:.4f}")
    
    print("\n" + "=" * 70)
    print("All tests completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
