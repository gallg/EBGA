#!/usr/bin/env python3
"""
Main test script for Compact Genetic Descent models.
Tests both regressor and classifier on benchmark datasets.
"""

import numpy as np
from sklearn.datasets import load_diabetes, load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, r2_score, 
    accuracy_score, classification_report
)
from EBGA.GDRegressor import CompactGeneticDescentRegressor
from EBGA.GDClassifier import CompactGeneticDescentClassifier


def test_regressor():
    """Test CompactGeneticDescentRegressor on diabetes dataset."""
    print("=" * 60)
    print("Testing CompactGeneticDescentRegressor")
    print("=" * 60)
    
    # Load and prepare data
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Standardize data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Create and train model with improved parameters
    print("\nTraining regressor on diabetes dataset...")
    print("Using compact version (no L1/L2 regularization)")
    model = CompactGeneticDescentRegressor(
        n_bins=15,
        max_iter=500,
        lr_mu=0.01,
        lr_sigma=0.001,
        entropy_awareness=0.05,
        calibration_interval=20,
        credit_factor=2.0,
        early_stopping=True,
        patience=30,
        calibration_size=30,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nTest Results:")
    print(f"  MSE: {mse:.4f}")
    print(f"  R²: {r2:.4f}")
    
    # Show some sample predictions
    print(f"\nSample predictions (first 5 test samples):")
    for i in range(5):
        print(f"  True: {y_test[i]:.2f}, Predicted: {y_pred[i]:.2f}")
    
    return model, mse, r2


def test_classifier():
    """Test CompactGeneticDescentClassifier on iris and breast cancer datasets."""
    print("\n" + "=" * 60)
    print("Testing CompactGeneticDescentClassifier")
    print("=" * 60)
    
    # Test on Iris dataset (multi-class)
    print("\n--- Iris Dataset (Multi-class) ---")
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Using compact version (no L2 regularization)")
    model_iris = CompactGeneticDescentClassifier(
        max_iter=500,
        lr_mu=0.01,
        lr_sigma=0.001,
        entropy_awareness=0.05,
        calibration_interval=20,
        credit_factor=2.0,
        early_stopping=True,
        patience=30,
        calibration_size=30,
        random_state=42
    )
    model_iris.fit(X_train_scaled, y_train)
    
    y_pred_iris = model_iris.predict(X_test_scaled)
    acc_iris = accuracy_score(y_test, y_pred_iris)
    print(f"Accuracy: {acc_iris:.4f}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred_iris)}")
    
    # Test on Breast Cancer dataset (binary classification)
    print("\n--- Breast Cancer Dataset (Binary) ---")
    cancer = load_breast_cancer()
    X, y = cancer.data, cancer.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model_cancer = CompactGeneticDescentClassifier(
        max_iter=500,
        lr_mu=0.01,
        lr_sigma=0.001,
        entropy_awareness=0.05,
        calibration_interval=20,
        credit_factor=2.0,
        early_stopping=True,
        patience=30,
        calibration_size=30,
        random_state=42
    )
    model_cancer.fit(X_train_scaled, y_train)
    
    y_pred_cancer = model_cancer.predict(X_test_scaled)
    acc_cancer = accuracy_score(y_test, y_pred_cancer)
    print(f"Accuracy: {acc_cancer:.4f}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred_cancer)}")
    
    return model_iris, acc_iris, model_cancer, acc_cancer


def main():
    """Run all tests."""
    print("Compact Genetic Descent - Benchmark Tests")
    print("=" * 60)
    print("Note: These models use the SIMPLIFIED compact version")
    print("(L1/L2 regularization has been removed)")
    print("Relying on sigma (σ) and entropy_awareness for natural regularization")
    print("=" * 60)
    
    # Test regressor
    regressor_model, mse, r2 = test_regressor()
    
    # Test classifier
    classifier_iris, acc_iris, classifier_cancer, acc_cancer = test_classifier()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nRegressor (Diabetes):")
    print(f"  MSE: {mse:.4f}")
    print(f"  R²: {r2:.4f}")
    print(f"\nClassifier (Iris - Multi-class):")
    print(f"  Accuracy: {acc_iris:.4f}")
    print(f"\nClassifier (Breast Cancer - Binary):")
    print(f"  Accuracy: {acc_cancer:.4f}")
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("Models are simplified (no L1/L2 regularization, using entropy_awareness)")
    print("=" * 60)


if __name__ == "__main__":
    main()
