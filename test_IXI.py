#!/usr/bin/env python3
"""
Test script for training GDRegressor and GDClassifier on IXI dataset.
Uses brain phenotype features to predict age (regression) and sex (classification).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, r2_score, 
    accuracy_score, classification_report
)
from EBGA.GDRegressor import CompactGeneticDescentRegressor
from EBGA.GDClassifier import CompactGeneticDescentClassifier


def load_and_prepare_data():
    """Load and prepare IXI dataset for training."""
    print("Loading IXI dataset...")
    
    # Load brain phenotype data
    phenotypes = pd.read_csv('data/ixi.csv')
    print(f"  Brain phenotypes: {phenotypes.shape[0]} samples, {phenotypes.shape[1]} features")
    
    # Load demographic data
    demo = pd.read_csv('data/final_demo.csv')
    print(f"  Demographic data: {demo.shape[0]} samples")
    
    # Merge datasets on ID
    # ixi.csv uses 'ID' column, final_demo.csv uses 'IXI_ID' column
    merged = pd.merge(
        phenotypes, 
        demo[['IXI_ID', 'SEX_ID (1=m, 2=f)']], 
        left_on='ID', 
        right_on='IXI_ID'
    )
    
    print(f"  Merged dataset: {merged.shape[0]} samples")
    
    # Extract features (all lh_* and rh_* brain volume columns)
    feature_cols = [col for col in merged.columns 
                  if col.startswith('lh_') or col.startswith('rh_')]
    X = merged[feature_cols].values
    
    # Extract targets
    y_age = merged['Age'].values  # For regression
    y_sex = merged['SEX_ID (1=m, 2=f)'].values  # For classification (1=m, 2=f)
    
    print(f"  Features: {len(feature_cols)} brain volume features")
    print(f"  Target (Age): range [{y_age.min():.2f}, {y_age.max():.2f}]")
    print(f"  Target (Sex): {len(np.unique(y_sex))} classes (1=m, 2=f)")
    
    return X, y_age, y_sex, feature_cols


def train_age_regressor(X, y_age):
    """Train GDRegressor to predict age from brain phenotypes."""
    print("\n" + "=" * 60)
    print("Training GDRegressor for Age Prediction")
    print("=" * 60)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_age, test_size=0.2, random_state=42
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train model
    print("\nTraining regressor on brain phenotypes...")
    model = CompactGeneticDescentRegressor(
        n_bins=8,
        max_iter=5000,
        lr_mu=0.001,
        lr_sigma=0.001,
        entropy_awareness=5.0,
        calibration_interval=21,
        credit_factor=20.0,
        early_stopping=False,
        patience=40,
        calibration_size=18,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nTest Results:")
    print(f"  MSE: {mse:.4f}")
    print(f"  R²: {r2:.4f}")
    
    # Show some sample predictions
    print(f"\nSample predictions (first 5 test samples):")
    for i in range(5):
        print(f"  True: {y_test[i]:.2f}, Predicted: {y_pred[i]:.2f}")
    
    return model, mse, r2, scaler


def train_sex_classifier(X, y_sex):
    """Train GDClassifier to predict sex from brain phenotypes."""
    print("\n" + "=" * 60)
    print("Training GDClassifier for Sex Prediction")
    print("=" * 60)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_sex, test_size=0.2, random_state=42
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train model
    print("\nTraining classifier on brain phenotypes...")
    model = CompactGeneticDescentClassifier(
        max_iter=3000,
        lr_mu=0.01,
        lr_sigma=0.01,
        entropy_awareness=1.0,
        calibration_interval=21,
        credit_factor=1.0,
        early_stopping=False,
        patience=30,
        calibration_size=30,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\nTest Results:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Male (1)', 'Female (2)']))
    
    return model, acc, scaler


def main():
    """Run all tests on IXI dataset."""
    print("IXI Dataset - GD Model Training")
    print("=" * 60)
    print("Predicting Age (regression) and Sex (classification)")
    print("from LH and RH brain phenotype features")
    print("=" * 60)
    
    # Load and prepare data
    X, y_age, y_sex, feature_cols = load_and_prepare_data()
    
    # Train age regressor
    regressor_model, mse, r2, regressor_scaler = train_age_regressor(X, y_age)
    
    # Train sex classifier
    classifier_model, acc, classifier_scaler = train_sex_classifier(X, y_sex)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nRegressor (Age Prediction):")
    print(f"  Features: {len(feature_cols)} brain volume features (LH + RH)")
    print(f"  MSE: {mse:.4f}")
    print(f"  R²: {r2:.4f}")
    print(f"\nClassifier (Sex Prediction):")
    print(f"  Features: {len(feature_cols)} brain volume features (LH + RH)")
    print(f"  Accuracy: {acc:.4f}")
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)
    
    return {
        'regressor': regressor_model,
        'regressor_scaler': regressor_scaler,
        'regressor_mse': mse,
        'regressor_r2': r2,
        'classifier': classifier_model,
        'classifier_scaler': classifier_scaler,
        'classifier_accuracy': acc,
        'features': feature_cols
    }


if __name__ == "__main__":
    results = main()
