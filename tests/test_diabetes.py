#!/usr/bin/env python3

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from EBGA.models import EBGARegressor


def run_test(random_state=42):
    # Set global random seed for reproducibility
    np.random.seed(random_state)
    
    print("=" * 70)
    print("TEST: Diabetes Dataset (Regression)")
    print("=" * 70)
    
    # Load data
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    
    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"\nDataset: Diabetes")
    print(f"Features: {X_train.shape[1]}, Target range: [{y.min():.2f}, {y.max():.2f}]")
    print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
    print(f"(sklearn LinearRegression baseline: R² ≈ 0.4526)")
    
    # Create model with explicit layers
    print("\nModel configuration:")
    print("  layers=[(1, 'linear')]")
    print("  normalize_output=True")
    print("  loss='mse'")
    print("  lr_mu=0.00055, lr_sigma=0.00008")
    print("  Layer-wise training: disabled")
    print("  Adaptive loss scale: enabled in optimizer")
    
    model = EBGARegressor(
        layers=[(1, 'linear')],
        normalize_output=True,
        loss='mse',
        lr_mu=0.00055,
        lr_sigma=0.00008,
        calibration_size=30,
        calibration_interval=50,
        layer_patience=0,
        max_iter=50000,
        early_stopping=False,
        patience=100,
        random_state=random_state
    )
    
    print("\nTraining...")
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nResults:")
    print(f"  R² Score: {r2:.4f}")
    
    return model, r2, scaler


if __name__ == "__main__":
    run_test()
