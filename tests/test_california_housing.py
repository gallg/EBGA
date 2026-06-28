#!/usr/bin/env python3

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

from EBGA.models import EBGARegressor


def run_test(random_state=42):
    # Set global random seed for reproducibility
    np.random.seed(random_state)
    
    print("=" * 70)
    print("TEST: California Housing Dataset (Regression)")
    print("=" * 70)
    
    # Load data
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    
    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"\nDataset: California Housing")
    print(f"Features: {X_train.shape[1]}, Target range: [{y.min():.2f}, {y.max():.2f}]")
    print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
    
    # sklearn baseline
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_y_pred = lr_model.predict(X_test)
    lr_r2 = r2_score(y_test, lr_y_pred)
    print(f"(sklearn LinearRegression baseline: R² = {lr_r2:.4f})")
    
    # Create model with feature extraction layers
    n_features = X_train.shape[1]
    print("\nModel configuration:")
    print(f"  layers=[({n_features}, 'relu'), (1, 'linear')]")
    print("  normalize_output=True")
    print("  loss='mse'")
    print("  lr_mu=0.00055, lr_sigma=0.00008")
    print("  Layer-wise training: enabled (evolutionary)")
    
    model = EBGARegressor(
        layers=[(n_features, 'relu'), (1, 'linear')],
        normalize_output=True,
        loss='mse',
        lr_mu=0.00055,
        lr_sigma=0.00008,
        calibration_size=30,
        calibration_interval=50,
        layer_patience=30,
        max_iter=50000,
        early_stopping=False,
        patience=100,
        random_state=random_state,
        use_layerwise=True
    )
    
    print("\nTraining...")
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nResults:")
    print(f"  R² Score: {r2:.4f}")
    print(f"  vs sklearn baseline: {r2:.4f} / {lr_r2:.4f} ({r2/lr_r2*100:.1f}%)")
    
    return model, r2, scaler


if __name__ == "__main__":
    run_test()
