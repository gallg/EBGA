#!/usr/bin/env python3

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import KFold, RandomizedSearchCV
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
    
    print(f"\nDataset: California Housing")
    print(f"Features: {X.shape[1]}, Target range: [{y.min():.2f}, {y.max():.2f}]")
    
    # Nested CV for sklearn LinearRegression baseline
    print("\n" + "-" * 70)
    print("BASELINE: sklearn LinearRegression with Nested CV")
    print("-" * 70)
    
    n_splits = 5
    outer_cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scaler = StandardScaler()
    
    lr_r2_scores = []
    for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        lr_model = LinearRegression()
        lr_model.fit(X_train_scaled, y_train)
        lr_y_pred = lr_model.predict(X_test_scaled)
        lr_r2 = r2_score(y_test, lr_y_pred)
        lr_r2_scores.append(lr_r2)
        print(f"  Fold {outer_fold + 1}: R² = {lr_r2:.4f}")
    
    avg_lr_r2 = np.mean(lr_r2_scores)
    std_lr_r2 = np.std(lr_r2_scores)
    print(f"LinearRegression Nested CV: R² = {avg_lr_r2:.4f} ± {std_lr_r2:.4f}")
    
    # EBGA with RandomizedSearchCV
    print("\n" + "-" * 70)
    print("EBGA: Cross-Validation with RandomizedSearchCV")
    print("-" * 70)
    
    # Define parameter distributions for random search
    param_distributions = {
        'lr_mu': [0.001, 0.0025, 0.005, 0.01],
        'lr_sigma': [0.0001, 0.00025, 0.0005, 0.001],
        'momentum': [0.0, 0.1, 0.3, 0.5, 0.7],
        'max_iter': [1000, 2000, 3000],
        'calibration_size': [10, 20, 30],
        'calibration_interval': [25, 50, 100],
    }
    
    # Setup outer CV
    outer_cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Arrays to store results
    r2_scores = []
    best_params_list = []
    
    # Nested CV loop with RandomizedSearchCV
    for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X)):
        print(f"\nOuter fold {outer_fold + 1}/{n_splits}")
        
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Scale data
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Create base model
        model = EBGARegressor(
            layers=[(3, 'sigmoid'), (1, 'linear')],
            normalize_output=True,
            loss='mse',
            sigma_min=0.001,
            sigma_max=1.0,
            layer_patience=30,
            early_stopping=False,
            patience=100,
            random_state=random_state + outer_fold * 100
        )
        
        # Inner CV for hyperparameter tuning with RandomizedSearchCV
        inner_cv = KFold(n_splits=3, shuffle=True, random_state=random_state + outer_fold)
        
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions,
            n_iter=5,
            cv=inner_cv,
            scoring='r2',
            n_jobs=None,
            random_state=random_state + outer_fold * 1000,
            verbose=0
        )
        
        # Fit search
        search.fit(X_train_scaled, y_train)
        
        # Get best model and parameters
        best_params = search.best_params_
        best_params_list.append(best_params)
        
        # Predict on test fold
        y_pred = search.predict(X_test_scaled)
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        r2_scores.append(r2)
        
        print(f"  Best params: {best_params}")
        print(f"  R2: {r2:.4f}")
    
    # Calculate final nested CV results
    avg_r2 = np.mean(r2_scores)
    std_r2 = np.std(r2_scores)
    
    print(f"\n{'='*60}")
    print("FINAL RESULTS - California Housing")
    print(f"{'='*60}")
    print(f"sklearn LinearRegression: R² = {avg_lr_r2:.4f} ± {std_lr_r2:.4f}")
    print(f"EBGA Regressor:         R² = {avg_r2:.4f} ± {std_r2:.4f}")
    print(f"Best params: {best_params_list[np.argmax(r2_scores)]}")
    
    return avg_r2, avg_lr_r2


if __name__ == "__main__":
    run_test()
