#!/usr/bin/env python3

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression

from EBGA.models import EBGARegressor
from EBGA.search import EvoHyperoptSearch


def run_test(random_state=42):
    # Set global random seed for reproducibility
    np.random.seed(random_state)
    
    print("=" * 70)
    print("TEST: Diabetes Dataset (Regression)")
    print("=" * 70)
    
    # Load data
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target
    
    print(f"\nDataset: Diabetes")
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
    
    avg_lr_r2 = np.mean(lr_r2_scores)
    std_lr_r2 = np.std(lr_r2_scores)
    print(f"LinearRegression Nested CV: R² = {avg_lr_r2:.4f} ± {std_lr_r2:.4f}")
    
    # EBGA with Nested CV and hyperparameter tuning
    print("\n" + "-" * 70)
    print("EBGA: Nested Cross-Validation with Hyperparameter Tuning")
    print("-" * 70)
    
    # Define search space
    param_distributions = {
        'lr_mu': (0.0001, 0.01, 'log-uniform'),
        'lr_sigma': (0.00001, 0.001, 'log-uniform'),
        'max_iter': [1000, 2000, 5000],
        'use_layerwise': [True, False]
    }
    
    # Setup outer CV
    outer_cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Arrays to store results
    r2_scores = []
    best_params_list = []
    
    # Nested CV loop
    for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X)):
        print(f"\nOuter fold {outer_fold + 1}/{n_splits}")
        
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Scale data
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Inner CV for hyperparameter tuning
        inner_cv = KFold(n_splits=3, shuffle=True, random_state=random_state + outer_fold)
        
        # Create search
        search = EvoHyperoptSearch(
            estimator=EBGARegressor(
                layers=[(8, 'relu'), (1, 'linear')],
                normalize_output=True,
                loss='mse',
                calibration_size=30,
                calibration_interval=50,
                layer_patience=30,
                early_stopping=True,
                patience=50,
                random_state=random_state + outer_fold * 100
            ),
            param_distributions=param_distributions,
            n_iter=5,
            cv=inner_cv,
            search_strategy='random',
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
    print("FINAL RESULTS - Diabetes")
    print(f"{'='*60}")
    print(f"sklearn LinearRegression: R² = {avg_lr_r2:.4f} ± {std_lr_r2:.4f}")
    print(f"EBGA Regressor:         R² = {avg_r2:.4f} ± {std_r2:.4f}")
    print(f"Best params: {best_params_list[np.argmax(r2_scores)]}")
    
    return avg_r2, avg_lr_r2


if __name__ == "__main__":
    run_test()
