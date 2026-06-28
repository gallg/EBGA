#!/usr/bin/env python3

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from EBGA.nn import Sequential
from EBGA.layers import Linear
from EBGA.losses import mse_loss
from EBGA.optimizer import MultiCandidateOptimizer


def run_test(random_state=42):
    # Set global random seed for reproducibility
    np.random.seed(random_state)
    
    print("=" * 70)
    print("TEST: Diabetes Dataset (Regression) with MultiCandidateOptimizer")
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
    
    n_features = X_train.shape[1]
    print(f"\nDataset: Diabetes")
    print(f"Features: {n_features}, Target range: [{y.min():.2f}, {y.max():.2f}]")
    print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
    print(f"(sklearn LinearRegression baseline: R² ≈ 0.4526)")
    
    # Build network - same as in test_diabetes.py
    network = Sequential(
        Linear(n_features, activation='relu'),
        Linear(1, activation='linear')
    )
    network.initialize(X_train.shape[1])
    
    # Create MultiCandidateOptimizer with same hyperparameters as test_diabetes.py
    # lr_mu=0.00055, lr_sigma=0.00008, calibration_size=30, calibration_interval=50
    # loss='mse', normalize_output=True
    optimizer = MultiCandidateOptimizer(
        param_dim=network.parameter_count(),
        n_candidates=3,
        lr_mu=0.00055,
        lr_sigma=0.00008,
        sigma_min=0.001,
        sigma_max=1.0,
        calibration_size=30,
        calibration_interval=50,
        credit_factor=2.0,
        sigma_regularization=0.0,
        random_state=random_state
    )
    
    print("\nModel configuration:")
    print(f"  layers=[({n_features}, 'relu'), (1, 'linear')]")
    print("  n_candidates=3")
    print("  normalize_output=True (applied in loss calculation)")
    print("  loss='mse'")
    print(f"  lr_mu={0.00055}, lr_sigma={0.00008}")
    print("  max_iter=50000")
    
    # Normalize output
    y_min, y_max = y_train.min(), y_train.max()
    y_normalized = (y_train - y_min) / (y_max - y_min + 1e-8)
    
    # Define loss function with MSE
    def loss_func(params):
        network.set_all_parameters(params)
        y_pred = network.forward(X_train).flatten()
        return mse_loss(y_pred, y_normalized)
    
    # Initialize optimizer
    optimizer.initialize()
    
    print("\nTraining...")
    n_iterations = 50000
    for iteration in range(n_iterations):
        loss = optimizer.step(loss_func, iteration=iteration)
        if (iteration + 1) % 10000 == 0:
            print(f"  Iteration {iteration + 1}: loss = {loss:.6f}")
    
    # Get final parameters
    final_params = optimizer.get_parameters()
    network.set_all_parameters(final_params)
    
    # Evaluate (with denormalization)
    y_pred_normalized = network.forward(X_test).flatten()
    y_pred = y_pred_normalized * (y_max - y_min) + y_min
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nResults:")
    print(f"  R² Score: {r2:.4f}")
    
    return network, r2, scaler


if __name__ == "__main__":
    run_test()
