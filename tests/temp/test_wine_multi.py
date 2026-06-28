#!/usr/bin/env python3

import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from EBGA.nn import Sequential
from EBGA.layers import Linear
from EBGA.losses import cross_entropy_loss
from EBGA.optimizer import MultiCandidateOptimizer


def run_test(random_state=42):
    # Set global random seed for reproducibility
    np.random.seed(random_state)
    
    print("=" * 70)
    print("TEST: Wine Dataset (Classification) with MultiCandidateOptimizer")
    print("=" * 70)
    
    # Load data
    wine = load_wine()
    X, y = wine.data, wine.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    
    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"\nDataset: Wine")
    print(f"Features: {X_train.shape[1]}, Classes: {len(np.unique(y))}")
    print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
    
    # Convert labels to one-hot for cross-entropy
    n_classes = len(np.unique(y))
    y_train_onehot = np.zeros((len(y_train), n_classes))
    y_train_onehot[np.arange(len(y_train)), y_train] = 1
    
    # Build network
    network = Sequential(
        Linear(10, activation='relu'),
        Linear(10, activation='relu'),
        Linear(n_classes, activation='softmax')
    )
    network.initialize(X_train.shape[1])
    
    # Create MultiCandidateOptimizer with same hyperparameters as EBGAClassifier
    optimizer = MultiCandidateOptimizer(
        param_dim=network.parameter_count(),
        n_candidates=3,
        lr_mu=0.05,
        lr_sigma=0.005,
        sigma_min=0.001,
        sigma_max=1.0,
        calibration_size=20,
        calibration_interval=25,
        credit_factor=2.0,
        sigma_regularization=0.0,
        random_state=random_state
    )
    
    print("\nModel configuration:")
    print("  layers=[(10, 'relu'), (10, 'relu'), (3, 'softmax')]")
    print("  n_candidates=3")
    print("  Optimizer: MultiCandidateOptimizer")
    
    # Define loss function
    def loss_func(params):
        network.set_all_parameters(params)
        y_pred = network.forward(X_train)
        y_pred = np.exp(y_pred - np.max(y_pred, axis=1, keepdims=True))
        y_pred = y_pred / np.sum(y_pred, axis=1, keepdims=True)
        return cross_entropy_loss(y_pred, y_train_onehot)
    
    # Initialize optimizer
    optimizer.initialize()
    
    print("\nTraining...")
    n_iterations = 2000
    for iteration in range(n_iterations):
        loss = optimizer.step(loss_func, iteration=iteration)
        if (iteration + 1) % 500 == 0:
            print(f"  Iteration {iteration + 1}: loss = {loss:.6f}")
    
    # Get final parameters
    final_params = optimizer.get_parameters()
    network.set_all_parameters(final_params)
    
    # Evaluate
    y_pred_logits = network.forward(X_test)
    y_pred = np.argmax(y_pred_logits, axis=1)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\nResults:")
    print(f"  Accuracy: {acc:.4f}")
    
    return network, acc, scaler


if __name__ == "__main__":
    run_test()
