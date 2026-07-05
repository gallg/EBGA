#!/usr/bin/env python3

import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import RidgeClassifier

from EBGA.models import EBGAClassifier


def run_test(random_state=42):
    # Set global random seed for reproducibility
    np.random.seed(random_state)
    
    print("=" * 70)
    print("TEST: Wine Dataset (Classification)")
    print("=" * 70)
    
    # Load data
    wine = load_wine()
    X, y = wine.data, wine.target
    n_classes = len(np.unique(y))
    
    print(f"\nDataset: Wine")
    print(f"Features: {X.shape[1]}, Classes: {n_classes}")
    
    # Nested CV for sklearn RidgeClassifier baseline
    print("\n" + "-" * 70)
    print("BASELINE: sklearn RidgeClassifier with Nested CV")
    print("-" * 70)
    
    n_splits = 5
    outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scaler = StandardScaler()
    
    rc_acc_scores = []
    for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        rc_model = RidgeClassifier(random_state=random_state + outer_fold)
        rc_model.fit(X_train_scaled, y_train)
        rc_y_pred = rc_model.predict(X_test_scaled)
        rc_acc = accuracy_score(y_test, rc_y_pred)
        rc_acc_scores.append(rc_acc)
        print(f"  Fold {outer_fold + 1}: Accuracy = {rc_acc:.4f}")
    
    avg_rc_acc = np.mean(rc_acc_scores)
    std_rc_acc = np.std(rc_acc_scores)
    print(f"RidgeClassifier Nested CV: Accuracy = {avg_rc_acc:.4f} ± {std_rc_acc:.4f}")
    
    # EBGA with RandomizedSearchCV
    print("\n" + "-" * 70)
    print("EBGA: Cross-Validation with RandomizedSearchCV")
    print("-" * 70)
    
    # Define parameter distributions for random search
    param_distributions = {
        'lr_mu': [0.001, 0.0025, 0.005, 0.01],
        'lr_sigma': [0.0001, 0.00025, 0.0005, 0.001],
        'momentum': [0.0, 0.1, 0.3, 0.5, 0.7],
        'max_iter': [200, 500, 1000],
        'calibration_size': [10, 20, 30],
        'calibration_interval': [25, 50, 100],
    }
    
    # Setup outer CV
    outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Arrays to store results
    acc_scores = []
    best_params_list = []
    
    # Nested CV loop with RandomizedSearchCV
    for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        print(f"\nOuter fold {outer_fold + 1}/{n_splits}")
        
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Scale data
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Create base model
        model = EBGAClassifier(
            layers=[(4, 'relu'), (n_classes, 'softmax')],
            n_classes=n_classes,
            sigma_min=0.001,
            sigma_max=1.0,
            credit_factor=2.0,
            early_stopping=False,
            patience=100,
            layer_patience=50,
            random_state=random_state + outer_fold * 100
        )
        
        # Inner CV for hyperparameter tuning with RandomizedSearchCV
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state + outer_fold)
        
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions,
            n_iter=5,
            cv=inner_cv,
            scoring='accuracy',
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
        acc = accuracy_score(y_test, y_pred)
        acc_scores.append(acc)
        
        print(f"  Best params: {best_params}")
        print(f"  Accuracy: {acc:.4f}")
    
    # Calculate final nested CV results
    avg_acc = np.mean(acc_scores)
    std_acc = np.std(acc_scores)
    
    print(f"\n{'='*60}")
    print("FINAL RESULTS - Wine")
    print(f"{'='*60}")
    print(f"sklearn RidgeClassifier: Accuracy = {avg_rc_acc:.4f} ± {std_rc_acc:.4f}")
    print(f"EBGA Classifier:         Accuracy = {avg_acc:.4f} ± {std_acc:.4f}")
    print(f"Best params: {best_params_list[np.argmax(acc_scores)]}")
    
    return avg_acc, avg_rc_acc


if __name__ == "__main__":
    run_test()
