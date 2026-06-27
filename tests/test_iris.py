#!/usr/bin/env python3

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import RidgeClassifier

from EBGA.models import EBGAClassifier


def run_test(random_state=42):
    # Set global random seed for reproducibility
    np.random.seed(random_state)
    
    print("=" * 70)
    print("TEST: Iris Dataset (Classification)")
    print("=" * 70)
    
    # Load data
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    
    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"\nDataset: {iris.DESCR.split('..')[0].strip()}")
    print(f"Features: {X_train.shape[1]}, Classes: {len(np.unique(y))}")
    print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
    
    # RidgeClassifier baseline
    ridge_clf = RidgeClassifier(alpha=1.0, random_state=random_state)
    ridge_clf.fit(X_train, y_train)
    ridge_acc = accuracy_score(y_test, ridge_clf.predict(X_test))
    print(f"(sklearn RidgeClassifier baseline: Accuracy = {ridge_acc:.4f})")
    
    # Create model with explicit layers
    print("\nModel configuration:")
    print("  layers=[(10, 'relu'), (10, 'relu'), (3, 'softmax')]")
    print("  n_classes=3")
    print("  Layer-wise training: enabled")
    
    model = EBGAClassifier(
        layers=[(10, 'relu'), (10, 'relu'), (3, 'softmax')],
        n_classes=3,
        max_iter=2000,
        lr_mu=0.01,
        lr_sigma=0.01,
        layer_patience=30,
        random_state=random_state
    )
    
    print("\nTraining...")
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\nResults:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  vs RidgeClassifier: {acc:.4f} / {ridge_acc:.4f} ({acc/ridge_acc*100:.1f}%)")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
    
    return model, acc, scaler


if __name__ == "__main__":
    run_test()
