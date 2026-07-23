#!/usr/bin/env python3
"""
Test ParallelEvaluator on California Housing with the same architecture
and parameters as the existing test, but using the custom nn path.

Supports optional layerwise training.
"""

import time
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from EBGA.nn import Sequential
from EBGA.layers import Dense
from EBGA.optimizer import CompactEvoOptimizer
from EBGA.parallel import ParallelEvaluator


def run_test(random_state=42, use_layerwise=False):
    np.random.seed(random_state)

    print("=" * 70)
    print("TEST: California Housing with ParallelEvaluator (6 jobs)")
    print(f"      Layerwise: {use_layerwise}")
    print("=" * 70)

    # Load data
    housing = fetch_california_housing()
    X, y = housing.data.astype(np.float64), housing.target.astype(np.float64)

    print(f"\nDataset: California Housing")
    print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Normalize output (same as EBGARegressor with normalize_output=True)
    y_min, y_max = np.min(y), np.max(y)
    y_normalized = (y - y_min) / (y_max - y_min + 1e-8)

    # Build network: same architecture as the test: [(3, 'sigmoid'), (1, 'linear')]
    net = Sequential(
        Dense(3, activation='sigmoid'),
        Dense(1, activation='linear'),
    )
    net.initialize(X_scaled.shape[1], scale_aware=y_normalized)

    print(f"Network parameters: {net.parameter_count()}")

    # Use the best params from the existing test as a starting point
    # (lr_mu=0.005, lr_sigma=0.0005, momentum=0.5, calibration_size=20)
    opt_config = dict(
        calibration_size=20,
        lr_mu=0.005,
        lr_sigma=0.0005,
        sigma_min=0.001,
        sigma_max=1.0,
        momentum=0.5,
        trust_region_radius=None,
        random_state=random_state,
    )

    overall_start = time.time()

    # Phase 1: Optional layer-wise pretraining
    if use_layerwise:
        layer_iters = 500
        print(f"\nLayer-wise pretraining ({layer_iters} iters per layer)...")
        net.layerwise_pretrain(
            X_scaled, y_normalized, loss='mse',
            layer_iters=layer_iters,
            optimizer_config=opt_config,
            n_jobs=6,
            random_state=random_state,
        )

    # Phase 2: Fine-tune / direct train with parallel evaluator
    opt = CompactEvoOptimizer(
        param_dim=net.parameter_count(),
        **opt_config,
    )
    opt.initialize(net.get_all_parameters())

    # Parallel evaluator with 6 jobs, no batching (full shard per worker)
    evaluator = ParallelEvaluator(
        net, X_scaled, y_normalized,
        loss='mse',
        n_jobs=6,
        batch_size=None,
        random_state=random_state,
    )

    # Train
    max_iter = 5000
    phase_label = "Fine-tuning" if use_layerwise else "Direct training"
    print(f"\n{phase_label} for {max_iter} iterations with batch_size=None, n_jobs=6...")
    print(f"{'Iter':>6}  {'Loss':>12}  {'Elapsed':>10}")
    print("-" * 32)

    start = time.time()
    losses = []
    with evaluator:
        for i in range(max_iter):
            loss = opt.step(iteration=i, evaluate_map=evaluator.evaluate_map)
            losses.append(loss)
            if (i + 1) % 500 == 0:
                elapsed = time.time() - start
                print(f"{i+1:>6}  {loss:>12.6f}  {elapsed:>8.1f}s")

    total_time = time.time() - overall_start
    print("-" * 32)
    print(f"{max_iter:>6}  {losses[-1]:>12.6f}  {total_time:>8.1f}s")

    # Set final parameters
    net.set_all_parameters(opt.get_parameters())

    # Predict and evaluate
    y_pred = net.forward(X_scaled)
    if net.output_size == 1:
        y_pred = y_pred.flatten()

    # Denormalize
    y_pred_denorm = y_pred * (y_max - y_min) + y_min

    r2 = r2_score(y, y_pred_denorm)
    print(f"\nR² score on full dataset: {r2:.4f}")
    print(f"Total training time: {total_time:.1f}s")
    print(f"Average time per iteration: {total_time / max_iter * 1000:.2f}ms")

    return r2


if __name__ == "__main__":
    run_test()
