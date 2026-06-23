"""
Simple test of Priority 2 with just 2 patches, 1 class
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import MinMaxScaler
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from EBGA.optimizer import MultiEvoOptimizer

np.random.seed(42)

print("="*70)
print("Simple Priority 2 Test: 2 patches, 1 class")
print("="*70)

# Load MNIST
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='pandas')
X_all = mnist.data.astype('float32')
y_all = mnist.target.astype('int')
X_all = MinMaxScaler().fit_transform(X_all)

# Class 0 only
mask = y_all == 0
X_class0 = X_all[mask][:500]
y_class0 = y_all[mask][:500]

print(f"Dataset: {len(X_class0)} samples, class 0")

N_PATCHES = 2
PATCH_SIZE = 784 // N_PATCHES

layer_configs = {0: [(PATCH_SIZE, 'sigmoid') for _ in range(N_PATCHES)]}

print(f"\nPatches: {N_PATCHES}, Patch size: {PATCH_SIZE}")

multi_opt = MultiEvoOptimizer(
    n_classes=1,
    layer_configs=layer_configs,
    lr_mu=0.01, lr_sigma=0.005,
    sigma_min=0.01, sigma_max=0.3,
    sigma_regularization=0.001,
    calibration_size=50, calibration_interval=10,
    credit_factor=2.0,
    random_state=42
)

# Initialize
initial_params = {0: []}
for p in range(N_PATCHES):
    start = p * PATCH_SIZE
    end = (p + 1) * PATCH_SIZE
    patch_mean = np.mean(X_class0[:, start:end], axis=0)
    initial_params[0].append(patch_mean)

multi_opt.initialize(initial_params_dict=initial_params)

# Check optimizer sizes
print("\nOptimizer sizes:")
for p in range(N_PATCHES):
    opt = multi_opt.get_optimizer(0, p)
    params = opt.get_parameters()
    print(f"  Patch {p}: {len(params)} params")

# Train
print("\nTraining...")
for c in range(1):
    for p in range(N_PATCHES):
        print(f"  Training class {c}, patch {p}...")
        
        def patch_loss(full_params):
            # full_params is the complete digit
            reconstructed = np.clip(full_params, 0.0, 1.0)
            distances = np.mean((X_class0 - reconstructed)**2, axis=1)
            return np.min(distances)
        
        for iteration in range(50):
            loss = multi_opt.step_layer(c, p, patch_loss, iteration=iteration)
            
            if iteration % 25 == 0:
                mu, sigma = multi_opt.get_optimizer(c, p).get_distribution_parameters()
                print(f"    Iter {iteration}: σ_mean = {np.mean(sigma):.4f}")

# Generate samples
print("\nGenerating 10 samples...")
samples = []
for _ in range(10):
    all_patch_params = []
    for p in range(N_PATCHES):
        opt = multi_opt.get_optimizer(0, p)
        mu, sigma = opt.get_distribution_parameters()
        sampled_params = mu + sigma * np.random.RandomState().randn(len(mu))
        all_patch_params.append(sampled_params)
    
    digit = np.concatenate(all_patch_params)
    digit = np.clip(digit, 0.0, 1.0)
    samples.append(digit)

samples = np.array(samples)

# Calculate diversity
pixel_std = np.mean(np.std(samples, axis=0))
print(f"Diversity: {pixel_std:.4f}")

# Similarity
nearest_distances = []
for sample in samples:
    distances = np.mean((sample[None] - X_class0)**2, axis=1)
    nearest_distances.append(np.min(distances))
mean_nearest_dist = np.mean(nearest_distances)

# Real-real distances
test_real_distances = []
for i in range(50):
    for j in range(50):
        if i != j:
            dist = np.mean((X_class0[i] - X_class0[j])**2)
            test_real_distances.append(dist)
mean_real_dist = np.mean(test_real_distances)
similarity_ratio = mean_nearest_dist / mean_real_dist

print(f"Similarity ratio: {similarity_ratio:.2f}")
print(f"Mean nearest distance: {mean_nearest_dist:.6f}")
print(f"Mean real distance: {mean_real_dist:.6f}")

# Save figure
os.makedirs('figures', exist_ok=True)
plt.figure(figsize=(15, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(samples[i].reshape(28, 28), cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
plt.suptitle(f'2 Patches, 1 Class\nDiversity={pixel_std:.3f}, Similarity={similarity_ratio:.2f}', fontsize=14)
plt.tight_layout()
plt.savefig('figures/priority2_simple.png', dpi=100, bbox_inches='tight')
plt.close()

print(f"\nSaved: figures/priority2_simple.png")
print("Done!")
