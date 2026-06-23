"""
Test Priority 1 Architecture v2: Direct Pixel Distribution with Bounds

Fixes from v1:
1. Add bounds to keep parameters in [0, 1] range
2. Better initialization
3. More appropriate loss function
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

from EBGA.optimizer import CompactEvoOptimizer

np.random.seed(42)

print("="*70)
print("Priority 1 v2: Direct Pixel Distribution with Bounds")
print("="*70)

# Load MNIST
print("\nLoading MNIST...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='pandas')
X_all = mnist.data.astype('float32')
y_all = mnist.target.astype('int')
X_all = MinMaxScaler().fit_transform(X_all)

# Use first 3 classes for testing
N_CLASSES = 3
SAMPLES_PER_CLASS = 500

mask = y_all < N_CLASSES
X_all = X_all[mask][:SAMPLES_PER_CLASS * N_CLASSES]
y_all = y_all[mask][:SAMPLES_PER_CLASS * N_CLASSES]

print(f"Dataset: {len(X_all)} samples, {N_CLASSES} classes ({SAMPLES_PER_CLASS} per class)")

# Separate by class
class_data = {c: X_all[y_all == c] for c in range(N_CLASSES)}

# Train one optimizer per class
class_optimizers = {}

print("\nTraining class distributions...")

# Configuration
SIGMA_REG = 0.001
SIGMA_MIN = 0.01
SIGMA_MAX = 0.3  # Lower to prevent too much noise
N_ITERATIONS = 500
PATIENCE = 50

for c in range(N_CLASSES):
    print(f"\n  Class {c}:")
    
    # Create optimizer for this class with BOUNDS
    optimizer = CompactEvoOptimizer(
        param_dim=784,
        lr_mu=0.01, lr_sigma=0.005,
        sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX,
        sigma_regularization=SIGMA_REG,
        calibration_size=50, calibration_interval=10,
        credit_factor=2.0,
        bounds=(0.0, 1.0),  # CRITICAL: Keep parameters in [0, 1]
        random_state=42
    )
    
    # Initialize with mean digit of this class
    mean_digit = np.mean(class_data[c], axis=0)
    # Ensure mean is in [0, 1]
    mean_digit = np.clip(mean_digit, 0.0, 1.0)
    optimizer.initialize(mean_digit)
    
    # Loss function: for parameter sample θ (a candidate digit),
    # find distance to nearest real digit in this class
    def loss_func(params):
        # params is a candidate digit (784 pixels)
        candidate = params.reshape(1, -1)
        # Clip candidate to valid range
        candidate = np.clip(candidate, 0.0, 1.0)
        
        # Distance to nearest real digit in class
        distances = np.mean((candidate - class_data[c])**2, axis=1)
        return np.min(distances)
    
    # Train
    best_loss = float('inf')
    patience_counter = 0
    
    for iteration in range(N_ITERATIONS):
        loss = optimizer.step(loss_func, iteration=iteration)
        
        if iteration % 50 == 0:
            current_loss = loss_func(optimizer.mu)
            mu, sigma = optimizer.get_distribution_parameters()
            print(f"    Iter {iteration}: loss = {current_loss:.6f}, σ_mean = {np.mean(sigma):.4f}")
            
            if current_loss < best_loss:
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"    Converged at iteration {iteration}")
                    break
    
    class_optimizers[c] = optimizer
    
    # Final results
    mu_final, sigma_final = optimizer.get_distribution_parameters()
    print(f"    Final: σ_mean = {np.mean(sigma_final):.4f}")
    
    # Check if mean is in valid range
    mu_clip_check = np.all((mu_final >= 0.0) & (mu_final <= 1.0))
    print(f"    Mean in [0,1]: {mu_clip_check}")

# ============================================================================
# Generation and Evaluation
# ============================================================================
print("\n" + "="*70)
print("Generation and Evaluation")
print("="*70)

os.makedirs('figures', exist_ok=True)

# Test 1: Generate samples from each class
print("\nTest 1: Generate 10 samples per class")
all_samples = []
all_labels = []

for c in range(N_CLASSES):
    opt = class_optimizers[c]
    mu, sigma = opt.get_distribution_parameters()
    
    rng = np.random.RandomState(42 + c)
    for _ in range(10):
        sampled_digit = mu + sigma * rng.randn(784)
        sampled_digit = np.clip(sampled_digit, 0.0, 1.0)  # Ensure valid
        all_samples.append(sampled_digit)
        all_labels.append(c)

all_samples = np.array(all_samples)
all_labels = np.array(all_labels)

# Calculate diversity per class
print("\nDiversity per class:")
for c in range(N_CLASSES):
    c_samples = all_samples[all_labels == c]
    pixel_std = np.std(c_samples, axis=0)
    mean_pixel_std = np.mean(pixel_std)
    print(f"  Class {c}: {mean_pixel_std:.4f}")

# Overall diversity
pixel_std_all = np.std(all_samples, axis=0)
mean_pixel_std_all = np.mean(pixel_std_all)
print(f"  Overall: {mean_pixel_std_all:.4f}")

# Visualize: Generated samples
plt.figure(figsize=(15, 10))
for i, (sample, label) in enumerate(zip(all_samples, all_labels)):
    row = i // 10
    col = i % 10
    plt.subplot(N_CLASSES, 10, i + 1)
    plt.imshow(sample.reshape(28, 28), cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
plt.suptitle(f'Direct Pixel Distribution with Bounds\nSIGMA_REG={SIGMA_REG}, σ_max={SIGMA_MAX}', fontsize=14)
plt.tight_layout()
plt.savefig('figures/priority1_v2_generated.png', dpi=100, bbox_inches='tight')
plt.close()

# Visualize: Mean digits
plt.figure(figsize=(15, 4))
for c in range(N_CLASSES):
    mu = class_optimizers[c].mu
    plt.subplot(1, N_CLASSES, c + 1)
    plt.imshow(mu.reshape(28, 28), cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    plt.title(f'Class {c}', fontsize=12)
plt.suptitle('Learned Mean Digits', fontsize=14)
plt.tight_layout()
plt.savefig('figures/priority1_v2_means.png', dpi=100, bbox_inches='tight')
plt.close()

# ============================================================================
# Test 2: Compare with Real Data
# ============================================================================
print("\nTest 2: Compare with real data")

# Real data statistics
real_means = {c: np.mean(class_data[c], axis=0) for c in range(N_CLASSES)}
real_stds = {c: np.std(class_data[c], axis=0) for c in range(N_CLASSES)}

# Generated data statistics (from sampling)
gen_means = {c: np.mean(all_samples[all_labels == c], axis=0) for c in range(N_CLASSES)}
gen_stds = {c: np.std(all_samples[all_labels == c], axis=0) for c in range(N_CLASSES)}

print("\nReal data pixel statistics:")
for c in range(N_CLASSES):
    print(f"  Class {c}: mean = {np.mean(real_means[c]):.4f}, std = {np.mean(real_stds[c]):.4f}")

print("\nGenerated data pixel statistics:")
for c in range(N_CLASSES):
    print(f"  Class {c}: mean = {np.mean(gen_means[c]):.4f}, std = {np.mean(gen_stds[c]):.4f}")

# MSE between real and generated means
print("\nMSE (real mean vs generated mean):")
for c in range(N_CLASSES):
    mse = np.mean((real_means[c] - gen_means[c])**2)
    print(f"  Class {c}: {mse:.6f}")

# ============================================================================
# Test 3: Similarity to real digits
# ============================================================================
print("\nTest 3: Similarity to real digits")

# For each generated sample, find distance to nearest real digit
nearest_distances = []
for sample in all_samples:
    distances = np.mean((sample[None] - X_all)**2, axis=1)
    nearest_distances.append(np.min(distances))

mean_nearest_dist = np.mean(nearest_distances)
std_nearest_dist = np.std(nearest_distances)
min_nearest_dist = np.min(nearest_distances)

print(f"  Mean distance to nearest real: {mean_nearest_dist:.6f}")
print(f"  Std distance: {std_nearest_dist:.6f}")
print(f"  Min distance: {min_nearest_dist:.6f}")

# Compare to real-real distances
test_real_distances = []
for i in range(50):
    for j in range(50):
        if i != j:
            dist = np.mean((X_all[i] - X_all[j])**2)
            test_real_distances.append(dist)

mean_real_dist = np.mean(test_real_distances)
std_real_dist = np.std(test_real_distances)

print(f"\nReal-real distances:")
print(f"  Mean: {mean_real_dist:.6f}")
print(f"  Std: {std_real_dist:.6f}")

similarity_ratio = mean_nearest_dist / mean_real_dist
print(f"\nSimilarity ratio: {similarity_ratio:.2f}")

# ============================================================================
# Test 4: Try different sigma_max values
# ============================================================================
print("\n" + "="*70)
print("Test 4: Different sigma_max values")
print("="*70)

for sigma_max in [0.1, 0.2, 0.3, 0.5]:
    print(f"\nsigma_max = {sigma_max}")
    
    optimizer = CompactEvoOptimizer(
        param_dim=784,
        lr_mu=0.01, lr_sigma=0.005,
        sigma_min=0.01, sigma_max=sigma_max,
        sigma_regularization=0.001,
        calibration_size=50, calibration_interval=10,
        credit_factor=2.0,
        bounds=(0.0, 1.0),
        random_state=42
    )
    
    mean_digit = np.clip(np.mean(class_data[0], axis=0), 0.0, 1.0)
    optimizer.initialize(mean_digit)
    
    def loss_func(params):
        candidate = np.clip(params.reshape(1, -1), 0.0, 1.0)
        distances = np.mean((candidate - class_data[0])**2, axis=1)
        return np.min(distances)
    
    for iteration in range(200):
        optimizer.step(loss_func, iteration=iteration)
    
    mu, sigma = optimizer.get_distribution_parameters()
    
    # Generate 10 samples
    rng = np.random.RandomState(42)
    samples = []
    for _ in range(10):
        sample = np.clip(mu + sigma * rng.randn(784), 0.0, 1.0)
        samples.append(sample)
    samples = np.array(samples)
    
    pixel_std = np.mean(np.std(samples, axis=0))
    
    print(f"  σ_mean = {np.mean(sigma):.4f}, diversity = {pixel_std:.4f}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print(f"\nArchitecture: Direct Pixel Distribution with Bounds")
print(f"Configuration: SIGMA_REG={SIGMA_REG}, SIGMA_MIN={SIGMA_MIN}, SIGMA_MAX={SIGMA_MAX}")

print(f"\nResults:")
print(f"  Overall diversity: {mean_pixel_std_all:.4f}")
print(f"  Similarity ratio: {similarity_ratio:.2f}")

# Final verdict
if mean_pixel_std_all > 0.05 and similarity_ratio < 1.5:
    print(f"\n✓ SUCCESS: Bounded direct pixel distribution works!")
    print(f"  Generated digits are diverse and realistic.")
elif mean_pixel_std_all > 0.05:
    print(f"\n⚠ PARTIAL: Diversity is good but similarity could be better.")
else:
    print(f"\n✗ FAILURE: Low diversity.")

print(f"\nNote: Check the generated figures to verify visual quality.")
print(f"  - figures/priority1_v2_generated.png")
print(f"  - figures/priority1_v2_means.png")

print("\n" + "="*70)
