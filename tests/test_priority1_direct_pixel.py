"""
Test Priority 1 Architecture: Direct Pixel Distribution

Approach: Use EBGA to learn a Gaussian distribution directly over pixel space.
- Each class has its own distribution N(μ_class, Σ_class)
- μ_class: mean digit for the class
- Σ_class: pixel-wise variance (diagonal covariance)
- Generate: sample pixel_i ~ N(μ_class[i], σ_class[i])

This is the simplest approach that leverages EBGA's natural distribution learning.
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
print("Priority 1: Direct Pixel Distribution Test")
print("="*70)

# Load MNIST
print("\nLoading MNIST...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='pandas')
X_all = mnist.data.astype('float32')
y_all = mnist.target.astype('int')
X_all = MinMaxScaler().fit_transform(X_all)

# Use first 3 classes for faster testing
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
SIGMA_REG = 0.001  # Try different values
SIGMA_MIN = 0.01
SIGMA_MAX = 0.5
N_ITERATIONS = 500
PATIENCE = 50

for c in range(N_CLASSES):
    print(f"\n  Class {c}:")
    
    # Create optimizer for this class
    optimizer = CompactEvoOptimizer(
        param_dim=784,
        lr_mu=0.01, lr_sigma=0.005,
        sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX,
        sigma_regularization=SIGMA_REG,
        calibration_size=50, calibration_interval=10,
        credit_factor=2.0,
        random_state=42
    )
    
    # Initialize with mean digit of this class
    mean_digit = np.mean(class_data[c], axis=0)
    optimizer.initialize(mean_digit)
    
    # Loss function: for parameter sample θ (a candidate digit),
    # find distance to nearest real digit in this class
    def loss_func(params):
        # params is a candidate digit (784 pixels)
        candidate = params.reshape(1, -1)
        
        # Distance to nearest real digit in class
        distances = np.mean((candidate - class_data[c])**2, axis=1)
        return np.min(distances)
    
    # Train
    best_loss = float('inf')
    patience_counter = 0
    losses = []
    
    for iteration in range(N_ITERATIONS):
        loss = optimizer.step(loss_func, iteration=iteration)
        losses.append(loss)
        
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
    print(f"    Final: σ_mean = {np.mean(sigma_final):.4f}, σ_min = {np.min(sigma_final):.4f}, σ_max = {np.max(sigma_final):.4f}")

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
        sampled_digit = np.clip(sampled_digit, 0, 1)  # Ensure valid pixel range
        all_samples.append(sampled_digit)
        all_labels.append(c)

all_samples = np.array(all_samples)
all_labels = np.array(all_labels)

# Calculate diversity per class
for c in range(N_CLASSES):
    c_samples = all_samples[all_labels == c]
    pixel_std = np.std(c_samples, axis=0)
    mean_pixel_std = np.mean(pixel_std)
    print(f"  Class {c}: diversity (mean pixel std) = {mean_pixel_std:.4f}")

# Overall diversity
pixel_std_all = np.std(all_samples, axis=0)
mean_pixel_std_all = np.mean(pixel_std_all)
print(f"  Overall diversity: {mean_pixel_std_all:.4f}")

# Visualize: Generated samples
plt.figure(figsize=(15, 10))
for i, (sample, label) in enumerate(zip(all_samples, all_labels)):
    row = i // 10
    col = i % 10
    plt.subplot(N_CLASSES, 10, i + 1)
    plt.imshow(sample.reshape(28, 28), cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
plt.suptitle(f'Direct Pixel Distribution\nSIGMA_REG={SIGMA_REG}, σ_min={SIGMA_MIN}, σ_max={SIGMA_MAX}', fontsize=14)
plt.tight_layout()
plt.savefig('figures/priority1_generated_samples.png', dpi=100, bbox_inches='tight')
plt.close()

# Visualize: Mean digits
plt.figure(figsize=(15, 4))
for c in range(N_CLASSES):
    mu = class_optimizers[c].mu
    plt.subplot(1, N_CLASSES, c + 1)
    plt.imshow(mu.reshape(28, 28), cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    plt.title(f'Class {c} Mean', fontsize=12)
plt.suptitle('Learned Mean Digits', fontsize=14)
plt.tight_layout()
plt.savefig('figures/priority1_mean_digits.png', dpi=100, bbox_inches='tight')
plt.close()

# Visualize: Sigma distributions per class
plt.figure(figsize=(15, 5))
for c in range(N_CLASSES):
    mu, sigma = class_optimizers[c].get_distribution_parameters()
    plt.subplot(1, N_CLASSES, c + 1)
    plt.hist(sigma, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    plt.axvline(np.mean(sigma), color='red', linestyle='--', label=f'mean={np.mean(sigma):.3f}')
    plt.xlabel('Sigma')
    plt.ylabel('Frequency')
    plt.title(f'Class {c}')
    plt.legend(fontsize=8)
plt.suptitle('Sigma Distributions per Class', fontsize=14)
plt.tight_layout()
plt.savefig('figures/priority1_sigma_distributions.png', dpi=100, bbox_inches='tight')
plt.close()

# ============================================================================
# Test 2: Compare with Real Data
# ============================================================================
print("\nTest 2: Compare generated with real data")

# Real data statistics
real_means = {c: np.mean(class_data[c], axis=0) for c in range(N_CLASSES)}
real_stds = {c: np.std(class_data[c], axis=0) for c in range(N_CLASSES)}

# Generated data statistics (from sampling)
gen_means = {c: np.mean(all_samples[all_labels == c], axis=0) for c in range(N_CLASSES)}
gen_stds = {c: np.std(all_samples[all_labels == c], axis=0) for c in range(N_CLASSES)}

print("\nReal data statistics:")
for c in range(N_CLASSES):
    print(f"  Class {c}: pixel mean = {np.mean(real_means[c]):.4f}, pixel std = {np.mean(real_stds[c]):.4f}")

print("\nGenerated data statistics:")
for c in range(N_CLASSES):
    print(f"  Class {c}: pixel mean = {np.mean(gen_means[c]):.4f}, pixel std = {np.mean(gen_stds[c]):.4f}")

# Calculate MSE between real and generated means
print("\nMean Squared Error (real mean vs generated mean):")
for c in range(N_CLASSES):
    mse = np.mean((real_means[c] - gen_means[c])**2)
    print(f"  Class {c}: MSE = {mse:.6f}")

# ============================================================================
# Test 3: Check if samples look like real digits
# ============================================================================
print("\nTest 3: Visual similarity to real digits")

# For each generated sample, find distance to nearest real digit
nearest_distances = []
for sample in all_samples:
    # Find nearest real digit (across all classes)
    all_real = X_all
    distances = np.mean((sample[None] - all_real)**2, axis=1)
    nearest_distances.append(np.min(distances))

mean_nearest_dist = np.mean(nearest_distances)
std_nearest_dist = np.std(nearest_distances)
min_nearest_dist = np.min(nearest_distances)

print(f"  Mean distance to nearest real digit: {mean_nearest_dist:.6f}")
print(f"  Std distance: {std_nearest_dist:.6f}")
print(f"  Min distance: {min_nearest_dist:.6f}")

# For comparison, calculate distances between real digits
test_real_distances = []
for i in range(50):  # Sample 50 real digits
    for j in range(50):
        if i != j:
            dist = np.mean((X_all[i] - X_all[j])**2)
            test_real_distances.append(dist)

mean_real_dist = np.mean(test_real_distances)
std_real_dist = np.std(test_real_distances)

print(f"\n  Mean distance between real digits: {mean_real_dist:.6f}")
print(f"  Std distance between real digits: {std_real_dist:.6f}")

similarity_ratio = mean_nearest_dist / mean_real_dist
print(f"\n  Similarity ratio (generated/real): {similarity_ratio:.2f}")
print(f"  Interpretation: {'✓ GOOD - Generated samples are similar to real digits' if similarity_ratio < 1.5 else '✗ POOR - Generated samples are far from real digits'}")

# ============================================================================
# Test 4: Try different sigma_regularization values
# ============================================================================
print("\n" + "="*70)
print("Test 4: Different sigma_regularization values")
print("="*70)

for sigma_reg in [0.0, 0.0001, 0.001, 0.01]:
    print(f"\nsigma_regularization = {sigma_reg}")
    
    # Train class 0 with this sigma_reg
    optimizer = CompactEvoOptimizer(
        param_dim=784,
        lr_mu=0.01, lr_sigma=0.005,
        sigma_min=0.01, sigma_max=0.5,
        sigma_regularization=sigma_reg,
        calibration_size=50, calibration_interval=10,
        credit_factor=2.0,
        random_state=42
    )
    
    mean_digit = np.mean(class_data[0], axis=0)
    optimizer.initialize(mean_digit)
    
    def loss_func(params):
        candidate = params.reshape(1, -1)
        distances = np.mean((candidate - class_data[0])**2, axis=1)
        return np.min(distances)
    
    for iteration in range(200):
        optimizer.step(loss_func, iteration=iteration)
    
    mu, sigma = optimizer.get_distribution_parameters()
    
    # Generate 10 samples
    rng = np.random.RandomState(42)
    samples = []
    for _ in range(10):
        sample = mu + sigma * rng.randn(784)
        samples.append(np.clip(sample, 0, 1))
    samples = np.array(samples)
    
    # Calculate diversity
    pixel_std = np.mean(np.std(samples, axis=0))
    
    print(f"  σ_mean = {np.mean(sigma):.4f}, diversity = {pixel_std:.4f}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print(f"\nArchitecture: Direct Pixel Distribution")
print(f"Configuration: SIGMA_REG={SIGMA_REG}, SIGMA_MIN={SIGMA_MIN}, SIGMA_MAX={SIGMA_MAX}")
print(f"Training: {N_ITERATIONS} iterations, {PATIENCE} patience")

print(f"\nResults:")
print(f"  Overall diversity: {mean_pixel_std_all:.4f}")
print(f"  Similarity to real digits: {similarity_ratio:.2f}x")

# Final verdict
if mean_pixel_std_all > 0.05 and similarity_ratio < 2.0:
    print(f"\n✓ SUCCESS: Priority 1 architecture produces diverse, realistic digits!")
    print(f"  The direct pixel distribution approach works.")
elif mean_pixel_std_all > 0.05:
    print(f"\n⚠ PARTIAL: Diversity is good but samples don't match real digits well.")
    print(f"  May need different loss function or training approach.")
else:
    print(f"\n✗ FAILURE: Low diversity or poor quality.")
    print(f"  Priority 1 doesn't work. Need to try Priority 2 or 3.")

print(f"\nSaved figures:")
print(f"  - figures/priority1_generated_samples.png")
print(f"  - figures/priority1_mean_digits.png")
print(f"  - figures/priority1_sigma_distributions.png")

print("\n" + "="*70)
print("Test completed!")
print("="*70)
