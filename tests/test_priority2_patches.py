"""
Test Priority 2 Architecture: Fewer, Larger Patches

Approach: Use MultiEvoOptimizer with fewer, larger patches.
- Instead of 16 patches × 10 classes = 160 optimizers
- Use 2-4 patches × 10 classes = 20-40 optimizers
- Each patch is larger and learns more meaningful features
- Train each patch on ALL class data (not individual digits)
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
print("Priority 2: Fewer, Larger Patches")
print("="*70)

# Load MNIST
print("\nLoading MNIST...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='pandas')
X_all = mnist.data.astype('float32')
y_all = mnist.target.astype('int')
X_all = MinMaxScaler().fit_transform(X_all)

# Use first 3 classes
N_CLASSES = 3
SAMPLES_PER_CLASS = 500

mask = y_all < N_CLASSES
X_all = X_all[mask][:SAMPLES_PER_CLASS * N_CLASSES]
y_all = y_all[mask][:SAMPLES_PER_CLASS * N_CLASSES]

print(f"Dataset: {len(X_all)} samples, {N_CLASSES} classes ({SAMPLES_PER_CLASS} per class)")

# Separate by class
class_data = {c: X_all[y_all == c] for c in range(N_CLASSES)}

# Try different numbers of patches
for N_PATCHES in [2, 4, 8]:
    print(f"\n{'='*70}")
    print(f"Testing with {N_PATCHES} patches per class")
    print(f"{'='*70}")
    
    PATCH_SIZE = 784 // N_PATCHES
    # Ensure patch size is valid
    assert N_PATCHES * PATCH_SIZE == 784, f"{N_PATCHES} * {PATCH_SIZE} != 784"
    TOTAL_OPTIMIZERS = N_CLASSES * N_PATCHES
    
    print(f"  Patches per class: {N_PATCHES}")
    print(f"  Patch size: {PATCH_SIZE} pixels")
    print(f"  Total optimizers: {TOTAL_OPTIMIZERS}")
    
    # Create layer configs
    layer_configs = {c: [(PATCH_SIZE, 'sigmoid') for _ in range(N_PATCHES)] for c in range(N_CLASSES)}
    
    # Create MultiEvoOptimizer
    multi_opt = MultiEvoOptimizer(
        n_classes=N_CLASSES,
        layer_configs=layer_configs,
        lr_mu=0.01, lr_sigma=0.005,
        sigma_min=0.01, sigma_max=0.3,
        sigma_regularization=0.001,
        calibration_size=50, calibration_interval=10,
        credit_factor=2.0,
        random_state=42
    )
    
    # Initialize with patch means for each class
    initial_params = {}
    for c in range(N_CLASSES):
        class_patches = []
        for p in range(N_PATCHES):
            start = p * PATCH_SIZE
            end = (p + 1) * PATCH_SIZE
            patch_mean = np.mean(class_data[c][:, start:end], axis=0)
            class_patches.append(patch_mean)
        initial_params[c] = class_patches
    
    multi_opt.initialize(initial_params_dict=initial_params)
    
    # Train each class
    print("\n  Training...")
    for c in range(N_CLASSES):
        for p in range(N_PATCHES):
            start = p * PATCH_SIZE
            end = (p + 1) * PATCH_SIZE
            
            # Train this patch on ALL data of this class
            # Note: full_params is ALREADY the concatenated parameters for all patches of this class
            # because step_layer builds it that way
            def patch_loss(full_params):
                # full_params is the complete digit (784 pixels) built from all patches
                reconstructed = np.clip(full_params, 0.0, 1.0)
                
                # Distance to nearest real digit in class
                # class_data[c] has shape (n_samples, 784)
                # reconstructed has shape (784,)
                distances = np.mean((class_data[c] - reconstructed)**2, axis=1)
                return np.min(distances)
            
            for iteration in range(100):
                loss = multi_opt.step_layer(c, p, patch_loss, iteration=iteration)
                
                if iteration % 50 == 0:
                    mu, sigma = multi_opt.get_optimizer(c, p).get_distribution_parameters()
                    print(f"    Class {c}, Patch {p}, Iter {iteration}: σ_mean = {np.mean(sigma):.4f}")
    
    # Generate samples
    print("\n  Generating samples...")
    all_samples = []
    all_labels = []
    
    for c in range(N_CLASSES):
        for _ in range(10):
            # Concatenate parameters from all patches
            all_patch_params = []
            for p in range(N_PATCHES):
                opt = multi_opt.get_optimizer(c, p)
                mu, sigma = opt.get_distribution_parameters()
                sampled_params = mu + sigma * np.random.RandomState().randn(len(mu))
                all_patch_params.append(sampled_params)
            
            digit = np.concatenate(all_patch_params)
            digit = np.clip(digit, 0.0, 1.0)
            all_samples.append(digit)
            all_labels.append(c)
    
    all_samples = np.array(all_samples)
    all_labels = np.array(all_labels)
    
    # Calculate diversity
    for c in range(N_CLASSES):
        c_samples = all_samples[all_labels == c]
        pixel_std = np.mean(np.std(c_samples, axis=0))
        print(f"    Class {c} diversity: {pixel_std:.4f}")
    
    # Overall diversity
    pixel_std_all = np.mean(np.std(all_samples, axis=0))
    print(f"    Overall diversity: {pixel_std_all:.4f}")
    
    # Similarity to real digits
    nearest_distances = []
    for sample in all_samples:
        distances = np.mean((sample[None] - X_all)**2, axis=1)
        nearest_distances.append(np.min(distances))
    mean_nearest_dist = np.mean(nearest_distances)
    
    # Real-real distances
    test_real_distances = []
    for i in range(50):
        for j in range(50):
            if i != j:
                dist = np.mean((X_all[i] - X_all[j])**2)
                test_real_distances.append(dist)
    mean_real_dist = np.mean(test_real_distances)
    similarity_ratio = mean_nearest_dist / mean_real_dist
    
    print(f"    Similarity ratio: {similarity_ratio:.2f}")
    
    # Save figure
    os.makedirs('figures', exist_ok=True)
    plt.figure(figsize=(15, 10))
    for i, (sample, label) in enumerate(zip(all_samples, all_labels)):
        row = i // 10
        col = i % 10
        plt.subplot(N_CLASSES, 10, i + 1)
        plt.imshow(sample.reshape(28, 28), cmap='gray', vmin=0, vmax=1)
        plt.axis('off')
    plt.suptitle(f'{N_PATCHES} Patches/Class ({TOTAL_OPTIMIZERS} Optimizers)\nSimilarity: {similarity_ratio:.2f}', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'figures/priority2_{N_PATCHES}patches.png', dpi=100, bbox_inches='tight')
    plt.close()
    
    # Final verdict
    if pixel_std_all > 0.05 and similarity_ratio < 1.5:
        print(f"    ✓ SUCCESS for {N_PATCHES} patches!")
    elif pixel_std_all > 0.05:
        print(f"    ⚠ PARTIAL for {N_PATCHES} patches")
    else:
        print(f"    ✗ FAILURE for {N_PATCHES} patches")

print("\n" + "="*70)
print("All Priority 2 tests completed!")
print("="*70)
