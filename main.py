import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from EBGA.GDGenerator import EvolutionaryAutoencoder
import os
import importlib

# Reload the module to ensure we have the latest changes
import EBGA.GDGenerator
importlib.reload(EBGA.GDGenerator)
from EBGA.GDGenerator import EvolutionaryAutoencoder

def load_mnist():
    """Load MNIST dataset and preprocess it."""
    print("Loading MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist.data, mnist.target
    # Normalize to [0, 1] and reshape
    X = X / 255.0
    X = X.reshape(-1, 28, 28)
    return X, y

def visualize_reconstructions(model, X_test, n_samples=5, save_path=None):
    """Visualize original and reconstructed images."""
    fig, axes = plt.subplots(2, n_samples, figsize=(12, 5))

    # Randomly select samples
    indices = np.random.choice(len(X_test), n_samples, replace=False)

    for i, idx in enumerate(indices):
        # Original image (already 2D)
        axes[0, i].imshow(X_test[idx], cmap='gray')
        axes[0, i].set_title('Original')
        axes[0, i].axis('off')

        # Reconstructed image (ensure 2D)
        recon = model.reconstruct(X_test[idx])
        if len(recon.shape) == 3 and recon.shape[0] == 1:  # If shape is (1, 28, 28)
            recon = recon[0]  # Remove the batch dimension
        axes[1, i].imshow(recon, cmap='gray')
        axes[1, i].set_title('Reconstruction')
        axes[1, i].axis('off')

    plt.suptitle('Original vs Reconstructed Images')
    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, 'reconstructions.png'))
    plt.show()

def visualize_generated_samples(model, n_samples=5, save_path=None):
    """Visualize generated samples."""
    fig, axes = plt.subplots(1, n_samples, figsize=(10, 2))

    # Generate and display samples
    samples = model.generate(n_samples)
    if len(samples.shape) == 3 and samples.shape[0] == n_samples:  # If shape is (n_samples, 28, 28)
        for i in range(n_samples):
            axes[i].imshow(samples[i], cmap='gray')
            axes[i].axis('off')
    else:  # If shape is different, try to reshape
        samples = np.reshape(samples, (n_samples, 28, 28))
        for i in range(n_samples):
            axes[i].imshow(samples[i], cmap='gray')
            axes[i].axis('off')

    plt.suptitle('Generated Samples')
    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, 'generated_samples.png'))
    plt.show()

def plot_loss_history(loss_history, target_loss, save_path=None):
    """Plot the loss history."""
    plt.figure(figsize=(8, 4))
    plt.plot(loss_history, label='Loss')
    plt.axhline(y=target_loss, color='r', linestyle='--', label='Target Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(f'Training Progress ({loss_history[0]:.4f} → {loss_history[-1]:.4f})')
    plt.legend()
    if save_path:
        plt.savefig(os.path.join(save_path, 'loss_history.png'))
    plt.show()

def main():
    # Create output directory
    output_dir = 'results_autoencoder'
    os.makedirs(output_dir, exist_ok=True)

    # Load and preprocess MNIST
    X, _ = load_mnist()

    # Use a subset for faster training
    X_train, X_test = train_test_split(X, test_size=0.99, random_state=42)  # Using 1% for training

    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")

    # Initialize and train the evolutionary autoencoder
    print("Initializing evolutionary autoencoder...")
    model = EvolutionaryAutoencoder(
        img_shape=(28, 28),
        latent_dim=5,
        target_loss=0.04,
        loss_metric='mae',
        lr_mu=0.009,
        lr_sigma=0.001,
        sigma_min=0.01,
        sigma_max=1.0,
        calibration_interval=25,
        credit_factor=2.0,
        calibration_size=20,
        random_state=42
    )

    print("Training evolutionary autoencoder...")
    # Train on a batch of samples for more stable training
    loss_history = model.fit(X_train[:100])  # Use first 100 samples

    # Visualize results
    print("Training completed. Visualizing results...")

    # Plot training progress
    plot_loss_history(loss_history, model.target_loss, save_path=output_dir)

    # Show reconstructions
    visualize_reconstructions(model, X_test, save_path=output_dir)

    # Show generated samples
    visualize_generated_samples(model, save_path=output_dir)

    # Save model and results
    np.save(os.path.join(output_dir, 'mu.npy'), model.mu)
    np.save(os.path.join(output_dir, 'sigma.npy'), model.sigma)
    np.save(os.path.join(output_dir, 'loss_history.npy'), loss_history)

    print("All done! Results saved in:", output_dir)

if __name__ == '__main__':
    main()
