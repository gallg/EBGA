import numpy as np
from EBGA.losses.gdloss import GeneticDescentLoss
from EBGA.utils import create_bins

class GeneticDescentRegressor:
    def __init__(self, n_bins=5, pop_size=50, max_iter=500,
                 lr_μ=0.08, lr_σ=0.001, surprise_weight=0.1, random_state=None):
        """
        Genetic Descent Regressor.

        Args:
            n_bins: Number of bins for discretization
            pop_size: Population size for genetic descent
            max_iter: Maximum number of iterations
            lr_μ: Learning rate for μ (mean parameters)
            lr_σ: Learning rate for σ (standard deviation)
            surprise_weight: Weight for the surprise term in loss
            random_state: Random seed for reproducibility
        """
        self.n_bins = n_bins
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.lr_μ = lr_μ
        self.lr_σ = lr_σ
        self.surprise_weight = surprise_weight
        self.random_state = random_state
        self.bin_edges_ = None
        self.μ = None
        self.σ = None
        self.loss_func = None

    def fit(self, X, y):
        """
        Fit the model to the training data.

        Args:
            X: Training features
            y: Training targets
        """
        # Set random state
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Create bins
        self.bin_edges_ = create_bins(y, self.n_bins)
        self.loss_func = GeneticDescentLoss(self.bin_edges_, self.surprise_weight)

        # Initialize parameters
        param_dim = self.n_bins * (X.shape[1] + 1)  # K*(d+1) for W and b
        self.μ = np.zeros(param_dim)
        self.σ = np.ones(param_dim)

        # Genetic Descent training loop
        for t in range(self.max_iter):
            # Sample perturbations
            noise = np.random.randn(self.pop_size, param_dim)
            perturbed_params = self.μ + self.σ * noise

            # Compute losses
            losses = np.array([self.loss_func(perturbed_params[i], X, y)
                              for i in range(self.pop_size)])

            # Estimate gradients
            grad_μ = np.mean(losses[:, None] * noise, axis=0)
            grad_σ = np.mean(losses[:, None] * (noise**2 - 1), axis=0)

            # Update distribution
            self.μ -= self.lr_μ * grad_μ
            self.σ *= np.exp(self.lr_σ * grad_σ)
            self.σ = np.clip(self.σ, 0.01, 1.0)

            # Logging
            if t % 20 == 0:
                print(f"Iteration {t}: Loss = {np.mean(losses):.4f}, σ_range = [{self.σ.min():.4f}, {self.σ.max():.4f}]")

    def predict(self, X):
        """
        Make predictions using the trained model.

        Args:
            X: Input features

        Returns:
            Predicted values
        """
        if self.μ is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")

        W_b = self.μ.reshape(self.n_bins, -1)
        W = W_b[:, :-1]
        b = W_b[:, -1]
        z = X @ W.T + b

        # Compute softmax probabilities
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        P = exp_z / np.sum(exp_z, axis=1, keepdims=True)

        # Calculate expected value
        centers = (self.bin_edges_[:-1] + self.bin_edges_[1:]) / 2
        return P @ centers

    def score(self, X, y):
        """
        Calculate R² score for the given data.

        Args:
            X: Input features
            y: Target values

        Returns:
            R² score
        """
        from sklearn.metrics import r2_score
        return r2_score(y, self.predict(X))
