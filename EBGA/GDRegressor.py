from scipy.special import softmax
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils import check_random_state
from sklearn.metrics import r2_score
from EBGA.utils import create_bins, calculate_loss

import numpy as np


class CompactGeneticDescentRegressor(BaseEstimator, RegressorMixin):
    """
    Compact Genetic Descent Regressor - A distribution-based evolutionary algorithm
    for regression with efficient compact updates.

    Parameters
    ----------
    n_bins : int, default=10
        Number of bins for target discretization.
    max_iter : int, default=500
        Maximum optimization iterations.
    lr_mu : float, default=0.05
        Learning rate for mean parameters.
    lr_sigma : float, default=0.005
        Learning rate for standard deviation.
    lambda_surprise : float, default=0.1
        Weight for surprise term in loss.
    sigma_min/max : float, default=0.01/1.0
        Bounds for standard deviation.
    calibration_interval : int, default=25
        How often to use population calibration.
    credit_factor : float, default=2.0
        Strength of credit assignment.
    early_stopping : bool, default=True
        Whether to use early stopping.
    patience : int, default=20
        Patience for early stopping.
    calibration_size : int, default=20
        Samples for calibration.
    random_state : int, default=None
        Random seed for reproducibility.
    """

    def __init__(self, n_bins=10, max_iter=500, lr_mu=0.05, lr_sigma=0.005,
                 lambda_surprise=0.1, sigma_min=0.01, sigma_max=1.0,
                 calibration_interval=25, credit_factor=2.0,
                 early_stopping=True, patience=20, calibration_size=20,
                 random_state=None):

        self.n_bins = n_bins
        self.max_iter = max_iter
        self.lr_mu = lr_mu
        self.lr_sigma = lr_sigma
        self.lambda_surprise = lambda_surprise
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.calibration_interval = calibration_interval
        self.credit_factor = credit_factor
        self.early_stopping = early_stopping
        self.patience = patience
        self.calibration_size = calibration_size
        self.random_state = random_state

    def fit(self, X, y):
        """Fit the CGD model to training data."""
        X, y = check_X_y(X, y)
        self._random_state = check_random_state(self.random_state)
        self._initialize(X, y)
        self._train(X, y)
        return self

    def predict(self, X):
        """Predict target values for input X."""
        check_is_fitted(self)
        X = check_array(X)

        W_b = self.mu_.reshape(self.n_bins, -1)
        W, b = W_b[:, :-1], W_b[:, -1]
        z = X @ W.T + b
        P = softmax(z, axis=1)
        centers = (self.bin_edges_[:-1] + self.bin_edges_[1:]) / 2
        return P @ centers

    def score(self, X, y):
        """Calculate R² score for predictions."""
        return r2_score(y, self.predict(X))

    def _initialize(self, X, y):
        """Initialize model parameters."""
        self.bin_edges_ = create_bins(y, self.n_bins)
        param_dim = self.n_bins * (X.shape[1] + 1)
        self.mu_ = np.zeros(param_dim)
        self.sigma_ = np.ones(param_dim)

    def _train(self, X, y):
        """Train the model using compact genetic updates."""
        best_loss, patience_counter = float('inf'), 0

        for gen in range(self.max_iter):
            # Use appropriate update method
            loss = (self._population_calibration(X, y)
                    if gen % self.calibration_interval == 0
                    else self._pairwise_update(X, y))

            # Early stopping check
            if self.early_stopping:
                current_loss = calculate_loss(self.mu_, X, y, self.bin_edges_,
                                            self.lambda_surprise, self.n_bins)
                if current_loss < best_loss:
                    best_loss, patience_counter = current_loss, 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        break

    def _population_calibration(self, X, y):
        """Use population samples to calibrate distribution parameters."""
        noise = self._random_state.randn(self.calibration_size, len(self.mu_))
        perturbed = self.mu_ + self.sigma_ * noise
        losses = np.array([calculate_loss(p, X, y, self.bin_edges_,
                                        self.lambda_surprise, self.n_bins)
                          for p in perturbed])

        # Update distribution parameters
        grad_mu = np.mean(losses[:, None] * noise, axis=0)
        grad_sigma = np.mean(losses[:, None] * (noise**2 - 1), axis=0)

        self.mu_ -= self.lr_mu * grad_mu
        self.sigma_ *= np.exp(self.lr_sigma * grad_sigma)
        self.sigma_ = np.clip(self.sigma_, self.sigma_min, self.sigma_max)

        return losses.mean()

    def _pairwise_update(self, X, y):
        """Perform compact pairwise update of distribution parameters."""
        # Sample and evaluate two individuals
        theta1 = self.mu_ + self.sigma_ * self._random_state.randn(len(self.mu_))
        theta2 = self.mu_ + self.sigma_ * self._random_state.randn(len(self.mu_))

        loss1 = calculate_loss(theta1, X, y, self.bin_edges_,
                             self.lambda_surprise, self.n_bins)
        loss2 = calculate_loss(theta2, X, y, self.bin_edges_,
                             self.lambda_surprise, self.n_bins)

        # Determine winner and loser
        winner, loser = (theta1, theta2) if loss1 < loss2 else (theta2, theta1)
        loss_diff = max(loss1, loss2) - min(loss1, loss2)

        # Credit assignment
        eps = 1e-8
        relative_improvement = loss_diff / (abs(loss1) + abs(loss2) + eps)
        update_strength = 1 + self.credit_factor * np.tanh(relative_improvement)

        # Parameter-specific adaptation
        winner_diff = winner - self.mu_
        loser_diff = loser - self.mu_
        update_mag = np.clip(np.abs(winner_diff) / (np.abs(loser_diff) + eps), 0.1, 10)

        # Update distribution parameters
        self.mu_ += self.lr_mu * update_strength * update_mag * winner_diff

        # Update sigma based on observed diversity
        observed_diversity = np.abs(winner - loser)
        self.sigma_ *= np.exp(self.lr_sigma * update_strength * update_mag *
                             (observed_diversity - self.sigma_))
        self.sigma_ = np.clip(self.sigma_, self.sigma_min, self.sigma_max)

        return (loss1 + loss2) / 2
