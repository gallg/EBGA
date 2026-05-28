from scipy.special import softmax
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils import check_random_state
from sklearn.preprocessing import LabelBinarizer

import numpy as np


class CompactGeneticDescentClassifier(BaseEstimator, ClassifierMixin):
    """
    Compact Genetic Descent Classifier for classification tasks.

    Implements an efficient distribution-based evolutionary algorithm
    that eliminates explicit population storage.

    Parameters
    ----------
    max_iter : int, default=500
        Maximum number of iterations.
    lr_mu : float, default=0.05
        Learning rate for mean parameters.
    lr_sigma : float, default=0.005
        Learning rate for standard deviation.
    sigma_min : float, default=0.01
        Minimum value for sigma parameters.
    sigma_max : float, default=1.0
        Maximum value for sigma parameters.
    calibration_interval : int, default=25
        How often to use population calibration vs pairwise updates.
    credit_factor : float, default=2.0
        Strength of credit assignment.
    early_stopping : bool, default=True
        Whether to use early stopping.
    patience : int, default=20
        Number of iterations without improvement before early stopping.
    calibration_size : int, default=20
        Number of samples for periodic calibration.
    lambda_l2 : float, default=0.0
        L2 regularization strength.
    random_state : int, default=None
        Random state for reproducibility.
    """

    def __init__(self, max_iter=500, lr_mu=0.05, lr_sigma=0.005,
                 sigma_min=0.01, sigma_max=1.0, calibration_interval=25,
                 credit_factor=2.0, early_stopping=True, patience=20,
                 calibration_size=20, lambda_l2=0.0, random_state=None):

        self.max_iter = max_iter
        self.lr_mu = lr_mu
        self.lr_sigma = lr_sigma
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.calibration_interval = calibration_interval
        self.credit_factor = credit_factor
        self.early_stopping = early_stopping
        self.patience = patience
        self.calibration_size = calibration_size
        self.lambda_l2 = lambda_l2
        self.random_state = random_state

    def fit(self, X, y):
        """
        Fit the classifier to the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input samples.
        y : array-like of shape (n_samples,)
            Target class labels.

        Returns
        -------
        self : object
            Fitted classifier.
        """
        # Validate input data
        X, y = check_X_y(X, y)

        # Set random state for reproducibility
        self._random_state = check_random_state(self.random_state)

        # Encode class labels
        self.label_binarizer_ = LabelBinarizer()
        y_onehot = self.label_binarizer_.fit_transform(y)
        self.classes_ = self.label_binarizer_.classes_
        self.n_classes_ = len(self.classes_)

        # Handle binary classification case
        if self.n_classes_ == 2:
            y_onehot = y_onehot[:, [1]]  # Take only positive class probabilities
            self.n_classes_ = 2  # Keep track of binary case

        # Initialize model parameters
        self._initialize_parameters(X)

        # Main training loop
        self._train_loop(X, y_onehot)

        return self

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        check_is_fitted(self)
        X = check_array(X)

        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, X):
        """
        Probability estimates for each class.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        check_is_fitted(self)
        X = check_array(X)

        W_b = self.μ_.reshape(self.n_classes_, -1)
        W = W_b[:, :-1]  # Weights
        b = W_b[:, -1]   # Biases

        # Calculate logits and apply softmax
        z = X @ W.T + b

        # Handle binary classification case
        if len(self.classes_) == 2:
            z = np.hstack([-z, z])  # Convert to two columns

        return softmax(z, axis=1)

    def score(self, X, y, sample_weight=None):
        """
        Return the mean accuracy on the given test data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        score : float
            Mean accuracy.
        """
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)

    def _initialize_parameters(self, X):
        """Initialize the distribution parameters."""
        param_dim = self.n_classes_ * (X.shape[1] + 1)  # K*(d+1)
        self.μ_ = np.zeros(param_dim)
        self.σ_ = np.ones(param_dim)

    def _loss(self, params, X, y):
        """
        Calculate the loss function.

        Parameters
        ----------
        params : array-like
            Model parameters.
        X : array-like
            Input features.
        y : array-like
            Target values (one-hot encoded).

        Returns
        -------
        loss : float
            Loss value.
        """
        W_b = params.reshape(self.n_classes_, -1)
        W = W_b[:, :-1]  # Weights
        b = W_b[:, -1]   # Biases

        z = X @ W.T + b

        # Handle binary classification case
        if y.shape[1] == 1:  # Binary case has single column
            z = np.hstack([-z, z])

        P = softmax(z, axis=1)

        # Cross-entropy loss
        ce_loss = -np.mean(np.sum(y * np.log(P + 1e-10), axis=1))

        # L2 regularization
        if self.lambda_l2 > 0:
            ce_loss += self.lambda_l2 * np.sum(W**2)

        return ce_loss

    def _train_loop(self, X, y):
        """Main training loop."""
        best_loss = float('inf')
        patience_counter = 0

        for gen in range(self.max_iter):
            # Alternate between update strategies
            if gen % self.calibration_interval == 0:
                loss = self._population_calibration(X, y)
            else:
                loss = self._pairwise_update(X, y)

            # Early stopping condition
            if self.early_stopping:
                current_loss = self._loss(self.μ_, X, y)
                if current_loss < best_loss:
                    best_loss = current_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.patience:
                    break

    def _population_calibration(self, X, y):
        """Population-based calibration using gradient estimation."""
        # Sample population
        noise = self._random_state.randn(self.calibration_size, len(self.μ_))
        perturbed_params = self.μ_ + self.σ_ * noise

        # Evaluate losses
        losses = np.array([self._loss(perturbed_params[i], X, y)
                          for i in range(self.calibration_size)])

        # Gradient estimation
        grad_mu = np.mean(losses[:, None] * noise, axis=0)
        grad_sigma = np.mean(losses[:, None] * (noise**2 - 1), axis=0)

        # Update distribution parameters
        self.μ_ -= self.lr_mu * grad_mu
        self.σ_ *= np.exp(self.lr_sigma * grad_sigma)
        self.σ_ = np.clip(self.σ_, self.sigma_min, self.sigma_max)

        return np.mean(losses)

    def _pairwise_update(self, X, y):
        """Compact pairwise update using two-sample comparisons."""
        # Sample two individuals
        theta1 = self.μ_ + self.σ_ * self._random_state.randn(len(self.μ_))
        theta2 = self.μ_ + self.σ_ * self._random_state.randn(len(self.μ_))

        # Evaluate both
        loss1 = self._loss(theta1, X, y)
        loss2 = self._loss(theta2, X, y)

        # Determine winner and loser
        winner, loser = (theta1, theta2) if loss1 < loss2 else (theta2, theta1)
        loss_diff = abs(loss1 - loss2)

        # Credit assignment
        eps = 1e-8
        relative_improvement = loss_diff / (abs(loss1) + abs(loss2) + eps)
        update_strength = 1 + self.credit_factor * np.tanh(relative_improvement)

        # Parameter-specific adaptation
        winner_diff = winner - self.μ_
        loser_diff = loser - self.μ_
        update_mag = np.clip(np.abs(winner_diff) / (np.abs(loser_diff) + eps), 0.1, 10)

        # Update parameters
        self.μ_ += self.lr_mu * update_strength * update_mag * winner_diff

        # Update sigma based on observed diversity
        observed_diversity = np.abs(winner - loser)
        self.σ_ *= np.exp(self.lr_sigma * update_strength * update_mag *
                         (observed_diversity - self.σ_))
        self.σ_ = np.clip(self.σ_, self.sigma_min, self.sigma_max)

        return (loss1 + loss2) / 2
