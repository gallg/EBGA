from scipy.special import softmax
import numpy as np


def create_bins(y, n_bins):
    """Create bins for target discretization."""
    y_range = y.max() - y.min()
    return np.linspace(y.min() - 0.1 * y_range, y.max() + 0.1 * y_range, n_bins + 1)


def calculate_loss(params, X, y, bin_edges, entropy_awareness, n_bins):
    """
    Calculate the loss function for CGD.

    Parameters:
    - params: model parameters
    - X: input features
    - y: target values
    - bin_edges: bin edges for discretization
    - entropy_awareness: weight for entropy term (uncertainty awareness)
    - n_bins: number of bins

    Returns:
    - Loss value
    """
    W_b = params.reshape(n_bins, -1)
    W = W_b[:, :-1]  # Weights
    b = W_b[:, -1]   # Biases

    # Softmax prediction
    z = X @ W.T + b
    P = softmax(z, axis=1)

    # Discretize target
    y_binned = np.digitize(y, bin_edges[:-1]) - 1
    y_binned = np.clip(y_binned, 0, n_bins - 1)
    y_onehot = np.eye(n_bins)[y_binned]

    # Cross-entropy loss
    ce_loss = -np.mean(np.sum(y_onehot * np.log(P + 1e-10), axis=1))

    # Surprise (entropy of predictions)
    entropy = -np.sum(P * np.log(P + 1e-10), axis=1)
    surprise_loss = np.mean(entropy)

    return ce_loss + entropy_awareness * surprise_loss

