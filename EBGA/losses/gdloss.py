import numpy as np

def softmax(z):
    """Compute softmax values for each set of scores in z."""
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

class GeneticDescentLoss:
    def __init__(self, bin_edges, surprise_weight=0.1):
        """
        Initialize the loss function for Genetic Descent regression.

        Args:
            bin_edges: Array of bin edges for discretizing the target variable
            surprise_weight: Weight for the surprise/entropy term in loss
        """
        self.bin_edges = bin_edges
        self.K = len(bin_edges) - 1
        self.surprise_weight = surprise_weight

    def __call__(self, params, X, y):
        """
        Compute the loss for given parameters.

        Args:
            params: Flattened array of model parameters
            X: Input features
            y: Target values

        Returns:
            Loss value
        """
        W_b = params.reshape(self.K, -1)
        W = W_b[:, :-1]
        b = W_b[:, -1]
        z = X @ W.T + b
        P = softmax(z)

        # Cross-entropy
        y_binned = np.digitize(y, self.bin_edges[:-1]) - 1
        y_binned = np.clip(y_binned, 0, self.K - 1)
        y_onehot = np.eye(self.K)[y_binned]
        ce_loss = -np.mean(np.sum(y_onehot * np.log(P + 1e-10), axis=1))

        # Surprise (entropy of P)
        entropy = -np.sum(P * np.log(P + 1e-10), axis=1)

        return ce_loss + self.surprise_weight * np.mean(entropy)

