"""
Activation functions for EBGA framework.
"""

import numpy as np


class Activation:
    """Base class for activation functions."""
    
    def __call__(self, x):
        """Apply activation function."""
        return self.forward(x)
    
    def forward(self, x):
        """Forward pass. To be implemented by subclasses."""
        raise NotImplementedError
    
    def backward(self, x):
        """Backward pass (derivative). Not used in gradient-free framework but provided for completeness."""
        raise NotImplementedError


class ReLU(Activation):
    """Rectified Linear Unit activation."""
    
    def forward(self, x):
        return np.maximum(0, x)
    
    def backward(self, x):
        return (x > 0).astype(float)


class Sigmoid(Activation):
    """Sigmoid activation."""
    
    def forward(self, x):
        return 1 / (1 + np.exp(-x))
    
    def backward(self, x):
        s = self.forward(x)
        return s * (1 - s)


class Tanh(Activation):
    """Hyperbolic tangent activation."""
    
    def forward(self, x):
        return np.tanh(x)
    
    def backward(self, x):
        return 1 - np.tanh(x)**2


class Linear(Activation):
    """Linear (identity) activation."""
    
    def forward(self, x):
        return x
    
    def backward(self, x):
        return np.ones_like(x)


class Softmax(Activation):
    """Softmax activation for multi-class classification."""
    
    def forward(self, x, axis=-1):
        # Subtract max for numerical stability
        x_exp = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return x_exp / np.sum(x_exp, axis=axis, keepdims=True)
    
    def backward(self, x):
        # Not typically used in gradient-free framework
        raise NotImplementedError("Softmax backward not implemented")


# Factory for creating activation instances
ACTIVATION_REGISTRY = {
    'relu': ReLU,
    'sigmoid': Sigmoid,
    'tanh': Tanh,
    'linear': Linear,
    'softmax': Softmax,
}


def get_activation(name):
    """Get activation function by name."""
    if name not in ACTIVATION_REGISTRY:
        raise ValueError(f"Unknown activation: {name}. Available: {list(ACTIVATION_REGISTRY.keys())}")
    return ACTIVATION_REGISTRY[name]()


# Functional API for backward compatibility
def relu(x):
    return ReLU().forward(x)


def sigmoid(x):
    return Sigmoid().forward(x)


def tanh(x):
    return Tanh().forward(x)


def linear(x):
    return Linear().forward(x)


def softmax(x, axis=-1):
    return Softmax().forward(x, axis=axis)
