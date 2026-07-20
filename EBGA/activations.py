import numpy as np


class Activation:
    
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        raise NotImplementedError


class ReLU(Activation):
    
    def forward(self, x):
        return np.maximum(0, x)


class Sigmoid(Activation):
    
    def forward(self, x):
        return 1 / (1 + np.exp(-x))


class Tanh(Activation):
    
    def forward(self, x):
        return np.tanh(x)


class Linear(Activation):
    
    def forward(self, x):
        return x


class Softmax(Activation):
    
    def forward(self, x, axis=-1):
        # Subtract max for numerical stability
        x_exp = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return x_exp / np.sum(x_exp, axis=axis, keepdims=True)


class LeakyReLU(Activation):
    """
    Leaky ReLU activation function.
    
    f(x) = x, if x > 0
    f(x) = alpha * x, if x <= 0
    
    Default alpha = 0.01
    """
    
    def __init__(self, alpha=0.01):
        self.alpha = alpha
    
    def forward(self, x):
        return np.where(x > 0, x, self.alpha * x)


class ELU(Activation):
    """
    Exponential Linear Unit activation function.
    
    f(x) = x, if x > 0
    f(x) = alpha * (exp(x) - 1), if x <= 0
    
    Default alpha = 1.0
    """
    
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def forward(self, x):
        return np.where(x > 0, x, self.alpha * (np.exp(x) - 1))


class SELU(Activation):
    """
    Scaled Exponential Linear Unit activation function.
    
    f(x) = lambda * x, if x > 0
    f(x) = lambda * alpha * exp(x), if x <= 0
    
    Default parameters from original paper:
    - lambda = 1.0507
    - alpha = 1.67326
    """
    
    def __init__(self, lambda_val=1.0507, alpha=1.67326):
        self.lambda_val = lambda_val
        self.alpha = alpha
    
    def forward(self, x):
        return np.where(x > 0, 
                       self.lambda_val * x, 
                       self.lambda_val * self.alpha * (np.exp(x) - 1))


class GELU(Activation):
    """
    Gaussian Error Linear Unit activation function.
    
    f(x) = x * Φ(x)
    where Φ(x) is the Gaussian CDF, approximated as:
    Φ(x) ≈ 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    
    This is the approximation commonly used in practice.
    """
    
    def forward(self, x):
        # Approximation of Gaussian CDF
        inner = np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)
        return 0.5 * x * (1 + np.tanh(inner))


class Swish(Activation):
    """
    Swish activation function.
    
    f(x) = x * sigmoid(x) = x / (1 + exp(-x))
    """
    
    def forward(self, x):
        return x / (1 + np.exp(-x))


# Factory for creating activation instances
ACTIVATION_REGISTRY = {
    'relu': ReLU,
    'sigmoid': Sigmoid,
    'tanh': Tanh,
    'linear': Linear,
    'softmax': Softmax,
    'leaky_relu': LeakyReLU,
    'elu': ELU,
    'selu': SELU,
    'gelu': GELU,
    'swish': Swish,
}


def get_activation(name):
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


def leaky_relu(x, alpha=0.01):
    return LeakyReLU(alpha=alpha).forward(x)


def elu(x, alpha=1.0):
    return ELU(alpha=alpha).forward(x)


def selu(x, lambda_val=1.0507, alpha=1.67326):
    return SELU(lambda_val=lambda_val, alpha=alpha).forward(x)


def gelu(x):
    return GELU().forward(x)


def swish(x):
    return Swish().forward(x)
