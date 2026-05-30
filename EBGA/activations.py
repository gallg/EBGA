import numpy as np


class Activation:
    
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        raise NotImplementedError
    
    def backward(self, x):
        raise NotImplementedError


class ReLU(Activation):
    
    def forward(self, x):
        return np.maximum(0, x)
    
    def backward(self, x):
        return (x > 0).astype(float)


class Sigmoid(Activation):
    
    def forward(self, x):
        return 1 / (1 + np.exp(-x))
    
    def backward(self, x):
        s = self.forward(x)
        return s * (1 - s)


class Tanh(Activation):
    
    def forward(self, x):
        return np.tanh(x)
    
    def backward(self, x):
        return 1 - np.tanh(x)**2


class Linear(Activation):
    
    def forward(self, x):
        return x
    
    def backward(self, x):
        return np.ones_like(x)


class Softmax(Activation):
    
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
