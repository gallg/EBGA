"""
Layer classes for EBGA framework.
"""

import numpy as np
from EBGA.activations import get_activation, ACTIVATION_REGISTRY


class Layer:
    """Base class for all layers."""
    
    def __init__(self):
        self.input_size = None
        self.output_size = None
        self.initialized = False
    
    def initialize(self, input_size):
        """Initialize layer parameters given input size."""
        raise NotImplementedError
    
    def forward(self, x):
        """Forward pass."""
        raise NotImplementedError
    
    def get_parameters(self):
        """Get layer parameters as a flat array."""
        raise NotImplementedError
    
    def set_parameters(self, params):
        """Set layer parameters from a flat array."""
        raise NotImplementedError
    
    def parameter_count(self):
        """Get number of parameters in this layer."""
        raise NotImplementedError


class Linear(Layer):
    """Fully connected linear layer."""
    
    def __init__(self, output_size, activation=None, use_bias=True):
        super().__init__()
        self.output_size = output_size
        self.activation = get_activation(activation) if activation else None
        self.use_bias = use_bias
    
    def initialize(self, input_size):
        """Initialize weights and biases."""
        self.input_size = input_size
        
        # Xavier/Glorot initialization
        limit = np.sqrt(6 / (self.input_size + self.output_size))
        self.W = np.random.uniform(-limit, limit, (self.output_size, self.input_size))
        
        if self.use_bias:
            self.b = np.zeros(self.output_size)
        else:
            self.b = None
        
        self.initialized = True
    
    def forward(self, x):
        """Forward pass: y = activation(Wx + b)."""
        if not self.initialized:
            self.initialize(x.shape[1])
        
        output = x @ self.W.T
        if self.b is not None:
            output += self.b
        
        if self.activation:
            output = self.activation(output)
        
        return output
    
    def get_parameters(self):
        """Get parameters as flat array."""
        params = self.W.flatten()
        if self.b is not None:
            params = np.concatenate([params, self.b])
        return params
    
    def set_parameters(self, params):
        """Set parameters from flat array."""
        param_count = self.output_size * self.input_size
        self.W = params[:param_count].reshape(self.output_size, self.input_size)
        
        if self.b is not None:
            self.b = params[param_count:param_count + self.output_size]
    
    def parameter_count(self):
        """Get total parameter count."""
        count = self.output_size * self.input_size
        if self.use_bias:
            count += self.output_size
        return count


class Flatten(Layer):
    """Flatten layer for converting 2D to 1D."""
    
    def __init__(self):
        super().__init__()
    
    def initialize(self, input_size):
        self.input_size = input_size
        self.output_size = input_size
        self.initialized = True
    
    def forward(self, x):
        if not self.initialized:
            self.initialize(x.shape[1])
        return x  # Flatten doesn't change anything for 2D input
    
    def get_parameters(self):
        return np.array([])
    
    def set_parameters(self, params):
        pass
    
    def parameter_count(self):
        return 0


# Factory functions
def linear(output_size, activation=None, use_bias=True):
    """Create a linear layer."""
    return Linear(output_size, activation=activation, use_bias=use_bias)


def flatten():
    """Create a flatten layer."""
    return Flatten()
