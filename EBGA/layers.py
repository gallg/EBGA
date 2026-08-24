import numpy as np
from EBGA.activations import get_activation


class Layer:

    def __init__(self):
        self.input_size = None
        self.output_size = None
        self.initialized = False

    def initialize(self, input_size, rng=None):
        raise NotImplementedError
    
    def forward(self, x):
        raise NotImplementedError
    
    def get_parameters(self):
        raise NotImplementedError
    
    def set_parameters(self, params):
        raise NotImplementedError
    
    def parameter_count(self):
        raise NotImplementedError


class Dense(Layer):
    
    def __init__(self, output_size, activation=None, use_bias=True):
        super().__init__()
        self.output_size = output_size
        # Handle activation: if it's a string, convert to activation instance; otherwise use as-is
        if activation is not None and isinstance(activation, str):
            self.activation = get_activation(activation)
        else:
            self.activation = activation
        self.use_bias = use_bias
    
    def initialize(self, input_size, rng=None):
        self.input_size = input_size
        if rng is None:
            rng = np.random.RandomState()
        limit = np.sqrt(6 / (self.input_size + self.output_size))
        self.W = rng.uniform(-limit, limit, (self.output_size, self.input_size))
        self.b = np.zeros(self.output_size) if self.use_bias else None
        self.initialized = True

    def forward(self, x):
        if not self.initialized:
            self.initialize(x.shape[1])
        
        output = x @ self.W.T
        if self.b is not None:
            output += self.b
        
        if self.activation:
            output = self.activation(output)
        
        return output
    
    def get_parameters(self):
        params = self.W.flatten()
        if self.b is not None:
            params = np.concatenate([params, self.b])
        return params
    
    def set_parameters(self, params):
        param_count = self.output_size * self.input_size
        self.W = params[:param_count].reshape(self.output_size, self.input_size)
        
        if self.b is not None:
            self.b = params[param_count:param_count + self.output_size]
    
    def parameter_count(self):
        count = self.output_size * self.input_size
        if self.use_bias:
            count += self.output_size
        return count


class Flatten(Layer):
    
    def __init__(self):
        super().__init__()
    
    def initialize(self, input_size, rng=None):
        self.input_size = input_size
        self.output_size = input_size
        self.initialized = True

    def forward(self, x):
        if not self.initialized:
            self.initialize(x.shape[1])
        return x
    
    def get_parameters(self):
        return np.array([])
    
    def set_parameters(self, params):
        pass
    
    def parameter_count(self):
        return 0
