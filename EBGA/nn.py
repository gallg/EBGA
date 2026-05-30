"""
Neural network module for EBGA framework.
Provides Sequential model for building networks layer by layer.
"""

import numpy as np
from EBGA.layers import Layer


class Sequential:
    """
    Sequential neural network that chains layers together.
    Similar to PyTorch's nn.Sequential.
    """
    
    def __init__(self, *layers):
        """
        Initialize sequential network with layers.
        
        Args:
            *layers: Variable number of Layer instances
        """
        self.layers = list(layers)
        self.initialized = False
    
    def initialize(self, input_size):
        """
        Initialize all layers with proper input sizes.
        
        Args:
            input_size: Size of input features
        """
        current_size = input_size
        for i, layer in enumerate(self.layers):
            layer.initialize(current_size)
            current_size = layer.output_size
        self.initialized = True
        self.input_size = input_size
        self.output_size = current_size
    
    def forward(self, x):
        """
        Forward pass through all layers.
        
        Args:
            x: Input data, shape (n_samples, input_size)
        
        Returns:
            Output after passing through all layers
        """
        if not self.initialized:
            self.initialize(x.shape[1])
        
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def get_all_parameters(self):
        """
        Get all parameters from all layers as a single flat array.
        
        Returns:
            Flat array of all parameters
        """
        all_params = []
        for layer in self.layers:
            all_params.append(layer.get_parameters())
        return np.concatenate(all_params)
    
    def set_all_parameters(self, params):
        """
        Set all parameters from a flat array.
        
        Args:
            params: Flat array of all parameters
        """
        offset = 0
        for layer in self.layers:
            param_count = layer.parameter_count()
            layer_params = params[offset:offset + param_count]
            layer.set_parameters(layer_params)
            offset += param_count
    
    def parameter_count(self):
        """
        Get total number of parameters.
        
        Returns:
            Total parameter count
        """
        return sum(layer.parameter_count() for layer in self.layers)
    
    def get_layer_parameters(self, layer_idx):
        """
        Get parameters for a specific layer.
        
        Args:
            layer_idx: Index of the layer
        
        Returns:
            Parameters for that layer
        """
        start = sum(l.parameter_count() for l in self.layers[:layer_idx])
        end = start + self.layers[layer_idx].parameter_count()
        return self.get_all_parameters()[start:end]
    
    def __len__(self):
        return len(self.layers)
    
    def __getitem__(self, idx):
        return self.layers[idx]
    
    def __repr__(self):
        return f"Sequential({[type(l).__name__ for l in self.layers]})"
