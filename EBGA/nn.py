import numpy as np


class Sequential:
    
    def __init__(self, *layers):
        self.layers = list(layers)
        self.initialized = False
    
    def initialize(self, input_size):
        current_size = input_size
        for i, layer in enumerate(self.layers):
            layer.initialize(current_size)
            current_size = layer.output_size
        self.initialized = True
        self.input_size = input_size
        self.output_size = current_size
    
    def forward(self, x):
        if not self.initialized:
            self.initialize(x.shape[1])
        
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def get_all_parameters(self):
        all_params = []
        for layer in self.layers:
            all_params.append(layer.get_parameters())
        return np.concatenate(all_params)
    
    def set_all_parameters(self, params):
        offset = 0
        for layer in self.layers:
            param_count = layer.parameter_count()
            layer_params = params[offset:offset + param_count]
            layer.set_parameters(layer_params)
            offset += param_count
    
    def parameter_count(self):
        return sum(layer.parameter_count() for layer in self.layers)
    
    def get_layer_parameters(self, layer_idx):
        start = sum(l.parameter_count() for l in self.layers[:layer_idx])
        end = start + self.layers[layer_idx].parameter_count()
        return self.get_all_parameters()[start:end]
    
    def __len__(self):
        return len(self.layers)
    
    def __getitem__(self, idx):
        return self.layers[idx]
    
    def __repr__(self):
        return f"Sequential({[type(l).__name__ for l in self.layers]})"
    
    def set_training(self, training):
        for layer in self.layers:
            if hasattr(layer, 'set_training'):
                layer.set_training(training)
