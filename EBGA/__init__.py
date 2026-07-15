from EBGA.models import EBGARegressor, EBGAClassifier
from EBGA.nn import Sequential
from EBGA.layers import Linear, Flatten
from EBGA.activations import (
    Activation, ReLU, Sigmoid, Tanh, Linear, Softmax,
    LeakyReLU, ELU, SELU, GELU, Swish,
    get_activation, relu, sigmoid, tanh, softmax,
    leaky_relu, elu, selu, gelu, swish
)
from EBGA.losses import (
    Loss, MSE, MAE, CrossEntropy, BinaryCrossEntropy,
    get_loss, mse_loss, mae_loss, cross_entropy_loss, bce_loss
)
from EBGA.optimizer import BaseEvoOptimizer, CompactEvoOptimizer, OptimizationResult
from EBGA.utils import save_model, load_model, save_network, load_network

__version__ = "0.1.2"

__all__ = [
    # Models
    'EBGARegressor',
    'EBGAClassifier',
    
    # Neural network
    'Sequential',
    
    # Layers
    'Layer',
    'Linear',
    'Flatten',
    
    # Activations
    'Activation',
    'ReLU',
    'Sigmoid',
    'Tanh',
    'Linear',
    'Softmax',
    'LeakyReLU',
    'ELU',
    'SELU',
    'GELU',
    'Swish',
    'get_activation',
    'relu',
    'sigmoid',
    'tanh',
    'softmax',
    'leaky_relu',
    'elu',
    'selu',
    'gelu',
    'swish',
    
    # Losses
    'Loss',
    'MSE',
    'MAE',
    'CrossEntropy',
    'BinaryCrossEntropy',
    'get_loss',
    'mse_loss',
    'mae_loss',
    'cross_entropy_loss',
    'bce_loss',
    
    # Optimizers
    'BaseEvoOptimizer',
    'CompactEvoOptimizer',
    'OptimizationResult',
    
    # Utils
    'save_model',
    'load_model',
    'save_network',
    'load_network',
]
