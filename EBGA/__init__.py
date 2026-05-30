from EBGA.models import EBGARegressor, EBGAClassifier
from EBGA.nn import Sequential
from EBGA.layers import Linear, Flatten
from EBGA.activations import (
    Activation, ReLU, Sigmoid, Tanh, Linear, Softmax,
    get_activation, relu, sigmoid, tanh, softmax
)
from EBGA.losses import (
    Loss, MSE, MAE, CrossEntropy, BinaryCrossEntropy,
    get_loss, mse_loss, mae_loss, cross_entropy_loss, bce_loss
)
from EBGA.optimizer import CompactEvoOptimizer

__version__ = "0.1.0"

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
    'get_activation',
    'relu',
    'sigmoid',
    'tanh',
    'softmax',
    
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
    
    # Optimizer
    'CompactEvoOptimizer',
]
