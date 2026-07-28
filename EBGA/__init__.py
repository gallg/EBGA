from EBGA.models import EBGARegressor, EBGAClassifier
from EBGA.nn import Sequential
from EBGA.layers import Dense, Flatten, Layer
from EBGA.dataset import Dataset
from EBGA.activations import (
    Activation, ReLU, Sigmoid, Tanh, Linear, Softmax,
    LeakyReLU, ELU, SELU, GELU, SiLU,
    get_activation,
)
from EBGA.losses import (
    Loss, MSE, MAE, CrossEntropy, BinaryCrossEntropy,
    get_loss,
)
from EBGA.optimizer import BaseEvoOptimizer, CompactEvoOptimizer
from EBGA.parallel import ParallelEvaluator
from EBGA.utils import save_model, load_model, save_network, load_network

__version__ = "0.2.3"

__all__ = [
    # Models
    'EBGARegressor',
    'EBGAClassifier',
    
    # Neural network
    'Sequential',
    
    # Dataset
    'Dataset',
    
    # Layers
    'Layer',
    'Dense',
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
    'SiLU',
    'get_activation',
    
    # Losses
    'Loss',
    'MSE',
    'MAE',
    'CrossEntropy',
    'BinaryCrossEntropy',
    'get_loss',
    
    # Optimizers
    'BaseEvoOptimizer',
    'CompactEvoOptimizer',
    
    # Parallel
    'ParallelEvaluator',
    
    # Utils
    'save_model',
    'load_model',
    'save_network',
    'load_network',
]
