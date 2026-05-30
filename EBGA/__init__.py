"""
EBGA: Evolutionary-Based Gradient-free Architecture

A framework for building and training neural networks using compact evolutionary algorithms.
Completely gradient-free, using distribution-based optimization.

Example:
    >>> from EBGA.models import EBGARegressor, EBGAClassifier
    >>> from EBGA.nn import Sequential
    >>> from EBGA.layers import Linear
    >>> from EBGA.activations import get_activation
    >>> from EBGA.losses import get_loss
    >>>
    >>> # Build a custom network
    >>> network = Sequential(
    ...     Linear(50, activation='relu'),
    ...     Linear(1, activation='linear')
    ... )
    >>>
    >>> # Or use pre-built models
    >>> regressor = EBGARegressor(
    ...     layers=[(50, 'sigmoid'), (1, 'linear')],
    ...     max_iter=1000
    ... )
    >>> regressor.fit(X_train, y_train)
    >>> predictions = regressor.predict(X_test)
"""

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
