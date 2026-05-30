# API Reference

This section contains complete documentation for all EBGA framework modules.

## Module Overview

| Module | Purpose | Key Classes/Functions |
|--------|---------|---------------------|
| [models](api/models.md) | High-level models | EBGARegressor, EBGAClassifier |
| [nn](api/nn.md) | Neural network container | Sequential |
| [layers](api/layers.md) | Layer implementations | Linear, Flatten |
| [activations](api/activations.md) | Activation functions | ReLU, Sigmoid, Tanh, Linear, Softmax |
| [losses](api/losses.md) | Loss functions | MSE, MAE, CrossEntropy, BinaryCrossEntropy |
| [optimizer](api/optimizer.md) | Optimization algorithm | CompactEvoOptimizer |

## Quick Import Guide

```python
# High-level models
from EBGA.models import EBGARegressor, EBGAClassifier

# Low-level components
from EBGA.nn import Sequential
from EBGA.layers import Linear, Flatten
from EBGA.activations import ReLU, Sigmoid, Tanh, Linear, Softmax
from EBGA.losses import MSE, MAE, CrossEntropy, BinaryCrossEntropy
from EBGA.optimizer import CompactEvoOptimizer

# Factory functions
from EBGA.activations import get_activation
from EBGA.losses import get_loss
```

## Usage Patterns

### Pattern 1: Quick Start (Recommended for Beginners)

```python
from EBGA.models import EBGARegressor

model = EBGARegressor(
    n_layers=2,
    h_dim=50,
    inner_activation='relu',
    output_activation='linear'
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### Pattern 2: Explicit Architecture

```python
from EBGA.models import EBGARegressor

model = EBGARegressor(
    layers=[(128, 'relu'), (64, 'relu'), (1, 'linear')],
    max_iter=10000
)
model.fit(X_train, y_train)
```

### Pattern 3: Custom Network (Advanced)

```python
from EBGA.nn import Sequential
from EBGA.layers import Linear
from EBGA.optimizer import CompactEvoOptimizer

# Build network
network = Sequential(
    Linear(64, activation='relu'),
    Linear(32, activation='tanh'),
    Linear(1, activation='linear')
)
network.initialize(input_size=X_train.shape[1])

# Create optimizer
optimizer = CompactEvoOptimizer(
    param_dim=network.parameter_count(),
    lr_mu=0.01,
    lr_sigma=0.001
)

# Custom training loop
def loss_func(params):
    network.set_all_parameters(params)
    return mae_loss(network.forward(X_train).flatten(), y_train)

optimizer.initialize()
for iteration in range(1000):
    optimizer.step(loss_func, iteration=iteration)
```

## See Also

- [models](api/models.md) - High-level model API
- [nn](api/nn.md) - Neural network container
- [layers](api/layers.md) - Layer implementations
- [activations](api/activations.md) - Activation functions
- [losses](api/losses.md) - Loss functions
- [optimizer](api/optimizer.md) - Optimizer implementation
