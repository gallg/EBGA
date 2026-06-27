# EBGA: Evolutionary-Based Gradient-Free Architecture

**EBGA** (**E**volutionary-**B**ased **G**radient-free **A**rchitecture) is a Python framework for training neural networks without gradients. It uses evolutionary computation to optimize parameters directly, making it suitable for problems where traditional gradient-based methods are ineffective or inappropriate.

## Overview

EBGA provides a simple, scikit-learn compatible interface for both regression and classification tasks, while maintaining full modularity for custom network architectures.

### Key Features

- **Gradient-free optimization** - No backpropagation, no gradient calculations
- **Modular architecture** - Build networks layer by layer
- **Scikit-learn compatible** - Familiar fit/predict interface
- **Layer-wise training** - Sequential training with plateau detection
- **Handles non-differentiable losses** - Works with any loss function
- **Built-in regularization** - L2 weight decay and dropout support

## Quick Start

### Installation

```bash
pip install -e .
```

Requirements: Python 3.8+, numpy, scikit-learn, scipy

### Regression Example

```python
from EBGA.models import EBGARegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

# Load data
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create and train model
model = EBGARegressor(
    layers=[(50, 'relu'), (30, 'relu')],
    max_iter=1000,
    random_state=42
)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
print(f"R² Score: {model.score(X_test, y_test):.4f}")
```

### Classification Example

```python
from EBGA.models import EBGAClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create and train model
model = EBGAClassifier(
    layers=[(10, 'relu'), (10, 'relu'), (3, 'softmax')],
    n_classes=3,
    max_iter=2000,
    random_state=42
)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
print(f"Accuracy: {model.score(X_test, y_test):.4f}")
```

## Architecture

### Core Components

- **CompactEvoOptimizer**: Distribution-based evolutionary optimizer
- **Sequential**: Modular neural network container
- **Layers**: Linear, Flatten
- **Activations**: ReLU, Sigmoid, Tanh, Linear, Softmax
- **Losses**: MSE, MAE, CrossEntropy, BinaryCrossEntropy
- **Utils**: Checkpointing functions (save_model, load_model, save_network, load_network)

### Model Building

EBGA supports both high-level models and custom network construction:

```python
from EBGA.nn import Sequential
from EBGA.layers import Linear
from EBGA.activations import get_activation
from EBGA.optimizer import CompactEvoOptimizer

# Build custom network
network = Sequential(
    Linear(64, activation='relu'),
    Linear(32, activation='tanh'),
    Linear(1, activation='linear')
)

# Create optimizer
optimizer = CompactEvoOptimizer(
    param_dim=network.parameter_count(),
    lr_mu=0.01,
    lr_sigma=0.001
)

# Initialize and train
network.initialize(input_size=X_train.shape[1])
optimizer.initialize()

for iteration in range(1000):
    loss = optimizer.step(loss_func, iteration=iteration)
```

## Hyperparameters

### Common Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `layers` | None | Explicit layer specification |
| `n_layers` | 1 | Number of hidden layers |
| `h_dim` | 50 | Size of hidden layers |
| `inner_activation` | 'relu' | Activation for hidden layers |
| `output_activation` | 'linear' / 'softmax' | Output activation |
| `max_iter` | 10000 (reg) / 500 (cls) | Maximum iterations |
| `lr_mu` | 0.03 (reg) / 0.05 (cls) | Learning rate for mean |
| `lr_sigma` | 0.03 (reg) / 0.005 (cls) | Learning rate for sigma |
| `sigma_min` | 0.001 | Minimum sigma |
| `sigma_max` | 1.0 | Maximum sigma |
| `calibration_size` | 30 (reg) / 20 (cls) | Population size |
| `calibration_interval` | 50 (reg) / 25 (cls) | Calibration frequency |

### Additional Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sigma_regularization` | 0.0 | Sigma diversity regularization |
| `normalize_output` | False (reg) | Normalize output to 0-1 range |
| `early_stopping` | True | Enable early stopping |
| `patience` | 100 (reg) / 20 (cls) | Early stopping patience |
| `layer_patience` | 50 | Layer-wise plateau detection patience |
| `credit_factor` | 2.0 | Credit assignment strength |

**Note:** Some parameters like `l2_penalty` and `dropout_rate` mentioned in earlier versions are not currently implemented in the core models.

## License

GNU General Public License v3.0 (GPL-3.0)
