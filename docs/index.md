# EBGA: Evolutionary Based Gradient Alignment

**EBGA** (**E**volutionary **B**ased **G**radient **A**lignment) is a Python framework for training neural networks without computing objective function gradients. It uses evolutionary computation with natural gradient updates on distribution parameters (μ, σ) to provide an alternative to classical gradient-based methods.

## Overview

EBGA provides a scikit-learn compatible interface for both regression and classification tasks, while maintaining full modularity for custom network architectures.

### Key Characteristics

- **Natural gradient optimization** - Uses natural gradients with respect to distribution parameters, not objective function gradients
- **Scikit-learn compatible** - Familiar fit/predict interface
- **Modular architecture** - Build networks layer by layer
- **Distribution-based optimization** - Parameters optimized through Gaussian distributions
- **Handles non-differentiable losses** - Works with any loss function
- **Flexible training modes** - Layer-wise or direct training
- **Built-in hyperparameter tuning** - Evolutionary search for optimal model configuration

### Available Models

- **EBGARegressor** - For regression tasks
- **EBGAClassifier** - For classification tasks
- **EvoHyperoptSearch** - For hyperparameter tuning with evolutionary search

### Available Optimizers

- **CompactEvoOptimizer** - Single distribution per parameter
- **MultiCandidateOptimizer** - Multiple candidate distributions per parameter

## Installation

```bash
pip install -e .
```

Requirements: Python 3.10+, numpy, scikit-learn

## Quick Start

### Regression

```python
from EBGA.models import EBGARegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = EBGARegressor(
    layers=[(50, 'relu'), (1, 'linear')],
    max_iter=1000,
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"R² Score: {model.score(X_test, y_test):.4f}")
```

### Classification

```python
from EBGA.models import EBGAClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = EBGAClassifier(
    layers=[(10, 'relu'), (10, 'relu'), (3, 'softmax')],
    n_classes=3,
    max_iter=2000,
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Accuracy: {model.score(X_test, y_test):.4f}")
```

## Building Custom Neural Networks

EBGA also provides low-level components for building custom neural networks. 
**Note: The current implementation supports only Dense (Linear) layers.**

```python
from EBGA.nn import Sequential
from EBGA.layers import Linear
from EBGA.optimizer import CompactEvoOptimizer
import numpy as np

# Build a custom network with Dense layers only
network = Sequential(
    Linear(64, activation='relu'),    # Dense layer: 64 units, ReLU activation
    Linear(32, activation='relu'),    # Dense layer: 32 units, ReLU activation
    Linear(1, activation='linear')     # Output layer: 1 unit, linear activation
)

# Initialize the network with input size
network.initialize(input_size=10)  # 10 input features

# Manually train the network (for advanced users)
optimizer = CompactEvoOptimizer(
    param_dim=network.parameter_count(),
    lr_mu=0.05,
    lr_sigma=0.005
)

# Get initial parameters
params = network.get_all_parameters()
optimizer.initialize(params)

# Define loss function
def loss_func(params):
    network.set_all_parameters(params)
    y_pred = network.forward(X_train)
    return np.mean((y_pred - y_train) ** 2)  # MSE

# Train
for iteration in range(1000):
    optimizer.step(loss_func, iteration=iteration)

# Get trained parameters
best_params = optimizer.get_parameters()
network.set_all_parameters(best_params)

# Make predictions
y_pred = network.forward(X_test)
```

**Important:** The current EBGA implementation only supports `Linear` (Dense) layers. Convolutional, recurrent, and other layer types are not yet available.

## Hyperparameter Tuning

EBGA models are hyperparameter-heavy, requiring optimization across multiple dimensions. The framework provides built-in hyperparameter search functionality:

```python
from EBGA.models import EBGARegressor
from EBGA.search import EvoHyperoptSearch
from sklearn.model_selection import train_test_split

# Load data
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Define search space
param_distributions = {
    'lr_mu': (0.0001, 0.01, 'log-uniform'),
    'lr_sigma': (0.00001, 0.001, 'log-uniform'),
    'max_iter': [1000, 5000, 10000],
    'use_layerwise': [True, False]
}

# Run hyperparameter search
search = EvoHyperoptSearch(
    estimator=EBGARegressor(layers=[(50, 'relu'), (1, 'linear')]),
    param_distributions=param_distributions,
    n_iter=10,
    cv=3,
    search_strategy='evolutionary',  # or 'random', 'hybrid'
    random_state=42
)
search.fit(X_train, y_train)

# Use best model
print(f"Best parameters: {search.best_params_}")
print(f"Best score: {search.best_score_:.4f}")
y_pred = search.predict(X_test)
```

**Search Strategies:**
- `'random'`: Simple random search (baseline)
- `'evolutionary'`: Evolutionary search with selection, crossover, mutation
- `'hybrid'`: Combines random exploration with evolutionary refinement

## License

GNU General Public License v3.0 (GPL-3.0)
