# EBGA: Evolutionary-Based Generative Adaptation

**EBGA** (**E**volutionary-**B**ased **G**enerative **A**daptation) is a Python framework for training neural networks without computing objective function gradients. It uses evolutionary computation with softmax-weighted recombination (NES-style) to provide an alternative to classical gradient-based methods.

## Overview

EBGA provides a scikit-learn compatible interface for both regression and classification tasks, while maintaining full modularity for custom network architectures.

### Key Characteristics

- **Softmax-weighted recombination** - Optimizes parameters using population-based evolutionary search with softmax-weighted selection
- **Scikit-learn compatible** - Familiar fit/predict interface
- **Modular architecture** - Build networks layer by layer
- **Distribution-based optimization** - Parameters optimized through Gaussian distributions
- **Handles non-differentiable losses** - Works with any loss function
- **Flexible training modes** - Layer-wise or direct training
- **Mini-batch training** - Train on large datasets with configurable batch sizes
- **Multi-core parallelism** - Parallel candidate evaluation for custom networks via `ParallelEvaluator`
- **Rich activation functions** - 10 built-in activation functions including ReLU, LeakyReLU, ELU, SELU, GELU, Swish


### Available Models

- **EBGARegressor** - For regression tasks
- **EBGAClassifier** - For classification tasks

### Available Optimizers

- **CompactEvoOptimizer** - Uses a single Gaussian distribution with diagonal covariance for parameter optimization via softmax-weighted recombination, with momentum, trust region constraints, and adaptive sigma.

---

## Installation

```bash
pip install -e .
```

Requirements: Python 3.10+, numpy, scikit-learn

---

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

### Mini-Batch Training

For large datasets with hundreds of thousands of samples, use the `batch_size` parameter:

```python
from EBGA.models import EBGARegressor

# Train with mini-batches of 64 samples
model = EBGARegressor(
    layers=[(100, 'relu'), (50, 'relu'), (1, 'linear')],
    batch_size=64,
    max_iter=1000,
    random_state=42
)
model.fit(X_train, y_train)
```

The framework automatically creates batches and cycles through them during training. If the last batch would be smaller than `batch_size`, it's merged with the previous batch.

### Activation Functions

EBGA provides 10 built-in activation functions. For classification tasks, **use `sigmoid` or `softmax` in the output layer** and other activations in hidden layers:

```python
# Recommended for classification
model = EBGAClassifier(
    layers=[
        (64, 'leaky_relu'),
        (32, 'gelu'),
        (10, 'swish'),
        (3, 'softmax')
    ],
    n_classes=3
)

# Recommended for regression
model = EBGARegressor(
    layers=[
        (64, 'selu'),
        (32, 'elu'),
        (1, 'linear')
    ]
)
```

A full list of available activation functions is provided in the [API Reference](api.md).

## Building Custom Neural Networks

EBGA exposes low-level components for building custom neural networks from scratch.


### Basic Custom Network

```python
from EBGA.nn import Sequential
from EBGA.layers import Dense
from EBGA.optimizer import CompactEvoOptimizer
from EBGA.parallel import ParallelEvaluator
import numpy as np

# Build a custom network
network = Sequential(
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
)

# Initialize the network with input size
network.initialize(input_size=10)

# Create optimizer
optimizer = CompactEvoOptimizer(
    param_dim=network.parameter_count(),
    lr_mu=0.05,
    lr_sigma=0.005
)
optimizer.initialize(network.get_all_parameters())

# Create parallel evaluator for multi-core candidate evaluation
evaluator = ParallelEvaluator(
    network, X_train, y_train,
    loss='mse',
    n_jobs=4,          # number of worker processes
    batch_size=None,   # None = full dataset per step
)

# Train with parallel candidate evaluation
with evaluator:
    for iteration in range(1000):
        optimizer.step(iteration=iteration, evaluate_map=evaluator.evaluate_map)

# Get trained parameters
best_params = optimizer.get_parameters()
network.set_all_parameters(best_params)

# Make predictions
y_pred = network.forward(X_test)
```

For sequential evaluation (single-core), use `n_jobs=1` or omit the `ParallelEvaluator` entirely.


### Scale-Aware Initialization

For improved convergence, pass training targets to `initialize()`:

```python
# Initialize with scale-aware output
network = Sequential(Dense(64, activation='relu'), Dense(1, activation='linear'))
network.initialize(input_size=10, scale_aware=y_train)
```


### Mini-Batch Training with Dataset

For mini-batch training with custom networks, use the `Dataset` class:

```python
from EBGA.dataset import Dataset
from EBGA.nn import Sequential
from EBGA.layers import Dense
from EBGA.optimizer import CompactEvoOptimizer

# Create dataset with batching
dataset = Dataset(X_train, y_train, batch_size=32, shuffle=True)

# Build network
network = Sequential(Dense(64, activation='relu'), Dense(1, activation='linear'))
network.initialize(input_size=X_train.shape[1], scale_aware=y_train)

# Create optimizer
optimizer = CompactEvoOptimizer(param_dim=network.parameter_count())
optimizer.initialize(network.get_all_parameters())

# Training loop with batches
for epoch in range(100):
    for X_batch, y_batch in dataset.batches():
        def batch_loss(params):
            network.set_all_parameters(params)
            y_pred = network.forward(X_batch).flatten()
            return np.mean((y_pred - y_batch) ** 2)
        optimizer.step(batch_loss)
```

**Important:** The current EBGA implementation only supports `Dense` layers. Convolutional, recurrent, and other layer types are not yet available.


## License

GNU General Public License v3.0 (GPL-3.0)
