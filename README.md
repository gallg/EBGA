# EBGA: Evolutionary Based Gradient Alignment

**EBGA** (**E**volutionary **B**ased **G**radient **A**lignment) is a machine learning framework that provides an alternative to classical gradient-based methods. It uses evolutionary computation with natural gradient updates on distribution parameters, enabling training of neural networks without computing objective function gradients.

## Overview

EBGA offers a familiar API to create neural network-based models, similar to known deep learning frameworks, while implementing optimization through evolutionary strategies with natural gradients. The framework also provides scikit-learn compatible interfaces for both regression and classification tasks.

### Key Characteristics

- **Natural gradient optimization** - Uses natural gradients with respect to distribution parameters (μ, σ), not objective function gradients
- **Scikit-learn compatible** - Familiar `fit`, `predict`, and `score` interface
- **Modular architecture** - Build networks with configurable layers and activations
- **Distribution-based optimization** - Parameters optimized through evolving Gaussian distributions
- **Handles non-differentiable losses** - Works with any loss function
- **Flexible training modes** - Supports both layer-wise and direct (all-layers-together) training
- **Built-in hyperparameter tuning** - Evolutionary search for optimal model configuration

### Available Models

- **EBGARegressor** - For regression tasks with continuous output
- **EBGAClassifier** - For classification tasks with discrete output
- **EvoHyperoptSearch** - For hyperparameter tuning with evolutionary search

### Available Optimizers

- **CompactEvoOptimizer** - Single Gaussian distribution with diagonal covariance for parameter optimization using natural gradient updates. Features include momentum, trust region constraints, and adaptive calibration.

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd EBGA

# Install in development mode
pip install -e .
```

### Dependencies

- Python 3.10+
- numpy
- scikit-learn

## Quick Start

```python
from EBGA.models import EBGARegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

# Load data
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create and train model
model = EBGARegressor(
    layers=[(50, 'relu'), (1, 'linear')],
    max_iter=1000,
    random_state=42
)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
print(f"R² Score: {model.score(X_test, y_test):.4f}")
```

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

This project is licensed under the **GNU General Public License v3.0** (GPL-3.0).

See [LICENSE](LICENSE) for the full license text.
