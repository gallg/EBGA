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

### Available Models

- **EBGARegressor** - For regression tasks
- **EBGAClassifier** - For classification tasks

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

## License

GNU General Public License v3.0 (GPL-3.0)
