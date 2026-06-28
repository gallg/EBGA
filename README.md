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

### Available Models

- **EBGARegressor** - For regression tasks with continuous output
- **EBGAClassifier** - For classification tasks with discrete output

### Available Optimizers

- **CompactEvoOptimizer** - Single gaussian distribution per parameter based on compact genetic algorithms.
- **MultiCandidateOptimizer** - Multiple candidate distributions per parameter for high-dimensional data.

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

## License

This project is licensed under the **GNU General Public License v3.0** (GPL-3.0).

See [LICENSE](LICENSE) for the full license text.
