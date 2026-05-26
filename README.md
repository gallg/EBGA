# EBGA: Evolutionary-Based Genetic Algorithms

A gradient-free evolutionary optimization library for regression tasks, implementing the Genetic Descent algorithm.

## Overview

Genetic Descent (GD) is a novel optimization algorithm that merges evolutionary computation with continuous optimization techniques. It achieves gradient descent-like behavior without requiring gradients or backpropagation, making it suitable for non-differentiable or black-box functions.

## Features

- **Gradient-free optimization**: No need for differentiable loss functions
- **Population-based search**: Maintains a distribution over parameters for robust exploration
- **Continuous convergence**: Smooth, downward-trending optimization similar to gradient descent
- **Built-in regularization**: Includes elastic net regularization and surprise-based exploration
- **Scikit-learn compatible API**: Easy integration with existing ML workflows

## Installation

```bash
git clone <repository-url>
cd EBGA
pip install -r requirements.txt
```

## Quick Start

```python
from sklearn.datasets import make_regression
from EBGA.genetic_descent import GeneticDescentRegressor

# Generate data
X, y = make_regression(n_samples=1000, n_features=6, noise=0.1, random_state=42)

# Create and fit model
gd = GeneticDescentRegressor(
    n_bins=5,
    pop_size=50,
    max_iter=500,
    lr_μ=0.08,
    lr_σ=0.001,
    random_state=42
)

gd.fit(X, y)

# Make predictions
predictions = gd.predict(X)
```

## Documentation

Detailed documentation is available in the `docs/` folder:

- `genetic_descent.md` - Mathematical foundations and algorithm details
- `ga_info.md` - Understanding Genetic Descent as a gradient-free optimizer
- `ga_intuition.md` - Intuitive explanations and examples

## Project Structure

```
EBGA/
├── EBGA/
│   ├── genetic_descent.py    # Main regressor implementation
│   ├── losses/               # Loss functions
│   └── utils.py              # Utility functions
├── docs/                     # Documentation
├── tests/                    # Test suite
└── main.py                   # Example usage
```

## Key Concepts

- **Distribution Optimization**: Optimizes a distribution over parameters rather than parameters directly
- **μ (mu)**: Mean of the parameter distribution (best guess)
- **σ (sigma)**: Standard deviation controlling exploration
- **λ-Surprise**: Regularization term encouraging exploration
- **Population-based Gradient Estimation**: Uses Monte Carlo samples to estimate gradients

## When to Use

- Non-differentiable or discontinuous objective functions
- Simulation-based optimization problems
- Black-box models where derivatives are unavailable
- Problems with stochastic evaluation
- When you need robust exploration capabilities

## License

[Specify your license here]