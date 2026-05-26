# EBGA: Energy-Based Genetic Algorithms

**EBGA** (**E**nergy-**B**ased **G**enetic **A**lgorithms) is a gradient-free optimization framework that combines evolutionary computation with energy-based models and information theory principles.

## What's Inside

This package provides:

🔹 **GeneticDescentRegressor** - Novel evolutionary regression algorithm
🔹 **Surprise-aware loss** - Information-theoretic loss function with uncertainty awareness
🔹 **Evolutionary optimization** - Distribution-based parameter evolution
🔹 **Advanced regularization** - Elastic Net (L1 + L2) regularization support
🔹 **Benchmarking tools** - For evaluating on standard datasets

## Key Features

- ✅ Gradient-free optimization (no backpropagation)
- ✅ Handles non-differentiable loss functions
- ✅ Built-in uncertainty modeling
- ✅ Works with noisy or discontinuous objectives
- ✅ Naturally parallelizable implementation

## Example Usage

```python
from ebga import GeneticDescentRegressor
from sklearn.datasets import load_diabetes

# Load data
data = load_diabetes()
X, y = data.data, data.target

# Create and train model
model = GeneticDescentRegressor(l2_lambda=0.01)
model.fit(X, y)

# Make predictions
predictions = model.predict(X_test)
```

EBGA provides a fresh alternative to traditional gradient-based methods, particularly suited for complex problems where gradients are unavailable or unreliable.
