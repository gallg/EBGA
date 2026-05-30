# Framework Overview

## What is EBGA?

**EBGA** (**E**nergy-**B**ased **G**enetic **A**lgorithms) is a Python framework for building and training neural networks using gradient-free evolutionary optimization. Unlike traditional deep learning frameworks that rely on backpropagation and gradient descent, EBGA uses compact genetic algorithms to optimize network parameters.

## Core Innovation

### The Problem with Traditional Approaches

Traditional neural network training has several limitations:

1. **Gradient Dependency**: Requires differentiable loss functions and smooth optimization landscapes
2. **Vanishing/Exploding Gradients**: Deep networks suffer from gradient instability
3. **Local Minima**: Gradient descent can get stuck in poor local optima
4. **Hyperparameter Sensitivity**: Learning rates, momentum, etc. require careful tuning
5. **Non-Differentiable Objectives**: Cannot handle loss functions with discontinuities or non-differentiable components

### The EBGA Solution

EBGA replaces gradient-based optimization with **distribution-based evolutionary computation**:

```
Traditional: θ_{t+1} = θ_t - η * ∇L(θ_t)
EBGA:       θ ~ N(μ, σ), update μ and σ based on fitness
```

Instead of maintaining a single parameter vector and updating it via gradients, EBGA maintains a **Gaussian distribution** over parameters and updates the distribution itself based on fitness evaluations.

## How It Works

### 1. Parameter Representation

Each parameter is represented as a Gaussian distribution:
- **μ (mean)**: The central value of the parameter
- **σ (standard deviation)**: The uncertainty/range of the parameter

### 2. Sampling and Evaluation

The algorithm samples candidate solutions from the distribution and evaluates their fitness (loss):

```python
# Sample from distribution
θ_candidate = μ + σ * ε, where ε ~ N(0, 1)

# Evaluate fitness
loss = loss_function(θ_candidate)
```

### 3. Distribution Update

Based on the fitness of samples, the algorithm updates μ and σ:
- **Good samples** (low loss): Pull μ toward them
- **Distribution width**: Adjust σ based on observed diversity

This is done via **natural gradient** updates, which account for the curvature of the parameter space.

## Key Features

### Gradient-Free Optimization
- No backpropagation required
- No gradient computation
- Works with any loss function, differentiable or not
- Naturally handles noisy objectives

### Compact Representation
- Maintains only μ and σ for each parameter (2N parameters instead of N)
- No population storage required
- Memory efficient

### Layer-Wise Training
- Trains layers sequentially until loss plateaus
- Avoids the "curse of dimensionality" in high-dimensional spaces
- Each layer learns meaningful features before the next

### Natural Regularization
- σ provides built-in uncertainty modeling
- Automatically adapts parameter ranges
- No need for L1/L2 regularization in most cases

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                        EBGA Framework                        │
├─────────────────┬─────────────────┬───────────────────┤
│    models.py     │    nn.py         │   optimizer.py     │
│  EBGARegressor   │  Sequential      │   CompactGenetic   │
│  EBGAClassifier  │                 │   Descent          │
├─────────────────┼─────────────────┼───────────────────┤
│    layers.py     │  activations.py  │    losses.py       │
│  Linear         │  ReLU, Sigmoid   │   MSE, MAE        │
│  Flatten        │  Tanh, Linear    │   CrossEntropy    │
│                 │  Softmax         │   BinaryCrossEnt. │
└─────────────────┴─────────────────┴───────────────────┘
```

## When to Use EBGA

### Use EBGA when:
- ✅ You need gradient-free optimization
- ✅ Your loss function is non-differentiable
- ✅ You're working with noisy or discontinuous objectives
- ✅ You want to explore alternative optimization paradigms
- ✅ You need built-in uncertainty modeling
- ✅ You're researching evolutionary computation

### Consider traditional frameworks when:
- You need maximum speed (EBGA is slower per iteration)
- You have well-behaved, differentiable loss functions
- You need state-of-the-art performance on standard benchmarks
- You require GPU acceleration

## Rationale

### Why Evolutionary Optimization?

Evolutionary algorithms have several inherent advantages:

1. **Global Search**: Can escape local minima that trap gradient methods
2. **No Gradient Requirements**: Work with any objective function
3. **Population-Based**: Naturally parallelizable
4. **Robust**: Less sensitive to initialization

### Why Compact?

Traditional genetic algorithms maintain a population of solutions, which is:
- Memory intensive (O(population_size × parameter_count))
- Computationally expensive (evaluate all population members)

The **compact** approach maintains only the distribution parameters (μ, σ), reducing memory from O(N × P) to O(2 × P) where N is population size and P is parameter count.

### Why Distribution-Based?

Maintaining a distribution over parameters provides:
- **Uncertainty modeling**: σ represents confidence in each parameter
- **Natural gradient**: Updates account for parameter space curvature
- **Adaptive exploration**: σ automatically adjusts based on fitness landscape

## Comparison to Other Approaches

| Feature | Gradient Descent | Traditional GA | EBGA |
|---------|-----------------|---------------|------|
| Gradient Required | ✅ Yes | ❌ No | ❌ No |
| Differentiable Loss | ✅ Yes | ❌ No | ❌ No |
| Local Minima | ❌ Gets stuck | ✅ Escapes | ✅ Escapes |
| Memory | O(P) | O(N×P) | O(2×P) |
| Population | ❌ Single point | ✅ Multiple | ❌ Single distribution |
| Speed | ✅ Fast | ❌ Slow | ⚠️ Medium |
| Parallelizable | ❌ Sequential | ✅ Yes | ✅ Yes |
| Non-differentiable | ❌ No | ✅ Yes | ✅ Yes |

## Mathematical Foundation

### Natural Gradient

The natural gradient accounts for the Riemannian structure of the parameter space:

```
∇_natural L = F^{-1} ∇L
```

where F is the Fisher information matrix, which for a Gaussian distribution is:

```
F = I (identity matrix for Gaussian)
```

Thus, the natural gradient update becomes:

```
μ_{t+1} = μ_t - η * F^{-1} ∇L ≈ μ_t - η * ∇L
```

### Distribution Update Rules

For a Gaussian distribution N(μ, σ), the update rules are:

```
# Mean update (toward better samples)
μ_{t+1} = μ_t - η_μ * E[(L - L̄) * z]

# Variance update (adapt exploration)
σ_{t+1} = σ_t * exp(η_σ * E[(L - L̄) * (z² - 1)])
```

where:
- L is the loss (fitness)
- L̄ is the average loss
- z = (θ - μ) / σ is the standardized parameter
- η_μ, η_σ are learning rates

This elegantly combines the benefits of evolutionary computation with the efficiency of gradient-like updates.
