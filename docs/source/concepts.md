# Core Concepts

This document explains the fundamental concepts behind the EBGA framework.

## Compact Genetic Algorithms

### Traditional Genetic Algorithms

Traditional Genetic Algorithms (GAs) work as follows:

1. **Initialize** a population of candidate solutions
2. **Evaluate** fitness of all candidates
3. **Select** the best candidates
4. **Recombine** (crossover) selected candidates
5. **Mutate** the offspring
6. **Replace** the population
7. Repeat

**Problems:**
- Memory: O(population_size × parameter_count)
- Computation: Evaluate entire population each generation
- Scalability: Doesn't scale well to high dimensions

### Compact Genetic Algorithms

The **compact** approach (Estximation of Distribution Algorithm, EDA) takes a different approach:

1. **Model** the population as a probability distribution
2. **Sample** new candidates from the distribution
3. **Update** the distribution based on fitness
4. Repeat

**Advantages:**
- Memory: O(parameter_count) - only store distribution parameters
- Scalability: Works well in high dimensions
- Efficiency: No need to maintain a population

### The EBGA Approach

EBGA uses a **Gaussian distribution** to model parameters:

```
θ ~ N(μ, diag(σ²))
```

Each parameter θ_i has its own mean μ_i and standard deviation σ_i.

## Optimization Process

### Population Calibration (Natural Gradient Step)

Periodically, EBGA performs a **population calibration** step:

1. Sample `calibration_size` candidates from N(μ, σ)
2. Evaluate loss for each candidate
3. Compute natural gradient update for μ and σ

The update equations are:

```python
# Sample candidates
θ_j = μ + σ * ε_j, where ε_j ~ N(0, I)

# Compute losses
L_j = loss(θ_j)

# Natural gradient for mean
∇μ = (1/calibration_size) * Σ (L_j - L̄) * ε_j
μ_new = μ - lr_mu * ∇μ

# Natural gradient for variance
∇σ = (1/calibration_size) * Σ (L_j - L̄) * (ε_j² - 1)
σ_new = σ * exp(lr_sigma * ∇σ)
```

This is a **natural gradient** update because it uses the Fisher information matrix of the Gaussian distribution.

### Pairwise Update (Efficient Step)

Between calibrations, EBGA performs **pairwise updates**:

1. Sample two candidates: θ₁, θ₂ ~ N(μ, σ)
2. Evaluate both: L₁ = loss(θ₁), L₂ = loss(θ₂)
3. Determine winner (lower loss) and loser
4. Update distribution toward winner

```python
if L₁ < L₂:
    winner, loser = θ₁, θ₂
    winner_loss, loser_loss = L₁, L₂
else:
    winner, loser = θ₂, θ₁
    winner_loss, loser_loss = L₂, L₁

# Credit assignment
improvement = (loser_loss - winner_loss) / (loser_loss + winner_loss + eps)
update_strength = 1 + credit_factor * tanh(improvement)

# Update mean
μ += lr_mu * update_strength * (winner - μ)

# Update variance
σ *= exp(lr_sigma * update_strength * (|winner - loser| - σ))
```

## Layer-Wise Training

### The Problem

Training all layers simultaneously in high-dimensional spaces:
- The optimization landscape becomes extremely complex
- Hard for evolutionary algorithms to make progress
- Credit assignment becomes noisy

### The Solution

EBGA trains layers **sequentially**:

1. Train Layer 1 until loss plateaus
2. Freeze Layer 1, train Layer 2 until loss plateaus
3. Continue until last layer
4. Fine-tune all layers together

**Plateau Detection:**
- Monitor loss for each layer
- If loss doesn't improve for `layer_patience` iterations, move to next layer
- Uses the same loss-based early stopping as the overall training

### Benefits

1. **Reduced Complexity**: Each layer optimizes a simpler problem
2. **Better Feature Learning**: Each layer learns meaningful features before the next
3. **Faster Convergence**: Avoids the "curse of dimensionality"
4. **Stability**: Less prone to getting stuck in poor regions

## Neural Network Architecture

### Building Networks

EBGA provides a **modular** neural network builder:

```python
from EBGA.nn import Sequential
from EBGA.layers import Linear

# Build a network
network = Sequential(
    Linear(64, activation='relu'),
    Linear(32, activation='tanh'),
    Linear(1, activation='linear')
)

# Initialize
network.initialize(input_size=10)
```

### Layer Types

**Linear Layer:**
```python
Linear(output_size, activation='relu', use_bias=True)
```
- Fully connected layer
- Xavier/Glorot initialization
- Optional bias
- Configurable activation

**Flatten Layer:**
```python
Flatten()
```
- Reshapes input (useful for CNN-like architectures)

### Activations

All standard activations are supported:
- `ReLU`: max(0, x)
- `Sigmoid`: 1 / (1 + exp(-x))
- `Tanh`: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
- `Linear`: x (identity)
- `Softmax`: exp(x) / sum(exp(x)) (for classification)

### Loss Functions

**Regression:**
- `MSE`: Mean Squared Error
- `MAE`: Mean Absolute Error

**Classification:**
- `CrossEntropy`: Cross-entropy loss
- `BinaryCrossEntropy`: Binary cross-entropy

## High-Level Models

### EBGARegressor

For regression tasks:

```python
model = EBGARegressor(
    n_layers=2,              # Number of hidden layers
    h_dim=50,                # Size of hidden layers
    inner_activation='relu', # Activation for hidden layers
    output_activation='linear',
    loss='mae',              # Loss function
    max_iter=10000,          # Training iterations
    lr_mu=0.03,              # Learning rate for μ
    lr_sigma=0.03,           # Learning rate for σ
    layer_patience=50,       # Patience for layer-wise training
    random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

### EBGAClassifier

For classification tasks:

```python
model = EBGAClassifier(
    n_layers=2,
    h_dim=50,
    inner_activation='relu',
    output_activation='softmax',
    n_classes=10,
    loss='cross_entropy',
    max_iter=2000,
    lr_mu=0.01,
    lr_sigma=0.01,
    layer_patience=30,
    random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)
```

## Parameter Space and Optimization

### Parameter Count

For a network with layers [d₁, d₂, ..., dₙ] and input dimension d_in:

```
Total parameters = Σ (d_i * d_{i-1} + d_i)
                 = Σ (weights + bias for each layer)
```

Example: Network with layers [64, 32, 1] and input=10:
- Layer 1: 64 × 10 weights + 64 biases = 704 parameters
- Layer 2: 32 × 64 weights + 32 biases = 2080 parameters
- Layer 3: 1 × 32 weights + 1 bias = 33 parameters
- **Total: 2817 parameters**

EBGA maintains 2 × 2817 = 5634 parameters (μ and σ for each).

### Search Space

With σ initialization of 0.1 and bounds [0.001, 1.0], each parameter explores:

```
Initial search range: μ ± 3σ ≈ μ ± 0.3
After adaptation: σ adjusts based on fitness landscape
```

The algorithm automatically adapts σ to match the scale of each parameter.

## Evolution vs. Gradients

### Traditional Gradient Descent

```
# Gradient update
θ_{t+1} = θ_t - η * ∇L(θ_t)

# Problems:
# - Requires ∇L(θ_t) to exist
# - Can get stuck in local minima
# - Sensitive to η (learning rate)
# - Vanishing/exploding gradients
```

### EBGA Evolution

```
# Distribution update
μ_{t+1} = μ_t - η_μ * E[(L - L̄) * (θ - μ) / σ]
σ_{t+1} = σ_t * exp(η_σ * E[(L - L̄) * ((θ - μ) / σ)² - 1)])

# Advantages:
# - No gradients required
# - Naturally escapes local minima
# - σ adapts automatically
# - No vanishing/exploding issues
```

### Key Differences

| Aspect | Gradient Descent | EBGA |
|--------|-----------------|------|
| Update direction | Negative gradient | Natural gradient from samples |
| Step size | Fixed (η) | Adaptive (σ) |
| Exploration | None (deterministic) | Built-in (sampling) |
| Local minima | Gets stuck | Can escape |
| Parallelization | Limited | Natural |

## Comparison to Other Evolutionary Approaches

### Differential Evolution (DE)

DE uses:
```
θ_new = θ_a + F * (θ_b - θ_c)
```
- Requires population of size N
- Memory: O(N × P)
- No distribution modeling
- Less efficient in high dimensions

### Particle Swarm Optimization (PSO)

PSO uses:
```
v_{t+1} = w * v_t + c1 * r1 * (p_best - θ_t) + c2 * r2 * (g_best - θ_t)
θ_{t+1} = θ_t + v_{t+1}
```
- Requires velocity tracking
- Memory: O(N × P)
- No uncertainty modeling

### Covariance Matrix Adaptation (CMA-ES)

CMA-ES uses:
```
θ ~ N(μ, C)
C_{t+1} = (1-c) * C_t + c * adaptive_update
```
- Maintains full covariance matrix
- Memory: O(P²)
- Excellent for small problems
- Doesn't scale to high dimensions

### EBGA Advantages

| Approach | Memory | Scalability | Gradient-Free | Distribution |
|----------|--------|-------------|--------------|-------------|
| DE | O(N×P) | Medium | ✅ Yes | ❌ No |
| PSO | O(N×P) | Medium | ✅ Yes | ❌ No |
| CMA-ES | O(P²) | Low | ✅ Yes | ✅ Yes |
| EBGA | O(2×P) | **High** | ✅ Yes | ✅ Yes (diagonal) |

EBGA offers the best **scalability** while maintaining distribution modeling and being completely gradient-free.
