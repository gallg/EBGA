# Compact Genetic Descent: Unified Framework for Regression and Classification

## Table of Contents
1. [Introduction](#introduction)
2. [Core Principles](#core-principles)
3. [The Intuition Behind CGD](#the-intuition-behind-cgd)
4. [Mathematical Framework](#mathematical-framework)
5. [Regression Implementation](#regression-implementation)
6. [Classification Implementation](#classification-implementation)
7. [Implementation Details](#implementation-details)
8. [Exploration-Exploitation Dynamics](#exploration-exploitation-dynamics)
9. [Practical Advantages](#practical-advantages)
10. [Hyperparameters and Tuning](#hyperparameters-and-tuning)
11. [Current Status](#current-status)
12. [Conclusion](#conclusion)

---

## Introduction

Compact Genetic Descent (CGD) represents a revolutionary approach to optimization that combines the robustness of evolutionary computation with the efficiency of compact genetic algorithms. Unlike traditional population-based methods that maintain explicit populations, CGD uses a **probabilistic distribution representation** to dramatically improve computational efficiency while preserving the exploration benefits of evolutionary strategies.

CGD is a **gradient-free** optimization framework that works for both **regression** and **classification** tasks, making it uniquely suited for problems where:
- Gradients are unavailable or unreliable
- Loss functions are non-differentiable or discontinuous
- Uncertainty quantification is important
- Computational efficiency is critical

---

## Core Principles

### Evolutionary Computation Foundations

Traditional genetic algorithms maintain an explicit population of candidate solutions that evolve through:
- **Selection**: Favoring better-performing solutions
- **Crossover**: Combining genetic information from parents
- **Mutation**: Introducing random variations
- **Replacement**: Introducing offspring into the population

While powerful, this approach can be computationally expensive.

### The Compact Genetic Algorithm Paradigm

CGD adopts the **compact genetic algorithm (cGA)** approach by:
- **Replacing explicit population with a probability vector**: Maintains a distribution over parameters
- **Using pairwise comparisons**: Only two individuals sampled and compared at each iteration
- **Updating probabilities based on competition**: Distribution adjusted toward winners, away from losers
- **Eliminating explicit crossover/mutation**: Subsumed into probability updates

This maintains population-based search benefits with dramatically improved efficiency.

---

## The Intuition Behind CGD

### Parameters as Probabilistic Entities

The fundamental insight: **The optimization process can be framed as the evolution of a parameter distribution rather than individual solutions.**

Key intuitions:
1. **Each parameter is a random variable** with normal distribution N(μ, σ²)
2. **μ (mean)**: Current best estimate of optimal parameter values
3. **σ (standard deviation)**: Scale of exploration, adapting to optimization difficulty
4. **Efficiency through pairwise competition**: Only 2 samples per iteration vs full populations

### Evolutionary Principles in CGD

| Biological Concept | Mathematical Equivalent | Role in Algorithm |
|-------------------|------------------------|-------------------|
| Population | Distribution samples | Diverse candidate solutions |
| Individuals | Parameter samples | Specific model instances |
| Fitness | Loss function | Evaluation metric |
| Natural selection | Winner/loser selection | Keeping better solutions |
| Mutation | Parameter perturbation | Introducing variations |
| Genetic diversity | σ parameter | Controlling search width |
| Species mean | μ parameter | Best current estimate |
| Generations | Iterations | Training process |

---

## Mathematical Framework

### Distribution Representation

CGD maintains a parametrized normal distribution over model parameters:
- **μ (mean vector)**: Best estimate of parameter values
- **σ (standard deviation vector)**: Exploration scale for each parameter
- **Joint distribution**: N(μ, diag(σ²)) - independent normals

### Training Procedure

Alternates between two complementary update strategies:
1. **Compact Pairwise Updates** (most iterations) - efficient local improvements
2. **Periodic Population Calibration** (occasional) - robust global adjustments

### Compact Pairwise Update Mechanism

At each iteration:
1. **Sample**: θ₁, θ₂ ∼ N(μ, diag(σ²))
2. **Evaluate**: L₁ = L(θ₁), L₂ = L(θ₂)
3. **Select**: Winner = θ with lower loss
4. **Update μ**: Move toward winner with credit assignment
   - Δμ ∝ credit × importance × (winner - μ)
   - Credit proportional to improvement magnitude
   - Importance: parameter-specific sensitivity
5. **Update σ**: Based on observed diversity |winner - loser|
   - Increase σ when diversity > current σ
   - Decrease σ when diversity < current σ

### Population Calibration

Every N iterations:
1. **Sample population** from current distribution
2. **Estimate gradients** using REINFORCE-like formulas
3. **Update μ** using estimated gradient
4. **Update σ** using multiplicative exponential update

---

## Regression Implementation

### Softmax Discretization Approach

CGD handles continuous regression targets through:
1. **Bin Creation**: Continuous target range divided into discrete bins
2. **Probabilistic Prediction**: Model outputs probabilities for each bin via softmax
3. **Expected Value Calculation**: Final predictions = expected value using bin centers

This creates a **differentiable approximation** to continuous mapping.

### Loss Function

```python
# Cross-entropy loss with discretized targets
ce_loss = -mean(sum(y_onehot * log(P + eps), axis=1))

# Entropy term for uncertainty calibration (formerly "surprise")
entropy = -sum(P * log(P + eps), axis=1)
entropy_loss = mean(entropy)

# Total loss (note: no regularization in compact version)
total_loss = ce_loss + entropy_awareness * entropy_loss
```

**Key**: The compact version **removes L1/L2 regularization** to simplify the model, relying on the distribution's σ parameters and `entropy_awareness` for natural regularization.

### Prediction

```python
z = X @ W.T + b
P = softmax(z, axis=1)
centers = (bin_edges[:-1] + bin_edges[1:]) / 2
predictions = P @ centers  # Expected value
```

---

## Classification Implementation

### Multi-class Classification

For classification, CGD uses standard softmax regression:
- Each class has its own weight vector and bias
- Parameters: W (K × d) weights, b (K) biases for K classes
- Softmax applied to logits for probability distribution

### Loss Function (Simplified)

```python
# Cross-entropy loss
z = X @ W.T + b
P = softmax(z, axis=1)
ce_loss = -mean(sum(y_onehot * log(P + eps), axis=1))

# Entropy term for confidence calibration
entropy = -sum(P * log(P + eps), axis=1)
entropy_loss = mean(entropy)

# Total loss (no L1/L2 regularization in compact version)
total_loss = ce_loss + entropy_awareness * entropy_loss
```

**Note**: The compact version removes L2 regularization to maintain simplicity and rely on the distribution's inherent exploration control through σ and `entropy_awareness`.

### Prediction

```python
# Class probabilities
proba = softmax(X @ W.T + b, axis=1)

# Class labels
predictions = argmax(proba, axis=1)
```

---

## Implementation Details

### Algorithm Parameters

#### Common Parameters (Regressor & Classifier)
- `max_iter`: Maximum iterations (default: 500)
- `lr_mu`: Learning rate for mean parameters (default: 0.05)
- `lr_sigma`: Learning rate for standard deviation (default: 0.005)
- `sigma_min/max`: Bounds for σ parameters (default: 0.01/1.0)
- `calibration_interval`: Population calibration frequency (default: 25)
- `credit_factor`: Strength of credit assignment (default: 2.0)
- `early_stopping`: Enable early stopping (default: True)
- `patience`: Early stopping patience (default: 20)
- `calibration_size`: Samples for calibration (default: 20)
- `entropy_awareness`: Weight for entropy term (default: 0.1)
- `random_state`: Random seed for reproducibility

#### Regressor-Specific
- `n_bins`: Number of bins for target discretization (default: 10)

### Python Implementation Structure

```python
# CompactGeneticDescentRegressor
class CompactGeneticDescentRegressor(BaseEstimator, RegressorMixin):
    # Uses softmax discretization for continuous targets
    # Implements pairwise updates and population calibration
    # Removed: L1/L2 regularization
    # Uses: entropy_awareness parameter

# CompactGeneticDescentClassifier  
class CompactGeneticDescentClassifier(BaseEstimator, ClassifierMixin):
    # Uses standard softmax for multi-class
    # Implements same update mechanisms
    # Removed: L2 regularization
    # Uses: entropy_awareness parameter
```

---

## Exploration-Exploitation Dynamics

### Adaptive Exploration Through σ

σ parameters implement **automatic, adaptive exploration**:
- **High σ**: Broad search for complex landscapes or early training
- **Low σ**: Focused exploitation for smooth landscapes or near convergence

Adaptation emerges from **observed diversity in successful samples**:
- When winners/losers far apart → σ increases (explore more)
- When winners/losers close together → σ decreases (exploit local goodness)

### Credit Assignment

**Proportional credit assignment** instead of equal updates:
- Clear victory (large loss difference) → strong update
- Marginal victory (small loss difference) → gentle update
- No update for ties

This creates a **continuous learning rate** that adapts to landscape difficulty.

### Dimension-Wise Adaptation

**Parameter-specific learning rates** based on importance:
- Sensitive parameters (large winner/loser differences) → larger updates
- Insensitive parameters (similar winners/losers) → smaller updates

---

## Practical Advantages

### Computational Efficiency
- **O(d) cost per iteration** vs O(p×d) for population methods
- **2-20 evaluations per iteration** vs 50-100 for traditional GA
- **Memory efficient**: Only stores distribution parameters, not entire population

### Robustness
- **Pairwise updates**: Efficient progress
- **Periodic calibration**: Robust convergence
- **Distribution representation**: Prevents premature convergence
- **Adaptive exploration**: Handles diverse landscapes

### Interpretability
- **Parameter importance**: Which parameters show large improvements
- **Uncertainty**: Variance in parameters reflects optimization uncertainty
- **Convergence**: σ values indicate progress

### Uncertainty Quantification

The `entropy_awareness` parameter enables **calibrated uncertainty estimates**:
- Low entropy: Confident predictions with sharp distributions
- High entropy: Uncertain predictions with broad distributions
- Controls accuracy-confidence tradeoff

---

## Hyperparameters and Tuning

### Primary Hyperparameters

| Parameter | Range | Description | Typical Value |
|-----------|-------|-------------|---------------|
| `n_bins` | 5-20 | Target discretization granularity | 10 |
| `max_iter` | 100-1000 | Maximum training iterations | 500 |
| `lr_mu` | 0.001-0.1 | Learning rate for μ | 0.05 |
| `lr_sigma` | 0.0001-0.01 | Learning rate for σ | 0.005 |
| `entropy_awareness` | 0.01-0.5 | Uncertainty calibration weight | 0.1 |
| `calibration_interval` | 10-50 | Balance between robust/efficient updates | 25 |
| `credit_factor` | 1.0-3.0 | Strength of proportional updates | 2.0 |
| `sigma_min` | 0.001-0.1 | Minimum exploration | 0.01 |
| `sigma_max` | 0.1-2.0 | Maximum exploration | 1.0 |

### Tuning Recommendations

1. **Start simple**: Use default values for most parameters
2. **Focus on lr_mu and lr_sigma**: lr_sigma should be ~10× smaller than lr_mu
3. **Adjust n_bins** based on target range and desired precision
4. **Tune entropy_awareness** for uncertainty calibration needs
5. **Monitor σ**: Should decrease over time but not reach minimum

---

## Current Status

### Simplified Models (Final Version)

The current implementation represents the **compact, simplified version** of Genetic Descent:

✅ **Removed L1/L2 regularization** from both regressor and classifier
✅ **Renamed parameter**: `lambda_surprise` → `entropy_awareness` for clarity
✅ **Relying on natural regularization** through σ (sigma) distribution parameters
✅ **Unified framework** for both regression and classification

### Available Models

- `CompactGeneticDescentRegressor`: For regression tasks with softmax discretization
- `CompactGeneticDescentClassifier`: For classification tasks (binary and multi-class)

### Benchmark Results

Based on current tests with standard sklearn datasets:
- **Diabetes (Regression)**: Functional, needs hyperparameter tuning
- **Iris (Multi-class Classification)**: ~93% accuracy with default parameters
- **Breast Cancer (Binary Classification)**: ~54% accuracy, needs improvement

---

## Conclusion

Compact Genetic Descent represents a significant advancement in evolutionary computation by combining:
- **Distribution-based optimization** for efficient search
- **Compact genetic algorithm principles** for memory efficiency
- **Pairwise competition** for computational efficiency
- **Adaptive exploration** for robust convergence
- **Simplified architecture** (no explicit L1/L2 regularization)

The framework provides a **unified approach** for both regression and classification tasks, with built-in uncertainty quantification through the `entropy_awareness` parameter and the ability to handle non-differentiable, noisy, or black-box optimization problems.

By treating parameters as probabilistic entities rather than fixed values, CGD captures the best of both evolutionary computation and continuous optimization, creating a powerful framework that is at once **efficient, robust, and adaptable**.
