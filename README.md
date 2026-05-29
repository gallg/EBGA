# EBGA: Compact Genetic Descent

**EBGA** (**E**nergy-**B**ased **G**enetic **A**lgorithms) is a gradient-free optimization framework that combines evolutionary computation with compact genetic algorithms.

## What's Inside

This package provides:

🔹 **CompactGeneticDescentRegressor** - Gradient-free evolutionary regression algorithm
🔹 **CompactGeneticDescentClassifier** - Gradient-free evolutionary classification algorithm
🔹 **Entropy-aware loss** - Information-theoretic loss function with uncertainty awareness
🔹 **Evolutionary optimization** - Distribution-based parameter evolution

## Key Features

- ✅ **Gradient-free optimization** (no backpropagation)
- ✅ Handles **non-differentiable loss functions**
- ✅ Built-in **uncertainty modeling** via entropy_awareness
- ✅ Works with **noisy or discontinuous objectives**
- ✅ Naturally **parallelizable implementation**
- ✅ **Simplified architecture** - No L1/L2 regularization (removed for compact version)
- ✅ **Unified framework** for regression and classification

### Benchmark Results

Based on current tests with standard sklearn datasets (with default parameters):
- **Diabetes (Regression)**: Functional, needs hyperparameter tuning
- **Iris (Multi-class Classification)**: ~93% accuracy
- **Breast Cancer (Binary Classification)**: ~54% accuracy

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd EBGA

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install numpy scikit-learn scipy
```

## Quick Start

### Regression Example

```python
from EBGA.GDRegressor import CompactGeneticDescentRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Standardize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = CompactGeneticDescentRegressor(
    n_bins=10,
    max_iter=200,
    entropy_awareness=0.1,
    random_state=42
)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
```

### Classification Example

```python
from EBGA.GDClassifier import CompactGeneticDescentClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Standardize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = CompactGeneticDescentClassifier(
    max_iter=200,
    entropy_awareness=0.1,
    random_state=42
)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
accuracy = model.score(X_test, y_test)
```

## Running Tests

Run the benchmark tests:

```bash
python main.py
```

This will test both the regressor and classifier on standard datasets and print performance metrics.

## Documentation

Full documentation is available in `docs/compact_genetic_descent.md` which covers:
- Core principles and intuition
- Mathematical framework
- Implementation details for regression and classification
- Hyperparameters and tuning guide
- When to use Compact Genetic Descent

## Hyperparameters

### Common Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_iter` | 500 | Maximum iterations |
| `lr_mu` | 0.05 | Learning rate for μ (mean) |
| `lr_sigma` | 0.005 | Learning rate for σ (std dev) |
| `entropy_awareness` | 0.1 | Entropy weight for uncertainty calibration |
| `calibration_interval` | 25 | Population calibration frequency |
| `credit_factor` | 2.0 | Strength of credit assignment |
| `early_stopping` | True | Enable early stopping |
| `patience` | 20 | Early stopping patience |

### Regressor-Specific

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_bins` | 10 | Number of bins for target discretization |
| `sigma_min` | 0.01 | Minimum σ value |
| `sigma_max` | 1.0 | Maximum σ value |

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

This project is open source. See LICENSE for details.
