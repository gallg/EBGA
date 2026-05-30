# Configuration Guide

This guide explains how to configure EBGA models for different tasks and datasets.

## Quick Start Configuration

### Regression

```python
from EBGA.models import EBGARegressor

# Basic configuration
model = EBGARegressor(
    n_layers=2,              # Number of hidden layers
    h_dim=50,                # Size of hidden layers
    inner_activation='relu', # Activation for hidden layers
    output_activation='linear',
    max_iter=10000,
    random_state=42
)
```

### Classification

```python
from EBGA.models import EBGAClassifier

# Basic configuration
model = EBGAClassifier(
    n_layers=2,
    h_dim=50,
    inner_activation='relu',
    output_activation='softmax',
    n_classes=10,             # Number of classes
    max_iter=2000,
    random_state=42
)
```

---

## Configuration Presets

### Small Dataset (n_samples < 1000)

```python
model = EBGARegressor(
    n_layers=1,
    h_dim=20,
    calibration_size=10,
    calibration_interval=10,
    max_iter=1000,
    lr_mu=0.05,
    lr_sigma=0.005
)
```

### Medium Dataset (1000 < n_samples < 10000)

```python
model = EBGARegressor(
    n_layers=2,
    h_dim=50,
    calibration_size=20,
    calibration_interval=25,
    max_iter=5000,
    lr_mu=0.03,
    lr_sigma=0.003
)
```

### Large Dataset (n_samples > 10000)

```python
model = EBGARegressor(
    n_layers=3,
    h_dim=100,
    calibration_size=30,
    calibration_interval=50,
    max_iter=20000,
    lr_mu=0.01,
    lr_sigma=0.001
)
```

---

## Configuration by Problem Type

### Simple Regression (Linear Relationship)

```python
model = EBGARegressor(
    n_layers=1,              # Single hidden layer is often enough
    h_dim=30,
    inner_activation='linear',  # Linear activation for simple relationships
    output_activation='linear',
    max_iter=5000
)
```

### Complex Regression (Non-linear Relationship)

```python
model = EBGARegressor(
    n_layers=3,              # More layers for complex patterns
    h_dim=64,
    inner_activation='relu',   # ReLU for non-linearity
    output_activation='linear',
    max_iter=10000
)
```

### Binary Classification

```python
model = EBGAClassifier(
    n_layers=2,
    h_dim=30,
    inner_activation='relu',
    output_activation='softmax',
    n_classes=2,
    max_iter=2000,
    calibration_size=20,
    calibration_interval=20
)
```

### Multi-class Classification (Many Classes)

```python
model = EBGAClassifier(
    n_layers=3,              # More layers for many classes
    h_dim=64,
    inner_activation='relu',
    output_activation='softmax',
    n_classes=100,           # Number of classes
    max_iter=5000,
    layer_patience=100      # More patience for deep networks
)
```

### High-Dimensional Data (Many Features)

```python
model = EBGARegressor(
    n_layers=3,
    h_dim=128,               # Wider layers for high dimensions
    inner_activation='relu',
    output_activation='linear',
    max_iter=20000,
    sigma_min=0.001,         # Prevent σ from collapsing
    sigma_max=1.0,
    credit_factor=5.0        # Stronger credit assignment
)
```

---

## Architecture Configuration

### Number of Layers (n_layers)

| n_layers | Use Case | Notes |
|----------|----------|-------|
| 1 | Simple patterns, small datasets | Fast, may not capture complex relationships |
| 2 | Most use cases | Good balance of power and speed |
| 3 | Complex patterns, large datasets | More powerful, needs more data |
| 4+ | Very complex patterns | Needs lots of data, careful tuning |

### Hidden Layer Size (h_dim)

| h_dim | Use Case | Parameters |
|-------|----------|------------|
| 10-30 | Small datasets, simple patterns | Few parameters, fast |
| 50-100 | Most use cases | Good balance |
| 128-256 | Complex patterns, large datasets | Many parameters, powerful |
| 512+ | Very large datasets | Needs lots of data |

**Rule of Thumb:**
- Start with h_dim = input_size / 2
- If overfitting: decrease h_dim
- If underfitting: increase h_dim

### Activation Functions

| Layer | Recommended | Notes |
|-------|-------------|-------|
| Hidden | ReLU | Fast, non-saturating, most common |
| Hidden | Tanh | Smooth, zero-centered, good alternative |
| Hidden | Sigmoid | Saturates, less common for hidden layers |
| Hidden | Linear | Only for very specific cases |
| Output (Regression) | Linear | Standard for regression |
| Output (Classification) | Softmax | Standard for multi-class |
| Output (Binary) | Sigmoid | Standard for binary classification |

---

## Optimizer Configuration

### Learning Rates

```python
# Standard values
lr_mu = 0.05    # Mean learning rate
lr_sigma = 0.005 # Sigma learning rate (typically 1/10 of lr_mu)
```

**Tuning:**
- If loss is decreasing too slowly: increase lr_mu
- If loss is oscillating: decrease lr_mu and lr_sigma
- lr_sigma should typically be 1/5 to 1/15 of lr_mu

### Distribution Bounds

```python
sigma_min = 0.001  # Prevent σ from collapsing to zero
sigma_max = 1.0    # Prevent σ from growing too large
```

**Tuning:**
- If parameters seem stuck: increase sigma_max
- If optimization is unstable: decrease sigma_max
- sigma_min should be small but not zero

### Calibration Settings

```python
calibration_size = 20    # Number of samples for natural gradient
calibration_interval = 25 # How often to calibrate
credit_factor = 2.0      # Strength of credit assignment
```

**Tuning:**
- Larger calibration_size: more accurate but slower
- Larger calibration_interval: faster but less stable
- credit_factor: higher values give more credit to large improvements

---

## Layer-Wise Training Configuration

### Layer Patience

```python
layer_patience = 50  # Patience for each layer
```

**Tuning:**
- Smaller values: faster training, less thorough per-layer
- Larger values: more thorough, slower overall
- Rule of thumb: layer_patience ≈ patience / n_layers

### Max Iterations

```python
max_iter = 10000  # Total iterations
```

**How it's used:**
- Without layer-wise: all iterations train all layers
- With layer-wise: iterations are divided among layers + final pass
- For n_layers with layer-wise: ~max_iter / (n_layers + 1) per layer + final pass

---

## Early Stopping Configuration

```python
early_stopping = True   # Enable early stopping
patience = 100         # Number of iterations without improvement
```

**Tuning:**
- patience should be large enough to allow recovery from plateaus
- patience too small: may stop too early
- patience too large: may waste computation

---

## Normalization Configuration

### Output Normalization (Regression Only)

```python
normalize_output = True  # Scale output to [0, 1]
```

**When to use:**
- When your target values have a known range
- When you want to constrain predictions
- When using certain loss functions

**Note:** The framework automatically denormalizes predictions to the original scale.

---

## Complete Configuration Examples

### Example 1: Diabetes Regression (Balanced)

```python
model = EBGARegressor(
    # Architecture
    n_layers=2,
    h_dim=50,
    inner_activation='relu',
    output_activation='linear',
    
    # Optimizer
    lr_mu=0.03,
    lr_sigma=0.003,
    sigma_min=0.001,
    sigma_max=1.0,
    calibration_size=30,
    calibration_interval=50,
    credit_factor=2.0,
    
    # Training
    max_iter=10000,
    early_stopping=True,
    patience=100,
    layer_patience=50,
    
    # Other
    loss='mae',
    normalize_output=False,
    random_state=42
)
```

### Example 2: Iris Classification (Small Dataset)

```python
model = EBGAClassifier(
    # Architecture
    n_layers=1,
    h_dim=10,
    inner_activation='relu',
    output_activation='softmax',
    n_classes=3,
    
    # Optimizer
    lr_mu=0.01,
    lr_sigma=0.001,
    sigma_min=0.001,
    sigma_max=1.0,
    calibration_size=20,
    calibration_interval=20,
    credit_factor=10.0,
    
    # Training
    max_iter=2000,
    early_stopping=True,
    patience=50,
    layer_patience=30,
    
    # Other
    loss='cross_entropy',
    random_state=42
)
```

### Example 3: Custom Architecture

```python
model = EBGARegressor(
    # Explicit architecture
    layers=[
        (128, 'relu'),
        (64, 'tanh'),
        (32, 'relu'),
        (1, 'linear')
    ],
    
    # Optimizer
    lr_mu=0.01,
    lr_sigma=0.001,
    
    # Training
    max_iter=20000,
    layer_patience=100,
    
    random_state=42
)
```

---

## Configuration Tips

1. **Start simple**: Begin with 1-2 layers and moderate h_dim
2. **Use defaults**: Most default values work well for many problems
3. **Monitor training**: Watch loss curves to detect issues
4. **Tune one at a time**: Change only one hyperparameter at a time
5. **Be patient**: Evolutionary optimization may need more iterations than gradient-based methods
6. **Use early stopping**: Saves computation and prevents overfitting
7. **Try layer-wise**: Often works better than training all layers at once
