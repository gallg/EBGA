# EBGA: Evolutionary Bayesian Geometric Adaptation

**EBGA** (**E**volutionary-**B**ayesian **G**eometric **A**daptation) is an optimization framework that combines evolutionary computation with Bayesian geometric principles and natural gradient methods.

## What's Inside

This package provides a **unified neural network framework** for both regression and classification:

🔹 **EBGARegressor** - Gradient-free evolutionary regression with continuous output
🔹 **EBGAClassifier** - Gradient-free evolutionary classification 
🔹 **Modular Neural Network Architecture** - Build custom networks with layers, activations, and losses
🔹 **Layer-wise Training** - Sequential layer training with plateau detection (enabled by default)
🔹 **CompactEvoOptimizer** - Distribution-based parameter evolution

## Key Features

- ✅ **Natural gradient optimization** (no backpropagation, uses distribution parameter gradients)
- ✅ **Modular architecture** - Similar to PyTorch but without traditional gradients
- ✅ **Continuous regression** - No binning, direct continuous output
- ✅ **Layer-wise training** - Train layers sequentially until loss plateaus (default)
- ✅ **Handles non-differentiable loss functions**
- ✅ **Built-in uncertainty modeling** via distribution parameters (μ, σ)
- ✅ **Works with noisy or discontinuous objectives**
- ✅ **Naturally parallelizable implementation**
- ✅ **Unified framework** for both regression and classification
- ✅ **Dropout regularization** - Built-in dropout layers for regularization
- ✅ **L2 regularization** - Optional weight decay for improved generalization

### Framework Modules

```
EBGA/
├── models.py      # EBGARegressor, EBGAClassifier (high-level models)
├── nn.py          # Sequential neural network
├── layers.py      # Layer classes (Linear, Flatten, etc.)
├── activations.py # Activation functions (ReLU, Sigmoid, Tanh, Linear, Softmax)
├── losses.py      # Loss functions (MSE, MAE, CrossEntropy, BinaryCrossEntropy)
├── optimizer.py   # CompactEvoOptimizer
└── utils.py       # Checkpointing utilities (save_model, load_model, etc.)
```

## Installation

### Option 1: Install from source (recommended)

```bash
# Clone the repository
git clone <repository-url>
cd EBGA

# Install in development mode
pip install -e .
```

This will install EBGA and all dependencies (numpy, scikit-learn).

### Option 2: Manual usage

```bash
# Clone the repository
git clone <repository-url>
cd EBGA

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install numpy scikit-learn
```

### Dependencies

- Python 3.10+
- numpy >= 2.4.6
- scikit-learn >= 1.8.0

## Quick Start

### Regression Example (Continuous Output)

```python
from EBGA.models import EBGARegressor
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

# Option 1: Simple configuration with n_layers and h_dim
model = EBGARegressor(
    n_layers=2,              # Number of hidden layers
    h_dim=50,                # Size of hidden layers
    inner_activation='relu', # Activation for hidden layers
    output_activation='linear',
    max_iter=1000,
    lr_mu=0.03,
    lr_sigma=0.03,
    random_state=42
)

# Option 2: Explicit layer specification
model = EBGARegressor(
    layers=[(50, 'relu'), (30, 'relu'), (1, 'linear')],
    max_iter=1000,
    random_state=42
)

model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
print(f"R² Score: {model.score(X_test, y_test):.4f}")
```

### Classification Example

```python
from EBGA.models import EBGAClassifier
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

# Create and train model with explicit layers
model = EBGAClassifier(
    layers=[(10, 'relu'), (10, 'relu'), (3, 'softmax')],
    n_classes=3,
    max_iter=2000,
    random_state=42
)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")

# Get class probabilities
y_proba = model.predict_proba(X_test)
```

### Building Custom Networks

```python
from EBGA.nn import Sequential
from EBGA.layers import Linear
from EBGA.activations import get_activation
from EBGA.losses import mae_loss
from EBGA.optimizer import CompactEvoOptimizer

# Build a custom network
network = Sequential(
    Linear(64, activation='relu'),
    Linear(32, activation='tanh'),
    Linear(1, activation='linear')
)

# Initialize
network.initialize(input_size=X_train.shape[1])

# Create optimizer
optimizer = CompactEvoOptimizer(
    param_dim=network.parameter_count(),
    lr_mu=0.01,
    lr_sigma=0.001
)

# Custom training loop
def loss_func(params):
    network.set_all_parameters(params)
    y_pred = network.forward(X_train)
    return mae_loss(y_pred.flatten(), y_train)

optimizer.initialize()
for iteration in range(1000):
    loss = optimizer.step(loss_func, iteration=iteration)
```

### Model Checkpointing

Save and load trained models and networks:

```python
from EBGA.utils import save_model, load_model, save_network, load_network

# High-level models
model = EBGARegressor(layers=[(50, 'relu'), (1, 'linear')])
model.fit(X_train, y_train)
save_model(model, 'regressor.npz')
loaded_model = load_model('regressor.npz')

# Custom networks
save_network(network, optimizer, 'custom_network.npz')
loaded_network, loaded_optimizer = load_network('custom_network.npz')
```

## Running Tests

Run the test suite on standard datasets (Iris, Breast Cancer, Wine):

```bash
python main.py
```

## Framework Architecture

### Layers Module
Provides various layer types:
- `Linear` - Fully connected layer with configurable activation
- `Flatten` - Flatten layer
- `Dropout` - Dropout layer for regularization

### Activations Module
Available activation functions:
- `ReLU` / `relu`
- `Sigmoid` / `sigmoid`
- `Tanh` / `tanh`
- `Linear` / `linear` (identity)
- `Softmax` / `softmax`

### Losses Module
Available loss functions:
- `MSE` / `mse_loss` - Mean Squared Error (regression)
- `MAE` / `mae_loss` - Mean Absolute Error (regression)
- `CrossEntropy` / `cross_entropy_loss` - Cross-entropy (classification)
- `BinaryCrossEntropy` / `bce_loss` - Binary cross-entropy

### Optimizer
`CompactEvoOptimizer` - The core evolutionary optimizer:
- Maintains Gaussian distribution over parameters (μ, σ)
- Uses natural gradient updates
- Supports population calibration and pairwise updates
- Natural gradient-based (no objective function gradients)

## Hyperparameters

### Common Parameters (Regressor and Classifier)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `layers` | None | Explicit layer specification as list of (size, activation) tuples |
| `n_layers` | 1 | Number of hidden layers (used if layers=None) |
| `h_dim` | 50 | Size of hidden layers (used if layers=None) |
| `inner_activation` | 'relu' | Activation for hidden layers |
| `output_activation` | 'linear' (reg) / 'softmax' (cls) | Activation for output layer |
| `max_iter` | 10000 (reg) / 500 (cls) | Maximum iterations |
| `lr_mu` | 0.03 (reg) / 0.05 (cls) | Learning rate for μ (mean) |
| `lr_sigma` | 0.03 (reg) / 0.005 (cls) | Learning rate for σ (std dev) |
| `sigma_min` | 0.001 | Minimum σ value |
| `sigma_max` | 1.0 | Maximum σ value |
| `sigma_regularization` | 0.0 | Sigma diversity regularization |
| `calibration_size` | 30 (reg) / 20 (cls) | Population size for calibration |
| `calibration_interval` | 50 (reg) / 25 (cls) | Population calibration frequency |
| `credit_factor` | 2.0 | Strength of credit assignment |
| `early_stopping` | True | Enable early stopping |
| `patience` | 100 (reg) / 20 (cls) | Early stopping patience |
| `layer_patience` | 50 | Patience for layer-wise plateau detection (layer-wise training is default) |

### Regressor-Specific

| Parameter | Default | Description |
|-----------|---------|-------------|
| `loss` | 'mae' | Loss function ('mse', 'mae') |
| `normalize_output` | False | Normalize output to 0-1 range |

### Classifier-Specific

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_classes` | None | Number of classes (auto-inferred if None) |
| `loss` | 'cross_entropy' | Loss function for classification |

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

This project is licensed under the **GNU General Public License v3.0** (GPL-3.0).

See [LICENSE](LICENSE) for the full license text.
