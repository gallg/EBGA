# API Reference

## Models

### EBGARegressor

Scikit-learn compatible regressor for neural network training with natural gradients.

**Additional Methods:**
- `get_params(deep=True)`: Get model parameters
- `set_params(**params)`: Set model parameters

**Notes:**
- Uses explicit layer specification via `layers` parameter
- Can use layer-wise training (`use_layerwise=True`) or direct training (`use_layerwise=False`)
- Supports normalization of output via `normalize_output=True`

```python
from EBGA.models import EBGARegressor

model = EBGARegressor(
    layers=[(50, 'relu'), (1, 'linear')],
    loss='mae',
    optimizer=CompactEvoOptimizer,
    use_layerwise=False,
    sigma_regularization=0.0,
    max_iter=10000,
    lr_mu=0.03,
    lr_sigma=0.03,
    sigma_min=0.001,
    sigma_max=1.0,
    calibration_size=30,
    calibration_interval=50,
    credit_factor=2.0,
    early_stopping=True,
    patience=100,
    layer_patience=50,
    normalize_output=False,
    random_state=None
)

model.fit(X, y)
y_pred = model.predict(X)
score = model.score(X, y)  # Returns R²
```

**Parameters:**
- `layers`: List of (size, activation) tuples for explicit layer specification. Each tuple defines (output_size, activation). Example: `[(50, 'relu'), (1, 'linear')]` for 1 hidden layer with 50 units and ReLU activation, plus output layer with linear activation
- `loss`: Loss function ('mse' or 'mae')
- `optimizer`: Optimizer class (currently only CompactEvoOptimizer is supported)
- `use_layerwise`: If True, use layer-wise training; if False, train all layers together
- `sigma_regularization`: Strength of sigma diversity regularization
- `max_iter`: Maximum training iterations
- `lr_mu`: Initial learning rate for mean parameters (adaptive during training)
- `lr_sigma`: Initial learning rate for sigma parameters (adaptive during training)
- `sigma_min`: Minimum sigma value
- `sigma_max`: Maximum sigma value
- `calibration_size`: Population size for calibration
- `calibration_interval`: Frequency of population calibration
- `credit_factor`: Strength of credit assignment
- `early_stopping`: Enable early stopping
- `patience`: Patience for early stopping
- `layer_patience`: Patience for layer-wise plateau detection
- `normalize_output`: Normalize output to 0-1 range
- `batch_size`: Batch size for mini-batch training. If None (default), uses full dataset. If set to an integer, trains on mini-batches. Last batch is merged with previous if too small.
- `random_state`: Random seed

---

### EBGAClassifier

Scikit-learn compatible classifier for neural network training with natural gradients.

```python
from EBGA.models import EBGAClassifier

model = EBGAClassifier(
    layers=[(50, 'relu'), (10, 'softmax')],
    n_classes=10,
    loss='cross_entropy',
    optimizer=CompactEvoOptimizer,
    use_layerwise=False,
    sigma_regularization=0.0,
    max_iter=500,
    lr_mu=0.05,
    lr_sigma=0.005,
    sigma_min=0.001,
    sigma_max=1.0,
    calibration_size=20,
    calibration_interval=50,
    credit_factor=2.0,
    early_stopping=True,
    patience=20,
    layer_patience=50,
    random_state=None
)

model.fit(X, y)
y_pred = model.predict(X)
y_proba = model.predict_proba(X)
score = model.score(X, y)  # Returns accuracy
```

**Parameters:** Same as EBGARegressor, plus:
- `n_classes`: Number of classes (auto-inferred if None)
- `loss`: Loss function ('cross_entropy')

**Additional Methods:**
- `get_params(deep=True)`: Get model parameters
- `set_params(**params)`: Set model parameters
- `predict_proba(X)`: Predict class probabilities

**Note on Activations for Classification:**
For best results with classification, use `sigmoid` for binary classification with 1 output neuron, or `softmax` for multi-class classification with N output neurons. Activations like LeakyReLU, ELU, SELU, GELU, and Swish should be used only in hidden layers, not in the output layer.

---

## Neural Network

## Neural Network

### Sequential

Container for sequential layers.

```python
from EBGA.nn import Sequential
from EBGA.layers import Linear

network = Sequential(layer1, layer2, ..., layerN)
network.initialize(input_size)
output = network.forward(X)
network.set_all_parameters(params)
params = network.get_all_parameters()
count = network.parameter_count()
```

---

## Layers

### Linear

Fully connected layer.

```python
from EBGA.layers import Linear

layer = Linear(output_size, activation=None, use_bias=True)
```

**Parameters:**
- `output_size`: Number of output units
- `activation`: Activation function name or callable
- `use_bias`: Whether to use bias term

### Flatten

Flatten layer.

```python
from EBGA.layers import Flatten

layer = Flatten()
```

---

## Activations

Available activation functions:
- ReLU / relu
- LeakyReLU / leaky_relu (configurable alpha, default=0.01)
- ELU / elu (configurable alpha, default=1.0)
- SELU / selu (self-normalizing, default lambda=1.0507, alpha=1.67326)
- GELU / gelu (Gaussian error linear unit)
- Swish / swish (x * sigmoid(x))
- Sigmoid / sigmoid
- Tanh / tanh
- Linear / linear
- Softmax / softmax

```python
from EBGA.activations import (
    ReLU, LeakyReLU, ELU, SELU, GELU, Swish,
    Sigmoid, Tanh, Linear, Softmax
)
from EBGA.activations import (
    relu, leaky_relu, elu, selu, gelu, swish,
    sigmoid, tanh, linear, softmax
)
from EBGA.activations import get_activation

# As classes
activation = LeakyReLU(alpha=0.03)
output = activation(x)

# As functions
output = leaky_relu(x, alpha=0.01)
output = gelu(x)

# Get by name
activation = get_activation('leaky_relu')
```

**Important**: For classification tasks, use `sigmoid` or `softmax` in the output layer. Other activations (LeakyReLU, ELU, SELU, GELU, Swish) should only be used in hidden layers to avoid numerical instability and incorrect probability outputs.

---

## Losses

Available loss functions:
- MSE / mse_loss (Mean Squared Error)
- MAE / mae_loss (Mean Absolute Error)
- CrossEntropy / cross_entropy_loss
- BinaryCrossEntropy / bce_loss

```python
from EBGA.losses import MSE, MAE, CrossEntropy, BinaryCrossEntropy
from EBGA.losses import mse_loss, mae_loss, cross_entropy_loss, bce_loss
from EBGA.losses import get_loss

# As classes
loss = MSE()
value = loss(y_pred, y_true)

# As functions
value = mse_loss(y_pred, y_true)

# Get by name
loss_fn = get_loss('mse')
```

---

## Optimizers

### CompactEvoOptimizer

Distribution-based evolutionary optimizer with single distribution per parameter.

```python
from EBGA.optimizer import CompactEvoOptimizer

optimizer = CompactEvoOptimizer(
    param_dim,
    lr_mu=0.05,
    lr_sigma=0.005,
    sigma_min=0.001,
    sigma_max=1.0,
    calibration_size=20,
    calibration_interval=50,
    credit_factor=2.0,
    sigma_regularization=0.0,
    momentum=0.5,
    trust_region_radius=None,
    random_state=None
)

optimizer.initialize(initial_params)
loss = optimizer.step(loss_func, iteration=iter)
params = optimizer.get_parameters()
```

**Parameters:**
- `param_dim`: Dimensionality of parameter space
- `lr_mu`: Initial learning rate for mean parameters (adaptive during training)
- `lr_sigma`: Initial learning rate for sigma parameters (adaptive during training)
- `sigma_min`: Minimum sigma value
- `sigma_max`: Maximum sigma value
- `calibration_size`: Population size for calibration
- `calibration_interval`: Frequency of population calibration (default: 50)
- `credit_factor`: Strength of credit assignment
- `sigma_regularization`: Strength of sigma regularization
- `momentum`: Momentum coefficient for velocity-based parameter updates (default: 0.5)
- `trust_region_radius`: Maximum allowed update norm (L2) per step for stability
- `random_state`: Random seed

**Methods:**
- `initialize(initial_params)`: Initialize optimizer with parameters
- `step(loss_func, iteration)`: Perform one optimization step
- `get_parameters()`: Get current mean parameters

**Attributes:**
- `mu`: Current mean parameters
- `sigma`: Current standard deviation parameters

---

## Utilities

### Checkpointing

```python
from EBGA.utils import save_model, load_model, save_network, load_network

# Save and load models
model = EBGARegressor(layers=[(10, 'relu'), (1, 'linear')])
model.fit(X_train, y_train)
save_model(model, 'model.npz')
loaded_model = load_model('model.npz')

# Save and load custom networks
network = Sequential(Linear(10, activation='relu'), Linear(1))
network.initialize(input_size=5)
optimizer = CompactEvoOptimizer(param_dim=network.parameter_count())
# ... train ...
save_network(network, optimizer, 'network.npz')
loaded_network, loaded_optimizer = load_network('network.npz')
```

**Functions:**
- `save_model(model, filepath)`: Save a trained EBGARegressor or EBGAClassifier
- `load_model(filepath)`: Load a saved model
- `save_network(network, optimizer, filepath)`: Save a custom network and optimizer
- `load_network(filepath)`: Load a saved network and optimizer
