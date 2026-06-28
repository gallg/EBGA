# API Reference

## Models

### EBGARegressor

```python
from EBGA.models import EBGARegressor

model = EBGARegressor(
    layers=None,
    n_layers=1,
    h_dim=50,
    inner_activation='relu',
    output_activation='linear',
    loss='mae',
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
- `layers`: List of (size, activation) tuples for explicit layer specification
- `n_layers`: Number of hidden layers (used if layers=None)
- `h_dim`: Size of hidden layers (used if layers=None)
- `inner_activation`: Activation function for hidden layers
- `output_activation`: Activation function for output layer
- `loss`: Loss function ('mse' or 'mae')
- `sigma_regularization`: Strength of sigma diversity regularization
- `max_iter`: Maximum training iterations
- `lr_mu`: Learning rate for mean parameters
- `lr_sigma`: Learning rate for sigma parameters
- `sigma_min`: Minimum sigma value
- `sigma_max`: Maximum sigma value
- `calibration_size`: Population size for calibration
- `calibration_interval`: Frequency of population calibration
- `credit_factor`: Strength of credit assignment
- `early_stopping`: Enable early stopping
- `patience`: Patience for early stopping
- `layer_patience`: Patience for layer-wise plateau detection
- `normalize_output`: Normalize output to 0-1 range
- `random_state`: Random seed

---

### EBGAClassifier

```python
from EBGA.models import EBGAClassifier

model = EBGAClassifier(
    layers=None,
    n_classes=None,
    n_layers=1,
    h_dim=50,
    inner_activation='relu',
    output_activation='softmax',
    loss='cross_entropy',
    sigma_regularization=0.0,
    max_iter=500,
    lr_mu=0.05,
    lr_sigma=0.005,
    sigma_min=0.001,
    sigma_max=1.0,
    calibration_size=20,
    calibration_interval=25,
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
- `loss`: Loss function ('cross_entropy' or 'bce')

---

## Neural Network

### Sequential

```python
from EBGA.nn import Sequential

network = Sequential(layer1, layer2, ..., layerN)
network.initialize(input_size)
network.forward(X)
network.set_all_parameters(params)
network.get_all_parameters()
network.parameter_count()
```

---

## Layers

### Linear

```python
from EBGA.layers import Linear

layer = Linear(
    output_size,
    activation=None,
    use_bias=True
)
```

**Parameters:**
- `output_size`: Number of output units
- `activation`: Activation function name or callable
- `use_bias`: Whether to use bias term

### Flatten

```python
from EBGA.layers import Flatten

layer = Flatten()
```

### Dropout

```python
from EBGA.layers import Dropout

layer = Dropout(rate)
```

**Parameters:**
- `rate`: Dropout probability (0-1)

---

## Activations

```python
from EBGA.activations import (
    ReLU, Sigmoid, Tanh, Linear, Softmax,
    relu, sigmoid, tanh, linear, softmax,
    get_activation
)

# As classes
activation = ReLU()
output = activation(x)

# As functions
output = relu(x)

# Get activation by name
activation = get_activation('relu')
```

---

## Losses

```python
from EBGA.losses import (
    MSE, MAE, CrossEntropy, BinaryCrossEntropy,
    mse_loss, mae_loss, cross_entropy_loss, bce_loss,
    get_loss
)

# As classes
loss = MSE()
value = loss(y_pred, y_true)

# As functions
value = mse_loss(y_pred, y_true)

# Get loss by name
loss_fn = get_loss('mse')
```

---

## Optimizer

### CompactEvoOptimizer

```python
from EBGA.optimizer import CompactEvoOptimizer

optimizer = CompactEvoOptimizer(
    param_dim,
    lr_mu=0.05,
    lr_sigma=0.005,
    sigma_min=0.001,
    sigma_max=1.0,
    calibration_size=20,
    calibration_interval=25,
    credit_factor=2.0,
    sigma_regularization=0.0,
    bounds=None,
    budget=None,
    random_state=None
)

optimizer.initialize(initial_params)
loss = optimizer.step(loss_func, iteration=iter)
params = optimizer.get_parameters()
optimizer.set_parameters(params)

# Or use minimize for complete optimization
result = optimizer.minimize(loss_func, initial_params=params, max_iter=1000)
```

**Parameters:**
- `param_dim`: Dimensionality of parameter space
- `lr_mu`: Learning rate for mean parameters
- `lr_sigma`: Learning rate for sigma parameters
- `sigma_min`: Minimum sigma value
- `sigma_max`: Maximum sigma value
- `calibration_size`: Population size for calibration
- `calibration_interval`: Frequency of population calibration
- `credit_factor`: Strength of credit assignment
- `sigma_regularization`: Strength of sigma regularization
- `bounds`: Tuple of (lower, upper) bounds for parameters
- `budget`: Maximum number of evaluations
- `random_state`: Random seed

**Methods:**
- `initialize(initial_params)`: Initialize optimizer with parameters
- `step(loss_func, iteration)`: Perform one optimization step
- `get_parameters()`: Get current mean parameters
- `set_parameters(params)`: Set mean parameters
- `minimize(func, initial_params, max_iter)`: Run complete optimization

**Attributes:**
- `mu`: Current mean parameters
- `sigma`: Current standard deviation parameters
- `num_evaluations`: Total number of evaluations performed

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

---

## Training Utilities

The EBGA framework includes internal utility functions used for the layer-wise training process:

### Training Functions
- `_create_layer_param_ranges(network)`: Create parameter ranges for each layer in the network
- `_create_layer_optimizer(...)`: Create a configured CompactEvoOptimizer for layer training
- `_train_single_layer(...)`: Train a single layer until plateau or max iterations
- `_train_all_layers_together(...)`: Train all layers together using full network optimizer
- `_build_layers_from_params_simple(...)`: Build layer configuration from simple parameters

These functions are used internally by the models and provide the foundation for the layer-wise training approach that enables efficient natural gradient optimization.
