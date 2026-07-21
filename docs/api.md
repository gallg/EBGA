# API Reference

## Models

#### EBGARegressor

Scikit-learn compatible regressor for evolutionary neural network training.

**Notes:**<br>
- Uses explicit layer specification via `layers` parameter<br>
- Supports layer-wise training (`use_layerwise=True`) or direct training<br>
- Supports output normalization via `normalize_output=True`<br>

```python
from EBGA.models import EBGARegressor

model = EBGARegressor(
    layers=[(50, 'relu'), (1, 'linear')],
    loss='mae',
    optimizer=CompactEvoOptimizer,
    use_layerwise=False,
    max_iter=10000,
    lr_mu=0.03,
    lr_sigma=0.03,
    sigma_min=0.001,
    sigma_max=1.0,
    calibration_size=10,
    early_stopping=True,
    patience=100,
    normalize_output=False,
    random_state=None
)

model.fit(X, y)
y_pred = model.predict(X)
score = model.score(X, y)  # Returns R²
```

**Parameters:**<br>
- `layers`: List of `(output_size, activation)` tuples, e.g., `[(50, 'relu'), (1, 'linear')]`<br>
- `loss`: Loss function ('mse' or 'mae')<br>
- `optimizer`: Optimizer class (currently only CompactEvoOptimizer is supported)<br>
- `use_layerwise`: If True, use layer-wise training; if False, train all layers together<br>
- `max_iter`: Maximum training iterations<br>
- `lr_mu`: Temperature for softmax weighting. Lower values make selection more greedy, higher values more uniform<br>
- `lr_sigma`: Learning rate for sigma adaptation<br>
- `sigma_min`: Minimum sigma value<br>
- `sigma_max`: Maximum sigma value<br>
- `calibration_size`: Population size per step (number of candidates sampled)<br>
- `early_stopping`: Enable early stopping<br>
- `patience`: Patience for early stopping<br>
- `normalize_output`: Normalize output to 0-1 range<br>
- `batch_size`: Batch size for mini-batch training. If `None`, uses the full dataset; last batch is merged with the previous one if too small<br>
- `momentum`: Momentum coefficient for velocity-based parameter updates<br>
- `trust_region_radius`: Maximum allowed update norm (L2) per step for stability<br>
- `random_state`: Random seed<br>

---

#### EBGAClassifier

Scikit-learn compatible classifier for evolutionary neural network training.

```python
from EBGA.models import EBGAClassifier

model = EBGAClassifier(
    layers=[(50, 'relu'), (10, 'softmax')],
    n_classes=10,
    loss='cross_entropy',
    optimizer=CompactEvoOptimizer,
    use_layerwise=False,
    max_iter=500,
    lr_mu=0.05,
    lr_sigma=0.005,
    sigma_min=0.001,
    sigma_max=1.0,
    calibration_size=10,
    early_stopping=True,
    patience=20,
    random_state=None
)

model.fit(X, y)
y_pred = model.predict(X)
y_proba = model.predict_proba(X)
score = model.score(X, y)  # Returns accuracy
```

**Parameters:** Same as EBGARegressor, plus:<br>
- `n_classes`: Number of classes (auto-inferred if None)<br>
- `loss`: Loss function ('cross_entropy')<br>

**Additional Methods:**<br>
- `predict_proba(X)`: Predict class probabilities<br>
- `get_params(deep=True)`: Get model parameters<br>
- `set_params(**params)`: Set model parameters<br>

**Note on Activations for Classification:**<br>
Use `sigmoid` for binary classification with 1 output neuron, or `softmax` for multi-class classification with N output neurons. Other activations should only be used in hidden layers.<br>

---

## Neural Network

#### Sequential

Container for sequential layers.

```python
from EBGA.nn import Sequential
from EBGA.layers import Dense

network = Sequential(layer1, layer2, ..., layerN)
network.initialize(input_size)
output = network.forward(X)
network.set_all_parameters(params)
params = network.get_all_parameters()
count = network.parameter_count()
```

---

## Layers

#### Dense

Fully connected layer.

```python
from EBGA.layers import Dense

layer = Dense(output_size, activation=None, use_bias=True)
```

**Parameters:**<br>
- `output_size`: Number of output units<br>
- `activation`: Activation function name or callable<br>
- `use_bias`: Whether to use bias term<br>

#### Flatten

Flatten layer.

```python
from EBGA.layers import Flatten

layer = Flatten()
```

---

## Activations

Available activation functions:<br>
- ReLU / relu<br>
- LeakyReLU / leaky_relu (configurable alpha, default=0.01)<br>
- ELU / elu (configurable alpha, default=1.0)<br>
- SELU / selu (self-normalizing, default lambda=1.0507, alpha=1.67326)<br>
- GELU / gelu (Gaussian error linear unit)<br>
- Swish / swish (x * sigmoid(x))<br>
- Sigmoid / sigmoid<br>
- Tanh / tanh<br>
- Linear / linear<br>
- Softmax / softmax<br>

```python
from EBGA.activations import (
    ReLU, LeakyReLU, ELU, SELU, GELU, Swish,
    Sigmoid, Tanh, Linear, Softmax
)
from EBGA.activations import (
    relu, leaky_relu, elu, selu, gelu, swish,
    sigmoid, tanh, linear, softmax, get_activation
)

# As classes
activation = LeakyReLU(alpha=0.03)
output = activation(x)

# As functions
output = leaky_relu(x, alpha=0.01)
output = gelu(x)

# Get by name
activation = get_activation('leaky_relu')
```

---


## Losses

Available loss functions:<br>
- MSE / mse_loss (Mean Squared Error)<br>
- MAE / mae_loss (Mean Absolute Error)<br>
- CrossEntropy / cross_entropy_loss<br>
- BinaryCrossEntropy / bce_loss<br>

```python
from EBGA.losses import MSE, MAE, CrossEntropy, BinaryCrossEntropy
from EBGA.losses import mse_loss, mae_loss, cross_entropy_loss, bce_loss, get_loss

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

#### CompactEvoOptimizer

Distribution-based evolutionary optimizer using softmax-weighted recombination (NES-style).

```python
from EBGA.optimizer import CompactEvoOptimizer

optimizer = CompactEvoOptimizer(
    param_dim,
    lr_mu=0.05,
    lr_sigma=0.005,
    sigma_min=0.001,
    sigma_max=1.0,
    calibration_size=10,
    momentum=0.5,
    trust_region_radius=None,
    random_state=None
)

optimizer.initialize(initial_params)
loss = optimizer.step(loss_func, iteration=iter)
params = optimizer.get_parameters()
```

**Parameters:**<br>
- `param_dim`: Dimensionality of parameter space<br>
- `lr_mu`: Temperature for softmax weighting. Lower values make selection more greedy, higher values more uniform<br>
- `lr_sigma`: Learning rate for sigma adaptation<br>
- `sigma_min`: Minimum sigma value<br>
- `sigma_max`: Maximum sigma value<br>
- `calibration_size`: Population size per step (number of candidates sampled)<br>
- `momentum`: Momentum coefficient for velocity-based parameter updates (default: 0.5)<br>
- `trust_region_radius`: Maximum allowed update norm (L2) per step for stability<br>
- `random_state`: Random seed<br>

**Methods:**<br>
- `initialize(initial_params)`: Initialize optimizer with parameters<br>
- `step(loss_func, iteration)`: Perform one optimization step (samples population, evaluates, softmax-weighted recombination)<br>
- `get_parameters()`: Get current mean parameters<br>

**Attributes:**<br>
- `mu`: Current mean parameters<br>
- `sigma`: Current standard deviation parameters<br>

---

## Dataset

#### Dataset

Dataset class for batching, similar to PyTorch's Dataset. For numpy arrays, pass X and y to the constructor. For custom datasets, subclass and implement `__len__` and `__getitem__`.

```python
from EBGA.dataset import Dataset

# Create dataset with batching
dataset = Dataset(X, y, batch_size=32, shuffle=True, random_state=42)

# Iterate over batches
for X_batch, y_batch in dataset.batches():
    # Process batch
    pass
```

**Parameters:**<br>
- `X`: Input features (optional, for array-based datasets)<br>
- `y`: Target values (optional, for array-based datasets)<br>
- `batch_size`: Number of samples per batch (None = full dataset)<br>
- `shuffle`: Shuffle data at each epoch<br>
- `random_state`: Random seed for shuffling<br>

**Methods:**<br>
- `batches()`: Yield (X_batch, y_batch) numpy arrays<br>

---

## Utilities

#### Checkpointing

```python
from EBGA.utils import save_model, load_model, save_network, load_network

# Save and load models
save_model(model, 'model.npz')
loaded_model = load_model('model.npz')

# Save and load custom networks
network = Sequential(Dense(10, activation='relu'), Dense(1))
network.initialize(input_size=5)
optimizer = CompactEvoOptimizer(param_dim=network.parameter_count())
# ... train ...
save_network(network, optimizer, 'network.npz')
loaded_network, loaded_optimizer = load_network('network.npz')
```

**Functions:**<br>
- `save_model(model, filepath)`: Save a trained EBGARegressor or EBGAClassifier<br>
- `load_model(filepath)`: Load a saved model<br>
- `save_network(network, optimizer, filepath)`: Save a custom network and optimizer<br>
- `load_network(filepath)`: Load a saved network and optimizer<br>
