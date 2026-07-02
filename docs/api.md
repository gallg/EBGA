# API Reference

## Models

### EBGARegressor

Scikit-learn compatible regressor for neural network training with natural gradients.

**Additional Methods:**
- `get_params(deep=True)`: Get model parameters
- `set_params(**params)`: Set model parameters

**Notes:**
- Supports both explicit layer specification via `layers` parameter and simple configuration via `n_layers`, `h_dim`, and `inner_activation`
- Can use layer-wise training (`use_layerwise=True`) or direct training (`use_layerwise=False`)
- Supports normalization of output via `normalize_output=True`

```python
from EBGA.models import EBGARegressor

model = EBGARegressor(
    layers=None,
    n_layers=1,
    h_dim=50,
    inner_activation='relu',
    output_activation='linear',
    loss='mae',
    optimizer=CompactEvoOptimizer,
    use_layerwise=False,
    n_candidates=None,
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
- `optimizer`: Optimizer class (CompactEvoOptimizer or MultiCandidateOptimizer)
- `use_layerwise`: If True, use layer-wise training; if False, train all layers together
- `n_candidates`: Number of candidates for MultiCandidateOptimizer (ignored for CompactEvoOptimizer)
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

Scikit-learn compatible classifier for neural network training with natural gradients.

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
    optimizer=CompactEvoOptimizer,
    use_layerwise=False,
    n_candidates=None,
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
- `loss`: Loss function ('cross_entropy')

**Additional Methods:**
- `get_params(deep=True)`: Get model parameters
- `set_params(**params)`: Set model parameters
- `predict_proba(X)`: Predict class probabilities

---

### EvoHyperoptSearch

Hyperparameter optimization for EBGA models using evolutionary algorithms.

```python
from EBGA.search import EvoHyperoptSearch
from EBGA.models import EBGARegressor

search = EvoHyperoptSearch(
    estimator=EBGARegressor(layers=[(50, 'relu'), (1, 'linear')]),
    param_distributions={
        'lr_mu': (0.0001, 0.01, 'log-uniform'),
        'lr_sigma': (0.00001, 0.001, 'log-uniform'),
        'max_iter': [1000, 5000, 10000],
        'use_layerwise': [True, False]
    },
    n_iter=10,
    cv=3,
    search_strategy='evolutionary',
    n_jobs=None,
    scoring=None,
    n_generations=5,
    tournament_size=3,
    elitism_count=1,
    crossover_rate=0.8,
    mutation_rate=0.2,
    mutation_scale=0.1,
    early_stopping_rounds=3,
    random_state=None,
    verbose=0
)

search.fit(X_train, y_train)
print(f"Best parameters: {search.best_params_}")
print(f"Best score: {search.best_score_}")
y_pred = search.predict(X_test)
```

**Parameters:**
- `estimator`: EBGA model or pipeline to optimize
- `param_distributions`: Dictionary mapping parameter names to their distributions:
  - Continuous: `(min, max)` or `(min, max, 'log-uniform')` or `(min, max, 'uniform')`
  - Discrete: `[choice1, choice2, ...]`
  - Boolean: `[True, False]`
  - Integer: `(min, max)` or `[choice1, choice2, ...]`
- `n_iter`: Number of parameter settings to sample (for random) or population size (for evolutionary)
- `cv`: Cross-validation strategy (int or cross-validator)
- `search_strategy`: `'random'`, `'evolutionary'`, or `'hybrid'`
- `n_jobs`: Number of parallel jobs for cross-validation
- `scoring`: Scoring metric (uses estimator's default if None)
- `n_generations`: Number of generations for evolutionary search
- `tournament_size`: Tournament size for selection in evolutionary search
- `elitism_count`: Number of best individuals to preserve between generations
- `crossover_rate`: Crossover rate for evolutionary search
- `mutation_rate`: Mutation rate for evolutionary search
- `mutation_scale`: Scale for mutation strength
- `early_stopping_rounds`: Stop if no improvement for this many generations
- `random_state`: Random seed for reproducibility
- `verbose`: Verbosity level (0=silent, 1=progress, 2=debug)

**Attributes:**
- `best_estimator_`: The best estimator found during search
- `best_params_`: The best parameter set found during search
- `best_score_`: The best score found during search
- `cv_results_`: Detailed cross-validation results
- `best_index_`: Index of the best parameter set
- `n_splits_`: Number of cross-validation splits

**Methods:**
- `fit(X, y)`: Run hyperparameter search
- `predict(X)`: Predict using the best estimator
- `predict_proba(X)`: Predict probabilities using the best estimator (if available)
- `score(X, y)`: Score using the best estimator
- `transform(X)`: Transform using the best estimator (for pipelines)
- `get_params(deep=True)`: Get parameters for this estimator
- `set_params(**params)`: Set parameters for this estimator

**Search Strategies:**
- `'random'`: Simple random search (baseline)
- `'evolutionary'`: Evolutionary search with selection, crossover, mutation
- `'hybrid'`: Start with random exploration, refine with evolutionary search

---

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
- Sigmoid / sigmoid
- Tanh / tanh
- Linear / linear
- Softmax / softmax

```python
from EBGA.activations import ReLU, Sigmoid, Tanh, Linear, Softmax
from EBGA.activations import relu, sigmoid, tanh, linear, softmax
from EBGA.activations import get_activation

# As classes
activation = ReLU()
output = activation(x)

# As functions
output = relu(x)

# Get by name
activation = get_activation('relu')
```

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
    calibration_interval=25,
    credit_factor=2.0,
    sigma_regularization=0.0,
    random_state=None
)

optimizer.initialize(initial_params)
loss = optimizer.step(loss_func, iteration=iter)
params = optimizer.get_parameters()
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
- `random_state`: Random seed

**Methods:**
- `initialize(initial_params)`: Initialize optimizer with parameters
- `step(loss_func, iteration)`: Perform one optimization step
- `get_parameters()`: Get current mean parameters

**Attributes:**
- `mu`: Current mean parameters
- `sigma`: Current standard deviation parameters

### MultiCandidateOptimizer

Distribution-based evolutionary optimizer with multiple candidate distributions per parameter.

```python
from EBGA.optimizer import MultiCandidateOptimizer

optimizer = MultiCandidateOptimizer(
    param_dim,
    n_candidates=3,
    lr_mu=0.05,
    lr_sigma=0.005,
    sigma_min=0.001,
    sigma_max=1.0,
    calibration_size=20,
    calibration_interval=25,
    credit_factor=2.0,
    sigma_regularization=0.0,
    random_state=None
)

optimizer.initialize(initial_params)
loss = optimizer.step(loss_func, iteration=iter)
params = optimizer.get_parameters()
```

**Parameters:** Same as CompactEvoOptimizer, plus:
- `n_candidates`: Number of candidate distributions per parameter (default: 3)

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
