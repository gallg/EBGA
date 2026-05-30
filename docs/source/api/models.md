# models Module

High-level neural network models for regression and classification.

## EBGARegressor

Evolutionary neural network for regression tasks.

### Class Signature

```python
class EBGARegressor(layers=None, n_layers=1, h_dim=50, inner_activation='relu',
                   output_activation='linear', loss='mae',
                   lr_mu=0.03, lr_sigma=0.03, sigma_min=0.001, sigma_max=1.0,
                   calibration_size=30, calibration_interval=50, credit_factor=2.0,
                   max_iter=10000, early_stopping=True, patience=100,
                   layer_patience=50, normalize_output=False, random_state=None)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `layers` | list of tuples | None | Network architecture as (size, activation) tuples. If None, uses n_layers/h_dim. |
| `n_layers` | int | 1 | Number of hidden layers (excluding output). Only used if layers=None. |
| `h_dim` | int | 50 | Size of each hidden layer. Only used if layers=None. |
| `inner_activation` | str | 'relu' | Activation for hidden layers. Options: 'relu', 'sigmoid', 'tanh', 'linear'. |
| `output_activation` | str | 'linear' | Activation for output layer. Options: 'relu', 'sigmoid', 'tanh', 'linear'. |
| `loss` | str or Loss | 'mae' | Loss function. Options: 'mse', 'mae'. |
| `lr_mu` | float | 0.03 | Learning rate for mean parameters. |
| `lr_sigma` | float | 0.03 | Learning rate for sigma parameters. |
| `sigma_min` | float | 0.001 | Minimum sigma value. |
| `sigma_max` | float | 1.0 | Maximum sigma value. |
| `calibration_size` | int | 30 | Number of samples for population calibration. |
| `calibration_interval` | int | 50 | How often to perform population calibration. |
| `credit_factor` | float | 2.0 | Strength of credit assignment in pairwise updates. |
| `max_iter` | int | 10000 | Maximum training iterations. |
| `early_stopping` | bool | True | Enable early stopping. |
| `patience` | int | 100 | Number of iterations to wait before early stopping. |
| `layer_patience` | int | 50 | Patience for layer-wise plateau detection. |
| `normalize_output` | bool | False | Scale output to 0-1 range. |
| `random_state` | int | None | Random seed for reproducibility. |

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `network_` | Sequential | The underlying neural network. |
| `optimizer_` | CompactEvoOptimizer | The optimizer instance. |
| `n_features_` | int | Number of input features. |
| `y_min_` | float | Minimum target value (if normalize_output=True). |
| `y_max_` | float | Maximum target value (if normalize_output=True). |

### Methods

#### fit(X, y)

Fit the model to training data.

**Parameters:**
- `X` : array-like, shape (n_samples, n_features) - Training input.
- `y` : array-like, shape (n_samples,) - Target values.

**Returns:**
- `self` : Returns the instance itself.

#### predict(X)

Predict target values for input X.

**Parameters:**
- `X` : array-like, shape (n_samples, n_features) - Input samples.

**Returns:**
- `y_pred` : array, shape (n_samples,) - Predicted values.

#### score(X, y)

Calculate R² score on test data.

**Parameters:**
- `X` : array-like, shape (n_samples, n_features) - Test input.
- `y` : array-like, shape (n_samples,) - True target values.

**Returns:**
- `score` : float - R² score (higher is better, max=1.0).

### Example

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

# Create and train model
model = EBGARegressor(
    n_layers=2,
    h_dim=50,
    inner_activation='relu',
    output_activation='linear',
    max_iter=5000,
    random_state=42
)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
score = model.score(X_test, y_test)
print(f"R² Score: {score:.4f}")
```

### Explicit Layers Example

```python
# Build network with explicit architecture
model = EBGARegressor(
    layers=[
        (128, 'relu'),
        (64, 'tanh'),
        (1, 'linear')
    ],
    max_iter=10000
)
model.fit(X_train, y_train)
```

---

## EBGAClassifier

Evolutionary neural network for classification tasks.

### Class Signature

```python
class EBGAClassifier(layers=None, n_classes=None, n_layers=1, h_dim=50, 
                    inner_activation='relu', output_activation='softmax', 
                    loss='cross_entropy',
                    lr_mu=0.05, lr_sigma=0.005, sigma_min=0.001, sigma_max=1.0,
                    calibration_size=20, calibration_interval=25, credit_factor=2.0,
                    max_iter=500, early_stopping=True, patience=20,
                    layer_patience=50, random_state=None)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `layers` | list of tuples | None | Network architecture as (size, activation) tuples. If None, uses n_layers/h_dim. |
| `n_classes` | int | None | Number of classes. If None, inferred from data. |
| `n_layers` | int | 1 | Number of hidden layers (excluding output). Only used if layers=None. |
| `h_dim` | int | 50 | Size of each hidden layer. Only used if layers=None. |
| `inner_activation` | str | 'relu' | Activation for hidden layers. |
| `output_activation` | str | 'softmax' | Activation for output layer. |
| `loss` | str or Loss | 'cross_entropy' | Loss function. Options: 'cross_entropy', 'binary_cross_entropy'. |
| `lr_mu` | float | 0.05 | Learning rate for mean parameters. |
| `lr_sigma` | float | 0.005 | Learning rate for sigma parameters. |
| `sigma_min` | float | 0.001 | Minimum sigma value. |
| `sigma_max` | float | 1.0 | Maximum sigma value. |
| `calibration_size` | int | 20 | Number of samples for population calibration. |
| `calibration_interval` | int | 25 | How often to perform population calibration. |
| `credit_factor` | float | 2.0 | Strength of credit assignment. |
| `max_iter` | int | 500 | Maximum training iterations. |
| `early_stopping` | bool | True | Enable early stopping. |
| `patience` | int | 20 | Number of iterations to wait before early stopping. |
| `layer_patience` | int | 50 | Patience for layer-wise plateau detection. |
| `random_state` | int | None | Random seed for reproducibility. |

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `network_` | Sequential | The underlying neural network. |
| `optimizer_` | CompactEvoOptimizer | The optimizer instance. |
| `n_features_` | int | Number of input features. |
| `n_classes_` | int | Number of classes. |
| `label_binarizer_` | LabelBinarizer | sklearn label binarizer for one-hot encoding. |

### Methods

#### fit(X, y)

Fit the model to training data.

**Parameters:**
- `X` : array-like, shape (n_samples, n_features) - Training input.
- `y` : array-like, shape (n_samples,) - Target class labels.

**Returns:**
- `self` : Returns the instance itself.

#### predict(X)

Predict class labels for input X.

**Parameters:**
- `X` : array-like, shape (n_samples, n_features) - Input samples.

**Returns:**
- `y_pred` : array, shape (n_samples,) - Predicted class labels (integers).

#### predict_proba(X)

Predict class probabilities for input X.

**Parameters:**
- `X` : array-like, shape (n_samples, n_features) - Input samples.

**Returns:**
- `y_proba` : array, shape (n_samples, n_classes) - Class probabilities.

#### score(X, y)

Calculate accuracy score on test data.

**Parameters:**
- `X` : array-like, shape (n_samples, n_features) - Test input.
- `y` : array-like, shape (n_samples,) - True class labels.

**Returns:**
- `score` : float - Accuracy (0-1, higher is better).

### Example

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

# Create and train model
model = EBGAClassifier(
    n_layers=1,
    h_dim=10,
    inner_activation='relu',
    output_activation='softmax',
    n_classes=3,
    max_iter=2000,
    random_state=42
)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")

# Get probabilities
y_proba = model.predict_proba(X_test)
```

### Explicit Layers Example

```python
# Build network with explicit architecture
model = EBGAClassifier(
    layers=[
        (64, 'relu'),
        (32, 'relu'),
        (3, 'softmax')  # 3 classes for Iris
    ],
    n_classes=3,
    max_iter=2000
)
model.fit(X_train, y_train)
```

---

## BaseModel

Abstract base class for EBGA models.

### Class Signature

```python
class BaseModel(layers=None, output_activation='linear',
               lr_mu=0.05, lr_sigma=0.005, sigma_min=0.001, sigma_max=1.0,
               calibration_size=20, calibration_interval=25, credit_factor=2.0,
               max_iter=500, early_stopping=True, patience=20, random_state=None)
```

### Methods

#### _build_network(input_size, output_size)

Build the neural network from layer specification.

**Parameters:**
- `input_size` : int - Size of input features.
- `output_size` : int - Size of output layer.

**Returns:**
- `Sequential` - The built neural network.

#### _build_layers_from_params()

Build layer specification from simple parameters (n_layers, h_dim, inner_activation).

**Returns:**
- `list` - List of (size, activation) tuples.

Only used when `layers=None`.

#### _create_loss_func(X, y)

Create a loss function closure for optimization.

**Parameters:**
- `X` : array-like - Input data.
- `y` : array-like - Target data.

**Returns:**
- `callable` - Loss function that takes parameters and returns loss.
