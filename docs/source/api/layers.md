# layers Module

Layer implementations for neural networks.

## Layer Base Class

Abstract base class for all layer types.

### Class Signature

```python
class Layer
```

### Methods (to be implemented by subclasses)

#### initialize(input_size)

Initialize layer parameters.

**Parameters:**
- `input_size` : int - Size of input features.

#### forward(x)

Forward pass through the layer.

**Parameters:**
- `x` : array-like - Input data.

**Returns:**
- `output` : array - Output after applying layer transformation.

#### get_parameters()

Get layer parameters as a flat array.

**Returns:**
- `params` : array - Flat array of parameters.

#### set_parameters(params)

Set layer parameters from a flat array.

**Parameters:**
- `params` : array - Flat array of parameters.

#### parameter_count()

Get number of parameters in the layer.

**Returns:**
- `count` : int - Number of parameters.

---

## Linear

Fully connected linear layer with optional activation.

### Class Signature

```python
class Linear(output_size, activation=None, use_bias=True)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_size` | int | Required | Number of output units. |
| `activation` | str or Activation | None | Activation function. Options: 'relu', 'sigmoid', 'tanh', 'linear', 'softmax', or Activation instance. |
| `use_bias` | bool | True | Whether to include a bias term. |

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `input_size` | int | Size of input (set during initialization). |
| `output_size` | int | Size of output. |
| `W` | array | Weight matrix, shape (output_size, input_size). |
| `b` | array | Bias vector, shape (output_size,). |
| `activation` | Activation | Activation function instance. |
| `use_bias` | bool | Whether bias is used. |
| `initialized` | bool | Whether the layer has been initialized. |

### Initialization

Uses **Xavier/Glorot initialization** for weights:

```
limit = sqrt(6 / (input_size + output_size))
W ~ Uniform(-limit, limit)
b = zeros(output_size)
```

This initialization helps maintain stable gradients in deep networks.

### Forward Pass

The forward pass computes:

```
output = x @ W.T + b
if activation is not None:
    output = activation(output)
```

### Example

```python
from EBGA.layers import Linear
import numpy as np

# Create layer
layer = Linear(output_size=32, activation='relu', use_bias=True)

# Initialize with input size
layer.initialize(input_size=64)

# Forward pass
x = np.random.randn(10, 64)  # 10 samples, 64 features
output = layer.forward(x)
print(f"Output shape: {output.shape}")  # (10, 32)

# Get parameters
params = layer.get_parameters()
print(f"Parameter count: {layer.parameter_count()}")
# 64 * 32 weights + 32 biases = 2080 parameters

# Set parameters
new_params = params * 0.9
layer.set_parameters(new_params)
```

### Without Bias

```python
# Create layer without bias
layer = Linear(output_size=32, activation='relu', use_bias=False)
layer.initialize(input_size=64)

# Parameter count: 64 * 32 = 2048 (no biases)
print(f"Parameter count: {layer.parameter_count()}")
```

---

## Flatten

Layer that reshapes input (identity operation for 2D inputs).

### Class Signature

```python
class Flatten
```

### Description

The Flatten layer currently performs an identity operation for 2D inputs. It's included for API compatibility and future extensibility (e.g., for handling multi-dimensional inputs).

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `input_size` | int | Size of input (set during initialization). |
| `output_size` | int | Size of output (same as input). |

### Methods

#### initialize(input_size)

Initialize the layer.

#### forward(x)

Return input unchanged.

#### get_parameters()

Return empty array (no parameters).

#### set_parameters(params)

No-op (no parameters to set).

#### parameter_count()

Return 0 (no parameters).

### Example

```python
from EBGA.layers import Flatten

flatten = Flatten()
flatten.initialize(input_size=100)

import numpy as np
x = np.random.randn(10, 100)
output = flatten.forward(x)

print(output.shape)  # (10, 100) - unchanged
print(flatten.parameter_count())  # 0
```

---

## Factory Functions

### linear()

Create a Linear layer.

```python
def linear(output_size, activation=None, use_bias=True)
```

**Parameters:**
- `output_size` : int - Number of output units.
- `activation` : str or Activation - Activation function.
- `use_bias` : bool - Whether to include bias.

**Returns:**
- `Linear` - A Linear layer instance.

### flatten()

Create a Flatten layer.

```python
def flatten()
```

**Returns:**
- `Flatten` - A Flatten layer instance.
