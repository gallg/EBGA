# nn Module

Neural network container for building sequential networks.

## Sequential

Container for chaining layers together in sequence, similar to PyTorch's `nn.Sequential`.

### Class Signature

```python
class Sequential(*layers)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `*layers` | Layer | Variable number of Layer instances to chain together. |

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `layers` | list | List of Layer instances. |
| `initialized` | bool | Whether the network has been initialized. |
| `input_size` | int | Size of input features (set after initialization). |
| `output_size` | int | Size of output (set after initialization). |

### Methods

#### initialize(input_size)

Initialize all layers with proper input/output sizes.

**Parameters:**
- `input_size` : int - Size of input features.

This method:
1. Sets `input_size` for the first layer
2. Propagates sizes through all layers
3. Calls `initialize()` on each layer
4. Sets `output_size` to the final layer's output size

#### forward(x)

Forward pass through all layers.

**Parameters:**
- `x` : array-like, shape (n_samples, input_size) - Input data.

**Returns:**
- `output` : array, shape (n_samples, output_size) - Output after all layers.

#### get_all_parameters()

Get all parameters from all layers as a single flat array.

**Returns:**
- `params` : array, shape (total_params,) - Concatenated parameters from all layers.

#### set_all_parameters(params)

Set all parameters from a flat array.

**Parameters:**
- `params` : array, shape (total_params,) - Flat array of parameters.

#### parameter_count()

Get total number of parameters in the network.

**Returns:**
- `count` : int - Total number of parameters.

#### get_layer_parameters(layer_idx)

Get parameters for a specific layer.

**Parameters:**
- `layer_idx` : int - Index of the layer.

**Returns:**
- `params` : array - Parameters for that layer.

### Special Methods

#### __len__()

Return the number of layers.

#### __getitem__(idx)

Access a layer by index.

**Parameters:**
- `idx` : int - Layer index.

**Returns:**
- `Layer` - The layer at the specified index.

### Example

```python
from EBGA.nn import Sequential
from EBGA.layers import Linear

# Build a network
network = Sequential(
    Linear(64, activation='relu'),
    Linear(32, activation='tanh'),
    Linear(1, activation='linear')
)

# Initialize with input size
network.initialize(input_size=10)

# Check properties
print(f"Number of layers: {len(network)}")
print(f"Input size: {network.input_size}")
print(f"Output size: {network.output_size}")
print(f"Total parameters: {network.parameter_count()}")

# Forward pass
import numpy as np
X = np.random.randn(5, 10)  # 5 samples, 10 features
output = network.forward(X)
print(f"Output shape: {output.shape}")  # (5, 1)

# Get and set parameters
params = network.get_all_parameters()
print(f"Parameter count: {len(params)}")

# Modify and set
params_new = params * 1.1  # Scale all parameters by 10%
network.set_all_parameters(params_new)

# Access individual layers
first_layer = network[0]
print(f"First layer output size: {first_layer.output_size}")
```

---

## Network Architecture Examples

### Simple Feedforward Network

```python
from EBGA.nn import Sequential
from EBGA.layers import Linear

network = Sequential(
    Linear(128, activation='relu'),
    Linear(64, activation='relu'),
    Linear(1, activation='linear')
)
network.initialize(input_size=20)
```

### Deep Network

```python
network = Sequential(
    Linear(256, activation='relu'),
    Linear(128, activation='relu'),
    Linear(64, activation='relu'),
    Linear(32, activation='relu'),
    Linear(1, activation='linear')
)
network.initialize(input_size=100)
```

### Network with Different Activations

```python
network = Sequential(
    Linear(64, activation='tanh'),
    Linear(32, activation='sigmoid'),
    Linear(1, activation='linear')
)
network.initialize(input_size=10)
```
