# activations Module

Activation functions for neural network layers.

## Activation Base Class

Abstract base class for all activation functions.

### Class Signature

```python
class Activation
```

### Methods

#### __call__(x)

Apply activation function (alias for forward).

#### forward(x)

Compute the activation.

**Parameters:**
- `x` : array-like - Input values.

**Returns:**
- `output` : array - Activated values.

#### backward(x)

Compute the derivative (provided for completeness, not used in gradient-free framework).

**Parameters:**
- `x` : array-like - Input values.

**Returns:**
- `derivative` : array - Derivative values.

---

## Available Activations

### ReLU

Rectified Linear Unit activation.

**Function:** f(x) = max(0, x)

**Class Signature:**
```python
class ReLU(Activation)
```

**Forward:**
```python
output = np.maximum(0, x)
```

**Backward:**
```python
derivative = (x > 0).astype(float)
```

**Properties:**
- Non-saturating (no vanishing gradient for positive inputs)
- Sparse activations (outputs exactly zero for half the inputs)
- Computationally efficient

**Example:**
```python
from EBGA.activations import ReLU

relu = ReLU()
x = np.array([-1, 0, 1, 2])
output = relu.forward(x)  # [0, 0, 1, 2]
```

---

### Sigmoid

Sigmoid (logistic) activation.

**Function:** f(x) = 1 / (1 + exp(-x))

**Class Signature:**
```python
class Sigmoid(Activation)
```

**Forward:**
```python
output = 1 / (1 + np.exp(-x))
```

**Backward:**
```python
s = 1 / (1 + np.exp(-x))
derivative = s * (1 - s)
```

**Properties:**
- Output range: (0, 1)
- S-shaped curve
- Saturates for large |x| (can cause vanishing gradients in deep networks)
- Useful for binary classification outputs

**Example:**
```python
from EBGA.activations import Sigmoid

sigmoid = Sigmoid()
x = np.array([-2, -1, 0, 1, 2])
output = sigmoid.forward(x)  # [0.119, 0.269, 0.5, 0.731, 0.881]
```

---

### Tanh

Hyperbolic tangent activation.

**Function:** f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))

**Class Signature:**
```python
class Tanh(Activation)
```

**Forward:**
```python
output = np.tanh(x)
```

**Backward:**
```python
derivative = 1 - np.tanh(x)**2
```

**Properties:**
- Output range: (-1, 1)
- Zero-centered (mean ~0 for random inputs)
- Saturates for large |x| but less severely than sigmoid
- Often preferred over sigmoid for hidden layers

**Example:**
```python
from EBGA.activations import Tanh

tanh = Tanh()
x = np.array([-2, -1, 0, 1, 2])
output = tanh.forward(x)  # [-0.964, -0.762, 0, 0.762, 0.964]
```

---

### Linear

Linear (identity) activation.

**Function:** f(x) = x

**Class Signature:**
```python
class Linear(Activation)
```

**Forward:**
```python
output = x
```

**Backward:**
```python
derivative = np.ones_like(x)
```

**Properties:**
- No transformation (identity function)
- Used for output layers in regression
- No saturation issues
- Derivative is always 1

**Example:**
```python
from EBGA.activations import Linear

linear = Linear()
x = np.array([1, 2, 3])
output = linear.forward(x)  # [1, 2, 3]
```

---

### Softmax

Softmax activation for multi-class classification.

**Function:** f(x)_i = exp(x_i) / sum(exp(x_j)) for all j

**Class Signature:**
```python
class Softmax(Activation)
```

**Forward:**
```python
x_exp = np.exp(x - np.max(x, axis=axis, keepdims=True))
output = x_exp / np.sum(x_exp, axis=axis, keepdims=True)
```

**Backward:**
Not implemented (not used in gradient-free framework).

**Properties:**
- Converts logits to probabilities
- Output values sum to 1 (valid probability distribution)
- Numerically stable (subtracts max before exp)
- Used for multi-class classification output

**Parameters:**
- `axis` : int - Axis along which to apply softmax. Default: -1 (last axis).

**Example:**
```python
from EBGA.activations import Softmax

softmax = Softmax()
x = np.array([[1, 2, 3], [1, 2, 3]])  # 2 samples, 3 classes
output = softmax.forward(x)
# [[0.090, 0.245, 0.665], [0.090, 0.245, 0.665]]
print(output.sum(axis=1))  # [1.0, 1.0]
```

---

## Factory Function

### get_activation(name)

Get activation function by name.

**Signature:**
```python
def get_activation(name: str) -> Activation
```

**Parameters:**
- `name` : str - Activation name. Options: 'relu', 'sigmoid', 'tanh', 'linear', 'softmax'.

**Returns:**
- `Activation` - An instance of the specified activation.

**Raises:**
- `ValueError` - If the activation name is not recognized.

**Example:**
```python
from EBGA.activations import get_activation

activation = get_activation('relu')
x = np.array([-1, 0, 1])
output = activation(x)
```

---

## Functional API

For convenience, functional versions of all activations are also available:

```python
from EBGA.activations import relu, sigmoid, tanh, linear, softmax

# ReLU
relu(np.array([-1, 0, 1]))  # [0, 0, 1]

# Sigmoid
sigmoid(np.array([-1, 0, 1]))  # [0.269, 0.5, 0.731]

# Tanh
tanh(np.array([-1, 0, 1]))  # [-0.762, 0, 0.762]

# Linear
linear(np.array([1, 2, 3]))  # [1, 2, 3]

# Softmax
softmax(np.array([[1, 2, 3]]), axis=1)  # [[0.090, 0.245, 0.665]]
```

---

## Choosing Activations

| Layer Type | Recommended Activations | Notes |
|------------|------------------------|-------|
| Hidden layers | ReLU, Tanh, Sigmoid | ReLU often best for deep networks |
| Regression output | Linear | Direct continuous output |
| Binary classification output | Sigmoid | Outputs in (0, 1) |
| Multi-class classification output | Softmax | Outputs sum to 1 |

**Best Practices:**
- Use **ReLU** for hidden layers (fast, non-saturating)
- Use **Linear** for regression output
- Use **Sigmoid** for binary classification output
- Use **Softmax** for multi-class classification output
- Experiment with **Tanh** for hidden layers if ReLU doesn't work well
