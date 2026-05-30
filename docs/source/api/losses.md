# losses Module

Loss functions for training neural networks.

## Loss Base Class

Abstract base class for all loss functions.

### Class Signature

```python
class Loss
```

### Methods

#### __call__(y_pred, y_true)

Compute loss (alias for forward).

#### forward(y_pred, y_true)

Compute the loss value.

**Parameters:**
- `y_pred` : array-like - Predicted values.
- `y_true` : array-like - True/target values.

**Returns:**
- `loss` : float - The computed loss value.

---

## Available Loss Functions

### MSE (Mean Squared Error)

Mean Squared Error for regression tasks.

**Function:** L = (1/n) * Σ (y_true - y_pred)²

**Class Signature:**
```python
class MSE(Loss)
```

**Forward:**
```python
loss = np.mean((y_true - y_pred) ** 2)
```

**Properties:**
- Commonly used for regression
- Sensitive to outliers (quadratic term)
- Smooth and differentiable everywhere
- Scale-dependent (larger errors contribute disproportionately)

**Example:**
```python
from EBGA.losses import MSE
import numpy as np

mse = MSE()
y_true = np.array([1, 2, 3, 4])
y_pred = np.array([1.1, 1.9, 3.2, 3.8])
loss = mse(y_pred, y_true)
print(f"MSE Loss: {loss:.4f}")
```

---

### MAE (Mean Absolute Error)

Mean Absolute Error for regression tasks.

**Function:** L = (1/n) * Σ |y_true - y_pred|

**Class Signature:**
```python
class MAE(Loss)
```

**Forward:**
```python
loss = np.mean(np.abs(y_true - y_pred))
```

**Properties:**
- Commonly used for regression
- Robust to outliers (linear term)
- Not differentiable at zero, but EBGA doesn't require gradients
- Scale-independent interpretation (direct error magnitude)
- Often preferred over MSE for robustness

**Example:**
```python
from EBGA.losses import MAE
import numpy as np

mae = MAE()
y_true = np.array([1, 2, 3, 4])
y_pred = np.array([1.1, 1.9, 3.2, 3.8])
loss = mae(y_pred, y_true)
print(f"MAE Loss: {loss:.4f}")
```

---

### CrossEntropy (Cross-Entropy)

Cross-entropy loss for multi-class classification.

**Function:** L = - (1/n) * Σ Σ y_true_ij * log(y_pred_ij)

**Class Signature:**
```python
class CrossEntropy(Loss)
```

**Forward:**
```python
# y_pred: probabilities (after softmax), shape (n_samples, n_classes)
# y_true: one-hot encoded, shape (n_samples, n_classes)
eps = 1e-10
loss = -np.mean(np.sum(y_true * np.log(y_pred + eps), axis=1))
```

**Properties:**
- Standard loss for multi-class classification
- Requires probability predictions (after softmax)
- Requires one-hot encoded targets
- Heavily penalizes confident wrong predictions
- Numerically stable (adds epsilon to prevent log(0))

**Example:**
```python
from EBGA.losses import CrossEntropy
from EBGA.activations import softmax
import numpy as np

cross_entropy = CrossEntropy()

# Logits (before softmax)
logits = np.array([[1, 2, 3], [3, 2, 1]])
# Apply softmax to get probabilities
probs = softmax(logits)

# One-hot encoded targets
y_true = np.array([[0, 0, 1], [1, 0, 0]])

loss = cross_entropy(probs, y_true)
print(f"Cross-Entropy Loss: {loss:.4f}")
```

---

### BinaryCrossEntropy (Binary Cross-Entropy)

Binary cross-entropy loss for binary classification.

**Function:** L = - (1/n) * Σ [y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred)]

**Class Signature:**
```python
class BinaryCrossEntropy(Loss)
```

**Forward:**
```python
eps = 1e-10
y_pred = np.clip(y_pred, eps, 1 - eps)
loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
```

**Properties:**
- Standard loss for binary classification
- Requires probability predictions in (0, 1)
- Works with binary targets (0 or 1)
- Numerically stable (clips predictions)

**Example:**
```python
from EBGA.losses import BinaryCrossEntropy
from EBGA.activations import sigmoid
import numpy as np

bce = BinaryCrossEntropy()

# Logits (before sigmoid)
logits = np.array([0.5, -0.5, 1.0, -1.0])
# Apply sigmoid to get probabilities
probs = sigmoid(logits)

# Binary targets
y_true = np.array([1, 0, 1, 0])

loss = bce(probs, y_true)
print(f"Binary Cross-Entropy Loss: {loss:.4f}")
```

---

## Factory Function

### get_loss(name)

Get loss function by name.

**Signature:**
```python
def get_loss(name: str) -> Loss
```

**Parameters:**
- `name` : str - Loss name. Options: 'mse', 'mae', 'cross_entropy', 'binary_cross_entropy'.

**Returns:**
- `Loss` - An instance of the specified loss function.

**Raises:**
- `ValueError` - If the loss name is not recognized.

**Example:**
```python
from EBGA.losses import get_loss
import numpy as np

loss_fn = get_loss('mae')
y_true = np.array([1, 2, 3])
y_pred = np.array([1.1, 1.9, 3.1])
loss = loss_fn(y_pred, y_true)
```

---

## Functional API

For convenience, functional versions of all loss functions are also available:

```python
from EBGA.losses import mse_loss, mae_loss, cross_entropy_loss, bce_loss
import numpy as np

# MSE
y_true = np.array([1, 2, 3])
y_pred = np.array([1.1, 1.9, 3.1])
loss = mse_loss(y_pred, y_true)

# MAE
loss = mae_loss(y_pred, y_true)

# Cross-Entropy (with softmax)
from EBGA.activations import softmax
logits = np.array([[1, 2, 3]])
probs = softmax(logits)
y_true_onehot = np.array([[0, 0, 1]])
loss = cross_entropy_loss(probs, y_true_onehot)

# Binary Cross-Entropy
from EBGA.activations import sigmoid
logits = np.array([0.5, -0.5])
probs = sigmoid(logits)
y_true_binary = np.array([1, 0])
loss = bce_loss(probs, y_true_binary)
```

---

## Choosing Loss Functions

| Task Type | Recommended Loss | Notes |
|-----------|----------------|-------|
| Regression (robust) | MAE | Less sensitive to outliers |
| Regression (smooth) | MSE | Standard, differentiable |
| Multi-class classification | CrossEntropy | Standard for multi-class |
| Binary classification | BinaryCrossEntropy | Standard for binary |
| Probabilistic regression | MSE | With appropriate output activation |

**Special Cases:**
- Use **MAE** when you have outliers in your data
- Use **MSE** when you want smoother optimization
- For classification, the loss must match your output activation (softmax → CrossEntropy, sigmoid → BinaryCrossEntropy)
