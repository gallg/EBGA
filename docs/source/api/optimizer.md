# optimizer Module

The core optimization algorithm: Compact Evolutionary Optimizer.

## CompactEvoOptimizer

Distribution-based evolutionary optimization algorithm that maintains a Gaussian distribution over parameters (μ, σ) and updates it using natural gradient.

### Class Signature

```python
class CompactEvoOptimizer(param_dim, lr_mu=0.05, lr_sigma=0.005,
                        sigma_min=0.001, sigma_max=1.0,
                        calibration_size=20, calibration_interval=25,
                        credit_factor=2.0, random_state=None)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `param_dim` | int | Required | Dimensionality of the parameter space. |
| `lr_mu` | float | 0.05 | Learning rate for mean (μ) parameters. |
| `lr_sigma` | float | 0.005 | Learning rate for standard deviation (σ) parameters. |
| `sigma_min` | float | 0.001 | Minimum value for σ (prevents collapse). |
| `sigma_max` | float | 1.0 | Maximum value for σ (prevents explosion). |
| `calibration_size` | int | 20 | Number of samples for population calibration. |
| `calibration_interval` | int | 25 | How often to perform population calibration (in iterations). |
| `credit_factor` | float | 2.0 | Strength of credit assignment in pairwise updates. |
| `random_state` | RandomState or int | None | Random number generator. |

### Methods

#### initialize(initial_params=None)
Initialize the distribution parameters.
- `initial_params`: array, optional - Initial parameter values.

#### step(loss_func, iteration=None)
Perform one optimization step.
- `loss_func`: callable - Function that computes loss given parameters.
- `iteration`: int, optional - Current iteration number.
- Returns: `avg_loss` - Average loss from the samples evaluated.

#### get_parameters()
Get current mean parameters. Returns array of shape (param_dim,).

#### get_distribution_parameters()
Get both μ and σ. Returns tuple of (mu, sigma) arrays.

#### set_parameters(params)
Set the mean parameters.
- `params`: array, shape (param_dim,) - New mean parameters.

#### state_dict()
Get optimizer state as a dictionary.

#### load_state_dict(state_dict)
Load optimizer state from a dictionary.
- `state_dict`: dict - Dictionary containing optimizer state.

### Example

```python
from EBGA.optimizer import CompactEvoOptimizer
import numpy as np

def loss_func(params):
    return np.sum((params - np.array([1.0, 1.0]))**2)

optimizer = CompactEvoOptimizer(param_dim=2, random_state=42)
optimizer.initialize()

for iteration in range(100):
    loss = optimizer.step(loss_func, iteration=iteration)

final_params = optimizer.get_parameters()
```

### Hyperparameter Tuning

**Learning Rates:**
- `lr_mu`: [0.001, 0.1] - Controls μ movement speed
- `lr_sigma`: [0.0001, 0.01] - Typically 1/10 of lr_mu
- Start with defaults: lr_mu=0.05, lr_sigma=0.005
- If oscillating: reduce both
- If not improving: increase lr_mu

**Distribution Bounds:**
- `sigma_min`: [0.001, 0.01] - Prevents σ collapse
- `sigma_max`: [0.1, 10.0] - Prevents σ explosion
- Start with sigma_min=0.001, sigma_max=1.0

**Calibration:**
- `calibration_size`: [10, 100] - More samples = more accurate but slower
- `calibration_interval`: [10, 100] - Less frequent = faster but less stable
- `credit_factor`: [1.0, 20.0] - Higher = more credit to large improvements
