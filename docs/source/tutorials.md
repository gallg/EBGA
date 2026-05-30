# Tutorials

Step-by-step guides to using the EBGA framework.

## Tutorial 1: Basic Regression

Learn how to use EBGA for a simple regression task.

### Goal
Predict diabetes disease progression using brain phenotype features.

### Steps

1. **Import the framework**
```python
from EBGA.models import EBGARegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
```

2. **Load and prepare data**
```python
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

3. **Create and train the model**
```python
model = EBGARegressor(
    n_layers=2,              # 2 hidden layers
    h_dim=50,                # 50 units per hidden layer
    inner_activation='relu', # ReLU activation for hidden layers
    output_activation='linear',
    max_iter=10000,
    random_state=42
)

model.fit(X_train, y_train)
```

4. **Evaluate**
```python
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2:.4f}")
```

### Complete Code

See `main.py` Tutorial 1 for the complete working example.

---

## Tutorial 2: Explicit Layer Architecture

Learn how to define custom network architectures.

### Simple Mode vs Explicit Mode

**Simple Mode** (automatic):
```python
model = EBGARegressor(
    n_layers=2,
    h_dim=50,
    inner_activation='relu'
)
# Creates: Linear(50, 'relu') -> Linear(50, 'relu') -> Linear(1, 'linear')
```

**Explicit Mode** (full control):
```python
model = EBGARegressor(
    layers=[
        (128, 'relu'),
        (64, 'tanh'),
        (1, 'linear')
    ]
)
# Creates exactly the specified architecture
```

### Example: Deep Network

```python
model = EBGARegressor(
    layers=[
        (256, 'relu'),
        (128, 'relu'),
        (64, 'relu'),
        (32, 'relu'),
        (1, 'linear')
    ],
    max_iter=20000
)
```

### Example: Mixed Activations

```python
model = EBGARegressor(
    layers=[
        (64, 'relu'),
        (32, 'sigmoid'),
        (1, 'linear')
    ]
)
```

### When to Use Each

- **Simple Mode**: Quick prototyping, standard architectures
- **Explicit Mode**: Custom architectures, research, specific layer sizes

---

## Tutorial 3: Building Custom Networks

Learn how to use the low-level API to build and train custom networks.

### Building a Network from Scratch

```python
from EBGA.nn import Sequential
from EBGA.layers import Linear
from EBGA.optimizer import CompactEvoOptimizer
from EBGA.losses import mae_loss
import numpy as np

# 1. Build network
network = Sequential(
    Linear(64, activation='relu'),
    Linear(32, activation='tanh'),
    Linear(1, activation='linear')
)

# 2. Initialize
network.initialize(input_size=X_train.shape[1])

# 3. Create optimizer
optimizer = CompactEvoOptimizer(
    param_dim=network.parameter_count(),
    lr_mu=0.03,
    lr_sigma=0.03,
    random_state=42
)

# 4. Initialize with scale-aware setup
initial_params = network.get_all_parameters()
initial_params[-1] = np.mean(y_train)  # Set output bias to target mean
optimizer.initialize(initial_params)
network.set_all_parameters(optimizer.get_parameters())

# 5. Define loss function
def loss_func(params):
    network.set_all_parameters(params)
    y_pred = network.forward(X_train).flatten()
    loss = mae_loss(y_pred, y_train)
    if np.any(np.abs(params) > 1e5):
        return float('inf')
    return loss

# 6. Train
for iteration in range(1000):
    optimizer.step(loss_func, iteration=iteration)

# 7. Predict
y_pred = network.forward(X_test).flatten()
```

### See Also

See `main.py` Tutorial 3 for a complete example with layer-wise training.

---

## Tutorial 4: Classification

Learn how to use EBGA for classification tasks.

### Binary Classification

```python
from EBGA.models import EBGAClassifier
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
X, y = cancer.data, cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = EBGAClassifier(
    n_layers=1,
    h_dim=10,
    inner_activation='relu',
    output_activation='softmax',
    n_classes=2,
    max_iter=2000,
    random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")

# Get class probabilities
y_proba = model.predict_proba(X_test)
```

### Multi-class Classification

```python
from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target

model = EBGAClassifier(
    layers=[(10, 'relu'), (3, 'softmax')],
    n_classes=3,
    max_iter=2000
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=iris.target_names))
```

---

## Tutorial 5: Layer-Wise Training

Understand how layer-wise training works and how to configure it.

### How It Works

1. Train Layer 1 until loss plateaus
2. Freeze Layer 1, train Layer 2 until loss plateaus
3. Continue for all layers
4. Fine-tune all layers together

### Configuration

```python
model = EBGARegressor(
    n_layers=3,
    h_dim=50,
    layer_patience=50,  # Patience for each layer
    max_iter=10000      # Total iterations (divided among layers)
)
```

### What layer_patience Controls

- If a layer's loss doesn't improve for `layer_patience` iterations, training moves to the next layer
- Smaller values: faster layer switching, less thorough per-layer training
- Larger values: more thorough per-layer training, slower overall progress

### Recommended Values

| Network Depth | layer_patience | Notes |
|---------------|----------------|-------|
| Shallow (1-2 layers) | 30-50 | Standard |
| Medium (3-5 layers) | 50-100 | More thorough |
| Deep (5+ layers) | 100-200 | Very thorough |

---

## Tutorial 6: Hyperparameter Tuning

Learn how to tune EBGA hyperparameters for better performance.

### Key Hyperparameters

| Hyperparameter | Typical Range | Effect |
|---------------|---------------|--------|
| `lr_mu` | [0.001, 0.1] | Controls μ update speed |
| `lr_sigma` | [0.0001, 0.01] | Controls σ adaptation speed |
| `sigma_min` | [0.001, 0.01] | Prevents σ collapse |
| `sigma_max` | [0.1, 10.0] | Prevents σ explosion |
| `calibration_size` | [10, 100] | More samples = more accurate |
| `calibration_interval` | [10, 100] | More frequent = more stable |
| `credit_factor` | [1.0, 20.0] | Higher = stronger credit assignment |
| `max_iter` | [1000, 100000] | Total training iterations |
| `layer_patience` | [30, 200] | Patience per layer |

### Tuning Strategy

1. **Start with defaults**
2. **Monitor training** - Check if loss is:
   - Decreasing too slowly: increase `lr_mu`
   - Oscillating: decrease `lr_mu` and `lr_sigma`
   - Not improving: increase `calibration_size` or `credit_factor`
3. **Check σ values** - They should:
   - Start around 0.1
   - Adapt to reasonable ranges (not too small or too large)
   - If σ values are stuck at bounds: adjust `sigma_min` or `sigma_max`

### Example: Grid Search

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'lr_mu': [0.01, 0.05, 0.1],
    'lr_sigma': [0.001, 0.005, 0.01],
    'calibration_interval': [20, 50, 100],
    'max_iter': [5000]
}

grid_search = GridSearchCV(
    EBGARegressor(n_layers=2, h_dim=50),
    param_grid,
    cv=3,
    scoring='r2',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
print(f"Best params: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.4f}")
```

---

## Tutorial 7: Working with IXI Dataset

Learn how to use EBGA with the IXI brain phenotype dataset.

### Load and Prepare Data

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
phenotypes = pd.read_csv('data/ixi.csv')
demo = pd.read_csv('data/final_demo.csv')

# Merge datasets
merged = pd.merge(
    phenotypes,
    demo[['IXI_ID', 'SEX_ID (1=m, 2=f)']],
    left_on='ID',
    right_on='IXI_ID'
)

# Extract features
feature_cols = [col for col in merged.columns 
                if col.startswith('lh_') or col.startswith('rh_')]
X = merged[feature_cols].values

# Extract targets
y_age = merged['Age'].values
y_sex = merged['SEX_ID (1=m, 2=f)'].values - 1  # Convert to 0-indexed
```

### Train Age Regressor

```python
from EBGA.models import EBGARegressor

X_train, X_test, y_train, y_test = train_test_split(X, y_age, test_size=0.2)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = EBGARegressor(
    layers=[(128, 'relu'), (64, 'relu'), (1, 'linear')],
    max_iter=5000,
    lr_mu=0.001,
    lr_sigma=0.001,
    layer_patience=50,
    random_state=42
)

model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
```

### Train Sex Classifier

```python
from EBGA.models import EBGAClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y_sex, test_size=0.2)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = EBGAClassifier(
    layers=[(64, 'relu'), (32, 'relu'), (2, 'softmax')],
    n_classes=2,
    max_iter=3000,
    lr_mu=0.01,
    lr_sigma=0.01,
    layer_patience=30,
    random_state=42
)

model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
```

### See Also

See `test_IXI.py` for the complete working example.

---

## Tutorial 8: Saving and Loading Models

Learn how to save and load trained models.

### Saving a Model

```python
import joblib

# Save the entire model
joblib.dump(model, 'model.pkl')

# Or save components separately
joblib.dump({
    'params': model.network_.get_all_parameters(),
    'mu': model.optimizer_.mu,
    'sigma': model.optimizer_.sigma,
    'config': {
        'layers': [(l.output_size, l.activation.__class__.__name__.lower()) 
                  for l in model.network_.layers],
        'lr_mu': model.optimizer_.lr_mu,
        'lr_sigma': model.optimizer_.lr_sigma,
        # ... other hyperparameters
    }
}, 'model_components.pkl')
```

### Loading a Model

```python
import joblib

# Load the entire model
model = joblib.load('model.pkl')

# Or rebuild from components
saved = joblib.load('model_components.pkl')
model = EBGARegressor(
    layers=saved['config']['layers'],
    lr_mu=saved['config']['lr_mu'],
    lr_sigma=saved['config']['lr_sigma']
)
model.network_.initialize(X_train.shape[1])
model.network_.set_all_parameters(saved['params'])
model.optimizer_.mu = saved['mu']
model.optimizer_.sigma = saved['sigma']
```

---

## Next Steps

1. **Try the examples** in `main.py` and `test_IXI.py`
2. **Experiment** with different architectures and hyperparameters
3. **Compare** EBGA with traditional frameworks on your datasets
4. **Contribute** to the framework by submitting issues or pull requests
5. **Explore** the advanced tutorials and API documentation
