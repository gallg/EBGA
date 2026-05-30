# Best Practices

This guide provides recommendations for using the EBGA framework effectively.

## General Best Practices

### 1. Start Simple

Begin with a simple architecture and gradually increase complexity:

```python
# Start here
model = EBGARegressor(n_layers=1, h_dim=30)

# Only increase if needed
model = EBGARegressor(n_layers=2, h_dim=50)
model = EBGARegressor(n_layers=3, h_dim=100)
```

### 2. Use Simple Mode for Prototyping

The simple mode (n_layers, h_dim) is great for quick experiments:

```python
# Quick prototyping
model = EBGARegressor(n_layers=2, h_dim=50, inner_activation='relu')

# Switch to explicit mode only when needed
model = EBGARegressor(layers=[(128, 'relu'), (64, 'tanh'), (1, 'linear')])
```

### 3. Always Standardize Input Features

Evolutionary algorithms work best with standardized inputs:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

### 4. Use Early Stopping

Prevents overfitting and saves computation:

```python
model = EBGARegressor(
    early_stopping=True,
    patience=100,
    max_iter=10000
)
```

### 5. Set Random State for Reproducibility

```python
model = EBGARegressor(random_state=42)
```

### 6. Use Layer-Wise Training

Layer-wise training is now the default and works well for most cases:

```python
model = EBGARegressor(
    n_layers=2,
    h_dim=50,
    layer_patience=50  # Adjust based on your needs
)
```

## Data Preparation

### Feature Scaling

**Always standardize features:**

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

**Why?** Evolutionary algorithms explore parameter space more effectively when features are on similar scales.

### Target Scaling (Regression)

**Consider normalizing targets:**

```python
from sklearn.preprocessing import MinMaxScaler

# Scale targets to [0, 1]
target_scaler = MinMaxScaler()
y_train = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test = target_scaler.transform(y_test.reshape(-1, 1)).flatten()

# Or use EBGA's built-in normalization
model = EBGARegressor(normalize_output=True)
```

### Handling Missing Values

**Always handle missing values before training:**

```python
# Option 1: Drop rows with missing values
df = df.dropna()

# Option 2: Impute missing values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
```

### Class Labels (Classification)

**Ensure class labels are 0-indexed:**

```python
# If your labels are 1-indexed (1, 2, 3...)
y = y - 1  # Convert to 0-indexed (0, 1, 2...)

# If your labels are strings
y = label_encoder.fit_transform(y)
```

## Model Configuration

### Architecture Design

**Rule of Thumb for h_dim:**
- Start with h_dim ≈ input_size / 2
- Decrease if overfitting
- Increase if underfitting

**Number of Layers:**
- 1 layer: Simple patterns
- 2 layers: Most use cases
- 3+ layers: Complex patterns with lots of data

### Activation Functions

**Recommended combinations:**

| Layer Type | Activation | Notes |
|------------|-----------|-------|
| Hidden | ReLU | Default, fast, non-saturating |
| Hidden | Tanh | Good alternative, zero-centered |
| Hidden | Sigmoid | Less common, can saturate |
| Output (Regression) | Linear | Always use for regression |
| Output (Classification) | Softmax | Use for multi-class |
| Output (Binary) | Sigmoid | Use for binary classification |

### Learning Rates

**Standard approach:**
```python
lr_mu = 0.05    # Mean learning rate
lr_sigma = 0.005  # Sigma learning rate (1/10 of lr_mu)
```

**Tuning:**
- If loss decreases too slowly: increase lr_mu
- If loss oscillates: decrease both lr_mu and lr_sigma
- If σ values are growing uncontrollably: decrease lr_sigma

### Distribution Bounds

```python
sigma_min = 0.001  # Prevent σ from collapsing
sigma_max = 1.0    # Prevent σ from growing too large
```

**Tuning:**
- If training is stuck: try increasing sigma_max
- If training is unstable: try decreasing sigma_max
- sigma_min should be small but not zero

### Layer-Wise Training

**When to use:** Always (it's the default now!)

**Configuration:**
```python
model = EBGARegressor(
    n_layers=3,
    layer_patience=50  # Patience per layer
)
```

**Tuning layer_patience:**
- Smaller: faster training, less thorough per-layer
- Larger: more thorough, slower overall
- Rule of thumb: layer_patience ≈ patience / n_layers

## Training

### Monitoring Training

**Track loss during training:**

```python
# The fit method will print progress
model.fit(X_train, y_train)

# Or monitor manually (not yet built-in)
# Consider adding verbose parameter in future
```

### Early Stopping

**Always use early stopping:**

```python
model = EBGARegressor(
    early_stopping=True,
    patience=100,
    max_iter=10000
)
```

**Choosing patience:**
- Small datasets: patience = 20-50
- Medium datasets: patience = 50-100
- Large datasets: patience = 100-200

### Training Time

**Evolutionary optimization is slower than gradient-based methods:**
- Expect training to take longer
- Each iteration requires multiple function evaluations
- But: no gradient computation, which can be expensive for deep networks

**Tips to speed up training:**
- Use smaller calibration_size (e.g., 10-20)
- Use larger calibration_interval (e.g., 50-100)
- Use fewer layers
- Use smaller h_dim
- Use early stopping

## Evaluation

### Metrics

**Regression:**
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
```

**Classification:**
```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
matrix = confusion_matrix(y_test, y_pred)
```

### Cross-Validation

**Use cross-validation for reliable evaluation:**

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print(f"Mean R²: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
```

## Debugging

### Check Input Data

**Verify data shapes and values:**

```python
print(f"X shape: {X_train.shape}")
print(f"y shape: {y_train.shape}")
print(f"X min: {X_train.min()}, max: {X_train.max()}")
print(f"y min: {y_train.min()}, max: {y_train.max()}")
print(f"Missing values in X: {np.isnan(X_train).sum()}")
print(f"Missing values in y: {np.isnan(y_train).sum()}")
```

### Check Network Output

**Verify network is producing reasonable outputs:**

```python
output = model.network_.forward(X_train[:5])
print(f"Output shape: {output.shape}")
print(f"Output min: {output.min()}, max: {output.max()}")
print(f"Output mean: {output.mean()}, std: {output.std()}")
```

### Check σ Values

**Ensure σ values are reasonable:**

```python
mu, sigma = model.optimizer_.get_distribution_parameters()
print(f"μ min: {mu.min()}, max: {mu.max()}, mean: {mu.mean()}")
print(f"σ min: {sigma.min()}, max: {sigma.max()}, mean: {sigma.mean()}")
print(f"σ at min bound: {(sigma == model.optimizer_.sigma_min).sum()}")
print(f"σ at max bound: {(sigma == model.optimizer_.sigma_max).sum()}")
```

**Warning signs:**
- Many σ values at sigma_min: lr_sigma may be too low
- Many σ values at sigma_max: lr_sigma may be too high or sigma_max too low
- σ values increasing over time: may need smaller lr_sigma

## Common Issues and Solutions

### Issue: Poor Performance

**Symptoms:** Low R²/accuracy, not improving with more iterations

**Possible causes and solutions:**
1. **Insufficient capacity**: Increase n_layers or h_dim
2. **Not enough iterations**: Increase max_iter
3. **Learning rates too small**: Increase lr_mu
4. **Features not scaled**: Standardize input features
5. **Wrong architecture**: Try different activations or layer sizes

### Issue: Training is Slow

**Symptoms:** Each iteration takes a long time

**Possible causes and solutions:**
1. **Large calibration_size**: Reduce to 10-20
2. **Frequent calibration**: Increase calibration_interval to 50-100
3. **Too many layers/parameters**: Reduce n_layers or h_dim
4. **Expensive loss function**: Use simpler loss (e.g., MAE instead of MSE)

### Issue: Loss is Oscillating

**Symptoms:** Loss goes up and down, doesn't converge

**Possible causes and solutions:**
1. **Learning rates too high**: Reduce lr_mu and lr_sigma
2. **Too few samples**: Increase calibration_size
3. **σ values too large**: Reduce sigma_max

### Issue: Training Stuck at Local Minimum

**Symptoms:** Loss plateaus at suboptimal value

**Note:** This is less common with EBGA than gradient-based methods, but can still happen.

**Possible causes and solutions:**
1. **Insufficient exploration**: Increase lr_sigma temporarily
2. **σ values too small**: Check if σ is at sigma_min
3. **Try different initialization**: Use different random_state
4. **Increase diversity**: Use larger calibration_size

### Issue: Overfitting

**Symptoms:** Good training performance, poor test performance

**Possible causes and solutions:**
1. **Too much capacity**: Reduce n_layers or h_dim
2. **Too many iterations**: Use early stopping with smaller patience
3. **Not enough data**: Use more data or simpler model
4. **Try regularization**: EBGA has some built-in regularization via σ

### Issue: Predictions are Constant

**Symptoms:** All predictions are the same value

**Possible causes and solutions:**
1. **σ values collapsed**: Check σ values, increase lr_sigma or sigma_min
2. **Output layer issue**: Verify output activation is correct
3. **Data issue**: Check that input features have variation

## Advanced Tips

### Warm Start

**Start from previously trained parameters:**

```python
# Train initial model
model1 = EBGARegressor(n_layers=2, h_dim=50, max_iter=1000)
model1.fit(X_train, y_train)

# Create new model with more capacity
model2 = EBGARegressor(n_layers=3, h_dim=100, max_iter=5000)

# Initialize with parameters from model1
model2.network_.initialize(X_train.shape[1])
model2.network_.set_all_parameters(model1.network_.get_all_parameters())

# Train further
model2.fit(X_train, y_train)
```

### Custom Loss Functions

**Define your own loss function:**

```python
from EBGA.losses import Loss

class CustomLoss(Loss):
    def forward(self, y_pred, y_true):
        # Your custom loss
        error = y_pred - y_true
        return np.mean(np.abs(error) + 0.1 * error**2)

model = EBGARegressor(loss=CustomLoss(), ...)
```

### Custom Activations

**Define your own activation function:**

```python
from EBGA.activations import Activation

class CustomActivation(Activation):
    def forward(self, x):
        return np.where(x > 0, x, 0.1 * x)  # Leaky ReLU-like
    
    def backward(self, x):
        return np.where(x > 0, 1, 0.1)

model = EBGARegressor(
    layers=[(50, CustomActivation()), (1, 'linear')],
    ...
)
```

### Callbacks

**Currently not built-in, but you can wrap the fit method:**

```python
def fit_with_callback(model, X, y, callback):
    # This would need to be implemented
    # For now, use the low-level API
    pass
```

## Contributing Best Practices

If you're contributing to the EBGA framework:

1. **Follow existing patterns**: Match the coding style of the framework
2. **Add tests**: Ensure new features work correctly
3. **Update documentation**: Document new features and changes
4. **Keep it simple**: Prefer simplicity over complexity
5. **Maintain compatibility**: Don't break existing APIs
6. **Test with benchmarks**: Run main.py and test_IXI.py to ensure nothing breaks
