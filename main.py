from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from EBGA.genetic_descent import GeneticDescentRegressor

# Generate data
X, y = make_regression(n_samples=1000, n_features=6, noise=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit model
gd = GeneticDescentRegressor(
    n_bins=5,
    pop_size=50,
    max_iter=500,
    lr_μ=0.08,
    lr_σ=0.001,
    random_state=42
)

print("Fitting Genetic Descent Regressor...")
gd.fit(X_train, y_train)

# Evaluate
print(f"\nTraining R²: {gd.score(X_train, y_train):.4f}")
print(f"Validation R²: {gd.score(X_val, y_val):.4f}")

# Make some predictions
sample_pred = gd.predict(X_val[:5])
print(f"\nSample predictions:\n{sample_pred}")
