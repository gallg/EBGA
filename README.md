# EBGA: Evolutionary-Based Generative Adaptation

**EBGA** (**E**volutionary-**B**ased **G**enerative **A**daptation) is a machine learning framework that provides an alternative to classical gradient-based methods. It uses evolutionary computation with softmax-weighted recombination (NES-style), enabling training of neural networks without computing objective function gradients.

## Installation

```bash
pip install -e .
```

Requires Python 3.10+, numpy, scikit-learn.

## Quick Start

```python
from EBGA.models import EBGARegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = EBGARegressor(
    layers=[(50, 'relu'), (1, 'linear')],
    max_iter=1000,
    random_state=42
)
model.fit(X_train, y_train)
print(f"R² Score: {model.score(X_test, y_test):.4f}")
```

For full documentation, examples, and API reference see [docs/index.md](docs/index.md).

## License

GNU General Public License v3.0 (GPL-3.0). See [LICENSE](LICENSE).
