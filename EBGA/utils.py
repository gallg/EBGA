import numpy as np

def create_bins(y, n_bins, strategy='quantile'):
    """
    Create bins for target variable discretization.

    Args:
        y: Target values
        n_bins: Number of bins
        strategy: Strategy for creating bins ('uniform' or 'quantile')

    Returns:
        Array of bin edges
    """
    if strategy == 'uniform':
        return np.linspace(y.min(), y.max(), n_bins + 1)
    elif strategy == 'quantile':
        return np.quantile(y, np.linspace(0, 1, n_bins + 1))
    else:
        raise ValueError(f"Unknown binning strategy: {strategy}")
