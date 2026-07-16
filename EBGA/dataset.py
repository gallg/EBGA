import numpy as np


class Dataset:
    """
    Dataset class for batching, similar to PyTorch's Dataset.

    For numpy arrays, pass X and y to the constructor.
    For custom datasets, subclass and implement __len__ and __getitem__.

    Args:
        X: Input features (optional, for array-based datasets)
        y: Target values (optional, for array-based datasets)
        batch_size: Number of samples per batch (None = full dataset)
        shuffle: Shuffle data at each epoch
        random_state: Random seed for shuffling
    """

    def __init__(self, X=None, y=None, batch_size=None, shuffle=False, random_state=None):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.X = np.asarray(X) if X is not None else None
        self.y = np.asarray(y) if y is not None else None

    def __len__(self):
        if self.X is not None:
            return len(self.X)
        raise NotImplementedError

    def __getitem__(self, idx):
        if self.X is not None and self.y is not None:
            return self.X[idx], self.y[idx]
        raise NotImplementedError

    def batches(self):
        """
        Yield (X_batch, y_batch) numpy arrays.
        """
        indices = np.arange(len(self))
        if self.shuffle:
            rng = np.random.RandomState(self.random_state)
            indices = rng.permutation(indices)

        batch_size = self.batch_size if self.batch_size is not None else len(self)

        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            X_list, y_list = [], []
            for idx in batch_indices:
                X_sample, y_sample = self[idx]
                X_list.append(X_sample)
                y_list.append(y_sample)
            yield np.array(X_list), np.array(y_list)
