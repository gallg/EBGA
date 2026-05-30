import numpy as np


class Loss:
    
    def __call__(self, y_pred, y_true):
        return self.forward(y_pred, y_true)
    
    def forward(self, y_pred, y_true):
        raise NotImplementedError


class MSE(Loss):
    
    def forward(self, y_pred, y_true):
        return np.mean((y_true - y_pred) ** 2)


class MAE(Loss):
    
    def forward(self, y_pred, y_true):
        return np.mean(np.abs(y_true - y_pred))


class CrossEntropy(Loss):
    
    def forward(self, y_pred, y_true):
        # y_pred: probabilities (after softmax), shape (n_samples, n_classes)
        # y_true: one-hot encoded, shape (n_samples, n_classes)
        eps = 1e-10
        return -np.mean(np.sum(y_true * np.log(y_pred + eps), axis=1))


class BinaryCrossEntropy(Loss):
    
    def forward(self, y_pred, y_true):
        eps = 1e-10
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


# Factory for creating loss instances
LOSS_REGISTRY = {
    'mse': MSE,
    'mae': MAE,
    'cross_entropy': CrossEntropy,
    'binary_cross_entropy': BinaryCrossEntropy,
}


def get_loss(name):
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss: {name}. Available: {list(LOSS_REGISTRY.keys())}")
    return LOSS_REGISTRY[name]()


# Functional API
def mse_loss(y_pred, y_true):
    return MSE().forward(y_pred, y_true)


def mae_loss(y_pred, y_true):
    return MAE().forward(y_pred, y_true)


def cross_entropy_loss(y_pred, y_true):
    return CrossEntropy().forward(y_pred, y_true)


def bce_loss(y_pred, y_true):
    return BinaryCrossEntropy().forward(y_pred, y_true)
