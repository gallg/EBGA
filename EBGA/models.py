import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils import check_random_state
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import LabelBinarizer

from EBGA.nn import Sequential
from EBGA.layers import Dense
from EBGA.losses import get_loss
from EBGA.optimizer import CompactEvoOptimizer


def _run_optimizer_training(optimizer, loss_func, max_iterations, early_stopping, patience):
    best_loss = float('inf')
    patience_counter = 0

    for iteration in range(max_iterations):
        loss = optimizer.step(loss_func, iteration=iteration)

        if early_stopping:
            current_loss = loss_func(optimizer.get_parameters())
            if current_loss < best_loss:
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

    return best_loss, patience_counter


class BaseModel(BaseEstimator):

    def __init__(self, layers,
                 lr_mu, lr_sigma, sigma_min, sigma_max,
                 calibration_size,
                 max_iter, early_stopping, patience,
                 random_state, use_layerwise, optimizer,
                 momentum, trust_region_radius, batch_size):

        self.layers = layers
        self.lr_mu = lr_mu
        self.lr_sigma = lr_sigma
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.calibration_size = calibration_size
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.patience = patience
        self.random_state = random_state
        self.use_layerwise = use_layerwise
        self.optimizer = optimizer
        self.momentum = momentum
        self.trust_region_radius = trust_region_radius
        self.batch_size = batch_size

        self._random_state = None
        self.network_ = None
        self.optimizer_ = None
        self.label_binarizer_ = None
        self.n_features_ = None
        self.n_classes_ = None

    def get_params(self, deep=True):
        return {
            'layers': self.layers,
            'lr_mu': self.lr_mu,
            'lr_sigma': self.lr_sigma,
            'sigma_min': self.sigma_min,
            'sigma_max': self.sigma_max,
            'calibration_size': self.calibration_size,
            'momentum': self.momentum,
            'trust_region_radius': self.trust_region_radius,
            'max_iter': self.max_iter,
            'early_stopping': self.early_stopping,
            'patience': self.patience,
            'random_state': self.random_state,
            'use_layerwise': self.use_layerwise,
            'optimizer': self.optimizer,
            'batch_size': self.batch_size
        }

    def set_params(self, **params):
        for param, value in params.items():
            if hasattr(self, param):
                setattr(self, param, value)
            else:
                if '__' in param:
                    pass
        return self

    def _get_optimizer_config(self):
        return {
            'lr_mu': self.lr_mu,
            'lr_sigma': self.lr_sigma,
            'sigma_min': self.sigma_min,
            'sigma_max': self.sigma_max,
            'calibration_size': self.calibration_size,
            'momentum': self.momentum,
            'trust_region_radius': self.trust_region_radius,
            'random_state': self._random_state
        }

    def _create_batches(self, X, y):
        if self.batch_size is None:
            return [(X, y)]

        n_samples = X.shape[0]
        batch_size = self.batch_size
        n_full_batches = n_samples // batch_size
        remainder = n_samples % batch_size

        if remainder > 0 and remainder < batch_size:
            batches = []
            for i in range(n_full_batches):
                start = i * batch_size
                if i == n_full_batches - 1:
                    end = start + batch_size + remainder
                else:
                    end = start + batch_size
                batches.append((X[start:end], y[start:end]))
            return batches
        else:
            batches = []
            for i in range(n_full_batches):
                start = i * batch_size
                end = start + batch_size
                batches.append((X[start:end], y[start:end]))
            return batches

    def _build_network(self, input_size, output_size):
        network_layers = []

        if self.layers is None or len(self.layers) == 0:
            network_layers.append(Dense(output_size, activation='linear'))
        else:
            for size, activation in self.layers:
                layer_activation = activation if activation is not None else 'linear'
                network_layers.append(Dense(size, activation=layer_activation))

        return Sequential(*network_layers)

    def _fit_layer_wise(self, X, y, loss_func=None):
        if loss_func is None:
            if self.batch_size is not None:
                loss_func = self._create_batched_loss_func(X, y)
            else:
                loss_func = self._create_loss_func(X, y)

        n_layers = len(self.network_.layers)
        self.network_.initialize(self.n_features_, scale_aware=y)

        optimizer_config = self._get_optimizer_config()

        last_layer = self.network_.layers[-1]
        output_size = last_layer.output_size
        output_activation = last_layer.activation

        if hasattr(output_activation, '__class__'):
            activation_name = output_activation.__class__.__name__.lower()
        else:
            activation_name = str(output_activation).lower() if output_activation else None

        is_classification = hasattr(self, 'n_classes_') and self.n_classes_ is not None
        if is_classification and activation_name != 'softmax':
            temp_output_activation = 'softmax'
        else:
            temp_output_activation = output_activation

        # Phase 1: Greedy layer-wise pretraining
        for layer_idx in range(n_layers):
            partial_layers = []

            for i in range(layer_idx + 1):
                main_layer = self.network_.layers[i]
                new_layer = Dense(main_layer.output_size, activation=main_layer.activation)
                partial_layers.append(new_layer)

            if layer_idx < n_layers - 1:
                temp_output_layer = Dense(output_size, activation=temp_output_activation)
                partial_layers.append(temp_output_layer)

            partial_network = Sequential(*partial_layers)
            partial_network.initialize(self.n_features_)

            if layer_idx > 0:
                main_params = self.network_.get_all_parameters()
                partial_params = []
                offset = 0
                for i in range(layer_idx + 1):
                    layer = self.network_.layers[i]
                    layer_param_count = layer.parameter_count()
                    partial_params.append(main_params[offset:offset + layer_param_count])
                    offset += layer_param_count

                if layer_idx < n_layers - 1:
                    temp_param_count = partial_network.layers[-1].parameter_count()
                    partial_params.append(
                        self._random_state.randn(temp_param_count) * 0.01
                    )

                partial_network.set_all_parameters(np.concatenate(partial_params))

            partial_optimizer = self.optimizer(
                param_dim=partial_network.parameter_count(),
                **optimizer_config
            )

            if self.batch_size is not None:
                batches = self._create_batches(X, y)
                batch_index = [0]

                def partial_loss(params):
                    idx = batch_index[0] % len(batches)
                    X_batch, y_batch = batches[idx]
                    batch_index[0] += 1

                    current = partial_network.get_all_parameters()
                    partial_network.set_all_parameters(params)
                    y_pred = partial_network.forward(X_batch)
                    if partial_network.output_size == 1:
                        y_pred = y_pred.flatten()
                    loss = self.loss_(y_pred, y_batch)
                    partial_network.set_all_parameters(current)
                    if np.any(np.abs(params) > 1e5):
                        return float('inf')
                    return loss
            else:
                def partial_loss(params):
                    current = partial_network.get_all_parameters()
                    partial_network.set_all_parameters(params)
                    y_pred = partial_network.forward(X)
                    if partial_network.output_size == 1:
                        y_pred = y_pred.flatten()
                    loss = self.loss_(y_pred, y)
                    partial_network.set_all_parameters(current)
                    if np.any(np.abs(params) > 1e5):
                        return float('inf')
                    return loss

            partial_optimizer.initialize()
            layer_iterations = self.max_iter // n_layers

            for iteration in range(layer_iterations):
                partial_optimizer.step(partial_loss, iteration=iteration)

            trained_params = partial_network.get_all_parameters()
            main_params = self.network_.get_all_parameters()

            param_offset = 0
            for i in range(layer_idx + 1):
                layer_param_count = self.network_.layers[i].parameter_count()
                main_params[param_offset:param_offset + layer_param_count] = \
                    trained_params[param_offset:param_offset + layer_param_count]
                param_offset += layer_param_count

            self.network_.set_all_parameters(main_params)

        # Phase 2: Fine-tuning all layers together
        final_iterations = self.max_iter // 2
        param_dim = self.network_.parameter_count()
        final_optimizer = self.optimizer(
            param_dim=param_dim,
            **optimizer_config
        )
        current_params = self.network_.get_all_parameters()
        final_optimizer.initialize(current_params)

        _run_optimizer_training(
            final_optimizer, loss_func, final_iterations, self.early_stopping, self.patience
        )

        self.network_.set_all_parameters(final_optimizer.get_parameters())
        self.optimizer_ = final_optimizer

    def _fit_direct(self, X, y, loss_func=None):
        if loss_func is None:
            if self.batch_size is not None:
                loss_func = self._create_batched_loss_func(X, y)
            else:
                loss_func = self._create_loss_func(X, y)

        self.network_.initialize(self.n_features_, scale_aware=y)

        optimizer_config = self._get_optimizer_config()

        final_optimizer = self.optimizer(
            param_dim=self.network_.parameter_count(),
            **optimizer_config
        )

        current_params = self.network_.get_all_parameters()
        final_optimizer.initialize(current_params)

        _run_optimizer_training(
            final_optimizer, loss_func, self.max_iter, self.early_stopping, self.patience
        )

        self.network_.set_all_parameters(final_optimizer.get_parameters())
        self.optimizer_ = final_optimizer

    def _create_loss_func(self, X, y):
        def loss_func(params):
            current = self.network_.get_all_parameters()
            self.network_.set_all_parameters(params)
            y_pred = self.network_.forward(X)
            if self.network_.output_size == 1:
                y_pred = y_pred.flatten()
            loss = self.loss_(y_pred, y)
            self.network_.set_all_parameters(current)
            if np.any(np.abs(params) > 1e5):
                return float('inf')
            return loss
        return loss_func

    def _wrap_with_batching(self, loss_func, X, y):
        batches = self._create_batches(X, y)
        batch_index = [0]

        def batched_loss_func(params):
            idx = batch_index[0] % len(batches)
            X_batch, y_batch = batches[idx]
            batch_index[0] += 1
            return loss_func(params, X_batch, y_batch)

        return batched_loss_func

    def _create_batched_loss_func(self, X, y):
        def base_loss(params, X_batch, y_batch):
            current = self.network_.get_all_parameters()
            self.network_.set_all_parameters(params)
            y_pred = self.network_.forward(X_batch)
            if self.network_.output_size == 1:
                y_pred = y_pred.flatten()
            loss = self.loss_(y_pred, y_batch)
            self.network_.set_all_parameters(current)
            if np.any(np.abs(params) > 1e5):
                return float('inf')
            return loss

        return self._wrap_with_batching(base_loss, X, y)


class EBGARegressor(BaseModel, RegressorMixin):
    """
    EBGA Regressor - Evolutionary neural network for regression.

    Parameters
    ----------
    layers : list of tuples, default=None
        Network architecture. Each tuple is (output_size, activation).
        Example: [(50, 'relu'), (1, 'linear')]
    loss : str or Loss, default='mae'
        Loss function name or instance.
    optimizer : class, default=CompactEvoOptimizer
        Optimizer class to use.
    use_layerwise : bool, default=False
        If True, train each layer greedily then fine-tune all together.
    lr_mu : float, default=0.03
        Temperature for softmax weighting.
    lr_sigma : float, default=0.03
        Learning rate for sigma adaptation.
    sigma_min : float, default=0.001
        Minimum sigma.
    sigma_max : float, default=1.0
        Maximum sigma.
    calibration_size : int, default=10
        Population size per step.
    max_iter : int, default=10000
        Maximum iterations.
    early_stopping : bool, default=True
        Enable early stopping.
    patience : int, default=100
        Patience for early stopping.
    normalize_output : bool, default=False
        If True, scale output to 0-1 range.
    random_state : int, optional
        Random seed.
    momentum : float, default=0.9
        Momentum coefficient.
    trust_region_radius : float, default=0.1
        Maximum update norm per step.
    batch_size : int, optional
        Batch size for mini-batch training.
    """

    def __init__(self, layers=None, loss='mae',
                 lr_mu=0.03, lr_sigma=0.03, sigma_min=0.001, sigma_max=1.0,
                 calibration_size=10,
                 max_iter=10000, early_stopping=True, patience=100,
                 normalize_output=False, random_state=None,
                 use_layerwise=False, optimizer=CompactEvoOptimizer,
                 momentum=0.9, trust_region_radius=0.1, batch_size=None):

        self.normalize_output = normalize_output

        super().__init__(
            layers=layers,
            lr_mu=lr_mu, lr_sigma=lr_sigma, sigma_min=sigma_min, sigma_max=sigma_max,
            calibration_size=calibration_size,
            max_iter=max_iter, early_stopping=early_stopping, patience=patience,
            random_state=random_state, use_layerwise=use_layerwise, optimizer=optimizer,
            momentum=momentum, trust_region_radius=trust_region_radius,
            batch_size=batch_size
        )

        self._loss_str = loss if isinstance(loss, str) else None
        if isinstance(loss, str):
            self.loss_ = get_loss(loss)
        else:
            self.loss_ = loss

    def get_params(self, deep=True):
        params = super().get_params(deep)
        params.update({
            'loss': self._loss_str if hasattr(self, '_loss_str') and self._loss_str is not None else 'mse',
            'normalize_output': self.normalize_output
        })
        return params

    def set_params(self, **params):
        if 'loss' in params:
            loss = params.pop('loss')
            if isinstance(loss, str):
                self.loss_ = get_loss(loss)
            else:
                self.loss_ = loss
        super().set_params(**params)
        return self

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self._random_state = check_random_state(self.random_state)
        self.n_features_ = X.shape[1]

        if self.normalize_output:
            self.y_min_ = np.min(y)
            self.y_max_ = np.max(y)
            y_normalized = (y - self.y_min_) / (self.y_max_ - self.y_min_ + 1e-8)
        else:
            y_normalized = y
            self.y_min_ = None
            self.y_max_ = None

        if self.layers is None or len(self.layers) == 0:
            output_size = 1
        else:
            output_size = self.layers[-1][0]

        self.network_ = self._build_network(self.n_features_, output_size=output_size)

        if self.use_layerwise:
            self._fit_layer_wise(X, y_normalized)
        else:
            self._fit_direct(X, y_normalized)

        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        output = self.network_.forward(X)
        if self.network_.output_size == 1:
            output = output.flatten()

        if self.normalize_output and self.y_min_ is not None and self.y_max_ is not None:
            output = output * (self.y_max_ - self.y_min_) + self.y_min_

        return output

    def score(self, X, y):
        return r2_score(y, self.predict(X))


class EBGAClassifier(BaseModel, ClassifierMixin):
    """
    EBGA Classifier - Evolutionary neural network for classification.

    Parameters
    ----------
    layers : list of tuples, default=None
        Network architecture. Each tuple is (output_size, activation).
        Example: [(50, 'relu'), (10, 'softmax')]
    n_classes : int, optional
        Number of classes. If None, inferred from data.
    loss : str or Loss, default='cross_entropy'
        Loss function name or instance.
    optimizer : class, default=CompactEvoOptimizer
        Optimizer class to use.
    use_layerwise : bool, default=False
        If True, train each layer greedily then fine-tune all together.
    lr_mu : float, default=0.05
        Temperature for softmax weighting.
    lr_sigma : float, default=0.005
        Learning rate for sigma adaptation.
    sigma_min : float, default=0.001
        Minimum sigma.
    sigma_max : float, default=1.0
        Maximum sigma.
    calibration_size : int, default=10
        Population size per step.
    max_iter : int, default=500
        Maximum iterations.
    early_stopping : bool, default=True
        Enable early stopping.
    patience : int, default=20
        Patience for early stopping.
    random_state : int, optional
        Random seed.
    momentum : float, default=0.5
        Momentum coefficient.
    trust_region_radius : float, default=None
        Maximum update norm per step.
    batch_size : int, optional
        Batch size for mini-batch training.
    """

    def __init__(self, layers=None, n_classes=None, loss='cross_entropy',
                 lr_mu=0.05, lr_sigma=0.005, sigma_min=0.001, sigma_max=1.0,
                 calibration_size=10,
                 max_iter=500, early_stopping=True, patience=20,
                 random_state=None, use_layerwise=False, optimizer=CompactEvoOptimizer,
                 momentum=0.5, trust_region_radius=None, batch_size=None):

        self.n_classes = n_classes

        super().__init__(
            layers=layers,
            lr_mu=lr_mu, lr_sigma=lr_sigma, sigma_min=sigma_min, sigma_max=sigma_max,
            calibration_size=calibration_size,
            max_iter=max_iter, early_stopping=early_stopping, patience=patience,
            random_state=random_state, use_layerwise=use_layerwise, optimizer=optimizer,
            momentum=momentum, trust_region_radius=trust_region_radius,
            batch_size=batch_size
        )

        self._loss_str = loss if isinstance(loss, str) else None
        if isinstance(loss, str):
            self.loss_ = get_loss(loss)
        else:
            self.loss_ = loss

    def get_params(self, deep=True):
        params = super().get_params(deep)
        params.update({
            'n_classes': self.n_classes,
            'loss': self._loss_str if hasattr(self, '_loss_str') and self._loss_str is not None else 'cross_entropy'
        })
        return params

    def set_params(self, **params):
        if 'loss' in params:
            loss = params.pop('loss')
            if isinstance(loss, str):
                self.loss_ = get_loss(loss)
            else:
                self.loss_ = loss
        super().set_params(**params)
        return self

    def _create_classification_loss_func(self, X, y_onehot):
        def class_base_loss(params, X_batch, y_batch):
            current = self.network_.get_all_parameters()
            self.network_.set_all_parameters(params)
            y_pred = self.network_.forward(X_batch)
            loss = self.loss_(y_pred, y_batch)
            self.network_.set_all_parameters(current)
            if np.any(np.abs(params) > 1e5):
                return float('inf')
            return loss

        if self.batch_size is not None:
            return self._wrap_with_batching(class_base_loss, X, y_onehot)
        else:
            def loss_func(params):
                return class_base_loss(params, X, y_onehot)
            return loss_func

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self._random_state = check_random_state(self.random_state)
        self.n_features_ = X.shape[1]

        if self.n_classes is None:
            self.n_classes_ = len(np.unique(y))
        else:
            self.n_classes_ = self.n_classes

        self.label_binarizer_ = LabelBinarizer()
        y_onehot = self.label_binarizer_.fit_transform(y)

        if self.layers is None or len(self.layers) == 0:
            output_size = self.n_classes_
        else:
            output_size = self.layers[-1][0]

        self.network_ = self._build_network(self.n_features_, output_size=output_size)

        loss_func = self._create_classification_loss_func(X, y_onehot)

        if self.use_layerwise:
            self._fit_layer_wise(X, y_onehot, loss_func=loss_func)
        else:
            self._fit_direct(X, y_onehot, loss_func=loss_func)

        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        output = self.network_.forward(X)

        last_layer = self.network_.layers[-1]
        activation = last_layer.activation

        if hasattr(activation, '__class__'):
            activation_name = activation.__class__.__name__.lower()
        else:
            activation_name = str(activation).lower() if activation else 'none'

        if activation_name == 'sigmoid':
            if output.shape[1] == 1:
                return (output >= 0.5).astype(int).flatten()
            elif self.n_classes_ == 2 and output.shape[1] == 2:
                import warnings
                warnings.warn(
                    f"Sigmoid activation with 2 output neurons for {self.n_classes_}-class "
                    f"classification is non-standard. Falling back to argmax.",
                    UserWarning
                )
                return np.argmax(output, axis=1)
            else:
                import warnings
                warnings.warn(
                    f"Sigmoid activation with {self.n_classes_} classes is non-standard. "
                    f"Falling back to argmax.",
                    UserWarning
                )
                return np.argmax(output, axis=1)
        else:
            return np.argmax(output, axis=1)

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        output = self.network_.forward(X)

        last_layer = self.network_.layers[-1]
        activation = last_layer.activation

        if hasattr(activation, '__class__'):
            activation_name = activation.__class__.__name__.lower()
        else:
            activation_name = str(activation).lower() if activation else 'none'

        if activation_name == 'softmax':
            return output
        else:
            output_exp = np.exp(output - np.max(output, axis=1, keepdims=True))
            return output_exp / np.sum(output_exp, axis=1, keepdims=True)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))
