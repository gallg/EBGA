import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils import check_random_state
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import LabelBinarizer

from EBGA.nn import Sequential
from EBGA.layers import Linear
from EBGA.activations import get_activation
from EBGA.losses import get_loss
from EBGA.optimizer import CompactEvoOptimizer


# =============================================================================
# UTILITY FUNCTIONS FOR LAYER-WISE TRAINING
# =============================================================================

def _build_layers_from_params_simple(n_layers, h_dim, inner_activation, output_activation):
    """
    Build layer configuration from simple parameters.
    
    Args:
        n_layers: Number of hidden layers
        h_dim: Size of each hidden layer
        inner_activation: Activation function for hidden layers
        output_activation: Activation function for output layer
        
    Returns:
        list: List of (size, activation) tuples
    """
    layers = []
    for i in range(n_layers):
        layers.append((h_dim, inner_activation))
    # Output layer
    layers.append((1, output_activation))
    return layers


def _create_layer_param_ranges(network):
    """
    Create parameter ranges for each layer in the network.
    
    Args:
        network: Sequential network instance
        
    Returns:
        list: List of (start, end) parameter index tuples for each layer
    """
    layer_param_ranges = []
    offset = 0
    for layer in network.layers:
        param_count = layer.parameter_count()
        layer_param_ranges.append((offset, offset + param_count))
        offset += param_count
    return layer_param_ranges


def _create_layer_optimizer(param_dim, lr_mu, lr_sigma, sigma_min, sigma_max,
                           calibration_size, calibration_interval, credit_factor,
                           sigma_regularization, random_state):
    """
    Create a CompactEvoOptimizer for a specific parameter dimension.
    
    Args:
        param_dim: Number of parameters to optimize
        lr_mu: Learning rate for mean
        lr_sigma: Learning rate for sigma
        sigma_min: Minimum sigma value
        sigma_max: Maximum sigma value
        calibration_size: Population size for calibration
        calibration_interval: Calibration frequency
        credit_factor: Credit assignment strength
        sigma_regularization: Sigma regularization strength
        random_state: Random state
        
    Returns:
        CompactEvoOptimizer: Configured optimizer instance
    """
    return CompactEvoOptimizer(
        param_dim=param_dim,
        lr_mu=lr_mu,
        lr_sigma=lr_sigma,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        calibration_size=calibration_size,
        calibration_interval=calibration_interval,
        credit_factor=credit_factor,
        sigma_regularization=sigma_regularization,
        random_state=random_state
    )


def _train_single_layer(layer_optimizer, layer_params, layer_loss_func,
                        max_iterations, layer_patience):
    """
    Train a single layer until plateau or max iterations.
    
    Args:
        layer_optimizer: Optimizer for this layer
        layer_params: Initial parameters for this layer
        layer_loss_func: Loss function for layer optimization
        max_iterations: Maximum number of iterations
        layer_patience: Patience for early stopping
        
    Returns:
        numpy.ndarray: Trained layer parameters
    """
    layer_optimizer.initialize(layer_params)
    
    best_layer_loss = float('inf')
    layer_patience_counter = 0
    
    for iteration in range(max_iterations):
        loss = layer_optimizer.step(layer_loss_func, iteration=iteration)
        
        # Check for plateau
        current_layer_loss = layer_loss_func(layer_optimizer.get_parameters())
        if current_layer_loss < best_layer_loss:
            best_layer_loss = current_layer_loss
            layer_patience_counter = 0
        else:
            layer_patience_counter += 1
            if layer_patience_counter >= layer_patience:
                break
    
    return layer_optimizer.get_parameters()


def _train_all_layers_together(network, optimizer_config, loss_func,
                               max_iterations, early_stopping, patience):
    """
    Train all layers together using the full network optimizer.
    
    Args:
        network: Sequential network instance
        optimizer_config: Dictionary with optimizer configuration
        loss_func: Loss function for full network
        max_iterations: Maximum number of iterations
        early_stopping: Whether to use early stopping
        patience: Patience for early stopping
        
    Returns:
        CompactEvoOptimizer: Trained optimizer with final parameters
    """
    param_dim = network.parameter_count()
    final_optimizer = CompactEvoOptimizer(
        param_dim=param_dim,
        lr_mu=optimizer_config['lr_mu'],
        lr_sigma=optimizer_config['lr_sigma'],
        sigma_min=optimizer_config['sigma_min'],
        sigma_max=optimizer_config['sigma_max'],
        calibration_size=optimizer_config['calibration_size'],
        calibration_interval=optimizer_config['calibration_interval'],
        credit_factor=optimizer_config['credit_factor'],
        sigma_regularization=optimizer_config['sigma_regularization'],
        random_state=optimizer_config['random_state']
    )
    
    # Initialize with current parameters
    current_params = network.get_all_parameters()
    final_optimizer.initialize(current_params)
    network.set_all_parameters(final_optimizer.get_parameters())
    
    # Train all layers together
    best_loss = float('inf')
    patience_counter = 0
    
    for iteration in range(max_iterations):
        loss = final_optimizer.step(loss_func, iteration=iteration)
        
        if early_stopping:
            current_loss = loss_func(final_optimizer.get_parameters())
            if current_loss < best_loss:
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
    
    return final_optimizer


class BaseModel(BaseEstimator):
    
    def __init__(self, layers=None, output_activation='linear',
                 lr_mu=0.05, lr_sigma=0.005, sigma_min=0.001, sigma_max=1.0,
                 calibration_size=20, calibration_interval=25, credit_factor=2.0,
                 sigma_regularization=0.0, max_iter=500, early_stopping=True, 
                 patience=20, random_state=None, layer_patience=50):
        
        # Store hyperparameters
        self.layers = layers
        self.output_activation = output_activation
        self.lr_mu = lr_mu
        self.lr_sigma = lr_sigma
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.calibration_size = calibration_size
        self.calibration_interval = calibration_interval
        self.credit_factor = credit_factor
        self.sigma_regularization = sigma_regularization
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.patience = patience
        self.random_state = random_state
        self.layer_patience = layer_patience
        
        # Will be initialized in fit()
        self._random_state = None
        self.network_ = None
        self.optimizer_ = None
        self.label_binarizer_ = None
        self.n_features_ = None
        self.n_classes_ = None
    
    def _get_optimizer_config(self):
        """Get optimizer configuration as a dictionary."""
        return {
            'lr_mu': self.lr_mu,
            'lr_sigma': self.lr_sigma,
            'sigma_min': self.sigma_min,
            'sigma_max': self.sigma_max,
            'calibration_size': self.calibration_size,
            'calibration_interval': self.calibration_interval,
            'credit_factor': self.credit_factor,
            'sigma_regularization': self.sigma_regularization,
            'random_state': self._random_state
        }
    
    def _build_network(self, input_size, output_size):
        """
        Build the neural network from layer specifications.
        
        Args:
            input_size: Number of input features
            output_size: Number of output units
            
        Returns:
            Sequential: Built network
        """
        network_layers = []
        
        if self.layers is None or len(self.layers) == 0:
            # Default: single output layer
            network_layers.append(Linear(output_size, activation=self.output_activation))
        else:
            # Build layers from specification - use sizes as specified
            for i, (size, activation) in enumerate(self.layers):
                if i == len(self.layers) - 1:
                    # Last layer: use specified size and activation
                    # If activation is None, use self.output_activation
                    layer_activation = activation if activation is not None else self.output_activation
                    network_layers.append(Linear(size, activation=layer_activation))
                else:
                    # Hidden layer: use specified size and activation
                    network_layers.append(Linear(size, activation=activation))
        
        return Sequential(*network_layers)
    
    def _initialize_parameters_with_scale_awareness(self, y):
        """
        Initialize network parameters with scale-aware output bias.
        
        Args:
            y: Target values for scale estimation
        """
        all_params = self.network_.get_all_parameters()
        target_mean = np.mean(y)
        # Set last parameter (output bias) to target mean
        all_params[-1] = target_mean
        self.network_.set_all_parameters(all_params)
    
    def _create_layer_loss_func(self, layer_params, start_param, end_param, full_loss_func):
        """
        Create a layer-specific loss function that only updates one layer.
        
        Args:
            layer_params: Parameters for this layer
            start_param: Start index of layer parameters in full parameter array
            end_param: End index of layer parameters in full parameter array
            full_loss_func: Full network loss function
            
        Returns:
            function: Layer-specific loss function
        """
        def layer_loss_func(layer_params):
            # Combine with other layers' parameters
            full_params = np.concatenate([
                self.network_.get_all_parameters()[:start_param],
                layer_params,
                self.network_.get_all_parameters()[end_param:]
            ])
            self.network_.set_all_parameters(full_params)
            return full_loss_func(full_params)
        return layer_loss_func
    
    def _update_layer_parameters(self, trained_layer_params, start_param, end_param):
        """
        Update network with trained layer parameters.
        
        Args:
            trained_layer_params: Trained parameters for the layer
            start_param: Start index of layer parameters
            end_param: End index of layer parameters
        """
        updated_all_params = np.concatenate([
            self.network_.get_all_parameters()[:start_param],
            trained_layer_params,
            self.network_.get_all_parameters()[end_param:]
        ])
        self.network_.set_all_parameters(updated_all_params)
    
    def _fit_layer_wise(self, X, y, loss_func=None):
        """
        Train network using layer-wise evolutionary optimization.
        
        This is the core training method used by both regressor and classifier.
        Subclasses can override this method or provide a custom loss_func.
        
        Args:
            X: Input data
            y: Target data
            loss_func: Optional custom loss function. If None, uses _create_loss_func
        """
        if loss_func is None:
            loss_func = self._create_loss_func(X, y)
        
        n_layers = len(self.network_.layers)
        
        # Initialize all layers
        self.network_.initialize(self.n_features_)
        
        # Initialize parameters with scale-aware output bias
        self._initialize_parameters_with_scale_awareness(y)
        
        # Get parameter ranges for each layer
        layer_param_ranges = _create_layer_param_ranges(self.network_)
        
        # Get optimizer configuration
        optimizer_config = self._get_optimizer_config()
        
        # Train layers sequentially
        for layer_idx in range(n_layers):
            start_param, end_param = layer_param_ranges[layer_idx]
            layer_param_dim = end_param - start_param
            
            # Create optimizer for this layer
            layer_optimizer = _create_layer_optimizer(
                param_dim=layer_param_dim,
                **optimizer_config
            )
            
            # Get current layer parameters
            current_all_params = self.network_.get_all_parameters()
            layer_params = current_all_params[start_param:end_param]
            
            # Create layer-specific loss function
            layer_loss_func = self._create_layer_loss_func(
                layer_params, start_param, end_param, loss_func
            )
            
            # Train this layer
            layer_iterations = self.max_iter // n_layers
            trained_layer_params = _train_single_layer(
                layer_optimizer, layer_params, layer_loss_func,
                layer_iterations, self.layer_patience
            )
            
            # Update network with trained layer parameters
            self._update_layer_parameters(trained_layer_params, start_param, end_param)
        
        # Final training pass on all layers together
        final_iterations = self.max_iter // 2
        final_optimizer = _train_all_layers_together(
            network=self.network_,
            optimizer_config=optimizer_config,
            loss_func=loss_func,
            max_iterations=final_iterations,
            early_stopping=self.early_stopping,
            patience=self.patience
        )
        
        # Set final parameters
        self.network_.set_all_parameters(final_optimizer.get_parameters())
        self.optimizer_ = final_optimizer  # Store optimizer for reference
    
    def _create_loss_func(self, X, y):
        """
        Create the default loss function for training.
        Override this method in subclasses for custom loss calculation.
        
        Args:
            X: Input data
            y: Target data
            
        Returns:
            function: Loss function that takes parameters and returns loss value
        """
        def loss_func(params):
            self.network_.set_all_parameters(params)
            y_pred = self.network_.forward(X)
            if self.network_.output_size == 1:
                y_pred = y_pred.flatten()
            loss = self.loss_(y_pred, y)
            # Add parameter clipping penalty for numerical stability
            if np.any(np.abs(params) > 1e5):
                return float('inf')
            return loss
        return loss_func


class EBGARegressor(BaseModel, RegressorMixin):
    """
    EBGA Regressor - Evolutionary neural network for regression.
    
    Can be configured either via explicit layers list or via simple parameters:
    - Use `layers` for full control over architecture
    - Use `n_layers` + `h_dim` for simple multi-layer networks

    Parameters:
        layers: list of tuples, default=None
            Network architecture. Each tuple is (output_size, activation)
            If None, uses n_layers and h_dim to build network.
        n_layers: int, default=1
            Number of hidden layers (excluding output layer).
            Only used if layers=None.
        h_dim: int, default=50
            Size of each hidden layer.
            Only used if layers=None.
        inner_activation: str, default='relu'
            Activation for hidden layers.
            Only used if layers=None.
        output_activation: str, default='linear'
            Activation for output layer
        loss: str or Loss, default='mae'
            Loss function name or instance
        lr_mu: float, default=0.03
            Learning rate for mean
        lr_sigma: float, default=0.03
            Learning rate for sigma
        sigma_min: float, default=0.001
            Minimum sigma
        sigma_max: float, default=1.0
            Maximum sigma
        calibration_size: int, default=30
            Population size for calibration
        calibration_interval: int, default=50
            Calibration frequency
        credit_factor: float, default=2.0
            Credit assignment strength
        max_iter: int, default=10000
            Maximum iterations
        early_stopping: bool, default=True
            Enable early stopping
        patience: int, default=100
            Patience for early stopping
        layer_patience: int, default=50
            Patience for layer-wise plateau detection.
        normalize_output: bool, default=False
            If True, scale output to 0-1 range.
        random_state: int, optional
            Random seed
        
    Example:
        >>> from EBGA.models import EBGARegressor
        >>> # Using simple parameters
        >>> model = EBGARegressor(
        ...     n_layers=2,
        ...     h_dim=50,
        ...     inner_activation='relu',
        ...     output_activation='linear',
        ...     max_iter=1000
        ... )
        >>> # Or using explicit layers
        >>> model = EBGARegressor(
        ...     layers=[(50, 'relu'), (1, 'linear')],
        ...     max_iter=1000
        ... )
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """
    
    def __init__(self, layers=None, n_layers=1, h_dim=50, inner_activation='relu',
                 output_activation='linear', loss='mae',
                 lr_mu=0.03, lr_sigma=0.03, sigma_min=0.001, sigma_max=1.0,
                 calibration_size=30, calibration_interval=50, credit_factor=2.0,
                 sigma_regularization=0.0, max_iter=10000, early_stopping=True, 
                 patience=100, layer_patience=50, normalize_output=False,
                 random_state=None):
        
        # Store new parameters
        self.n_layers = n_layers
        self.h_dim = h_dim
        self.inner_activation = inner_activation
        self.layer_patience = layer_patience
        self.normalize_output = normalize_output
        self.output_activation = output_activation
        
        # Build layers if not provided
        if layers is None:
            layers = self._build_layers_from_params()
        
        super().__init__(
            layers=layers, output_activation=output_activation,
            lr_mu=lr_mu, lr_sigma=lr_sigma, sigma_min=sigma_min, sigma_max=sigma_max,
            calibration_size=calibration_size, calibration_interval=calibration_interval,
            credit_factor=credit_factor, sigma_regularization=sigma_regularization,
            max_iter=max_iter, early_stopping=early_stopping, patience=patience, 
            random_state=random_state, layer_patience=layer_patience
        )
        
        # Set up loss function
        if isinstance(loss, str):
            self.loss_ = get_loss(loss)
        else:
            self.loss_ = loss
    
    def _build_layers_from_params(self):
        """Build layer configuration from n_layers, h_dim, and activations."""
        return _build_layers_from_params_simple(
            self.n_layers, self.h_dim, self.inner_activation, self.output_activation
        )
    
    def fit(self, X, y):
        """
        Fit the regressor to training data.
        
        Args:
            X: Input features
            y: Target values
            
        Returns:
            self: Fitted regressor
        """
        X, y = check_X_y(X, y)
        self._random_state = check_random_state(self.random_state)
        self.n_features_ = X.shape[1]
        
        # Normalize target if requested
        if self.normalize_output:
            self.y_min_ = np.min(y)
            self.y_max_ = np.max(y)
            y_normalized = (y - self.y_min_) / (self.y_max_ - self.y_min_ + 1e-8)
        else:
            y_normalized = y
            self.y_min_ = None
            self.y_max_ = None
        
        # Build network
        self.network_ = self._build_network(self.n_features_, output_size=1)
        self.network_.initialize(self.n_features_)
        
        # Always use layer-wise training
        self._fit_layer_wise(X, y_normalized)
        
        return self
    
    def predict(self, X):
        """
        Predict target values for input data.
        
        Args:
            X: Input features
            
        Returns:
            numpy.ndarray: Predicted target values
        """
        check_is_fitted(self)
        X = check_array(X)
        output = self.network_.forward(X)
        if self.network_.output_size == 1:
            output = output.flatten()
        
        # Denormalize output if normalization was used during training
        if self.normalize_output and self.y_min_ is not None and self.y_max_ is not None:
            output = output * (self.y_max_ - self.y_min_) + self.y_min_
        
        return output
    
    def score(self, X, y):
        """
        Compute R² score on test data.
        
        Args:
            X: Input features
            y: True target values
            
        Returns:
            float: R² score
        """
        return r2_score(y, self.predict(X))


class EBGAClassifier(BaseModel, ClassifierMixin):
    """
    EBGA Classifier - Evolutionary neural network for classification.
    
    Can be configured either via explicit layers list or via simple parameters:
    - Use `layers` for full control over architecture
    - Use `n_layers` + `h_dim` for simple multi-layer networks

    Parameters:
        layers: list of tuples, default=None
            Network architecture. Each tuple is (output_size, activation)
            If None, uses n_layers and h_dim to build network.
        n_classes: int, optional
            Number of classes. If None, inferred from data.
        n_layers: int, default=1
            Number of hidden layers (excluding output layer).
            Only used if layers=None.
        h_dim: int, default=50
            Size of each hidden layer.
            Only used if layers=None.
        inner_activation: str, default='relu'
            Activation for hidden layers.
            Only used if layers=None.
        output_activation: str, default='softmax'
            Activation for output layer
        loss: str or Loss, default='cross_entropy'
            Loss function name or instance
        lr_mu: float, default=0.05
            Learning rate for mean
        lr_sigma: float, default=0.005
            Learning rate for sigma
        sigma_min: float, default=0.001
            Minimum sigma
        sigma_max: float, default=1.0
            Maximum sigma
        calibration_size: int, default=20
            Population size for calibration
        calibration_interval: int, default=25
            Calibration frequency
        credit_factor: float, default=2.0
            Credit assignment strength
        max_iter: int, default=500
            Maximum iterations
        early_stopping: bool, default=True
            Enable early stopping
        patience: int, default=20
            Patience for early stopping
        layer_patience: int, default=50
            Patience for layer-wise plateau detection.
        random_state: int, optional
            Random seed
        
    Example:
        >>> from EBGA.models import EBGAClassifier
        >>> # Using simple parameters
        >>> model = EBGAClassifier(
        ...     n_layers=2,
        ...     h_dim=50,
        ...     inner_activation='relu',
        ...     n_classes=10,
        ...     output_activation='softmax'
        ... )
        >>> # Or using explicit layers
        >>> model = EBGAClassifier(
        ...     layers=[(50, 'relu'), (10, 'relu')],
        ...     n_classes=10,
        ...     output_activation='softmax'
        ... )
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """
    
    def __init__(self, layers=None, n_classes=None, n_layers=1, h_dim=50, inner_activation='relu',
                 output_activation='softmax', loss='cross_entropy',
                 lr_mu=0.05, lr_sigma=0.005, sigma_min=0.001, sigma_max=1.0,
                 calibration_size=20, calibration_interval=25, credit_factor=2.0,
                 sigma_regularization=0.0, max_iter=500, early_stopping=True, 
                 patience=20, layer_patience=50,
                 random_state=None):
        
        # Store new parameters
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.h_dim = h_dim
        self.inner_activation = inner_activation
        self.layer_patience = layer_patience
        self.output_activation = output_activation
        
        # Build layers if not provided
        if layers is None:
            layers = self._build_layers_from_params()
        
        super().__init__(
            layers=layers, output_activation=output_activation,
            lr_mu=lr_mu, lr_sigma=lr_sigma, sigma_min=sigma_min, sigma_max=sigma_max,
            calibration_size=calibration_size, calibration_interval=calibration_interval,
            credit_factor=credit_factor, sigma_regularization=sigma_regularization,
            max_iter=max_iter, early_stopping=early_stopping, patience=patience, 
            random_state=random_state, layer_patience=layer_patience
        )
        
        # Set up loss function
        if isinstance(loss, str):
            self.loss_ = get_loss(loss)
        else:
            self.loss_ = loss
    
    def _build_layers_from_params(self):
        """Build layer configuration from n_layers, h_dim, and activations."""
        layers = []
        for i in range(self.n_layers):
            layers.append((self.h_dim, self.inner_activation))
        # Output layer - size will be set to n_classes in fit()
        layers.append((1, self.output_activation))  # Will be resized in fit
        return layers
    
    def _create_classification_loss_func(self, X, y_onehot):
        """
        Create classification-specific loss function with softmax activation.
        
        Args:
            X: Input data
            y_onehot: One-hot encoded target labels
            
        Returns:
            function: Loss function for classification
        """
        def loss_func(params):
            self.network_.set_all_parameters(params)
            y_pred = self.network_.forward(X)
            # Apply softmax if needed
            if self.output_activation == 'softmax':
                y_pred = np.exp(y_pred - np.max(y_pred, axis=1, keepdims=True))
                y_pred = y_pred / np.sum(y_pred, axis=1, keepdims=True)
            loss = self.loss_(y_pred, y_onehot)
            # Add parameter clipping penalty for numerical stability
            if np.any(np.abs(params) > 1e5):
                return float('inf')
            return loss
        return loss_func
    
    def fit(self, X, y):
        """
        Fit the classifier to training data.
        
        Args:
            X: Input features
            y: Target class labels
            
        Returns:
            self: Fitted classifier
        """
        X, y = check_X_y(X, y)
        self._random_state = check_random_state(self.random_state)
        self.n_features_ = X.shape[1]
        
        # Infer number of classes if not specified
        if self.n_classes is None:
            self.n_classes_ = len(np.unique(y))
        else:
            self.n_classes_ = self.n_classes
        
        # Set up label binarizer for one-hot encoding
        self.label_binarizer_ = LabelBinarizer()
        y_onehot = self.label_binarizer_.fit_transform(y)
        
        # Build network
        self.network_ = self._build_network(self.n_features_, output_size=self.n_classes_)
        self.network_.initialize(self.n_features_)
        
        # Use shared layer-wise training with classification-specific loss
        loss_func = self._create_classification_loss_func(X, y_onehot)
        self._fit_layer_wise(X, y_onehot, loss_func=loss_func)
        
        return self
    
    def predict(self, X):
        """
        Predict class labels for input data.
        
        Args:
            X: Input features
            
        Returns:
            numpy.ndarray: Predicted class labels
        """
        check_is_fitted(self)
        X = check_array(X)
        output = self.network_.forward(X)
        # Return class with highest score
        return np.argmax(output, axis=1)
    
    def predict_proba(self, X):
        """
        Predict class probabilities for input data.
        
        Args:
            X: Input features
            
        Returns:
            numpy.ndarray: Class probabilities (softmax output)
        """
        check_is_fitted(self)
        X = check_array(X)
        output = self.network_.forward(X)
        # Apply softmax
        output = np.exp(output - np.max(output, axis=1, keepdims=True))
        output = output / np.sum(output, axis=1, keepdims=True)
        return output
    
    def score(self, X, y):
        """
        Compute accuracy score on test data.
        
        Args:
            X: Input features
            y: True class labels
            
        Returns:
            float: Accuracy score
        """
        return accuracy_score(y, self.predict(X))
