import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils import check_random_state
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import LabelBinarizer

from EBGA.nn import Sequential
from EBGA.layers import Linear
from EBGA.losses import get_loss
from EBGA.optimizer import CompactEvoOptimizer, MultiCandidateOptimizer


# =============================================================================
# UTILITY FUNCTIONS
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


def _create_optimizer(optimizer_class, param_dim, optimizer_config, n_candidates=None):
    """
    Create an optimizer of the specified class.
    
    Args:
        optimizer_class: The optimizer class to instantiate (e.g., CompactEvoOptimizer)
        param_dim: Number of parameters to optimize
        optimizer_config: Dictionary with optimizer configuration parameters
        n_candidates: Number of candidates (for MultiCandidateOptimizer, ignored for others)
        
    Returns:
        Optimizer instance of the specified class
    """
    # Handle MultiCandidateOptimizer which needs n_candidates
    if optimizer_class == MultiCandidateOptimizer:
        if n_candidates is None:
            n_candidates = 3  # Default for MultiCandidate
        return optimizer_class(
            param_dim=param_dim,
            n_candidates=n_candidates,
            **optimizer_config
        )
    else:
        return optimizer_class(
            param_dim=param_dim,
            **optimizer_config
        )



def _create_stateless_loss_func(network, loss_func):
    """
    Create a stateless loss function that prevents network state pollution.
    
    This function wraps the given loss function to ensure the network state
    is preserved after each loss evaluation, preventing side effects during optimization.
    
    Args:
        network: Sequential network instance
        loss_func: The base loss function to wrap (takes params, returns loss)
        
    Returns:
        function: Stateless loss function
    """
    def stateless_loss_func(params):
        current = network.get_all_parameters()
        network.set_all_parameters(params)
        loss = loss_func(params)
        network.set_all_parameters(current)
        return loss
    return stateless_loss_func


def _run_optimizer_training(optimizer, stateless_loss_func, max_iterations, early_stopping, patience):
    """
    Run the optimizer training loop with early stopping support.
    
    Args:
        optimizer: The optimizer instance to train
        stateless_loss_func: Stateless loss function for evaluation
        max_iterations: Maximum number of iterations
        early_stopping: Whether to use early stopping
        patience: Patience for early stopping
        
    Returns:
        tuple: (best_loss, patience_counter) after training
    """
    best_loss = float('inf')
    patience_counter = 0
    
    for iteration in range(max_iterations):
        loss = optimizer.step(stateless_loss_func, iteration=iteration)
        
        if early_stopping:
            current_loss = stateless_loss_func(optimizer.get_parameters())
            if current_loss < best_loss:
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
    
    return best_loss, patience_counter



def _train_all_layers_together(network, optimizer_class, optimizer_config, loss_func,
                               max_iterations, early_stopping, patience, n_candidates=None):
    """
    Train all layers together using the specified optimizer class.
    
    Args:
        network: Sequential network instance
        optimizer_class: The optimizer class to use
        optimizer_config: Dictionary with optimizer configuration
        loss_func: Loss function for full network
        max_iterations: Maximum number of iterations
        early_stopping: Whether to use early stopping
        patience: Patience for early stopping
        n_candidates: Number of candidates (for MultiCandidateOptimizer)
        
    Returns:
        Optimizer instance with final parameters
    """
    param_dim = network.parameter_count()
    
    # Create optimizer of specified class using utility function
    final_optimizer = _create_optimizer(
        optimizer_class, param_dim, optimizer_config, n_candidates
    )
    
    # Initialize with current parameters
    current_params = network.get_all_parameters()
    final_optimizer.initialize(current_params)
    
    # Create stateless loss function to prevent state pollution
    stateless_loss_func = _create_stateless_loss_func(network, loss_func)
    
    # Train all layers together
    _run_optimizer_training(
        final_optimizer, stateless_loss_func, max_iterations, early_stopping, patience
    )
    
    return final_optimizer


class BaseModel(BaseEstimator):
    
    def __init__(self, layers=None, output_activation='linear',
                 lr_mu=0.05, lr_sigma=0.005, sigma_min=0.001, sigma_max=1.0,
                 calibration_size=20, calibration_interval=25, credit_factor=2.0,
                 sigma_regularization=0.0, max_iter=500, early_stopping=True, 
                 patience=20, random_state=None, layer_patience=50,
                 use_layerwise=False, optimizer=CompactEvoOptimizer, n_candidates=None):
        
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
        self.use_layerwise = use_layerwise
        self.optimizer = optimizer
        self.n_candidates = n_candidates
        
        # Will be initialized in fit()
        self._random_state = None
        self.network_ = None
        self.optimizer_ = None
        self.label_binarizer_ = None
        self.n_features_ = None
        self.n_classes_ = None
    
    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        
        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        
        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        params = {
            'layers': self.layers,
            'output_activation': self.output_activation,
            'lr_mu': self.lr_mu,
            'lr_sigma': self.lr_sigma,
            'sigma_min': self.sigma_min,
            'sigma_max': self.sigma_max,
            'calibration_size': self.calibration_size,
            'calibration_interval': self.calibration_interval,
            'credit_factor': self.credit_factor,
            'sigma_regularization': self.sigma_regularization,
            'max_iter': self.max_iter,
            'early_stopping': self.early_stopping,
            'patience': self.patience,
            'random_state': self.random_state,
            'layer_patience': self.layer_patience,
            'use_layerwise': self.use_layerwise,
            'optimizer': self.optimizer,
            'n_candidates': self.n_candidates
        }
        return params
    
    def set_params(self, **params):
        """
        Set the parameters of this estimator.
        
        Parameters
        ----------
        **params : dict
            Estimator parameters.
        
        Returns
        -------
        self : object
            Estimator instance.
        """
        for param, value in params.items():
            if hasattr(self, param):
                setattr(self, param, value)
            else:
                # Handle nested parameters (for pipelines)
                if '__' in param:
                    # This will be handled by sklearn's set_params
                    pass
        return self
    
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
    
    def _fit_layer_wise(self, X, y, loss_func=None):
        """
        Train network using layer-wise evolutionary optimization with greedy pretraining.
        
        Each layer is trained as if it plus all previous layers form a complete model
        that predicts the final output. For intermediate layers, a temporary output
        layer is added to enable direct prediction from the current layer.
        
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
        
        # Initialize first layer
        self.network_.initialize(self.n_features_)
        self._initialize_parameters_with_scale_awareness(y)
        
        # Get optimizer configuration
        optimizer_config = self._get_optimizer_config()
        
        # Determine if this is classification (need one-hot for temporary networks)
        is_classification = hasattr(self, 'n_classes_') and self.n_classes_ is not None
        if is_classification:
            output_size = self.n_classes_
            output_activation = 'softmax'
        else:
            output_size = 1
            output_activation = 'linear'
        
        # Create base loss function for partial networks
        if is_classification:
            # y is already one-hot encoded for classification
            base_loss_func = lambda y_pred: self.loss_(y_pred, y)
        else:
            base_loss_func = lambda y_pred: self.loss_(y_pred, y)
        
        # Phase 1: Greedy layer-wise pretraining
        for layer_idx in range(n_layers):
            # Build partial network: create fresh copies of layers[0..layer_idx]
            partial_layers = []
            
            for i in range(layer_idx + 1):
                main_layer = self.network_.layers[i]
                # Create a fresh copy of the layer with the same activation
                new_layer = Linear(main_layer.output_size, activation=main_layer.activation)
                partial_layers.append(new_layer)
            
            # Add temporary output layer if not the final layer
            if layer_idx < n_layers - 1:
                temp_output_layer = Linear(output_size, activation=output_activation)
                partial_layers.append(temp_output_layer)
            
            # Create partial network
            partial_network = Sequential(*partial_layers)
            partial_network.initialize(self.n_features_)
            
            # Copy trained parameters from previous layers
            if layer_idx > 0:
                # Get parameters from main network for layers 0..layer_idx
                main_params = self.network_.get_all_parameters()
                partial_params = []
                offset = 0
                for i in range(layer_idx + 1):
                    layer = self.network_.layers[i]
                    layer_param_count = layer.parameter_count()
                    partial_params.append(main_params[offset:offset + layer_param_count])
                    offset += layer_param_count
                
                # Add random initialization for temporary output layer
                if layer_idx < n_layers - 1:
                    temp_param_count = partial_network.layers[-1].parameter_count()
                    partial_params.append(
                        np.random.randn(temp_param_count) * 0.01
                    )
                
                partial_network.set_all_parameters(np.concatenate(partial_params))
            
            # Create optimizer for this partial network
            partial_optimizer = _create_optimizer(
                self.optimizer,
                param_dim=partial_network.parameter_count(),
                optimizer_config=optimizer_config,
                n_candidates=self.n_candidates
            )
            
            # Define loss function for partial network
            def partial_loss(params):
                partial_network.set_all_parameters(params)
                y_pred = partial_network.forward(X)
                if partial_network.output_size == 1:
                    y_pred = y_pred.flatten()
                loss = base_loss_func(y_pred)
                # Add parameter clipping penalty for numerical stability
                if np.any(np.abs(params) > 1e5):
                    return float('inf')
                return loss
            
            # Train partial network
            partial_optimizer.initialize()
            layer_iterations = self.max_iter // n_layers
            
            for iteration in range(layer_iterations):
                partial_optimizer.step(partial_loss, iteration=iteration)
            
            # Copy trained parameters back to main network (only the non-temporary layers)
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
        final_optimizer = _train_all_layers_together(
            network=self.network_,
            optimizer_class=self.optimizer,
            optimizer_config=optimizer_config,
            loss_func=loss_func,
            max_iterations=final_iterations,
            early_stopping=self.early_stopping,
            patience=self.patience,
            n_candidates=self.n_candidates
        )
        
        # Set final parameters
        self.network_.set_all_parameters(final_optimizer.get_parameters())
        self.optimizer_ = final_optimizer  # Store optimizer for reference
    
    def _fit_direct(self, X, y, loss_func=None):
        """
        Train network with all layers together (direct training).
        
        Args:
            X: Input data
            y: Target data
            loss_func: Optional custom loss function. If None, uses _create_loss_func
        """
        if loss_func is None:
            loss_func = self._create_loss_func(X, y)
        
        # Initialize network
        self.network_.initialize(self.n_features_)
        self._initialize_parameters_with_scale_awareness(y)
        
        # Get optimizer configuration
        optimizer_config = self._get_optimizer_config()
        
        # Create optimizer of specified class for full network using utility function
        n_candidates = self.n_candidates if self.n_candidates is not None else 3
        final_optimizer = _create_optimizer(
            self.optimizer,
            param_dim=self.network_.parameter_count(),
            optimizer_config=optimizer_config,
            n_candidates=n_candidates
        )
        
        # Create stateless loss function to prevent state pollution
        stateless_loss_func = _create_stateless_loss_func(self.network_, loss_func)
        
        # Initialize optimizer
        current_params = self.network_.get_all_parameters()
        final_optimizer.initialize(current_params)
        
        # Train
        _run_optimizer_training(
            final_optimizer, stateless_loss_func, self.max_iter, self.early_stopping, self.patience
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
        optimizer: class, default=CompactEvoOptimizer
            Optimizer class to use. Can be CompactEvoOptimizer or MultiCandidateOptimizer.
        use_layerwise: bool, default=False
            If True, use layer-wise training (train each layer in isolation, then all together).
            If False, use direct training (train all layers together from start).
        n_candidates: int, optional
            Number of candidates for MultiCandidateOptimizer. Ignored for other optimizers.
            If None and optimizer is MultiCandidateOptimizer, defaults to 3.
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
                 random_state=None, use_layerwise=False, optimizer=CompactEvoOptimizer,
                 n_candidates=None):
        
        # Store new parameters
        self.n_layers = n_layers
        self.h_dim = h_dim
        self.inner_activation = inner_activation
        self.layer_patience = layer_patience
        self.normalize_output = normalize_output
        self.output_activation = output_activation
        self.use_layerwise = use_layerwise
        self.optimizer = optimizer
        self.n_candidates = n_candidates
        
        # Build layers if not provided
        if layers is None:
            layers = self._build_layers_from_params()
        
        super().__init__(
            layers=layers, output_activation=output_activation,
            lr_mu=lr_mu, lr_sigma=lr_sigma, sigma_min=sigma_min, sigma_max=sigma_max,
            calibration_size=calibration_size, calibration_interval=calibration_interval,
            credit_factor=credit_factor, sigma_regularization=sigma_regularization,
            max_iter=max_iter, early_stopping=early_stopping, patience=patience, 
            random_state=random_state, layer_patience=layer_patience,
            use_layerwise=use_layerwise, optimizer=optimizer, n_candidates=n_candidates
        )
        
        # Set up loss function
        self._loss_str = loss if isinstance(loss, str) else None
        if isinstance(loss, str):
            self.loss_ = get_loss(loss)
        else:
            self.loss_ = loss
    
    def _build_layers_from_params(self):
        """Build layer configuration from n_layers, h_dim, and activations."""
        return _build_layers_from_params_simple(
            self.n_layers, self.h_dim, self.inner_activation, self.output_activation
        )
    
    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        
        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        
        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        params = super().get_params(deep)
        params.update({
            'n_layers': self.n_layers,
            'h_dim': self.h_dim,
            'inner_activation': self.inner_activation,
            'loss': self._loss_str if hasattr(self, '_loss_str') and self._loss_str is not None else 'mse',
            'normalize_output': self.normalize_output
        })
        return params
    
    def set_params(self, **params):
        """
        Set the parameters of this estimator.
        
        Parameters
        ----------
        **params : dict
            Estimator parameters.
        
        Returns
        -------
        self : object
            Estimator instance.
        """
        # Handle loss separately
        if 'loss' in params:
            loss = params.pop('loss')
            if isinstance(loss, str):
                self.loss_ = get_loss(loss)
            else:
                self.loss_ = loss
        
        # Handle other parameters
        super().set_params(**params)
        return self
    
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
        
        # Choose training strategy
        if self.use_layerwise:
            self._fit_layer_wise(X, y_normalized)
        else:
            self._fit_direct(X, y_normalized)
        
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
        optimizer: class, default=CompactEvoOptimizer
            Optimizer class to use. Can be CompactEvoOptimizer or MultiCandidateOptimizer.
        use_layerwise: bool, default=False
            If True, use layer-wise training (train each layer in isolation, then all together).
            If False, use direct training (train all layers together from start).
        n_candidates: int, optional
            Number of candidates for MultiCandidateOptimizer. Ignored for other optimizers.
            If None and optimizer is MultiCandidateOptimizer, defaults to 3.
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
                 random_state=None, use_layerwise=False, optimizer=CompactEvoOptimizer,
                 n_candidates=None):
        
        # Store new parameters
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.h_dim = h_dim
        self.inner_activation = inner_activation
        self.layer_patience = layer_patience
        self.output_activation = output_activation
        self.use_layerwise = use_layerwise
        self.optimizer = optimizer
        self.n_candidates = n_candidates
        
        # Build layers if not provided
        if layers is None:
            layers = self._build_layers_from_params()
        
        super().__init__(
            layers=layers, output_activation=output_activation,
            lr_mu=lr_mu, lr_sigma=lr_sigma, sigma_min=sigma_min, sigma_max=sigma_max,
            calibration_size=calibration_size, calibration_interval=calibration_interval,
            credit_factor=credit_factor, sigma_regularization=sigma_regularization,
            max_iter=max_iter, early_stopping=early_stopping, patience=patience, 
            random_state=random_state, layer_patience=layer_patience,
            use_layerwise=use_layerwise, optimizer=optimizer, n_candidates=n_candidates
        )
        
        # Set up loss function
        self._loss_str = loss if isinstance(loss, str) else None
        if isinstance(loss, str):
            self.loss_ = get_loss(loss)
        else:
            self.loss_ = loss
    
    def _build_layers_from_params(self):
        """Build layer configuration from n_layers, h_dim, and activations."""
        return _build_layers_from_params_simple(
            self.n_layers, self.h_dim, self.inner_activation, self.output_activation
        )
    
    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        
        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        
        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        params = super().get_params(deep)
        params.update({
            'n_classes': self.n_classes,
            'n_layers': self.n_layers,
            'h_dim': self.h_dim,
            'inner_activation': self.inner_activation,
            'loss': self._loss_str if hasattr(self, '_loss_str') and self._loss_str is not None else 'cross_entropy'
        })
        return params
    
    def set_params(self, **params):
        """
        Set the parameters of this estimator.
        
        Parameters
        ----------
        **params : dict
            Estimator parameters.
        
        Returns
        -------
        self : object
            Estimator instance.
        """
        # Handle loss separately
        if 'loss' in params:
            loss = params.pop('loss')
            if isinstance(loss, str):
                self.loss_ = get_loss(loss)
            else:
                self.loss_ = loss
        
        # Handle other parameters
        super().set_params(**params)
        return self
    
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
        
        # Choose training strategy
        if self.use_layerwise:
            # Use layer-wise training with classification-specific loss
            loss_func = self._create_classification_loss_func(X, y_onehot)
            self._fit_layer_wise(X, y_onehot, loss_func=loss_func)
        else:
            # Use direct training with classification-specific loss
            loss_func = self._create_classification_loss_func(X, y_onehot)
            self._fit_direct(X, y_onehot, loss_func=loss_func)
        
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
