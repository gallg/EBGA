import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils import check_random_state
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import LabelBinarizer

from EBGA.nn import Sequential
from EBGA.layers import Linear
from EBGA.losses import get_loss
from EBGA.optimizer import CompactEvoOptimizer


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _create_optimizer(optimizer_class, param_dim, optimizer_config):
    """
    Create an optimizer of the specified class.
    
    Args:
        optimizer_class: The optimizer class to instantiate (e.g., CompactEvoOptimizer)
        param_dim: Number of parameters to optimize
        optimizer_config: Dictionary with optimizer configuration parameters
        
    Returns:
        Optimizer instance of the specified class
    """
    return optimizer_class(
        param_dim=param_dim,
        **optimizer_config
    )



def _run_optimizer_training(optimizer, loss_func, max_iterations, early_stopping, patience):
    """
    Run the optimizer training loop with early stopping support.
    
    Args:
        optimizer: The optimizer instance to train
        loss_func: Loss function for evaluation
        max_iterations: Maximum number of iterations
        early_stopping: Whether to use early stopping
        patience: Patience for early stopping
        
    Returns:
        tuple: (best_loss, patience_counter) after training
    """
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



def _train_all_layers_together(network, optimizer_class, optimizer_config, loss_func,
                               max_iterations, early_stopping, patience):
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
        
    Returns:
        Optimizer instance with final parameters
    """
    param_dim = network.parameter_count()
    
    # Create optimizer of specified class using utility function
    final_optimizer = _create_optimizer(
        optimizer_class, param_dim, optimizer_config
    )
    
    # Initialize with current parameters
    current_params = network.get_all_parameters()
    final_optimizer.initialize(current_params)
    
    # Train all layers together
    _run_optimizer_training(
        final_optimizer, loss_func, max_iterations, early_stopping, patience
    )
    
    return final_optimizer


class BaseModel(BaseEstimator):
    
    def __init__(self, layers,
                 lr_mu, lr_sigma, sigma_min, sigma_max,
                 calibration_size, calibration_interval, credit_factor,
                 sigma_regularization, max_iter, early_stopping, 
                 patience, random_state, layer_patience,
                 use_layerwise, optimizer,
                 momentum, trust_region_radius, batch_size):
        
        # Store hyperparameters
        self.layers = layers
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
        self.momentum = momentum
        self.trust_region_radius = trust_region_radius
        self.batch_size = batch_size
        
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
            'lr_mu': self.lr_mu,
            'lr_sigma': self.lr_sigma,
            'sigma_min': self.sigma_min,
            'sigma_max': self.sigma_max,
            'calibration_size': self.calibration_size,
            'calibration_interval': self.calibration_interval,
            'credit_factor': self.credit_factor,
            'sigma_regularization': self.sigma_regularization,
            'momentum': self.momentum,
            'trust_region_radius': self.trust_region_radius,
            'max_iter': self.max_iter,
            'early_stopping': self.early_stopping,
            'patience': self.patience,
            'random_state': self.random_state,
            'layer_patience': self.layer_patience,
            'use_layerwise': self.use_layerwise,
            'optimizer': self.optimizer,
            'batch_size': self.batch_size
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
            'momentum': self.momentum,
            'trust_region_radius': self.trust_region_radius,
            'random_state': self._random_state
        }
    
    def _create_batches(self, X, y):
        """
        Create batches from the dataset.
        
        If batch_size is None, returns the full dataset as a single batch.
        Otherwise, splits into batches of the specified size.
        If the last batch would be too small (< batch_size), merges it with the previous batch.
        
        Args:
            X: Input features
            y: Target values
            
        Returns:
            list: List of (X_batch, y_batch) tuples
        """
        if self.batch_size is None:
            return [(X, y)]
        
        n_samples = X.shape[0]
        batch_size = self.batch_size
        
        # Calculate number of batches
        n_full_batches = n_samples // batch_size
        remainder = n_samples % batch_size
        
        # If remainder is too small, merge with last batch
        if remainder > 0 and remainder < batch_size:
            # Merge remainder with the last full batch
            batches = []
            for i in range(n_full_batches):
                start = i * batch_size
                # Last batch gets extra samples
                if i == n_full_batches - 1:
                    end = start + batch_size + remainder
                else:
                    end = start + batch_size
                batches.append((X[start:end], y[start:end]))
            return batches
        else:
            # No remainder or exact multiple
            batches = []
            for i in range(n_full_batches):
                start = i * batch_size
                end = start + batch_size
                batches.append((X[start:end], y[start:end]))
            return batches
    
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
            # Default: single output layer with linear activation
            network_layers.append(Linear(output_size, activation='linear'))
        else:
            # Build layers from specification - use sizes and activations as specified
            for i, (size, activation) in enumerate(self.layers):
                layer_activation = activation if activation is not None else 'linear'
                network_layers.append(Linear(size, activation=layer_activation))
        
        return Sequential(*network_layers)
    
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
            if self.batch_size is not None:
                loss_func = self._create_batched_loss_func(X, y)
            else:
                loss_func = self._create_loss_func(X, y)
        
        n_layers = len(self.network_.layers)
        
        # Initialize first layer with scale-aware output
        self.network_.initialize(self.n_features_, scale_aware=y)
        
        # Get optimizer configuration
        optimizer_config = self._get_optimizer_config()
        
        # Get output size and activation from the actual last layer
        last_layer = self.network_.layers[-1]
        output_size = last_layer.output_size
        output_activation = last_layer.activation
        
        # If output_activation is an activation object, get its name
        if hasattr(output_activation, '__class__'):
            activation_name = output_activation.__class__.__name__.lower()
        else:
            activation_name = str(output_activation).lower() if output_activation else None
            
        # Determine activation for temporary networks
        is_classification = hasattr(self, 'n_classes_') and self.n_classes_ is not None
        if is_classification and activation_name != 'softmax':
            # Use softmax for temporary output layer only if main output isn't already softmax
            temp_output_activation = 'softmax'
        else:
            # For regression or when main output already handles classification
            # Use the same activation as the main network's last layer
            temp_output_activation = output_activation
        
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
                temp_output_layer = Linear(output_size, activation=temp_output_activation)
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
                        self._random_state.randn(temp_param_count) * 0.01
                    )
                
                partial_network.set_all_parameters(np.concatenate(partial_params))
            
            # Create optimizer for this partial network
            partial_optimizer = _create_optimizer(
                self.optimizer,
                param_dim=partial_network.parameter_count(),
                optimizer_config=optimizer_config
            )
            
            # Define loss function for partial network
            if self.batch_size is not None:
                # With batching: create batches and cycle through them
                batches = self._create_batches(X, y)
                batch_index = [0]
                
                def partial_loss(params):
                    # Get current batch
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
                    # Add parameter clipping penalty for numerical stability
                    if np.any(np.abs(params) > 1e5):
                        return float('inf')
                    return loss
            else:
                # Without batching: use full dataset
                def partial_loss(params):
                    current = partial_network.get_all_parameters()
                    partial_network.set_all_parameters(params)
                    y_pred = partial_network.forward(X)
                    if partial_network.output_size == 1:
                        y_pred = y_pred.flatten()
                    loss = self.loss_(y_pred, y)
                    partial_network.set_all_parameters(current)
                    # Add parameter clipping penalty for numerical stability
                    if np.any(np.abs(params) > 1e5):
                        return float('inf')
                    return loss
            
            # Train partial network
            partial_optimizer.initialize()
            # Each layer gets max_iter // n_layers iterations.
            # Total for layer-wise pretraining: n_layers * (max_iter // n_layers) ~ max_iter.
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
        # Fine-tuning uses max_iter // 2 additional iterations.
        # Total for layer-wise mode: ~1.5 * max_iter (pretraining + fine-tuning).
        # Direct training uses exactly max_iter iterations for the same setting.
        final_iterations = self.max_iter // 2
        final_optimizer = _train_all_layers_together(
            network=self.network_,
            optimizer_class=self.optimizer,
            optimizer_config=optimizer_config,
            loss_func=loss_func,
            max_iterations=final_iterations,
            early_stopping=self.early_stopping,
            patience=self.patience
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
            if self.batch_size is not None:
                loss_func = self._create_batched_loss_func(X, y)
            else:
                loss_func = self._create_loss_func(X, y)
        
        # Initialize network with scale-aware output
        self.network_.initialize(self.n_features_, scale_aware=y)
        
        # Get optimizer configuration
        optimizer_config = self._get_optimizer_config()
        
        # Create optimizer of specified class for full network using utility function
        final_optimizer = _create_optimizer(
            self.optimizer,
            param_dim=self.network_.parameter_count(),
            optimizer_config=optimizer_config
        )
        
        # Initialize optimizer
        current_params = self.network_.get_all_parameters()
        final_optimizer.initialize(current_params)
        
        # Train
        _run_optimizer_training(
            final_optimizer, loss_func, self.max_iter, self.early_stopping, self.patience
        )
        
        # Set final parameters
        self.network_.set_all_parameters(final_optimizer.get_parameters())
        self.optimizer_ = final_optimizer  # Store optimizer for reference
    
    def _create_loss_func(self, X, y):
        """
        Create the default loss function for training.
        Override this method in subclasses for custom loss calculation.
        
        The returned function is stateless: it saves the current network state,
        evaluates the loss on the given parameters, then restores the original state.
        
        Args:
            X: Input data
            y: Target data
            
        Returns:
            function: Loss function that takes parameters and returns loss value
        """
        def loss_func(params):
            current = self.network_.get_all_parameters()
            self.network_.set_all_parameters(params)
            y_pred = self.network_.forward(X)
            if self.network_.output_size == 1:
                y_pred = y_pred.flatten()
            loss = self.loss_(y_pred, y)
            self.network_.set_all_parameters(current)
            # Add parameter clipping penalty for numerical stability
            if np.any(np.abs(params) > 1e5):
                return float('inf')
            return loss
        return loss_func
    
    def _wrap_with_batching(self, loss_func, X, y):
        """
        Wrap a loss function with batching support.
        
        Creates a new loss function that cycles through batches of the data.
        Each call to the returned function uses the next batch in sequence.
        
        Args:
            loss_func: Base loss function that takes (params, X_batch, y_batch)
            X: Full input data
            y: Full target data
            
        Returns:
            function: Batched loss function that takes parameters and returns loss value
        """
        # Create batches
        batches = self._create_batches(X, y)
        batch_index = [0]  # Use list to allow mutation in nested function
        
        def batched_loss_func(params):
            # Get current batch
            idx = batch_index[0] % len(batches)
            X_batch, y_batch = batches[idx]
            batch_index[0] += 1
            
            # Call the base loss function with batch data
            return loss_func(params, X_batch, y_batch)
        
        return batched_loss_func
    
    def _create_batched_loss_func(self, X, y):
        """
        Create a batched loss function that cycles through batches.
        
        Each call to the returned loss function uses a different batch for evaluation.
        The batches are cycled through in a round-robin fashion.
        
        Args:
            X: Input data
            y: Target data
            
        Returns:
            function: Batched loss function that takes parameters and returns loss value
        """
        # Base loss function for the main network
        def base_loss(params, X_batch, y_batch):
            current = self.network_.get_all_parameters()
            self.network_.set_all_parameters(params)
            y_pred = self.network_.forward(X_batch)
            if self.network_.output_size == 1:
                y_pred = y_pred.flatten()
            loss = self.loss_(y_pred, y_batch)
            self.network_.set_all_parameters(current)
            # Add parameter clipping penalty for numerical stability
            if np.any(np.abs(params) > 1e5):
                return float('inf')
            return loss
        
        return self._wrap_with_batching(base_loss, X, y)


class EBGARegressor(BaseModel, RegressorMixin):
    """
    EBGA Regressor - Evolutionary neural network for regression.
    
    Uses explicit layers list for network architecture configuration.

    Parameters:
        layers: list of tuples, default=None
            Network architecture. Each tuple is (output_size, activation)
            Example: [(50, 'relu'), (1, 'linear')] for 1 hidden layer with 50 units and ReLU, output with linear activation
        loss: str or Loss, default='mae'
            Loss function name or instance
        optimizer: class, default=CompactEvoOptimizer
            Optimizer class to use. Currently only CompactEvoOptimizer is supported.
        use_layerwise: bool, default=False
            If True, use layer-wise training (train each layer in isolation, then all together).
            If False, use direct training (train all layers together from start).
        lr_mu: float, default=0.03
            Initial learning rate for mean (adaptive during training)
        lr_sigma: float, default=0.03
            Initial learning rate for sigma (adaptive during training)
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
        >>> model = EBGARegressor(
        ...     layers=[(50, 'relu'), (1, 'linear')],
        ...     max_iter=1000
        ... )
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """
    
    def __init__(self, layers=None, loss='mae',
                 lr_mu=0.03, lr_sigma=0.03, sigma_min=0.001, sigma_max=1.0,
                 calibration_size=30, calibration_interval=50, credit_factor=2.0,
                 sigma_regularization=0.0, max_iter=10000, early_stopping=True, 
                 patience=100, layer_patience=50, normalize_output=False,
                 random_state=None, use_layerwise=False, optimizer=CompactEvoOptimizer,
                 momentum=0.9, trust_region_radius=0.1, batch_size=None):
        
        self.normalize_output = normalize_output
        
        super().__init__(
            layers=layers,
            lr_mu=lr_mu, lr_sigma=lr_sigma, sigma_min=sigma_min, sigma_max=sigma_max,
            calibration_size=calibration_size, calibration_interval=calibration_interval,
            credit_factor=credit_factor, sigma_regularization=sigma_regularization,
            max_iter=max_iter, early_stopping=early_stopping, patience=patience, 
            random_state=random_state, layer_patience=layer_patience,
            use_layerwise=use_layerwise, optimizer=optimizer,
            momentum=momentum, trust_region_radius=trust_region_radius,
            batch_size=batch_size
        )
        
        # Set up loss function
        self._loss_str = loss if isinstance(loss, str) else None
        if isinstance(loss, str):
            self.loss_ = get_loss(loss)
        else:
            self.loss_ = loss
    
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
        
        # Check if user provided layers - if so, use the last layer's output size
        # If not, use 1 as output size for regression
        if self.layers is None or len(self.layers) == 0:
            output_size = 1
        else:
            # Use the size specified in the last layer
            output_size = self.layers[-1][0]
        
        # Build network
        self.network_ = self._build_network(self.n_features_, output_size=output_size)
        
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
    
    Uses explicit layers list for network architecture configuration.

    Parameters:
        layers: list of tuples, default=None
            Network architecture. Each tuple is (output_size, activation)
            Example: [(50, 'relu'), (10, 'softmax')] for 1 hidden layer with 50 units and ReLU, output with softmax activation
        n_classes: int, optional
            Number of classes. If None, inferred from data.
        loss: str or Loss, default='cross_entropy'
            Loss function name or instance
        optimizer: class, default=CompactEvoOptimizer
            Optimizer class to use. Currently only CompactEvoOptimizer is supported.
        use_layerwise: bool, default=False
            If True, use layer-wise training (train each layer in isolation, then all together).
            If False, use direct training (train all layers together from start).
        lr_mu: float, default=0.05
            Initial learning rate for mean (adaptive during training)
        lr_sigma: float, default=0.005
            Initial learning rate for sigma (adaptive during training)
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
        >>> model = EBGAClassifier(
        ...     layers=[(50, 'relu'), (10, 'softmax')],
        ...     n_classes=10
        ... )
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """
    
    def __init__(self, layers=None, n_classes=None, loss='cross_entropy',
                 lr_mu=0.05, lr_sigma=0.005, sigma_min=0.001, sigma_max=1.0,
                 calibration_size=20, calibration_interval=25, credit_factor=2.0,
                 sigma_regularization=0.0, max_iter=500, early_stopping=True, 
                 patience=20, layer_patience=50,
                 random_state=None, use_layerwise=False, optimizer=CompactEvoOptimizer,
                 momentum=0.5, trust_region_radius=None, batch_size=None):
        
        self.n_classes = n_classes
        
        super().__init__(
            layers=layers,
            lr_mu=lr_mu, lr_sigma=lr_sigma, sigma_min=sigma_min, sigma_max=sigma_max,
            calibration_size=calibration_size, calibration_interval=calibration_interval,
            credit_factor=credit_factor, sigma_regularization=sigma_regularization,
            max_iter=max_iter, early_stopping=early_stopping, patience=patience, 
            random_state=random_state, layer_patience=layer_patience,
            use_layerwise=use_layerwise, optimizer=optimizer,
            momentum=momentum, trust_region_radius=trust_region_radius,
            batch_size=batch_size
        )
        
        # Set up loss function
        self._loss_str = loss if isinstance(loss, str) else None
        if isinstance(loss, str):
            self.loss_ = get_loss(loss)
        else:
            self.loss_ = loss
    
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
        Create classification-specific loss function.
        Uses network output as-is, relying on the network's activation.
        
        Args:
            X: Input data
            y_onehot: One-hot encoded target labels
            
        Returns:
            function: Loss function for classification
        """
        # Base loss function for classification
        def class_base_loss(params, X_batch, y_batch):
            current = self.network_.get_all_parameters()
            self.network_.set_all_parameters(params)
            y_pred = self.network_.forward(X_batch)
            loss = self.loss_(y_pred, y_batch)
            self.network_.set_all_parameters(current)
            # Add parameter clipping penalty for numerical stability
            if np.any(np.abs(params) > 1e5):
                return float('inf')
            return loss
        
        # Use batching if batch_size is set
        if self.batch_size is not None:
            return self._wrap_with_batching(class_base_loss, X, y_onehot)
        else:
            # Without batching, use full dataset
            def loss_func(params):
                return class_base_loss(params, X, y_onehot)
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
        
        # Check if user provided layers - if so, use the last layer's output size
        # If not, use n_classes_ as output size
        if self.layers is None or len(self.layers) == 0:
            output_size = self.n_classes_
        else:
            # Use the size specified in the last layer
            output_size = self.layers[-1][0]
        
        # Build network
        self.network_ = self._build_network(self.n_features_, output_size=output_size)
        
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
        
        Handles different output activations:
        - softmax: returns argmax of network output
        - sigmoid with 2 classes: applies 0.5 threshold
        - sigmoid with >2 classes: warns and falls back to softmax + argmax
        - other: returns argmax
        
        Args:
            X: Input features
            
        Returns:
            numpy.ndarray: Predicted class labels
        """
        check_is_fitted(self)
        X = check_array(X)
        output = self.network_.forward(X)
        
        # Get the actual activation of the last layer
        last_layer = self.network_.layers[-1]
        activation = last_layer.activation
        
        # Get activation name
        if hasattr(activation, '__class__'):
            activation_name = activation.__class__.__name__.lower()
        else:
            activation_name = str(activation).lower() if activation else 'none'
        
        if activation_name == 'sigmoid':
            # For sigmoid activation
            if output.shape[1] == 1:
                # Single output neuron with sigmoid: binary classification using 0.5 threshold
                return (output >= 0.5).astype(int).flatten()
            elif self.n_classes_ == 2 and output.shape[1] == 2:
                # Two output neurons with sigmoid for binary classification
                # This is non-standard but we'll handle it by using argmax
                import warnings
                warnings.warn(
                    f"Sigmoid activation with 2 output neurons for {self.n_classes_}-class classification is non-standard. "
                    f"Consider using (1, 'sigmoid') for binary or ({self.n_classes_}, 'softmax') for multi-class. "
                    f"Falling back to argmax.",
                    UserWarning
                )
                return np.argmax(output, axis=1)
            else:
                # Multi-class with sigmoid: warn and fall back to argmax
                import warnings
                warnings.warn(
                    f"Sigmoid activation with {self.n_classes_} classes and {output.shape[1]} outputs is not standard. "
                    f"Consider using ({self.n_classes_}, 'softmax'). Falling back to argmax.",
                    UserWarning
                )
                return np.argmax(output, axis=1)
        else:
            # For softmax, linear, or other activations: use argmax
            return np.argmax(output, axis=1)
    
    def predict_proba(self, X):
        """
        Predict class probabilities for input data.
        
        Handles different output activations:
        - softmax: returns network output as-is (already probabilities)
        - sigmoid: applies softmax to convert to probabilities
        - linear: applies softmax to convert to probabilities
        - other: applies softmax to convert to probabilities
        
        Args:
            X: Input features
            
        Returns:
            numpy.ndarray: Class probabilities
        """
        check_is_fitted(self)
        X = check_array(X)
        output = self.network_.forward(X)
        
        # Get the actual activation of the last layer
        last_layer = self.network_.layers[-1]
        activation = last_layer.activation
        
        # Get activation name
        if hasattr(activation, '__class__'):
            activation_name = activation.__class__.__name__.lower()
        else:
            activation_name = str(activation).lower() if activation else 'none'
        
        if activation_name == 'softmax':
            # Network already outputs probabilities
            return output
        else:
            # Apply softmax to get probabilities
            output_exp = np.exp(output - np.max(output, axis=1, keepdims=True))
            return output_exp / np.sum(output_exp, axis=1, keepdims=True)
    
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
