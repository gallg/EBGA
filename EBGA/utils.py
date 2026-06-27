import numpy as np


def _to_python_scalar(val):
    """Convert numpy scalar to Python type."""
    if isinstance(val, np.ndarray):
        if val.ndim == 0:
            return val.item()
        return val.tolist()
    return val


def save_model(model, filepath):
    """
    Save a trained EBGA model to a file.
    
    Args:
        model: Trained EBGARegressor or EBGAClassifier instance
        filepath: Path to save the model (should end with .npz)
    """
    def act_to_str(act):
        if act is None:
            return None
        if isinstance(act, str):
            return act
        return type(act).__name__.lower()
    
    network_params = model.network_.get_all_parameters()
    optimizer_state = model.optimizer_.state_dict()
    
    state = {
        'type': type(model).__name__,
        'network_params': network_params,
        'optimizer_state': optimizer_state,
        'n_features': int(model.n_features_),
        'layers': [(size, act_to_str(act)) for size, act in model.layers] if model.layers else None,
        'output_activation': act_to_str(model.output_activation) if model.output_activation else None,
        'lr_mu': float(model.lr_mu),
        'lr_sigma': float(model.lr_sigma),
        'sigma_min': float(model.sigma_min),
        'sigma_max': float(model.sigma_max),
        'calibration_size': int(model.calibration_size),
        'calibration_interval': int(model.calibration_interval),
        'credit_factor': float(model.credit_factor),
        'sigma_regularization': float(model.sigma_regularization),
        'max_iter': int(model.max_iter),
    }
    
    if hasattr(model, 'y_min_') and model.y_min_ is not None:
        state['y_min'] = float(model.y_min_)
        state['y_max'] = float(model.y_max_)
        state['normalize_output'] = bool(model.normalize_output)
    
    if hasattr(model, 'n_classes_') and model.n_classes_ is not None:
        state['n_classes'] = int(model.n_classes_)
    
    if hasattr(model, 'label_binarizer_') and model.label_binarizer_ is not None:
        state['classes'] = _to_python_scalar(model.label_binarizer_.classes_)
    
    np.savez(filepath, **state)


def load_model(filepath):
    """
    Load a saved EBGA model from a file.
    
    Args:
        filepath: Path to the saved model file
        
    Returns:
        model: Loaded EBGARegressor or EBGAClassifier instance
    """
    from EBGA.nn import Sequential
    from EBGA.layers import Linear
    from EBGA.activations import get_activation
    from EBGA.optimizer import CompactEvoOptimizer
    from sklearn.preprocessing import LabelBinarizer
    
    state = np.load(filepath, allow_pickle=True)
    
    model_type = _to_python_scalar(state['type'])
    n_features = int(_to_python_scalar(state['n_features']))
    
    # Handle layers
    layers_raw = _to_python_scalar(state['layers'])
    layers_config = []
    if isinstance(layers_raw, np.ndarray):
        for row in layers_raw:
            size = int(_to_python_scalar(row[0]))
            act = _to_python_scalar(row[1]) if len(row) > 1 else None
            layers_config.append((size, act))
    elif isinstance(layers_raw, list):
        for item in layers_raw:
            if isinstance(item, (list, tuple)):
                size = int(_to_python_scalar(item[0]))
                act = _to_python_scalar(item[1]) if len(item) > 1 else None
                layers_config.append((size, act))
    if not layers_config:
        layers_config = []
    
    output_activation = _to_python_scalar(state['output_activation'])
    
    # Build network
    network_layers = []
    n_layers = len(layers_config)
    
    for i, (size, activation) in enumerate(layers_config):
        act = output_activation if i == n_layers - 1 else activation
        if act is not None and not isinstance(act, str):
            act = str(act)
        network_layers.append(Linear(size, activation=act if act else None))
    
    network = Sequential(*network_layers)
    network.initialize(n_features)
    network.set_all_parameters(state['network_params'])
    
    # Reconstruct optimizer
    optimizer = CompactEvoOptimizer(
        param_dim=network.parameter_count(),
        lr_mu=float(_to_python_scalar(state['lr_mu'])),
        lr_sigma=float(_to_python_scalar(state['lr_sigma'])),
        sigma_min=float(_to_python_scalar(state['sigma_min'])),
        sigma_max=float(_to_python_scalar(state['sigma_max'])),
        calibration_size=int(_to_python_scalar(state['calibration_size'])),
        calibration_interval=int(_to_python_scalar(state['calibration_interval'])),
        credit_factor=float(_to_python_scalar(state['credit_factor'])),
        sigma_regularization=float(_to_python_scalar(state['sigma_regularization'])),
    )
    opt_state = state['optimizer_state']
    if isinstance(opt_state, np.ndarray) and opt_state.ndim == 0:
        opt_state = opt_state.item()
    optimizer.load_state_dict(opt_state)
    
    # Create model
    if model_type == 'EBGARegressor':
        from EBGA.models import EBGARegressor
        model = EBGARegressor.__new__(EBGARegressor)
    else:
        from EBGA.models import EBGAClassifier
        model = EBGAClassifier.__new__(EBGAClassifier)
    
    model.layers = layers_config
    model.output_activation = output_activation
    model.lr_mu = float(_to_python_scalar(state['lr_mu']))
    model.lr_sigma = float(_to_python_scalar(state['lr_sigma']))
    model.sigma_min = float(_to_python_scalar(state['sigma_min']))
    model.sigma_max = float(_to_python_scalar(state['sigma_max']))
    model.calibration_size = int(_to_python_scalar(state['calibration_size']))
    model.calibration_interval = int(_to_python_scalar(state['calibration_interval']))
    model.credit_factor = float(_to_python_scalar(state['credit_factor']))
    model.sigma_regularization = float(_to_python_scalar(state['sigma_regularization']))
    model.max_iter = int(_to_python_scalar(state['max_iter']))
    model.n_features_ = n_features
    model.network_ = network
    model.optimizer_ = optimizer
    
    # Initialize attributes that might be needed by the model
    model.early_stopping = True  # Default
    model.patience = 20  # Default
    model.layer_patience = 50  # Default
    model.random_state = None
    model._random_state = None
    model.label_binarizer_ = None
    model.n_classes_ = None
    
    # Initialize normalize_output for regressor
    model.normalize_output = False
    
    if 'y_min' in state:
        model.y_min_ = float(_to_python_scalar(state['y_min']))
        model.y_max_ = float(_to_python_scalar(state['y_max']))
        model.normalize_output = bool(_to_python_scalar(state.get('normalize_output', False)))
    
    if 'n_classes' in state:
        model.n_classes_ = int(_to_python_scalar(state['n_classes']))
    
    if 'classes' in state:
        model.label_binarizer_ = LabelBinarizer()
        model.label_binarizer_.classes_ = _to_python_scalar(state['classes'])
    
    model.network_.set_training(False)
    
    return model


def save_network(network, optimizer, filepath):
    """
    Save a custom network and its optimizer to a file.
    
    Args:
        network: Sequential network instance
        optimizer: CompactEvoOptimizer instance
        filepath: Path to save the network (should end with .npz)
    """
    layer_configs = []
    for layer in network.layers:
        config = {
            'type': type(layer).__name__,
            'output_size': int(layer.output_size),
        }
        if hasattr(layer, 'use_bias'):
            config['use_bias'] = bool(layer.use_bias)
        if hasattr(layer, 'activation'):
            act = layer.activation
            config['activation'] = None if act is None else (act if isinstance(act, str) else type(act).__name__.lower())
        layer_configs.append(config)
    
    state = {
        'type': 'Sequential',
        'input_size': int(network.input_size),
        'network_params': network.get_all_parameters(),
        'optimizer_state': optimizer.state_dict(),
        'optimizer_params': {
            'lr_mu': float(optimizer.lr_mu),
            'lr_sigma': float(optimizer.lr_sigma),
            'sigma_min': float(optimizer.sigma_min),
            'sigma_max': float(optimizer.sigma_max),
            'calibration_size': int(optimizer.calibration_size),
            'calibration_interval': int(optimizer.calibration_interval),
            'credit_factor': float(optimizer.credit_factor),
            'sigma_regularization': float(optimizer.sigma_regularization),
        },
        'layer_configs': layer_configs,
    }
    
    np.savez(filepath, **state)


def load_network(filepath):
    """
    Load a saved custom network and optimizer from a file.
    
    Args:
        filepath: Path to the saved network file
        
    Returns:
        network: Loaded Sequential network instance
        optimizer: Loaded CompactEvoOptimizer instance
    """
    state = np.load(filepath, allow_pickle=True)
    
    from EBGA.nn import Sequential
    from EBGA.layers import Linear, Flatten
    from EBGA.optimizer import CompactEvoOptimizer
    from EBGA.activations import get_activation
    
    layers = []
    for config in _to_python_scalar(state['layer_configs']):
        layer_type = _to_python_scalar(config['type'])
        if layer_type == 'Linear':
            act = _to_python_scalar(config.get('activation'))
            if act is not None and not isinstance(act, str):
                act = str(act)
            layer = Linear(
                output_size=int(_to_python_scalar(config['output_size'])),
                activation=act if act else None,
                use_bias=bool(_to_python_scalar(config.get('use_bias', True)))
            )
        elif layer_type == 'Flatten':
            layer = Flatten()

        else:
            raise ValueError(f"Unknown layer type: {layer_type}")
        layers.append(layer)
    
    network = Sequential(*layers)
    network.initialize(int(_to_python_scalar(state['input_size'])))
    network.set_all_parameters(state['network_params'])
    
    opt_params = _to_python_scalar(state['optimizer_params'])
    optimizer = CompactEvoOptimizer(
        param_dim=network.parameter_count(),
        lr_mu=float(opt_params['lr_mu']),
        lr_sigma=float(opt_params['lr_sigma']),
        sigma_min=float(opt_params['sigma_min']),
        sigma_max=float(opt_params['sigma_max']),
        calibration_size=int(opt_params['calibration_size']),
        calibration_interval=int(opt_params['calibration_interval']),
        credit_factor=float(opt_params['credit_factor']),
        sigma_regularization=float(opt_params['sigma_regularization']),
    )
    opt_state = state['optimizer_state']
    if isinstance(opt_state, np.ndarray) and opt_state.ndim == 0:
        opt_state = opt_state.item()
    optimizer.load_state_dict(opt_state)
    
    return network, optimizer
