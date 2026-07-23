import numpy as np

from EBGA.layers import Dense
from EBGA.optimizer import CompactEvoOptimizer


def _make_loss_func(network, X, y, loss):
    """Create a sequential loss function for a given network."""
    def loss_func(params):
        current = network.get_all_parameters()
        network.set_all_parameters(params)
        y_pred = network.forward(X)
        if network.output_size == 1:
            y_pred = y_pred.flatten()
        loss_val = loss(y_pred, y)
        network.set_all_parameters(current)
        if np.any(np.abs(params) > 1e5):
            return float('inf')
        return loss_val
    return loss_func


class Sequential:
    
    def __init__(self, *layers):
        self.layers = list(layers)
        self.initialized = False
    
    def initialize(self, input_size, scale_aware=None):
        """
        Initialize network parameters.

        Args:
            input_size: Number of input features
            scale_aware: Target values for scale-aware output initialization.
                       If provided, the last parameter is set to mean(scale_aware).
        """
        current_size = input_size
        for i, layer in enumerate(self.layers):
            layer.initialize(current_size)
            current_size = layer.output_size
        self.initialized = True
        self.input_size = input_size
        self.output_size = current_size

        if scale_aware is not None:
            all_params = self.get_all_parameters()
            # Set the bias of the last layer to the mean of the target.
            # For multi-output layers, set all bias elements.
            last_layer = self.layers[-1]
            if last_layer.parameter_count() > 0:
                n_bias = last_layer.output_size if hasattr(last_layer, 'use_bias') and last_layer.use_bias else 0
                if n_bias > 0:
                    all_params[-n_bias:] = float(np.mean(scale_aware))
            self.set_all_parameters(all_params)
    
    def forward(self, x):
        if not self.initialized:
            self.initialize(x.shape[1])
        
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def get_all_parameters(self):
        all_params = []
        for layer in self.layers:
            all_params.append(layer.get_parameters())
        return np.concatenate(all_params)
    
    def set_all_parameters(self, params):
        offset = 0
        for layer in self.layers:
            param_count = layer.parameter_count()
            layer_params = params[offset:offset + param_count]
            layer.set_parameters(layer_params)
            offset += param_count
    
    def parameter_count(self):
        return sum(layer.parameter_count() for layer in self.layers)
    
    def get_layer_parameters(self, layer_idx):
        start = sum(l.parameter_count() for l in self.layers[:layer_idx])
        end = start + self.layers[layer_idx].parameter_count()
        return self.get_all_parameters()[start:end]
    
    def copy_layer_parameters(self, source, layer_idx):
        """Copy parameters for a single layer from source network into self."""
        src_start = sum(l.parameter_count() for l in source.layers[:layer_idx])
        src_end = src_start + source.layers[layer_idx].parameter_count()
        dst_start = sum(l.parameter_count() for l in self.layers[:layer_idx])
        dst_end = dst_start + self.layers[layer_idx].parameter_count()
        params = self.get_all_parameters()
        params[dst_start:dst_end] = source.get_all_parameters()[src_start:src_end]
        self.set_all_parameters(params)

    def layerwise_pretrain(self, X, y, loss, n_classes=None, layer_iters=500,
                           optimizer_cls=None, optimizer_config=None,
                           n_jobs=1, random_state=None, verbose=True):
        """
        Greedy layer-wise pretraining.

        Trains each layer sequentially, building a partial network for each
        layer (up to that layer + a temporary output layer). After all layers
        are pretrained, the network is ready for fine-tuning.

        When ``n_jobs > 1``, each layer's candidate evaluations are
        parallelized across ``n_jobs`` worker processes via a per-layer
        ``ParallelEvaluator``. Layers still train sequentially.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            Target values.
        loss : Loss instance
            Loss function to use for training.
        n_classes : int, optional
            Number of output classes. Required for classification tasks
            (softmax output). For regression, inferred from output size.
        layer_iters : int, default=500
            Number of iterations per layer.
        optimizer_cls : class, default=CompactEvoOptimizer
            Optimizer class to use for each partial network.
        optimizer_config : dict, optional
            Configuration dict passed to the optimizer constructor.
            If None, uses defaults.
        n_jobs : int, default=1
            Number of worker processes for parallel candidate evaluation.
            When 1, candidates are evaluated sequentially.
        random_state : int or RandomState, optional
            Random state for initializing temp layer parameters.
        verbose : bool, default=True
            If True, print progress during training.
        """
        if optimizer_cls is None:
            optimizer_cls = CompactEvoOptimizer
        if optimizer_config is None:
            optimizer_config = {}

        if isinstance(loss, str):
            from EBGA.losses import get_loss
            loss = get_loss(loss)

        n_layers = len(self.layers)
        output_size = self.layers[-1].output_size

        # Determine temp output activation
        last_activation = self.layers[-1].activation
        if last_activation is not None and hasattr(last_activation, '__class__'):
            act_name = last_activation.__class__.__name__.lower()
        else:
            act_name = str(last_activation).lower() if last_activation else 'linear'

        if n_classes is not None and act_name != 'softmax':
            temp_activation = 'softmax'
        else:
            temp_activation = act_name

        if verbose:
            print(f"Layer-wise pretraining ({n_layers} layers, {layer_iters} iters each)")

        for layer_idx in range(n_layers):
            if verbose:
                print(f"  Training layer {layer_idx + 1}/{n_layers}...")

            # Build partial network: layers [0..layer_idx] + temp output
            partial_layers = []
            for i in range(layer_idx + 1):
                l = self.layers[i]
                partial_layers.append(Dense(l.output_size, activation=l.activation))

            # Add temp output layer (only if not the last layer)
            if layer_idx < n_layers - 1:
                temp_output = Dense(output_size, activation=temp_activation)
                partial_layers.append(temp_output)

            partial_net = Sequential(*partial_layers)
            partial_net.initialize(X.shape[1])

            # Copy already-trained params from main network
            if layer_idx > 0:
                main_params = self.get_all_parameters()
                partial_params = []
                offset = 0
                for i in range(layer_idx + 1):
                    pc = self.layers[i].parameter_count()
                    partial_params.append(main_params[offset:offset + pc])
                    offset += pc
                if layer_idx < n_layers - 1:
                    temp_pc = partial_net.layers[-1].parameter_count()
                    rng = np.random.RandomState(random_state)
                    partial_params.append(rng.randn(temp_pc) * 0.01)
                partial_net.set_all_parameters(np.concatenate(partial_params))

            # Create optimizer for this partial network
            partial_opt = optimizer_cls(
                param_dim=partial_net.parameter_count(),
                **optimizer_config,
            )
            partial_opt.initialize(partial_net.get_all_parameters())

            # Train — parallel or sequential
            if n_jobs > 1:
                from EBGA.parallel import ParallelEvaluator
                with ParallelEvaluator(
                    partial_net, X, y, loss=loss,
                    n_jobs=n_jobs, batch_size=None,
                    random_state=random_state,
                ) as evaluator:
                    for i in range(layer_iters):
                        partial_opt.step(iteration=i, evaluate_map=evaluator.evaluate_map)
            else:
                partial_loss = _make_loss_func(partial_net, X, y, loss)
                for i in range(layer_iters):
                    partial_opt.step(partial_loss, iteration=i)

            # Copy trained params back to main network
            trained_params = partial_net.get_all_parameters()
            main_params = self.get_all_parameters()
            offset = 0
            for i in range(layer_idx + 1):
                pc = self.layers[i].parameter_count()
                main_params[offset:offset + pc] = trained_params[offset:offset + pc]
                offset += pc
            self.set_all_parameters(main_params)

    def __len__(self):
        return len(self.layers)
    
    def __getitem__(self, idx):
        return self.layers[idx]
    
    def __repr__(self):
        return f"Sequential({[type(l).__name__ for l in self.layers]})"
