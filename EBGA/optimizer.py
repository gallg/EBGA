import numpy as np


class BudgetExceededError(Exception):
    pass


class BaseEvoOptimizer:
    """
    Base class for evolutionary optimizers with shared infrastructure.
    
    Provides common functionality for budget management, parameter clipping,
    loss scale tracking, and callback handling.
    """
    
    def __init__(self, param_dim, lr_mu=0.05, lr_sigma=0.005,
                 sigma_min=0.001, sigma_max=1.0,
                 calibration_size=20, calibration_interval=25,
                 credit_factor=2.0, sigma_regularization=0.0,
                 bounds=None, budget=None, random_state=None):
        # Store common parameters
        self.param_dim = param_dim
        self.lr_mu = lr_mu
        self.lr_sigma = lr_sigma
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.calibration_size = calibration_size
        self.calibration_interval = calibration_interval
        self.credit_factor = credit_factor
        self.sigma_regularization = sigma_regularization
        self.bounds = bounds
        self.budget = budget
        
        # Random state
        if random_state is None:
            self.rng = np.random.RandomState()
        elif isinstance(random_state, int):
            self.rng = np.random.RandomState(random_state)
        else:
            self.rng = random_state
        
        # Tracking
        self.num_evaluations = 0
        self.callbacks = {"tell": []}
        
        # Loss scale for adaptive learning rates
        self.loss_scale = 1.0
        self.loss_scale_decay = 0.99
    
    def register_callback(self, event, callback):
        """Register a callback function for an event."""
        if event in self.callbacks:
            self.callbacks[event].append(callback)
    
    def _check_budget(self, evals_this_step):
        """Check if budget is exceeded."""
        if self.budget is not None and self.num_evaluations + evals_this_step > self.budget:
            raise BudgetExceededError("Optimization budget exceeded")
    
    def _update_loss_scale(self, losses):
        """Update loss scale estimate using exponential moving average."""
        self.loss_scale = (
            self.loss_scale_decay * self.loss_scale +
            (1 - self.loss_scale_decay) * np.mean(np.abs(losses))
        )
    
    def _clip_parameters(self):
        """Clip parameters to bounds (to be implemented by subclasses)."""
        raise NotImplementedError
    
    def step(self, loss_func, iteration=None):
        """
        Perform one optimization step.
        
        Returns:
            float: Mean loss of evaluated samples
        """
        is_calibration = iteration is None or iteration % self.calibration_interval == 0
        evals_this_step = self.calibration_size if is_calibration else 2
        
        self._check_budget(evals_this_step)
        
        if is_calibration:
            loss = self._population_step(loss_func)
        else:
            loss = self._pairwise_step(loss_func)
        
        self.num_evaluations += evals_this_step
        self._clip_parameters()
        
        # Trigger callbacks with current parameters
        params = self.get_parameters()
        for callback in self.callbacks["tell"]:
            callback(self, params.copy(), loss)
        
        return loss
    
    def _population_step(self, loss_func):
        """Population calibration step (to be implemented by subclasses)."""
        raise NotImplementedError
    
    def _pairwise_step(self, loss_func):
        """Pairwise update step (to be implemented by subclasses)."""
        raise NotImplementedError
    
    def get_parameters(self):
        """Get current parameters (to be implemented by subclasses)."""
        raise NotImplementedError
    
    def set_parameters(self, params):
        """Set current parameters (to be implemented by subclasses)."""
        raise NotImplementedError
    
    def initialize(self, initial_params=None):
        """Initialize distribution parameters (to be implemented by subclasses)."""
        raise NotImplementedError
    
    def get_distribution_parameters(self):
        """Get distribution parameters (to be implemented by subclasses)."""
        raise NotImplementedError
    
    def state_dict(self):
        """Get state dictionary (to be implemented by subclasses)."""
        raise NotImplementedError
    
    def load_state_dict(self, state_dict):
        """Load state from dictionary (to be implemented by subclasses)."""
        raise NotImplementedError
    
    def minimize(self, func, initial_params=None, max_iter=None):
        """
        Run complete optimization.
        
        Returns:
            OptimizationResult: Result with final parameters and loss
        """
        self.initialize(initial_params)
        
        if max_iter is None:
            max_iter = self.budget if self.budget else 1000
        
        for iteration in range(max_iter):
            try:
                loss = self.step(func, iteration=iteration)
            except BudgetExceededError:
                break
        
        return OptimizationResult(
            value=self.get_parameters().copy(),
            loss=loss,
            num_evaluations=self.num_evaluations
        )


class CompactEvoOptimizer(BaseEvoOptimizer):
    """
    Compact Evolutionary Optimizer.
    
    Implements a single Gaussian distribution with diagonal covariance for
    parameter optimization using natural gradient updates.
    
    Parameters:
        param_dim: int
            Dimensionality of the parameter space
        lr_mu: float, default=0.05
            Initial learning rate for mean parameters (adaptive during training)
        lr_sigma: float, default=0.005
            Initial learning rate for standard deviation parameters (adaptive during training)
        sigma_min: float, default=0.001
            Minimum standard deviation
        sigma_max: float, default=1.0
            Maximum standard deviation
        calibration_size: int, default=20
            Number of samples for population calibration
        calibration_interval: int, default=25
            How often to perform population calibration
        credit_factor: float, default=2.0
            Strength of credit assignment for update magnitude
        sigma_regularization: float, default=0.0
            Strength of sigma diversity regularization. If > 0, adds a term that
            encourages larger sigma values to prevent regression to mean.
            Uses -log(sigma) regularization similar to VAEs.
        momentum: float, default=0.5
            Momentum coefficient for velocity-based parameter updates.
            If 0, no momentum is used. Values close to 1 provide smoother updates.
        trust_region_radius: float, default=None
            Maximum allowed update norm (L2) per step. Updates exceeding this
            radius are clipped proportionally to maintain stability.
            Set to None or 0 to disable trust region clipping. Recommended values
            are in range [0.01, 1.0] for typical problems.
        bounds: tuple, optional
            (lower, upper) bounds for each parameter
        budget: int, optional
            Maximum number of evaluations
        random_state: RandomState, optional
            Random number generator
    """
    
    def __init__(self, param_dim, lr_mu=0.05, lr_sigma=0.005,
                 sigma_min=0.001, sigma_max=1.0,
                 calibration_size=20, calibration_interval=50,
                 credit_factor=2.0, sigma_regularization=0.0,
                 momentum=0.5, trust_region_radius=None,
                 bounds=None, budget=None, random_state=None):
        
        super().__init__(param_dim, lr_mu, lr_sigma, sigma_min, sigma_max,
                        calibration_size, calibration_interval,
                        credit_factor, sigma_regularization,
                        bounds, budget, random_state)
        
        # Momentum parameters
        self.momentum = momentum
        self.velocity = None
        
        # Trust region
        self.trust_region_radius = trust_region_radius
        
        # Initialize distribution parameters
        self.mu = None
        self.sigma = None
    
    def initialize(self, initial_params=None):
        """Initialize the distribution parameters."""
        if initial_params is not None:
            self.mu = np.array(initial_params, dtype=float)
        else:
            self.mu = self.rng.randn(self.param_dim) * 0.01
        
        # Initialize sigma to constant value; per-dimension adaptation happens during optimization
        self.sigma = np.ones(self.param_dim) * 0.1
        self.velocity = np.zeros(self.param_dim) if self.momentum > 0 else None
        self.num_evaluations = 0
    
    def _clip_parameters(self):
        """Clip mu and sigma to bounds."""
        if self.bounds is not None:
            lower, upper = self.bounds
            self.mu = np.clip(self.mu, lower, upper)
            self.sigma = np.clip(self.sigma, 0, np.inf)
    
    def _apply_trust_region(self, update):
        """Apply trust region constraint to update vector."""
        if self.trust_region_radius is None or self.trust_region_radius <= 0:
            return update
        
        update_norm = np.linalg.norm(update)
        if update_norm > self.trust_region_radius:
            # Clip proportionally
            update = update * (self.trust_region_radius / (update_norm + 1e-8))
        return update
    
    def _population_step(self, loss_func):
        """
        Population calibration step for single Gaussian distribution.
        
        Samples multiple candidates, evaluates them, and performs natural
        gradient updates on mu and sigma.
        """
        noise = self.rng.randn(self.calibration_size, self.param_dim)
        perturbed = self.mu + self.sigma * noise
        
        # Evaluate all perturbed samples
        losses = np.array([loss_func(p) for p in perturbed])
        
        self._update_loss_scale(losses)
        
        # Natural gradient update for Gaussian distribution
        grad_mu = np.mean(losses[:, None] * noise, axis=0)
        grad_sigma = np.mean(losses[:, None] * (noise**2 - 1), axis=0)
        
        # Sigma regularization: add term to encourage larger sigma (prevents regression to mean)
        # Gradient of -log(sigma) w.r.t. sigma is +1/sigma, scaled by sigma_regularization strength
        if self.sigma_regularization > 0:
            grad_sigma += self.sigma_regularization * (1.0 / (self.sigma + 1e-8))
        
        # Adaptive learning rates: scale by inverse of loss scale
        # When loss is large, gradients are large, so we need smaller effective LR
        # When loss is small, gradients are small, so we need larger effective LR
        adaptive_lr_mu = self.lr_mu / (self.loss_scale + 1e-8)
        adaptive_lr_sigma = self.lr_sigma / (self.loss_scale + 1e-8)
        
        # Compute update
        update = adaptive_lr_mu * grad_mu
        
        # Apply momentum if enabled
        if self.momentum > 0 and self.velocity is not None:
            self.velocity = self.momentum * self.velocity + update
            update = self.velocity
        
        # Apply trust region constraint
        update = self._apply_trust_region(update)
        
        # Update distribution
        self.mu -= update
        self.sigma *= np.exp(adaptive_lr_sigma * grad_sigma)
        self.sigma = np.clip(self.sigma, self.sigma_min, self.sigma_max)
        
        return np.mean(losses)
    
    def _pairwise_step(self, loss_func):
        """
        Pairwise update step for single Gaussian distribution.
        
        Samples two candidates, compares their losses, and updates the distribution
        towards the better candidate.
        """
        # Sample two candidates
        theta1 = self.mu + self.sigma * self.rng.randn(self.param_dim)
        theta2 = self.mu + self.sigma * self.rng.randn(self.param_dim)
        
        loss1 = loss_func(theta1)
        loss2 = loss_func(theta2)
        
        # Determine winner and loser
        if loss1 < loss2:
            winner, loser = theta1, theta2
            winner_loss, loser_loss = loss1, loss2
        else:
            winner, loser = theta2, theta1
            winner_loss, loser_loss = loss2, loss1
        
        # Credit assignment based on absolute improvement, normalized by loss scale
        eps = 1e-8
        absolute_improvement = loser_loss - winner_loss
        
        # Normalize improvement by running loss scale estimate
        # This makes credit assignment invariant to the absolute loss magnitude
        normalized_improvement = absolute_improvement / (self.loss_scale + eps)
        update_strength = 1 + self.credit_factor * np.tanh(normalized_improvement)
        
        # Update loss scale estimate (exponential moving average)
        # Use absolute value for consistency (losses are always positive for MSE/MAE)
        avg_loss = (loss1 + loss2) / 2
        self._update_loss_scale(np.array([avg_loss]))
        
        # Parameter-specific adaptation
        winner_diff = winner - self.mu
        loser_diff = loser - self.mu
        update_mag = np.clip(np.abs(winner_diff) / (np.abs(loser_diff) + eps), 0.1, 10)
        
        # Compute update
        update = self.lr_mu * update_strength * update_mag * winner_diff
        
        # Apply momentum if enabled
        if self.momentum > 0 and self.velocity is not None:
            self.velocity = self.momentum * self.velocity + update
            update = self.velocity
        
        # Apply trust region constraint
        update = self._apply_trust_region(update)
        
        # Update mean towards winner
        self.mu += update
        
        # Update sigma based on observed diversity
        observed_diversity = np.abs(winner - loser)
        sigma_update = self.lr_sigma * update_strength * update_mag * (observed_diversity - self.sigma)
        
        # Sigma regularization: add term to encourage larger sigma (prevents regression to mean)
        # Gradient of -log(sigma) w.r.t. sigma is +1/sigma, scaled by sigma_regularization strength
        if self.sigma_regularization > 0:
            sigma_update += self.sigma_regularization * (1.0 / (self.sigma + 1e-8))
        
        self.sigma *= np.exp(sigma_update)
        self.sigma = np.clip(self.sigma, self.sigma_min, self.sigma_max)
        
        return avg_loss
    
    def get_parameters(self):
        """Return the current mean parameters."""
        return self.mu
    
    def get_distribution_parameters(self):
        """Return the distribution parameters (mu, sigma)."""
        return self.mu, self.sigma
    
    def set_parameters(self, params):
        """Set the mean parameters."""
        self.mu = np.array(params)
        # Reset velocity when parameters are explicitly set
        if self.momentum > 0:
            self.velocity = np.zeros(self.param_dim)
    
    def state_dict(self):
        """Return state dictionary for saving."""
        return {
            'mu': self.mu,
            'sigma': self.sigma,
            'velocity': self.velocity,
            'lr_mu': self.lr_mu,
            'lr_sigma': self.lr_sigma,
            'sigma_min': self.sigma_min,
            'sigma_max': self.sigma_max,
            'sigma_regularization': self.sigma_regularization,
            'momentum': self.momentum,
            'trust_region_radius': self.trust_region_radius,
            'num_evaluations': self.num_evaluations,
            'loss_scale': self.loss_scale,
        }
    
    def load_state_dict(self, state_dict):
        """Load state from dictionary."""
        self.mu = state_dict['mu']
        self.sigma = state_dict['sigma']
        self.velocity = state_dict.get('velocity', None)
        self.lr_mu = state_dict.get('lr_mu', self.lr_mu)
        self.lr_sigma = state_dict.get('lr_sigma', self.lr_sigma)
        self.sigma_min = state_dict.get('sigma_min', self.sigma_min)
        self.sigma_max = state_dict.get('sigma_max', self.sigma_max)
        self.sigma_regularization = state_dict.get('sigma_regularization', self.sigma_regularization)
        self.momentum = state_dict.get('momentum', self.momentum)
        self.trust_region_radius = state_dict.get('trust_region_radius', self.trust_region_radius)
        self.num_evaluations = state_dict.get('num_evaluations', 0)
        self.loss_scale = state_dict.get('loss_scale', 1.0)



class OptimizationResult:
    """Result of an optimization run."""
    def __init__(self, value, loss, num_evaluations=0):
        self.value = value
        self.loss = loss
        self.num_evaluations = num_evaluations
