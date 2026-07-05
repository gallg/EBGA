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
            Learning rate for mean parameters
        lr_sigma: float, default=0.005
            Learning rate for standard deviation parameters
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


class MultiCandidateOptimizer(BaseEvoOptimizer):
    """
    Multi-candidate evolutionary optimizer with adaptive weighting.
    
    Maintains K candidate solutions, each with its own mean and diagonal
    standard deviation. Samples from candidates proportionally to their
    performance-based weights, enabling escape from local optima while
    maintaining computational efficiency.
    
    Parameters:
        param_dim: int
            Dimensionality of the parameter space
        n_candidates: int, default=3
            Number of candidate distributions to maintain
        beta: float, default=1.0
            Temperature parameter for weight softmax (higher = more aggressive weight concentration)
        alpha: float, default=0.1
            EMA decay rate for candidate loss tracking
        lr_mu: float, default=0.05
            Learning rate for mean parameters
        lr_sigma: float, default=0.005
            Learning rate for standard deviation parameters
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
            Strength of sigma diversity regularization
        bounds: tuple, optional
            (lower, upper) bounds for each parameter
        budget: int, optional
            Maximum number of evaluations
        random_state: RandomState, optional
            Random number generator
    """
    
    def __init__(self, param_dim, n_candidates=3,
                 beta=1.0, alpha=0.1,
                 lr_mu=0.05, lr_sigma=0.005,
                 sigma_min=0.001, sigma_max=1.0,
                 calibration_size=20, calibration_interval=25,
                 credit_factor=2.0, sigma_regularization=0.0,
                 bounds=None, budget=None, random_state=None):
        
        super().__init__(param_dim, lr_mu, lr_sigma, sigma_min, sigma_max,
                        calibration_size, calibration_interval,
                        credit_factor, sigma_regularization,
                        bounds, budget, random_state)
        
        self.n_candidates = n_candidates
        self.beta = beta
        self.alpha = alpha
        
        # Candidate parameters: list of arrays
        self.mus = []
        self.sigmas = []
        
        # Candidate performance tracking
        self.ema_losses = np.zeros(n_candidates)
        self.weights = np.ones(n_candidates) / n_candidates
    
    def initialize(self, initial_params=None):
        """Initialize all candidate distributions."""
        if initial_params is not None:
            base_mu = np.array(initial_params, dtype=float)
        else:
            base_mu = self.rng.randn(self.param_dim) * 0.01
        
        base_sigma = np.ones(self.param_dim) * 0.1
        
        self.mus = []
        self.sigmas = []
        for i in range(self.n_candidates):
            # Spread candidates around initial point
            offset = self.rng.randn(self.param_dim) * 0.5
            self.mus.append(base_mu + offset)
            self.sigmas.append(base_sigma.copy())
        
        # Reset performance tracking
        self.ema_losses = np.zeros(self.n_candidates)
        self.weights = np.ones(self.n_candidates) / self.n_candidates
        self.num_evaluations = 0
        self.loss_scale = 1.0
    
    def _clip_parameters(self):
        """Clip all candidate parameters to bounds."""
        if self.bounds is not None:
            lower, upper = self.bounds
            for i in range(self.n_candidates):
                self.mus[i] = np.clip(self.mus[i], lower, upper)
                self.sigmas[i] = np.clip(self.sigmas[i], 0, np.inf)
    
    def _sample_from_candidate(self, candidate_idx):
        """Sample a single parameter vector from candidate k."""
        return self.mus[candidate_idx] + self.sigmas[candidate_idx] * self.rng.randn(self.param_dim)
    
    def _sample_candidate_idx(self):
        """Sample a candidate index based on current weights."""
        return self.rng.choice(self.n_candidates, p=self.weights)
    
    def _update_weights(self, candidate_idx, loss):
        """
        Update candidate weights based on observed loss.
        
        Uses softmax of negative EMA losses to ensure weights sum to 1
        and stay in (0,1).
        """
        # Update EMA loss for this candidate
        self.ema_losses[candidate_idx] = (
            self.alpha * loss +
            (1 - self.alpha) * self.ema_losses[candidate_idx]
        )
        
        # Recompute all weights (softmax of negative EMA losses)
        weighted_losses = -self.beta * self.ema_losses
        max_weighted = np.max(weighted_losses)  # For numerical stability
        exp_weights = np.exp(weighted_losses - max_weighted)
        self.weights = exp_weights / np.sum(exp_weights)
    
    def _population_step(self, loss_func):
        """
        Population calibration step: sample from all candidates,
        evaluate, and update each candidate based on its samples.
        
        Returns:
            float: Mean loss across all samples
        """
        # Sample from each candidate
        all_losses = []
        candidate_samples = []  # List of arrays: samples per candidate
        candidate_losses = []  # List of arrays: losses per candidate
        
        for k in range(self.n_candidates):
            k_samples = []
            k_losses = []
            for _ in range(self.calibration_size):
                sample = self._sample_from_candidate(k)
                k_samples.append(sample)
                k_losses.append(loss_func(sample))
                all_losses.append(k_losses[-1])
            candidate_samples.append(np.array(k_samples))
            candidate_losses.append(np.array(k_losses))
        
        # Update loss scale estimate
        self._update_loss_scale(all_losses)
        
        # Update each candidate based on its samples
        for k in range(self.n_candidates):
            self._update_candidate_population(
                k,
                candidate_samples[k],
                candidate_losses[k]
            )
            
            # Update weights for each sample
            for sample_loss in candidate_losses[k]:
                self._update_weights(k, sample_loss)
        
        return np.mean(all_losses)
    
    def _update_candidate_population(self, candidate_idx, samples, losses):
        """
        Update candidate k using natural gradient from population samples.
        
        Parameters:
            candidate_idx: int
                Index of candidate to update
            samples: numpy.ndarray
                Array of samples from this candidate, shape (n_samples, param_dim)
            losses: numpy.ndarray
                Array of loss values for the samples, shape (n_samples,)
        """
        mu_k = self.mus[candidate_idx]
        sigma_k = self.sigmas[candidate_idx]
        
        # Centered and normalized samples: z = (x - mu) / sigma
        z = (samples - mu_k) / (sigma_k + 1e-8)
        
        # Natural gradient for diagonal Gaussian
        # These are the analytical gradients of the expected loss w.r.t. distribution parameters
        grad_mu = np.mean(losses[:, None] * z, axis=0)
        grad_sigma = np.mean(losses[:, None] * (z**2 - 1), axis=0)
        
        # Sigma regularization: add term to encourage larger sigma (prevents regression to mean)
        # Gradient of -log(sigma) w.r.t. sigma is +1/sigma
        if self.sigma_regularization > 0:
            grad_sigma += self.sigma_regularization * (1.0 / (sigma_k + 1e-8))
        
        # Adaptive learning rates: scale by inverse of loss scale
        adaptive_lr_mu = self.lr_mu / (self.loss_scale + 1e-8)
        adaptive_lr_sigma = self.lr_sigma / (self.loss_scale + 1e-8)
        
        # Update distribution parameters
        self.mus[candidate_idx] -= adaptive_lr_mu * grad_mu
        self.sigmas[candidate_idx] *= np.exp(adaptive_lr_sigma * grad_sigma)
        self.sigmas[candidate_idx] = np.clip(
            self.sigmas[candidate_idx],
            self.sigma_min,
            self.sigma_max
        )
    
    def _pairwise_step(self, loss_func):
        """
        Pairwise update step: sample two candidates, compare, update both.
        
        Returns:
            float: Mean loss of the two samples
        """
        # Sample two candidates (possibly the same)
        k1 = self._sample_candidate_idx()
        k2 = self._sample_candidate_idx()
        
        theta1 = self._sample_from_candidate(k1)
        theta2 = self._sample_from_candidate(k2)
        
        loss1 = loss_func(theta1)
        loss2 = loss_func(theta2)
        
        # Determine winner and loser
        if loss1 < loss2:
            winner, loser = theta1, theta2
            winner_loss, loser_loss = loss1, loss2
            winner_candidate = k1
        else:
            winner, loser = theta2, theta1
            winner_loss, loser_loss = loss2, loss1
            winner_candidate = k2
        
        # Update loss scale estimate
        avg_loss = (loss1 + loss2) / 2
        self._update_loss_scale(np.array([avg_loss]))
        
        # Credit assignment
        eps = 1e-8
        absolute_improvement = loser_loss - winner_loss
        normalized_improvement = absolute_improvement / (self.loss_scale + eps)
        update_strength = 1 + self.credit_factor * np.tanh(normalized_improvement)
        
        # Update both candidates
        for k, theta, loss in [(k1, theta1, loss1), (k2, theta2, loss2)]:
            # Parameter-specific adaptation
            winner_diff = winner - self.mus[k]
            loser_diff = loser - self.mus[k]
            update_mag = np.clip(
                np.abs(winner_diff) / (np.abs(loser_diff) + eps),
                0.1, 10
            )
            
            # Update mean: stronger update for winner's candidate
            if k == winner_candidate:
                lr_factor = 1.0
            else:
                lr_factor = 0.5  # Loser's candidate gets weaker update
            
            self.mus[k] += (
                lr_factor * self.lr_mu * update_strength * update_mag * winner_diff
            )
            
            # Update sigma based on observed diversity
            observed_diversity = np.abs(winner - loser)
            sigma_update = (
                self.lr_sigma * update_strength * update_mag *
                (observed_diversity - self.sigmas[k])
            )
            
            # Sigma regularization
            if self.sigma_regularization > 0:
                sigma_update += self.sigma_regularization * (1.0 / (self.sigmas[k] + 1e-8))
            
            self.sigmas[k] *= np.exp(sigma_update)
            self.sigmas[k] = np.clip(
                self.sigmas[k],
                self.sigma_min,
                self.sigma_max
            )
            
            # Update weight based on this sample's loss
            self._update_weights(k, loss)
        
        return (loss1 + loss2) / 2
    
    def get_parameters(self):
        """Return weighted average of all candidate means."""
        return np.sum(
            [w * mu for w, mu in zip(self.weights, self.mus)],
            axis=0
        )
    
    def set_parameters(self, params):
        """Set all candidate means to params, spread out."""
        params = np.array(params)
        for i in range(self.n_candidates):
            # Spread candidates around the new parameters
            offset = self.rng.randn(self.param_dim) * 0.1
            self.mus[i] = params + offset
            # Initialize or keep existing sigma
            if i >= len(self.sigmas):
                self.sigmas.append(np.ones(self.param_dim) * 0.1)
    
    def get_distribution_parameters(self):
        """Return all candidate parameters for inspection."""
        return [m.copy() for m in self.mus], [s.copy() for s in self.sigmas]
    
    def state_dict(self):
        """Return state dictionary for saving."""
        return {
            'mus': [m.copy() for m in self.mus],
            'sigmas': [s.copy() for s in self.sigmas],
            'ema_losses': self.ema_losses.copy(),
            'weights': self.weights.copy(),
            'lr_mu': self.lr_mu,
            'lr_sigma': self.lr_sigma,
            'sigma_min': self.sigma_min,
            'sigma_max': self.sigma_max,
            'sigma_regularization': self.sigma_regularization,
            'num_evaluations': self.num_evaluations,
            'loss_scale': self.loss_scale,
            'n_candidates': self.n_candidates,
            'beta': self.beta,
            'alpha': self.alpha,
        }
    
    def load_state_dict(self, state_dict):
        """Load state from dictionary."""
        self.mus = [np.array(m) for m in state_dict['mus']]
        self.sigmas = [np.array(s) for s in state_dict['sigmas']]
        self.ema_losses = state_dict['ema_losses']
        self.weights = state_dict['weights']
        self.lr_mu = state_dict.get('lr_mu', self.lr_mu)
        self.lr_sigma = state_dict.get('lr_sigma', self.lr_sigma)
        self.sigma_min = state_dict.get('sigma_min', self.sigma_min)
        self.sigma_max = state_dict.get('sigma_max', self.sigma_max)
        self.sigma_regularization = state_dict.get('sigma_regularization', self.sigma_regularization)
        self.num_evaluations = state_dict.get('num_evaluations', 0)
        self.loss_scale = state_dict.get('loss_scale', 1.0)
        self.n_candidates = state_dict.get('n_candidates', self.n_candidates)
        self.beta = state_dict.get('beta', self.beta)
        self.alpha = state_dict.get('alpha', self.alpha)


class OptimizationResult:
    """Result of an optimization run."""
    def __init__(self, value, loss, num_evaluations=0):
        self.value = value
        self.loss = loss
        self.num_evaluations = num_evaluations
