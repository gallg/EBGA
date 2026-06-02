import numpy as np


class CompactEvoOptimizer:
    """
    Compact Evolutionary Optimizer.
    
    A gradient-free evolutionary optimization algorithm that maintains
    a Gaussian distribution over parameters (mean mu, std sigma) and
    updates it based on fitness evaluations using natural gradient updates.
    
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
        random_state: RandomState, optional
            Random number generator
    """
    
    def __init__(self, param_dim, lr_mu=0.05, lr_sigma=0.005,
                 sigma_min=0.001, sigma_max=1.0,
                 calibration_size=20, calibration_interval=25,
                 credit_factor=2.0, random_state=None):
        self.param_dim = param_dim
        self.lr_mu = lr_mu
        self.lr_sigma = lr_sigma
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.calibration_size = calibration_size
        self.calibration_interval = calibration_interval
        self.credit_factor = credit_factor
        
        if random_state is None:
            self.rng = np.random.RandomState()
        elif isinstance(random_state, int):
            self.rng = np.random.RandomState(random_state)
        else:
            self.rng = random_state
        
        # Initialize distribution parameters
        self.mu = None
        self.sigma = None
        
        # Running estimate of loss scale for adaptive credit assignment
        self.loss_scale = 1.0
        self.loss_scale_decay = 0.99
    
    def initialize(self, initial_params=None):
        if initial_params is not None:
            self.mu = np.array(initial_params, dtype=float)
        else:
            self.mu = self.rng.randn(self.param_dim) * 0.01
        
        self.sigma = np.ones(self.param_dim) * 0.1
    
    def step(self, loss_func, iteration=None):
        if iteration is None or iteration % self.calibration_interval == 0:
            return self._population_calibration_step(loss_func)
        else:
            return self._pairwise_update_step(loss_func)
    
    def _population_calibration_step(self, loss_func):
        noise = self.rng.randn(self.calibration_size, self.param_dim)
        perturbed = self.mu + self.sigma * noise
        
        # Evaluate all perturbed samples
        losses = np.array([loss_func(p) for p in perturbed])
        
        # Update loss scale estimate (exponential moving average of loss magnitude)
        # Use mean of evaluated losses for consistency with pairwise step
        self.loss_scale = self.loss_scale_decay * self.loss_scale + (1 - self.loss_scale_decay) * np.mean(np.abs(losses))
        
        # Natural gradient update for Gaussian distribution
        grad_mu = np.mean(losses[:, None] * noise, axis=0)
        grad_sigma = np.mean(losses[:, None] * (noise**2 - 1), axis=0)
        
        # Adaptive learning rates: scale by inverse of loss scale
        # When loss is large, gradients are large, so we need smaller effective LR
        # When loss is small, gradients are small, so we need larger effective LR
        adaptive_lr_mu = self.lr_mu / (self.loss_scale + 1e-8)
        adaptive_lr_sigma = self.lr_sigma / (self.loss_scale + 1e-8)
        
        # Update distribution
        self.mu -= adaptive_lr_mu * grad_mu
        self.sigma *= np.exp(adaptive_lr_sigma * grad_sigma)
        self.sigma = np.clip(self.sigma, self.sigma_min, self.sigma_max)
        
        return np.mean(losses)
    
    def _pairwise_update_step(self, loss_func):
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
        self.loss_scale = self.loss_scale_decay * self.loss_scale + (1 - self.loss_scale_decay) * np.abs(avg_loss)
        
        # Parameter-specific adaptation
        winner_diff = winner - self.mu
        loser_diff = loser - self.mu
        update_mag = np.clip(np.abs(winner_diff) / (np.abs(loser_diff) + eps), 0.1, 10)
        
        # Update mean towards winner
        self.mu += self.lr_mu * update_strength * update_mag * winner_diff
        
        # Update sigma based on observed diversity
        observed_diversity = np.abs(winner - loser)
        self.sigma *= np.exp(self.lr_sigma * update_strength * update_mag *
                            (observed_diversity - self.sigma))
        self.sigma = np.clip(self.sigma, self.sigma_min, self.sigma_max)
        
        return (loss1 + loss2) / 2
    
    def get_parameters(self):
        return self.mu
    
    def get_distribution_parameters(self):
        return self.mu, self.sigma
    
    def set_parameters(self, params):
        self.mu = np.array(params)
    
    def state_dict(self):
        return {
            'mu': self.mu,
            'sigma': self.sigma,
            'lr_mu': self.lr_mu,
            'lr_sigma': self.lr_sigma,
            'sigma_min': self.sigma_min,
            'sigma_max': self.sigma_max,
        }
    
    def load_state_dict(self, state_dict):
        self.mu = state_dict['mu']
        self.sigma = state_dict['sigma']
        self.lr_mu = state_dict.get('lr_mu', self.lr_mu)
        self.lr_sigma = state_dict.get('lr_sigma', self.lr_sigma)
        self.sigma_min = state_dict.get('sigma_min', self.sigma_min)
        self.sigma_max = state_dict.get('sigma_max', self.sigma_max)
