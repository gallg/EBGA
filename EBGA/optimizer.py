import numpy as np


class BaseEvoOptimizer:
    """Base class for evolutionary optimizers."""

    def __init__(self, param_dim, lr_mu=0.05, lr_sigma=0.005,
                 sigma_min=0.001, sigma_max=1.0,
                 calibration_size=10, n_jobs=1, random_state=None):
        self.param_dim = param_dim
        self.lr_mu = lr_mu
        self.lr_sigma = lr_sigma
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.calibration_size = calibration_size
        self.n_jobs = n_jobs

        if random_state is None:
            self.rng = np.random.RandomState()
        elif isinstance(random_state, int):
            self.rng = np.random.RandomState(random_state)
        else:
            self.rng = random_state

    def step(self, loss_func, iteration=None):
        raise NotImplementedError

    def get_parameters(self):
        raise NotImplementedError

    def set_parameters(self, params):
        raise NotImplementedError

    def initialize(self, initial_params=None):
        raise NotImplementedError

    def get_distribution_parameters(self):
        raise NotImplementedError

    def state_dict(self):
        raise NotImplementedError

    def load_state_dict(self, state_dict):
        raise NotImplementedError


class CompactEvoOptimizer(BaseEvoOptimizer):
    """
    Compact Evolutionary Optimizer.

    Implements a single Gaussian distribution with diagonal covariance for
    parameter optimization using softmax-weighted recombination (NES-style).

    Every step samples a population of candidates from N(mu, sigma^2 I),
    evaluates them, and computes a softmax-weighted average as the new mean.

    Parameters
    ----------
    param_dim : int
        Dimensionality of the parameter space.
    lr_mu : float, default=0.05
        Temperature for softmax weighting. Lower values make selection more
        greedy, higher values more uniform.
    lr_sigma : float, default=0.005
        Learning rate for sigma adaptation.
    sigma_min : float, default=0.001
        Minimum standard deviation.
    sigma_max : float, default=1.0
        Maximum standard deviation.
    calibration_size : int, default=10
        Number of candidates sampled per step (population size).
    momentum : float, default=0.5
        Momentum coefficient for velocity-based updates. If 0, no momentum.
    trust_region_radius : float, default=None
        Maximum allowed update norm (L2) per step. None or 0 disables.
    n_jobs : int, default=1
        Number of parallel workers for candidate evaluation. Passed through
        to ``ParallelEvaluator`` when used together. The optimizer itself
        does not manage parallelism — use ``ParallelEvaluator`` to wrap
        the loss function for data-parallel evaluation across workers.
    random_state : RandomState, optional
        Random number generator.
    """

    def __init__(self, param_dim, lr_mu=0.05, lr_sigma=0.005,
                 sigma_min=0.001, sigma_max=1.0,
                 calibration_size=10,
                 momentum=0.5, trust_region_radius=None,
                 n_jobs=1, random_state=None):

        super().__init__(param_dim, lr_mu, lr_sigma, sigma_min, sigma_max,
                        calibration_size, n_jobs, random_state)

        self.momentum = momentum
        self.trust_region_radius = trust_region_radius
        self.mu = None
        self.sigma = None
        self.velocity = None

    def initialize(self, initial_params=None):
        if initial_params is not None:
            self.mu = np.array(initial_params, dtype=float)
        else:
            self.mu = self.rng.randn(self.param_dim) * 0.01
        self.sigma = np.ones(self.param_dim) * 0.1
        self.velocity = np.zeros(self.param_dim) if self.momentum > 0 else None

    def _apply_trust_region(self, update):
        if self.trust_region_radius is None or self.trust_region_radius <= 0:
            return update
        norm = np.linalg.norm(update)
        if norm > self.trust_region_radius:
            update = update * (self.trust_region_radius / (norm + 1e-8))
        return update

    def step(self, loss_func=None, iteration=None, evaluate_map=None):
        """
        Perform one optimization step.

        Parameters
        ----------
        loss_func : callable, optional
            Loss function: ``loss_func(params) -> float``. Required when
            ``evaluate_map`` is not provided.
        iteration : int, optional
            Current iteration number (used by some schedulers).
        evaluate_map : callable, optional
            If provided, used to evaluate all candidates in parallel.
            Signature: ``evaluate_map(candidates) -> np.ndarray`` where
            ``candidates`` is a list of parameter vectors and the return
            value is an array of loss values. When None, candidates are
            evaluated sequentially via ``loss_func``.
        """
        pop_size = self.calibration_size
        noise = self.rng.randn(pop_size, self.param_dim)
        candidates = self.mu + self.sigma * noise

        if evaluate_map is not None:
            losses = np.array(evaluate_map(candidates))
        else:
            losses = np.array([loss_func(p) for p in candidates])

        # Softmax-weighted recombination
        centered = losses - np.mean(losses)
        std = np.std(losses) + 1e-8
        logits = -centered / std / max(self.lr_mu, 1e-8)
        logits -= np.max(logits)
        weights = np.exp(logits)
        weights /= np.sum(weights) + 1e-8

        new_mu = np.sum(weights[:, None] * candidates, axis=0)
        update = new_mu - self.mu

        if self.momentum > 0 and self.velocity is not None:
            self.velocity = self.momentum * self.velocity + update
            update = self.velocity

        update = self._apply_trust_region(update)
        self.mu += update

        # Sigma adaptation
        weighted_dev = np.sqrt(np.sum(weights[:, None] * (candidates - self.mu)**2, axis=0))
        sigma_target = np.clip(weighted_dev, self.sigma_min, self.sigma_max)
        self.sigma += self.lr_sigma * (sigma_target - self.sigma)
        self.sigma = np.clip(self.sigma, self.sigma_min, self.sigma_max)

        return np.mean(losses)

    def get_parameters(self):
        return self.mu

    def get_distribution_parameters(self):
        return self.mu, self.sigma

    def set_parameters(self, params):
        self.mu = np.array(params)
        if self.momentum > 0:
            self.velocity = np.zeros(self.param_dim)

    def state_dict(self):
        return {
            'mu': self.mu,
            'sigma': self.sigma,
            'velocity': self.velocity,
            'lr_mu': self.lr_mu,
            'lr_sigma': self.lr_sigma,
            'sigma_min': self.sigma_min,
            'sigma_max': self.sigma_max,
            'momentum': self.momentum,
            'trust_region_radius': self.trust_region_radius,
            'n_jobs': self.n_jobs,
        }

    def load_state_dict(self, state_dict):
        self.mu = state_dict['mu']
        self.sigma = state_dict['sigma']
        self.velocity = state_dict.get('velocity', None)
        self.lr_mu = state_dict.get('lr_mu', self.lr_mu)
        self.lr_sigma = state_dict.get('lr_sigma', self.lr_sigma)
        self.sigma_min = state_dict.get('sigma_min', self.sigma_min)
        self.sigma_max = state_dict.get('sigma_max', self.sigma_max)
        self.momentum = state_dict.get('momentum', self.momentum)
        self.trust_region_radius = state_dict.get('trust_region_radius', self.trust_region_radius)
        self.n_jobs = state_dict.get('n_jobs', 1)
