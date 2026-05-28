import numpy as np
from sklearn.utils import check_random_state
from sklearn.metrics import mean_absolute_error, mean_squared_error

def sigmoid(x):
    """Numerically stable sigmoid function."""
    # Clip input to prevent overflow
    x = np.clip(x, -50, 50)  # exp(-50) is already a very small number (~1.9e-22)
    return np.where(
        x >= 0,
        1 / (1 + np.exp(-x)),
        np.exp(x) / (1 + np.exp(x))
    )

class EvolutionaryAutoencoder:
    """
    An evolutionary autoencoder that learns a compact latent representation
    using Compact Genetic Descent.
    """
    def __init__(self, img_shape=(28, 28), latent_dim=32,
                 target_loss=0.1, max_iter_safeguard=10000,
                 loss_metric='mse', lr_mu=0.05, lr_sigma=0.005,
                 sigma_min=0.01, sigma_max=1.0,
                 calibration_interval=25, credit_factor=2.0,
                 calibration_size=20, random_state=None):
        """
        Parameters
        ----------
        img_shape : tuple, default=(28, 28)
            Shape of input images.
        latent_dim : int, default=32
            Dimension of the latent space.
        target_loss : float, default=0.1
            Target loss value to reach for stopping training.
        max_iter_safeguard : int, default=10000
            Maximum number of iterations as a safeguard.
        loss_metric : str, default='mse'
            Loss metric to use ('mse' or 'mae').
        lr_mu : float, default=0.05
            Learning rate for mean parameters.
        lr_sigma : float, default=0.005
            Learning rate for standard deviation.
        sigma_min/max : float, default=0.01/1.0
            Bounds for standard deviation.
        calibration_interval : int, default=25
            How often to use population calibration.
        credit_factor : float, default=2.0
            Strength of credit assignment.
        calibration_size : int, default=20
            Number of samples for calibration.
        random_state : int, default=None
            Random seed for reproducibility.
        """
        self.img_shape = img_shape
        self.n_pixels = np.prod(img_shape)
        self.latent_dim = latent_dim
        self.target_loss = target_loss
        self.max_iter_safeguard = max_iter_safeguard
        self.loss_metric = loss_metric.lower()
        self.lr_mu = lr_mu
        self.lr_sigma = lr_sigma
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.calibration_interval = calibration_interval
        self.credit_factor = credit_factor
        self.calibration_size = calibration_size
        self.random_state = random_state

        # Initialize model parameters
        self._initialize_parameters()

    def _initialize_parameters(self):
        """Initialize the encoding and decoding parameters."""
        self.rng = check_random_state(self.random_state)

        # Encoder parameters scaled to prevent large initial outputs
        self.encoder_mu = {
            'weights': self.rng.randn(self.n_pixels, self.latent_dim) * 0.01,
            'bias': self.rng.randn(self.latent_dim) * 0.01
        }
        self.encoder_logvar = {
            'weights': self.rng.randn(self.n_pixels, self.latent_dim) * 0.001,  # Smaller for logvar
            'bias': -3.0 * np.ones(self.latent_dim)  # Initialize to small variance (~0.05)
        }

        # Decoder parameters
        self.decoder = {
            'weights': self.rng.randn(self.latent_dim, self.n_pixels) * 0.01,
            'bias': self.rng.randn(self.n_pixels) * 0.01
        }

        # Distribution parameters for evolutionary optimization
        self.param_names = [
            'encoder_mu_weights', 'encoder_mu_bias',
            'encoder_logvar_weights', 'encoder_logvar_bias',
            'decoder_weights', 'decoder_bias'
        ]

        # Initialize mu and sigma for all parameters
        self.mu = self._get_flat_parameters()
        self.sigma = np.ones_like(self.mu) * 0.05  # Smaller initial sigma

    def _get_flat_parameters(self):
        """Flatten all model parameters into a single vector."""
        params = []

        # Encoder mean
        params.extend(self.encoder_mu['weights'].flatten())
        params.extend(self.encoder_mu['bias'].flatten())

        # Encoder log-variance
        params.extend(self.encoder_logvar['weights'].flatten())
        params.extend(self.encoder_logvar['bias'].flatten())

        # Decoder
        params.extend(self.decoder['weights'].flatten())
        params.extend(self.decoder['bias'].flatten())

        return np.array(params)

    def _set_flat_parameters(self, flat_params):
        """Set model parameters from a flattened vector."""
        idx = 0

        try:
            # Encoder mean
            end = idx + self.encoder_mu['weights'].size
            self.encoder_mu['weights'] = flat_params[idx:end].reshape(self.encoder_mu['weights'].shape)
            idx = end

            end = idx + self.encoder_mu['bias'].size
            self.encoder_mu['bias'] = flat_params[idx:end]
            idx = end

            # Encoder log-variance
            end = idx + self.encoder_logvar['weights'].size
            self.encoder_logvar['weights'] = flat_params[idx:end].reshape(self.encoder_logvar['weights'].shape)
            idx = end

            end = idx + self.encoder_logvar['bias'].size
            self.encoder_logvar['bias'] = flat_params[idx:end]
            idx = end

            # Decoder
            end = idx + self.decoder['weights'].size
            self.decoder['weights'] = flat_params[idx:end].reshape(self.decoder['weights'].shape)
            idx = end

            end = idx + self.decoder['bias'].size
            self.decoder['bias'] = flat_params[idx:end]
        except Exception as e:
            print(f"Error while setting parameters: {e}")
            print(f"Index: {idx}, End: {end}, Flat params shape: {flat_params.shape}")
            raise

    def encode(self, x):
        """Encode input into latent mean and log-variance."""
        x_flat = x.reshape(-1, self.n_pixels)

        # Calculate latent mean
        latent_mu = np.dot(x_flat, self.encoder_mu['weights']) + self.encoder_mu['bias']

        # Calculate latent log-variance
        latent_logvar = np.dot(x_flat, self.encoder_logvar['weights']) + self.encoder_logvar['bias']
        latent_logvar = np.clip(latent_logvar, -10, 10)  # Prevent extreme variances

        return latent_mu, latent_logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick to sample from N(mu, var)."""
        std = np.exp(0.5 * logvar)
        std = np.clip(std, 1e-6, 1e6)  # Prevent extreme std values
        eps = self.rng.randn(*mu.shape)
        return mu + eps * std

    def decode(self, z):
        """Decode latent vector into image."""
        # Simple linear decoder with sigmoid
        recon = np.dot(z, self.decoder['weights']) + self.decoder['bias']
        return sigmoid(recon).reshape(self.img_shape)

    def forward(self, x):
        """Forward pass: x -> z -> x_recon."""
        # Encode
        mu, logvar = self.encode(x)

        # Sample latent
        z = self.reparameterize(mu, logvar)

        # Decode
        recon = self.decode(z)

        return recon

    def calculate_loss(self, x, recon):
        """Calculate loss between original and reconstructed image."""
        x_flat = x.reshape(-1, self.n_pixels)
        recon_flat = recon.reshape(-1, self.n_pixels)

        if self.loss_metric == 'mse':
            loss = mean_squared_error(x_flat, recon_flat)
        elif self.loss_metric == 'mae':
            loss = mean_absolute_error(x_flat, recon_flat)
        else:
            raise ValueError("loss_metric must be 'mse' or 'mae'")

        return loss

    def fit(self, X_train):
        """Train the evolutionary autoencoder."""
        loss_history = []
        best_loss = float('inf')
        iteration = 0

        print(f"Training until {self.loss_metric.upper()} reaches {self.target_loss}...")

        while best_loss > self.target_loss and iteration < self.max_iter_safeguard:
            iteration += 1

            if iteration % self.calibration_interval == 0:
                # Use population calibration for robustness
                loss = self._population_calibration(X_train)
            else:
                # Use efficient pairwise updates
                loss = self._pairwise_update(X_train)

            loss_history.append(loss)

            # Track best loss
            if loss < best_loss:
                best_loss = loss

            # Print progress
            if iteration % 100 == 0:
                print(f"Iteration {iteration}: Current {self.loss_metric} = {loss:.4f}, Best = {best_loss:.4f}")

            # Check if target is reached
            if best_loss <= self.target_loss:
                print(f"Target {self.loss_metric.upper()} of {self.target_loss} reached at iteration {iteration}!")
                break

        # If safeguard is hit
        if iteration >= self.max_iter_safeguard:
            print(f"Max iterations safeguard ({self.max_iter_safeguard}) reached. Best {self.loss_metric.upper()}: {best_loss:.4f}")

        return loss_history

    def _pairwise_update(self, X_train):
        """Perform compact pairwise update."""
        # Sample two parameter sets
        theta1 = self.mu + self.sigma * self.rng.randn(len(self.mu))
        theta2 = self.mu + self.sigma * self.rng.randn(len(self.mu))

        # Set parameters and calculate losses
        try:
            self._set_flat_parameters(theta1)
            recon1 = self.forward(X_train[0])  # Use first training sample
            loss1 = self.calculate_loss(X_train[0], recon1)

            self._set_flat_parameters(theta2)
            recon2 = self.forward(X_train[0])  # Use first training sample
            loss2 = self.calculate_loss(X_train[0], recon2)
        except Exception as e:
            print(f"Error during pairwise update: {e}")
            print(f"mu range: [{self.mu.min():.4f}, {self.mu.max():.4f}]")
            print(f"sigma range: [{self.sigma.min():.4f}, {self.sigma.max():.4f}]")
            print(f"theta1 range: [{theta1.min():.4f}, {theta1.max():.4f}]")
            print(f"theta2 range: [{theta2.min():.4f}, {theta2.max():.4f}]")
            raise

        # Determine winner and loser (lower loss is better)
        winner, loser = (theta1, theta2) if loss1 < loss2 else (theta2, theta1)
        improvement = abs(loss1 - loss2)

        # Clip improvement to prevent extreme updates
        improvement = np.clip(improvement, 0, 1.0)

        # Credit assignment
        credit = 1 + self.credit_factor * np.tanh(improvement)
        winner_flat = winner
        loser_flat = loser

        # Update magnitude based on parameter sensitivity
        eps = 1e-8
        update_mag = np.clip(
            np.abs(winner_flat - self.mu) / (np.abs(loser_flat - self.mu) + eps),
            0.1, 10.0
        )

        # Update mu (move toward winner)
        self.mu += self.lr_mu * credit * update_mag * (winner_flat - self.mu)

        # Update sigma based on observed diversity - numerically stable implementation
        observed_diversity = np.abs(winner_flat - loser_flat)
        observed_diversity = np.clip(observed_diversity, 1e-6, 1e6)  # Prevent extreme values
        sigma_update = self.lr_sigma * credit * update_mag * (observed_diversity - self.sigma)
        sigma_update = np.clip(sigma_update, -0.1, 0.1)  # Limit update magnitude
        self.sigma = self.sigma * np.exp(sigma_update)  # More stable than direct exponentiation
        self.sigma = np.clip(self.sigma, self.sigma_min, self.sigma_max)

        # Set the model parameters to current mu
        self._set_flat_parameters(self.mu)

        return (loss1 + loss2) / 2

    def _population_calibration(self, X_train):
        """Use population samples to calibrate distribution parameters."""
        # Sample population from current distribution
        noise = self.rng.randn(self.calibration_size, len(self.mu))
        perturbed = self.mu + self.sigma * noise

        # Create array to store losses
        losses = np.zeros(self.calibration_size)

        # Calculate loss for each perturbed sample
        for i in range(self.calibration_size):
            try:
                self._set_flat_parameters(perturbed[i])
                recon = self.forward(X_train[0])  # Use first training sample
                losses[i] = self.calculate_loss(X_train[0], recon)
            except Exception as e:
                print(f"Error during population calibration for sample {i}: {e}")
                losses[i] = float('inf')  # Assign worst possible loss

        # Remove invalid losses
        valid_idx = np.isfinite(losses)
        if not np.any(valid_idx):
            print("Warning: All samples in population calibration produced invalid losses!")
            return float('inf')

        noise = noise[valid_idx]
        losses = losses[valid_idx]
        if len(losses) < 2:
            print("Warning: Not enough valid samples for population calibration!")
            return float('inf')

        # Normalize losses to prevent extreme values in gradient estimation
        loss_mean = np.mean(losses)
        loss_std = np.std(losses)
        if loss_std < 1e-6:
            # If all losses are the same, no gradient information
            return loss_mean

        # Standardize losses
        loss_std = np.clip(loss_std, 1e-6, 1e6)
        standardized_losses = (losses - loss_mean) / loss_std

        # Estimate gradients via REINFORCE (minimizing loss)
        grad_mu = np.mean(standardized_losses[:, None] * noise, axis=0)
        grad_sigma = np.mean(standardized_losses[:, None] * (noise**2 - 1), axis=0)

        # Clip gradients to prevent extreme updates
        grad_mu = np.clip(grad_mu, -1.0, 1.0)
        grad_sigma = np.clip(grad_sigma, -1.0, 1.0)

        # Update parameters
        self.mu -= self.lr_mu * grad_mu
        self.sigma *= np.exp(-self.lr_sigma * grad_sigma)
        self.sigma = np.clip(self.sigma, self.sigma_min, self.sigma_max)

        # Set the model parameters to current mu
        self._set_flat_parameters(self.mu)

        return np.mean(losses)

    def generate(self, n_samples=1):
        """Generate new samples from the learned latent distribution."""
        # Sample from prior (N(0, I)) and decode
        z = self.rng.randn(n_samples, self.latent_dim)
        return self.decode(z)

    def reconstruct(self, x):
        """Reconstruct an input image."""
        mu, _ = self.encode(x)
        return self.decode(mu)  # Use mean for reconstruction
