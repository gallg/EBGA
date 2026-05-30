# EBGA Framework Documentation

**EBGA** (**E**volutionary-**B**ased **G**radient-free **A**rchitecture) is a neural network framework using distribution-based evolutionary optimization.

## Documentation Contents

1. **[Framework Overview](overview.md)** - Introduction, rationale, and key differences from other frameworks
2. **[Core Concepts](concepts.md)** - How the framework works, evolutionary principles
3. **[API Reference](api.md)** - Complete documentation for all modules
   - [models](api/models.md) - EBGARegressor, EBGAClassifier
   - [nn](api/nn.md) - Sequential neural network
   - [layers](api/layers.md) - Layer classes
   - [activations](api/activations.md) - Activation functions
   - [losses](api/losses.md) - Loss functions
   - [optimizer](api/optimizer.md) - CompactEvoOptimizer
4. **[Tutorials](tutorials.md)** - Step-by-step guides
5. **[Comparison](comparison.md)** - How EBGA differs from traditional approaches

## Quick Links

- [Installation Guide](installation.md)
- [Configuration Guide](configuration.md)
- [Best Practices](best_practices.md)

## Framework Philosophy

EBGA represents a fundamentally different approach to training neural networks:

- **No Gradients**: Uses evolutionary computation instead of backpropagation
- **Distribution-Based**: Maintains a Gaussian distribution over parameters (μ, σ)
- **Compact**: Uses natural gradient updates without maintaining a population
- **Universal**: Works with any loss function, including non-differentiable ones

This makes EBGA particularly suitable for:
- Problems with noisy or discontinuous objectives
- Non-differentiable loss functions
- Situations where gradient computation is problematic
- Research into alternative optimization paradigms
