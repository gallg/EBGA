# EBGA Package Naming and Multi-Distribution Support

This document summarizes the discussion on renaming the **Evolutionary Bayesian Geometric Adaptation (EBGA)** package and extending it to support multiple distributions beyond Gaussians.

---

## 📌 1. Package Naming Discussion

### **Problem with Previous Name: "Evolutionary-Based Gradient-Free Algorithms (EBGA)"**
- **Misleading**: EBGA is **not gradient-free** in the traditional sense (e.g., compared to SGD or Backpropagation).
  - It **does not compute gradients of the objective function** (e.g., `∇L/∇θ`), making it gradient-free *with respect to the objective*.
  - However, it **does compute gradients of the expected fitness** with respect to the **distribution parameters** (e.g., `μ`, `Σ` for a Gaussian).
- **Confusing**: The term "gradient-free" typically implies no gradients are used at all (e.g., genetic algorithms, CMA-ES). EBGA uses **natural gradients**, which are a specific type of gradient.

---

### **Proposed Corrections to the EBGA Acronym**

| **Acronym** | **Expanded Name**                          | **Why It Fits**                                                                                     | **Pros**                                                                                     | **Cons**                                                                                     |
|-------------|--------------------------------------------|---------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| **EBGA**    | **E**volutionary **B**ayesian **G**eometric **A**daptation | Emphasizes Bayesian principles and geometric (Riemannian) updates.                          | Accurate, modern, and highlights key features (Bayesian + geometry).                       | Less intuitive for users unfamiliar with information geometry.                              |
| **EBGA**    | **E**volutionary **B**ased **G**radient **A**lignment       | Focuses on aligning updates with the natural gradient.                                         | Clear and technically accurate.                                                               | Slightly less catchy.                                                                         |
| **NEGA**    | **N**atural **E**volutionary **G**radient **A**lgorithm    | Directly references natural gradients and evolutionary framework.                             | Clear, concise, and widely understandable.                                                  | Loses the "E" for Evolutionary (though still implied).                                       |
| **IGEA**    | **I**nformation-**G**eometric **E**volutionary **A**lgorithm | Highlights the information-geometric foundation (Fisher matrix, Riemannian metrics).        | Theoretically rigorous and appealing to researchers.                                      | Less intuitive for practitioners.                                                          |
| **FEGA**    | **F**isher **E**volutionary **G**radient **A**lgorithm     | Explicitly mentions the Fisher matrix, a key component of natural gradients.                  | Emphasizes the unique feature (Fisher matrix).                                               | Niche appeal (may not resonate with all users).                                               |
| **REGA**    | **R**iemannian **E**volutionary **G**radient **A**lgorithm | Highlights the Riemannian geometry aspect.                                                   | Mathematically precise.                                                                     | Less familiar to non-specialists.                                                             |
| **BEGA**    | **B**ayesian **E**volutionary **G**radient **A**lgorithm   | Combines Bayesian, evolutionary, and gradient-based ideas.                                     | Broad appeal (Bayesian + evolutionary).                                                      | Slightly redundant (Bayesian and evolutionary overlap).                                      |

---

### **Recommended Naming Strategy**
1. **If keeping "EBGA"**:
   - Use **"Evolutionary Bayesian Geometric Adaptation"** (EBGA).
   - **Why?** Accurate, modern, and highlights the Bayesian and geometric aspects.

2. **If open to a new acronym**:
   - Use **"Natural Evolutionary Gradient Algorithm" (NEGA)**.
   - **Why?** Clear, concise, and directly describes the hybrid nature (natural gradients + evolutionary framework).

3. **For marketing or broader appeal**:
   - Use **"Information-Geometric Evolutionary Algorithm" (IGEA)** or **"Fisher Evolutionary Gradient Algorithm" (FEGA)** to emphasize the theoretical foundations.

---

## 📌 2. Supporting Multiple Distributions in EBGA

### **Why Move Beyond Gaussians?**
Gaussians are simple and effective for many problems, but they have **limitations** that can be addressed by supporting other distributions:

| **Limitation**               | **Problem**                                                                                     | **Solution**                                                                                     |
|------------------------------|-------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|
| **Unimodal**                 | Can only represent one peak (mode).                                                            | Use **Gaussian Mixture Models (GMMs)** or **normalizing flows**.                              |
| **Symmetric**                | Assumes symmetric uncertainty around the mean.                                     | Use **Log-Normal, Beta, or Gamma** distributions.                                               |
| **Light-tailed**             | Rarely samples far from the mean (poor exploration).                                         | Use **Student’s t, Laplace, or Cauchy** distributions.                                          |
| **Fixed Support**            | Assigns non-zero probability everywhere (inefficient for bounded parameters).             | Use **Beta (for [0,1]), Poisson (for counts), or Categorical (for discrete choices)**.         |
| **No Correlations**          | Independent Gaussians cannot model parameter dependencies.                                    | Use a **full covariance Gaussian** or **GMM**.                                                  |
| **High-Dimensional Cost**    | Covariance matrices are expensive to store/invert in high dimensions.                      | Use **diagonal approximations, K-FAC, or adaptive methods**.                                   |

---

### **Proposed Distributions and Use Cases**

| **Distribution**          | **When to Use**                                                                                     | **Example Use Case**                          | **Fisher Matrix Complexity** |
|---------------------------|---------------------------------------------------------------------------------------------------|-----------------------------------------------|--------------------------------|
| **Gaussian**              | Simple problems, unimodal objectives, or as a starting point.                                   | General-purpose optimization.                 | Low (analytical).              |
| **Gaussian Mixture (GMM)** | Multi-modal objectives (e.g., multiple good solutions).                                      | Neural Architecture Search (NAS), RL.         | High (approximate numerically). |
| **Log-Normal**            | Positive parameters (e.g., learning rates, variances).                                         | Hyperparameter tuning.                        | Medium (analytical).           |
| **Beta**                  | Bounded parameters (e.g., dropout rates in [0, 1]).                                              | Hyperparameter tuning.                        | Medium (analytical).           |
| **Student’s t**           | Heavy-tailed exploration (e.g., early stages of RL or black-box optimization).                 | Reinforcement Learning (RL).                  | High (approximate numerically). |
| **Categorical**           | Discrete parameters (e.g., optimizer choice, number of layers).                                | Neural Architecture Search (NAS).            | Low (analytical).               |
| **Poisson**               | Count data (e.g., number of layers in a neural network).                                       | Neural Architecture Search (NAS).            | Medium (analytical).           |
| **Gamma**                 | Positive, skewed parameters (e.g., time constants).                                             | Physics simulations.                          | Medium (analytical).           |
| **Normalizing Flows**     | Arbitrary, complex distributions.                                                                | Black-box optimization.                       | Very High (numerical).          |
| **Kernel Density Estimation (KDE)** | Non-parametric, data-driven distributions.                                                      | Black-box optimization.                       | Very High (numerical).          |

---

### **Implementation Roadmap**

#### **Phase 1: Basic Extensions (Low Effort, High Impact)**
1. **Add Support for Common Distributions**:
   - **Log-Normal**: For positive parameters (e.g., learning rates, variances).
   - **Beta**: For bounded parameters (e.g., dropout rates in [0, 1]).
   - **Categorical**: For discrete parameters (e.g., optimizer choice, number of layers).
   - **Student’s t**: For heavy-tailed exploration.

2. **Implementation Steps**:
   - Define a **base class** for distributions with methods for:
     - Sampling (`sample()`).
     - PDF evaluation (`pdf()`).
     - Fisher matrix computation (`fisher_matrix()`).
     - Parameter updates (`update()`).
   - Use **automatic differentiation** (e.g., PyTorch, JAX) to compute Fisher matrices for non-Gaussian distributions.

3. **Example Code Structure**:
   ```python
   from abc import ABC, abstractmethod
   import numpy as np
   
   class BaseDistribution(ABC):
       @abstractmethod
       def sample(self, n_samples=1):
           pass
       
       @abstractmethod
       def pdf(self, x):
           pass
       
       @abstractmethod
       def fisher_matrix(self, n_samples=1000):
           pass
       
       @abstractmethod
       def update(self, natural_grad, gamma, F_inv):
           pass
   
   class Gaussian(BaseDistribution):
       def __init__(self, mu, sigma):
           self.mu = mu
           self.sigma = sigma
       
       def sample(self, n_samples=1):
           return np.random.normal(self.mu, self.sigma, n_samples)
       
       def pdf(self, x):
           return (1 / (self.sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - self.mu) / self.sigma) ** 2)
       
       def fisher_matrix(self, n_samples=1000):
           # Analytical Fisher matrix for Gaussian: [[1/sigma², 0], [0, 2/sigma⁴]]
           return np.array([[1 / self.sigma**2, 0], [0, 2 / self.sigma**4]])
       
       def update(self, natural_grad, gamma, F_inv):
           grad = F_inv @ natural_grad
           self.mu += gamma * grad[0]
           self.sigma += gamma * grad[1]
   
   class LogNormal(BaseDistribution):
       def __init__(self, mu, sigma):
           self.mu = mu
           self.sigma = sigma
       
       def sample(self, n_samples=1):
           return np.random.lognormal(mean=self.mu, sigma=self.sigma, size=n_samples)
       
       def pdf(self, x):
           from scipy.stats import lognorm
           return lognorm.pdf(x, s=self.sigma, scale=np.exp(self.mu))
       
       def fisher_matrix(self, n_samples=1000):
           # Approximate Fisher matrix via sampling (Monte Carlo)
           theta_samples = self.sample(n_samples)
           log_theta = np.log(theta_samples)
           grad_mu = (log_theta - self.mu) / self.sigma**2
           grad_sigma = -1/self.sigma + (log_theta - self.mu)**2 / self.sigma**3
           F = np.zeros((2, 2))
           for g_mu, g_sigma in zip(grad_mu, grad_sigma):
               F += np.outer([g_mu, g_sigma], [g_mu, g_sigma])
           return F / n_samples
       
       def update(self, natural_grad, gamma, F_inv):
           grad = F_inv @ natural_grad
           self.mu += gamma * grad[0]
           self.sigma += gamma * grad[1]
   
   class Categorical(BaseDistribution):
       def __init__(self, probs):
           self.probs = np.array(probs)
       
       def sample(self, n_samples=1):
           return np.random.choice(len(self.probs), size=n_samples, p=self.probs)
       
       def pdf(self, x):
           return self.probs[x]
       
       def fisher_matrix(self, n_samples=1000):
           # Fisher matrix for Categorical: F_ij = -p_i p_j for i != j, F_ii = p_i (1 - p_i)
           F = -np.outer(self.probs, self.probs)
           np.fill_diagonal(F, self.probs * (1 - self.probs))
           return F
       
       def update(self, natural_grad, gamma, F_inv):
           grad = F_inv @ natural_grad
           self.probs += gamma * grad
           self.probs = np.exp(self.probs) / np.sum(np.exp(self.probs))  # Softmax normalization
   ```

---

#### **Phase 2: Advanced Extensions (Medium Effort, High Impact)**
1. **Full Covariance Gaussian**:
   - Replace independent Gaussians with a **single Gaussian with full covariance matrix** (`Σ`).
   - **Use case**: Problems where parameters are **correlated** (e.g., learning rate and batch size).

2. **Gaussian Mixture Model (GMM)**:
   - Use a **mixture of Gaussians** to model multi-modal distributions.
   - **Use case**: Problems with **multiple good solutions** (e.g., NAS, RL).

3. **Implementation Steps**:
   - Use libraries like `scikit-learn` or `PyTorch` for GMMs.
   - Approximate the Fisher matrix for GMMs numerically or use diagonal approximations.

4. **Example Code for GMM**:
   ```python
   from sklearn.mixture import GaussianMixture
   
   class GMM_EBGA:
       def __init__(self, n_components=2, n_params=2):
           self.gmm = GaussianMixture(n_components=n_components, covariance_type='full')
           self.n_params = n_params
           # Initialize with random samples
           self.gmm.fit(np.random.randn(100, n_params))
       
       def sample(self, n_samples=1):
           samples, _ = self.gmm.sample(n_samples)
           return samples[0] if n_samples == 1 else samples
       
       def update(self, fitness_samples, gamma=0.1):
           thetas = np.array([sample for sample, _ in fitness_samples])
           fitness = np.array([fitness for _, fitness in fitness_samples])
           # Refit the GMM to the samples, weighted by fitness
           self.gmm.fit(thetas, sample_weight=fitness)
   ```

---

#### **Phase 3: Adaptive and Non-Parametric Extensions (High Effort, High Impact)**
1. **Adaptive Distribution Switching**:
   - Start with a **simple distribution** (e.g., Gaussian).
   - Dynamically switch to a **more complex distribution** (e.g., GMM, normalizing flow) if the problem requires it.
   - **Use case**: Problems where the **optimal distribution changes during optimization**.

2. **Normalizing Flows**:
   - Use **normalizing flows** to transform a simple distribution (e.g., Gaussian) into a **complex, arbitrary distribution**.
   - **Use case**: Problems with **highly non-Gaussian or multi-modal** distributions.

3. **Kernel Density Estimation (KDE)**:
   - Non-parametrically estimate the distribution from samples.
   - **Use case**: Problems where the **true distribution is unknown and complex**.

4. **Example Code for Adaptive EBGA**:
   ```python
   class AdaptiveEBGA:
       def __init__(self, n_params):
           self.mode = 'gaussian'  # 'gaussian', 'gmm', 'flow'
           self.gaussian = Gaussian(mu=np.zeros(n_params), sigma=np.ones(n_params))
           self.gmm = GMM_EBGA(n_components=2, n_params=n_params)
           # Initialize normalizing flow (pseudocode)
           self.flow = NormalizingFlow(n_params)
       
       def update(self, fitness_samples):
           if self.mode == 'gaussian':
               # Check if the landscape is multi-modal
               thetas = np.array([s for s, _ in fitness_samples])
               if self.is_multimodal(thetas):
                   self.mode = 'gmm'
                   self.gmm.gmm.fit(thetas)
               else:
                   # Update Gaussian
                   natural_grad = self.compute_natural_gradient(fitness_samples)
                   F = self.gaussian.fisher_matrix()
                   self.gaussian.update(natural_grad, gamma=0.1, F_inv=np.linalg.inv(F))
           
           elif self.mode == 'gmm':
               self.gmm.update(fitness_samples)
               # Check if GMM is sufficient or if we need a flow
               if self.needs_flow(fitness_samples):
                   self.mode = 'flow'
                   self.flow.fit(thetas)
           
           elif self.mode == 'flow':
               self.flow.update(fitness_samples)
       
       def is_multimodal(self, thetas, threshold=0.5):
           # Simple heuristic: Check if samples are clustered into multiple groups
           from sklearn.cluster import KMeans
           kmeans = KMeans(n_clusters=2).fit(thetas)
           return np.min(kmeans.cluster_centers_) < threshold
       
       def needs_flow(self, fitness_samples):
           # Placeholder for more complex heuristics
           return False
   ```

---

### **Practical Considerations**

#### **1. Computational Cost**
| **Distribution**          | **Sampling Cost** | **Fisher Matrix Cost** | **Update Cost** |
|---------------------------|-------------------|-------------------------|-----------------|
| Gaussian (independent)    | Low               | Low (analytical)        | Low             |
| Full Covariance Gaussian | Medium            | Medium (analytical)     | Medium          |
| GMM                       | Medium            | High (numerical)        | High            |
| Normalizing Flows        | High              | Very High (numerical)   | Very High       |
| KDE                       | High              | Very High (numerical)   | Very High       |

- **Solution**: Use **diagonal approximations** for the Fisher matrix or **K-FAC** for high-dimensional problems.

#### **2. Fisher Matrix Computation**
- For **Gaussians**, the Fisher matrix is **analytical** and easy to compute.
- For **other distributions**, use:
  - **Automatic differentiation** (e.g., PyTorch, JAX).
  - **Monte Carlo approximation** (sample and compute empirically).
  - **Diagonal approximation** (ignore off-diagonal terms).

#### **3. Sampling Efficiency**
- **Independent distributions**: Easy to sample from.
- **GMM**: Requires sampling from a mixture (first pick a component, then sample from it).
- **Normalizing flows**: Sampling is efficient but requires more computation.

#### **4. Convergence**
- **Independent distributions**: May converge slowly if parameters are correlated.
- **Full covariance Gaussian**: Faster convergence for correlated parameters.
- **GMM**: Can converge to better solutions in multi-modal landscapes but may require more samples.

---

### **Recommended Distribution Support Roadmap**

| **Phase** | **Distributions to Add**               | **Use Cases**                                                                                     | **Effort** |
|-----------|----------------------------------------|---------------------------------------------------------------------------------------------------|------------|
| 1         | Log-Normal, Beta, Categorical          | Hyperparameter tuning, discrete parameters.                                                     | Low        |
| 2         | Full Covariance Gaussian, Student’s t  | Correlated parameters, heavy-tailed exploration.                                               | Medium     |
| 3         | GMM, Normalizing Flows                  | Multi-modal problems, arbitrary distributions.                                               | High       |
| 4         | Adaptive Switching                      | Dynamic problems where the optimal distribution changes during optimization.          | High       |

---

### **Example Use Cases for Multi-Distribution EBGA**

| **Use Case**                          | **Recommended Distributions**               | **Why?**                                                                                     |
|---------------------------------------|---------------------------------------------|---------------------------------------------------------------------------------------------|
| **Neural Architecture Search (NAS)** | GMM, Categorical, Log-Normal                 | Multi-modal (multiple good architectures), discrete (number of layers), positive (learning rate). |
| **Reinforcement Learning (RL)**      | Student’s t, Full Covariance Gaussian       | Heavy-tailed exploration (early stages), correlated parameters (policy weights).         |
| **Hyperparameter Tuning**             | Log-Normal, Beta, Categorical               | Positive (learning rate), bounded (dropout rate), discrete (optimizer choice).               |
| **Black-Box Optimization**            | Normalizing Flows, GMM                       | Arbitrary, complex, or multi-modal objective functions.                                       |
| **Bayesian Optimization**             | Full Covariance Gaussian, GMM               | Correlated parameters, multi-modal landscapes.                                               |

---

## 🎯 Summary of Recommendations

### **1. Package Naming**
- **If keeping "EBGA"**: Use **"Evolutionary Bayesian Geometric Adaptation"** to clarify the Bayesian and geometric aspects.
- **If open to a new name**: Use **"Natural Evolutionary Gradient Algorithm" (NEGA)** for clarity and broad appeal.

### **2. Multi-Distribution Support**
- **Start with Phase 1**: Add support for **Log-Normal, Beta, Categorical, and Student’s t** distributions. These cover most common use cases (hyperparameter tuning, RL) with minimal effort.
- **Move to Phase 2**: Implement **full covariance Gaussian and GMM** for problems with correlated parameters or multi-modality.
- **Phase 3 (Optional)**: Add **normalizing flows or adaptive switching** for highly complex or dynamic problems.

### **3. Implementation Tips**
- Use **object-oriented design** (base class for distributions) to make the code modular and extensible.
- Leverage **automatic differentiation** (PyTorch, JAX) to compute Fisher matrices for non-Gaussian distributions.
- Use **approximations** (diagonal Fisher matrix, K-FAC) to reduce computational cost in high dimensions.
- **Monitor performance** and adaptively switch distributions if needed (e.g., from Gaussian to GMM).

---

## 📚 References and Further Reading
- **Natural Gradients**: [Amari, 1998](https://www.jmlr.org/papers/volume10/amari98a/amari98a.pdf) (Natural Gradient Works Efficiently in Learning).
- **Gaussian Mixture Models**: [Bishop, 2006](https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/) (Pattern Recognition and Machine Learning).
- **Normalizing Flows**: [Papamakarios et al., 2019](https://arxiv.org/abs/1912.02762) (Normalizing Flows for Probabilistic Modeling and Inference).
- **Information Geometry**: [Amari & Nagaoka, 2000](https://www.springer.com/gp/book/9784431548590) (Methods of Information Geometry).

---

## 💡 Final Thoughts
- **Naming**: Choose a name that **accurately reflects the method** (e.g., EBGA as "Evolutionary Bayesian Geometric Adaptation" or NEGA as "Natural Evolutionary Gradient Algorithm").
- **Distributions**: Extending EBGA to support **multiple distributions** will **significantly improve its flexibility and performance** on real-world problems. Start with **Phase 1 distributions** (Log-Normal, Beta, Categorical) and gradually add more complex ones as needed.
- **Adaptive Methods**: Consider **dynamically switching distributions** based on the problem’s characteristics (e.g., from Gaussian to GMM for multi-modal landscapes).

This roadmap ensures that your EBGA package is **both theoretically sound and practically powerful**.

---

## 📌 3. Implementation: MultiCandidateOptimizer

### **Overview**
The MultiCandidateOptimizer has been implemented to provide multi-modal distribution support without the overhead of full GMM machinery. It maintains K candidate solutions, each with its own diagonal Gaussian distribution (mean `mu_k` and standard deviation `sigma_k`).

### **Key Design Decisions**

1. **Inheritance Architecture**:
   - Created `BaseEvoOptimizer` class containing shared infrastructure (budget management, parameter clipping, loss scale tracking, callback handling)
   - `CompactEvoOptimizer` and `MultiCandidateOptimizer` both inherit from `BaseEvoOptimizer`
   - This eliminates code duplication while maintaining the same interface

2. **Lightweight Multi-Modality**:
   - Each candidate uses diagonal covariance (O(d) storage per candidate)
   - Total storage: O(K × d) where K is number of candidates (typically 3-5)
   - No full covariance matrices (O(d²)) required

3. **Adaptive Weighting**:
   - Weights based on softmax of negative EMA losses
   - Online adaptation via exponential moving average (EMA)
   - Avoids degeneracy: weights always sum to 1 and stay in (0,1)
   - Parameters: `beta` (temperature), `alpha` (EMA decay rate)

4. **Natural Gradient Updates**:
   - Same update principles as CompactEvoOptimizer
   - Population calibration: natural gradient for each candidate independently
   - Pairwise updates: both candidates updated, winner's candidate gets stronger update

### **Interface Compatibility**
Both optimizers implement the same interface:
- `initialize(initial_params)`
- `step(loss_func, iteration)`
- `get_parameters()`
- `set_parameters(params)`
- `minimize(func, initial_params, max_iter)`
- `state_dict()` / `load_state_dict()`

This means existing models (`EBGARegressor`, `EBGAClassifier`) can use either optimizer transparently.

### **Usage Example**

```python
from EBGA.nn import Sequential
from EBGA.layers import Linear
from EBGA.optimizer import MultiCandidateOptimizer

network = Sequential(
    Linear(64, activation='relu'),
    Linear(1)
)
network.initialize(input_size=10)

# Use multi-candidate optimizer
optimizer = MultiCandidateOptimizer(
    param_dim=network.parameter_count(),
    n_candidates=3,        # Number of search heads
    beta=1.0,              # Weight temperature
    alpha=0.1,             # EMA decay rate
    lr_mu=0.01,
    lr_sigma=0.001
)

# Standard training loop
optimizer.initialize()
for iteration in range(1000):
    loss = optimizer.step(loss_func, iteration=iteration)
```

### **Advantages Over Full GMM**

| Aspect | MultiCandidateOptimizer | Full GMM |
|--------|------------------------|----------|
| Storage | O(K × d) | O(K × d²) |
| Sampling | O(K) per sample | O(K × d) per sample |
| Multi-modality | Yes | Yes |
| Implementation | ~200 lines | Complex |
| Dependencies | None (pure numpy) | sklearn or custom |
| High-dimensional | Excellent | Poor (O(d²) storage) |

### **Performance Characteristics**
- **Escape from local optima**: K candidates explore K different regions simultaneously
- **Adaptive exploration**: Poor-performing candidates get lower sampling probability
- **Efficient in high dimensions**: Linear storage in parameter dimension
- **Transparent to models**: Zero changes required to existing EBGA models

### **Hyperparameter Recommendations**
- `n_candidates`: Start with 3-5 for most problems
- `beta`: 1.0 (higher values concentrate weights more aggressively)
- `alpha`: 0.1 (smoother adaptation with lower values)
- Other parameters: Same as CompactEvoOptimizer