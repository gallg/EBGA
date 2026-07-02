"""
Hyperparameter Search for EBGA Models

This module provides hyperparameter tuning functionality for EBGA models,
supporting both evolutionary and random search strategies. The search classes
are compatible with scikit-learn and can be used in pipelines.

Classes
-------
EvoHyperoptSearch : Main hyperparameter search class
    Combines evolutionary and random search strategies for efficient
    hyperparameter optimization of EBGA models.

ParameterSpace : Parameter space handler
    Manages mixed parameter types and conversions between search space
    and model parameters.
"""

import numpy as np
import warnings

from sklearn.base import BaseEstimator, MetaEstimatorMixin, clone
from sklearn.utils.validation import check_X_y, check_random_state
from sklearn.model_selection import cross_val_score, check_cv
from sklearn.metrics import check_scoring
from sklearn.utils.metaestimators import available_if


class ParameterSpace:
    """
    Helper class for managing hyperparameter search spaces.
    
    Handles mixed parameter types (continuous, discrete, boolean) and provides
    utilities for sampling, decoding, and validating parameter combinations.
    
    Parameters
    ----------
    param_distributions : dict
        Dictionary mapping parameter names to their distributions.
        - Continuous: (min, max) or (min, max, 'log-uniform') or (min, max, 'uniform')
        - Discrete: list of choices
        - Boolean: [True, False]
        - Integer: (min, max) or list of choices
    random_state : int or RandomState, optional
        Random number generator seed for reproducibility.
    """
    
    SUPPORTED_DISTRIBUTIONS = {'uniform', 'log-uniform'}
    
    def __init__(self, param_distributions, random_state=None):
        self.param_distributions = param_distributions
        self.random_state = check_random_state(random_state)
        self._validate_distributions()
        self._param_names = list(param_distributions.keys())
        self._param_types = {}
        self._param_ranges = {}
        
        for name, dist in param_distributions.items():
            self._param_types[name] = self._infer_type(dist)
            self._param_ranges[name] = dist
    
    def _infer_type(self, distribution):
        """Infer parameter type from distribution."""
        if isinstance(distribution, list):
            if len(distribution) == 2 and all(isinstance(x, bool) for x in distribution):
                return 'boolean'
            elif all(isinstance(x, int) for x in distribution):
                return 'integer'
            elif all(isinstance(x, str) for x in distribution):
                return 'categorical'
            else:
                return 'categorical'
        elif isinstance(distribution, tuple):
            if len(distribution) == 2:
                # Check if both are numeric
                if all(isinstance(x, (int, float)) for x in distribution):
                    return 'continuous'
                else:
                    return 'categorical'
            elif len(distribution) == 3:
                if (isinstance(distribution[0], (int, float)) and 
                    isinstance(distribution[1], (int, float)) and
                    isinstance(distribution[2], str)):
                    return 'continuous'
                else:
                    return 'categorical'
        else:
            raise ValueError(f"Unsupported distribution format: {distribution}")
    
    def _validate_distributions(self):
        """Validate parameter distributions."""
        for name, dist in self.param_distributions.items():
            if isinstance(dist, tuple) and len(dist) >= 3:
                if dist[2] not in self.SUPPORTED_DISTRIBUTIONS:
                    raise ValueError(
                        f"Unsupported distribution '{dist[2]}' for parameter '{name}'. "
                        f"Supported: {self.SUPPORTED_DISTRIBUTIONS}"
                    )
    
    @property
    def param_names(self):
        """List of parameter names."""
        return self._param_names
    
    @property
    def n_params(self):
        """Number of parameters."""
        return len(self._param_names)
    
    def sample(self, n_samples=1):
        """
        Sample random parameter combinations.
        
        Parameters
        ----------
        n_samples : int, default=1
            Number of samples to generate.
        
        Returns
        -------
        list of dict
            List of sampled parameter dictionaries.
        """
        if n_samples == 1:
            return [self._sample_single()]
        
        return [self._sample_single() for _ in range(n_samples)]
    
    def _sample_single(self):
        """Sample a single parameter combination."""
        params = {}
        for name in self._param_names:
            params[name] = self._sample_param(name)
        return params
    
    def _sample_param(self, param_name):
        """Sample a single parameter value."""
        dist = self._param_ranges[param_name]
        param_type = self._param_types[param_name]
        
        if param_type == 'continuous':
            return self._sample_continuous(dist)
        elif param_type == 'categorical':
            return self._sample_categorical(dist)
        elif param_type == 'boolean':
            return self._sample_boolean(dist)
        elif param_type == 'integer':
            return self._sample_integer(dist)
        else:
            return self._sample_categorical(dist)
    
    def _sample_continuous(self, dist):
        """Sample from continuous distribution."""
        if len(dist) == 2:
            min_val, max_val = dist
            distribution = 'uniform'
        else:
            min_val, max_val, distribution = dist[:3]
        
        if distribution == 'log-uniform':
            # Sample in log space and exponentiate
            log_min, log_max = np.log(min_val), np.log(max_val)
            value = np.exp(self.random_state.uniform(log_min, log_max))
        else:  # uniform
            value = self.random_state.uniform(min_val, max_val)
        
        return float(value)
    
    def _sample_categorical(self, dist):
        """Sample from categorical distribution."""
        if isinstance(dist, list):
            choices = dist
        elif isinstance(dist, tuple):
            choices = list(dist[:2])  # Handle (min, max) as choices
        else:
            raise ValueError(f"Unsupported categorical distribution: {dist}")
        
        return choices[self.random_state.randint(0, len(choices))]
    
    def _sample_boolean(self, dist):
        """Sample from boolean distribution."""
        return bool(self.random_state.randint(0, 2))
    
    def _sample_integer(self, dist):
        """Sample from integer distribution."""
        if isinstance(dist, list):
            return dist[self.random_state.randint(0, len(dist))]
        elif isinstance(dist, tuple) and len(dist) == 2:
            min_val, max_val = dist
            return int(self.random_state.randint(min_val, max_val + 1))
        else:
            return self._sample_categorical(dist)
    
    def decode_genes(self, genes):
        """
        Decode gene representation to parameter dictionary.
        
        Parameters
        ----------
        genes : list
            List of gene values.
        
        Returns
        -------
        dict
            Parameter dictionary.
        """
        if len(genes) != self.n_params:
            raise ValueError(f"Expected {self.n_params} genes, got {len(genes)}")
        
        params = {}
        for i, name in enumerate(self._param_names):
            gene = genes[i]
            dist = self._param_ranges[name]
            param_type = self._param_types[name]
            
            if param_type == 'continuous':
                params[name] = float(gene)
            elif param_type == 'categorical':
                if isinstance(dist, list):
                    params[name] = dist[int(gene) % len(dist)]
                else:
                    params[name] = gene
            elif param_type == 'boolean':
                params[name] = bool(int(gene) > 0.5)
            elif param_type == 'integer':
                params[name] = int(gene)
            else:
                params[name] = gene
        
        return params
    
    def encode_params(self, params):
        """
        Encode parameter dictionary to gene representation.
        
        Parameters
        ----------
        params : dict
            Parameter dictionary.
        
        Returns
        -------
        list
            List of gene values.
        """
        genes = []
        for name in self._param_names:
            if name not in params:
                # Use default from distribution
                gene = self._sample_param(name)
            else:
                value = params[name]
                dist = self._param_ranges[name]
                param_type = self._param_types[name]
                
                if param_type == 'continuous':
                    gene = float(value)
                elif param_type == 'categorical':
                    if isinstance(dist, list):
                        try:
                            gene = float(dist.index(value))
                        except ValueError:
                            gene = 0.0
                    else:
                        gene = value
                elif param_type == 'boolean':
                    gene = 1.0 if value else 0.0
                elif param_type == 'integer':
                    gene = float(value)
                else:
                    gene = float(value)
            
            genes.append(gene)
        
        return genes


class SearchIndividual:
    """
    Individual in evolutionary search population.
    
    Represents a hyperparameter configuration with its fitness score.
    
    Parameters
    ----------
    genes : list
        List of gene values representing parameter configuration.
    fitness : float, optional
        Cached fitness score (higher is better).
    """
    
    def __init__(self, genes, fitness=None):
        self.genes = list(genes)
        self.fitness = fitness
    
    def __repr__(self):
        return f"SearchIndividual(genes={self.genes[:5]}..., fitness={self.fitness:.4f})"
    
    def copy(self):
        """Create a copy of this individual."""
        return SearchIndividual(self.genes, self.fitness)


class EvoHyperoptSearch(BaseEstimator, MetaEstimatorMixin):
    """
    Evolutionary hyperparameter optimization for EBGA models.
    
    This search class provides efficient hyperparameter tuning using
    evolutionary algorithms, optimized for EBGA's mixed parameter space
    (continuous, discrete, boolean parameters). It's designed to be
    computationally efficient and works seamlessly with scikit-learn.
    
    The search can use different strategies:
    - 'random': Simple random search (baseline)
    - 'evolutionary': Evolutionary search with selection, crossover, mutation
    - 'hybrid': Start with random exploration, refine with evolutionary search
    
    Parameters
    ----------
    estimator : estimator
        The EBGA model or pipeline to optimize.
    param_distributions : dict
        Dictionary with parameters to search and their distributions.
        - Continuous: (min, max) or (min, max, 'log-uniform') or (min, max, 'uniform')
        - Discrete: list of choices
        - Boolean: [True, False]
        - Integer: (min, max) or list of choices
    n_iter : int, default=20
        Number of parameter settings to sample (for random) or population
        size (for evolutionary).
    cv : int or cross-validator, default=3
        Cross-validation strategy.
    search_strategy : str, default='evolutionary'
        Search strategy: 'random', 'evolutionary', or 'hybrid'.
    n_jobs : int, default=None
        Number of parallel jobs for cross-validation.
    scoring : str or callable, default=None
        Scoring metric. If None, uses estimator's default scoring.
    n_generations : int, default=5
        Number of generations for evolutionary search.
    tournament_size : int, default=3
        Tournament size for selection in evolutionary search.
    elitism_count : int, default=1
        Number of best individuals to preserve between generations.
    crossover_rate : float, default=0.8
        Crossover rate for evolutionary search.
    mutation_rate : float, default=0.2
        Mutation rate for evolutionary search.
    mutation_scale : float, default=0.1
        Scale for mutation strength.
    early_stopping_rounds : int, default=3
        Stop evolutionary search if no improvement for this many generations.
    random_state : int, default=None
        Random seed for reproducibility.
    verbose : int, default=0
        Verbosity level (0=silent, 1=progress, 2=debug).
    
    Attributes
    ----------
    best_estimator_ : estimator
        The best estimator found during search.
    best_params_ : dict
        The best parameter set found during search.
    best_score_ : float
        The best score found during search.
    cv_results_ : dict
        Detailed cross-validation results.
    best_index_ : int
        Index of the best parameter set.
    n_splits_ : int
        Number of cross-validation splits.
    """
    
    def __init__(self, estimator, param_distributions, n_iter=20, cv=3,
                 search_strategy='evolutionary', n_jobs=None, scoring=None,
                 n_generations=5, tournament_size=3, elitism_count=1,
                 crossover_rate=0.8, mutation_rate=0.2, mutation_scale=0.1,
                 early_stopping_rounds=3, random_state=None, verbose=0):
        
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.cv = cv
        self.search_strategy = search_strategy
        self.n_jobs = n_jobs
        self.scoring = scoring
        self.n_generations = n_generations
        self.tournament_size = tournament_size
        self.elitism_count = elitism_count
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self.verbose = verbose
    
    def fit(self, X, y, groups=None):
        """
        Run hyperparameter search.
        
        Parameters
        ----------
        X : array-like
            Training data features.
        y : array-like
            Training data targets.
        groups : array-like, default=None
            Group labels for cross-validation.
        
        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y)
        
        # Initialize random state
        random_state = check_random_state(self.random_state)
        
        # Create parameter space
        self.param_space_ = ParameterSpace(
            self.param_distributions, 
            random_state=random_state
        )
        
        # Setup CV
        cv = check_cv(self.cv, y, classifier=hasattr(self.estimator, '_estimator_type') and 
                      self.estimator._estimator_type in ['classifier'])
        self.n_splits_ = cv.get_n_splits()
        
        # Setup scoring
        scoring = check_scoring(self.estimator, self.scoring)
        
        # Store search metadata
        self._search_metadata = {
            'random_state': random_state,
            'cv': cv,
            'scoring': scoring,
            'n_iter': self.n_iter,
            'n_splits': self.n_splits_,
            'strategy': self.search_strategy
        }
        
        # Store X and y for later use in _process_results
        self._fit_X = X
        self._fit_y = y
        
        # Run appropriate search strategy
        if self.search_strategy == 'random':
            self._run_random_search(X, y, cv, scoring, random_state)
        elif self.search_strategy == 'evolutionary':
            self._run_evolutionary_search(X, y, cv, scoring, random_state)
        elif self.search_strategy == 'hybrid':
            self._run_hybrid_search(X, y, cv, scoring, random_state)
        else:
            raise ValueError(f"Unknown search strategy: {self.search_strategy}. "
                           f"Supported: 'random', 'evolutionary', 'hybrid'")
        
        return self
    
    def _run_random_search(self, X, y, cv, scoring, random_state):
        """Run random search."""
        if self.verbose > 0:
            print(f"Running random search with {self.n_iter} iterations...")
        
        # Sample parameter combinations
        param_list = self.param_space_.sample(self.n_iter)
        
        # Evaluate each parameter set
        results = []
        for i, params in enumerate(param_list):
            if self.verbose > 0:
                print(f"  Iteration {i+1}/{self.n_iter}")
            
            score, std = self._evaluate_params(X, y, cv, scoring, params)
            results.append({
                'params': params,
                'mean_score': score,
                'std_score': std
            })
        
        self._process_results(results)
    
    def _run_evolutionary_search(self, X, y, cv, scoring, random_state):
        """Run evolutionary search."""
        if self.verbose > 0:
            print(f"Running evolutionary search with {self.n_iter} population, "
                  f"{self.n_generations} generations...")
        
        # Initialize population
        population = []
        for _ in range(self.n_iter):
            params = self.param_space_.sample(1)[0]
            individual = SearchIndividual(
                self.param_space_.encode_params(params)
            )
            population.append(individual)
        
        # Evolutionary loop
        best_score = -np.inf
        best_params = None
        no_improvement_count = 0
        all_results = []
        
        for generation in range(self.n_generations):
            if self.verbose > 0:
                print(f"  Generation {generation+1}/{self.n_generations}")
            
            # Evaluate population
            generation_results = []
            for individual in population:
                params = self.param_space_.decode_genes(individual.genes)
                score, std = self._evaluate_params(X, y, cv, scoring, params)
                individual.fitness = score
                generation_results.append({
                    'params': params,
                    'mean_score': score,
                    'std_score': std
                })
                
                # Track best
                if score > best_score:
                    best_score = score
                    best_params = params
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
            
            all_results.extend(generation_results)
            
            # Check early stopping
            if no_improvement_count >= self.early_stopping_rounds:
                if self.verbose > 0:
                    print(f"  Early stopping after {generation+1} generations")
                break
            
            # Create next generation
            population = self._evolve_population(population, random_state)
        
        self._process_results(all_results)
    
    def _run_hybrid_search(self, X, y, cv, scoring, random_state):
        """Run hybrid search (random exploration + evolutionary refinement)."""
        if self.verbose > 0:
            print(f"Running hybrid search: random exploration + evolutionary refinement")
        
        # Phase 1: Random exploration
        random_iter = max(5, self.n_iter // 2)  # At least 5 random samples
        if self.verbose > 0:
            print(f"  Phase 1: Random exploration ({random_iter} iterations)")
        
        param_list = self.param_space_.sample(random_iter)
        results = []
        for i, params in enumerate(param_list):
            if self.verbose > 0:
                print(f"    Iteration {i+1}/{random_iter}")
            score, std = self._evaluate_params(X, y, cv, scoring, params)
            results.append({
                'params': params,
                'mean_score': score,
                'std_score': std
            })
        
        # Phase 2: Evolutionary refinement on promising region
        if self.verbose > 0:
            print(f"  Phase 2: Evolutionary refinement")
        
        # Find top 50% of random samples
        sorted_results = sorted(results, key=lambda x: x['mean_score'], reverse=True)
        top_results = sorted_results[:len(sorted_results) // 2]
        
        # Create initial population from top results + new random samples
        population = []
        for result in top_results:
            individual = SearchIndividual(
                self.param_space_.encode_params(result['params'])
            )
            individual.fitness = result['mean_score']
            population.append(individual)
        
        # Fill remaining population with new random samples
        while len(population) < self.n_iter:
            params = self.param_space_.sample(1)[0]
            individual = SearchIndividual(
                self.param_space_.encode_params(params)
            )
            population.append(individual)
        
        # Evolutionary refinement
        best_score = -np.inf
        no_improvement_count = 0
        
        for generation in range(self.n_generations):
            if self.verbose > 0:
                print(f"    Generation {generation+1}/{self.n_generations}")
            
            generation_results = []
            for individual in population:
                if individual.fitness is None:  # Not yet evaluated
                    params = self.param_space_.decode_genes(individual.genes)
                    score, std = self._evaluate_params(X, y, cv, scoring, params)
                    individual.fitness = score
                    generation_results.append({
                        'params': params,
                        'mean_score': score,
                        'std_score': std
                    })
                    
                    if score > best_score:
                        best_score = score
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1
            
            results.extend(generation_results)
            
            if no_improvement_count >= self.early_stopping_rounds:
                if self.verbose > 0:
                    print(f"    Early stopping after {generation+1} generations")
                break
            
            population = self._evolve_population(population, random_state)
        
        self._process_results(results)
    
    def _evolve_population(self, population, random_state):
        """Create next generation through selection, crossover, and mutation."""
        # Sort population by fitness (descending)
        sorted_pop = sorted(population, key=lambda x: x.fitness if x.fitness is not None else -np.inf, 
                          reverse=True)
        
        new_population = []
        
        # Elitism: keep best individuals
        for i in range(self.elitism_count):
            if i < len(sorted_pop):
                new_population.append(sorted_pop[i].copy())
        
        # Fill rest of population with offspring
        while len(new_population) < len(population):
            # Selection
            parent1, parent2 = self._tournament_selection(sorted_pop, random_state)
            
            # Crossover
            if random_state.random() < self.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2, random_state)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # Mutation
            child1 = self._mutate(child1, random_state)
            child2 = self._mutate(child2, random_state)
            
            new_population.extend([child1, child2])
        
        # Trim to population size
        return new_population[:len(population)]
    
    def _tournament_selection(self, population, random_state):
        """Tournament selection."""
        # Ensure tournament size doesn't exceed population size
        actual_tournament_size = min(self.tournament_size, len(population))
        
        tournament1 = random_state.choice(population, size=actual_tournament_size, replace=False)
        tournament2 = random_state.choice(population, size=actual_tournament_size, replace=False)
        
        winner1 = max(tournament1, key=lambda x: x.fitness if x.fitness is not None else -np.inf)
        winner2 = max(tournament2, key=lambda x: x.fitness if x.fitness is not None else -np.inf)
        
        return winner1, winner2
    
    def _crossover(self, parent1, parent2, random_state):
        """Blend crossover (BLX-alpha) for continuous parameters, uniform for others."""
        child1_genes = []
        child2_genes = []
        
        for i, (g1, g2) in enumerate(zip(parent1.genes, parent2.genes)):
            if random_state.random() < 0.5:
                # BLX-alpha crossover
                min_gene = min(g1, g2)
                max_gene = max(g1, g2)
                range_gene = max_gene - min_gene
                alpha = 0.5  # BLX-alpha parameter
                
                lower = min_gene - alpha * range_gene
                upper = max_gene + alpha * range_gene
                
                c1 = random_state.uniform(lower, upper)
                c2 = random_state.uniform(lower, upper)
            else:
                # Uniform crossover
                c1, c2 = g1, g2
            
            child1_genes.append(c1)
            child2_genes.append(c2)
        
        return SearchIndividual(child1_genes), SearchIndividual(child2_genes)
    
    def _mutate(self, individual, random_state):
        """Gaussian mutation with boundary handling."""
        mutated_genes = []
        
        for i, gene in enumerate(individual.genes):
            if random_state.random() < self.mutation_rate:
                # Gaussian mutation
                scale = self.mutation_scale
                mutation = random_state.normal(0, scale)
                mutated_gene = gene + mutation
                mutated_genes.append(mutated_gene)
            else:
                mutated_genes.append(gene)
        
        return SearchIndividual(mutated_genes)
    
    def _evaluate_params(self, X, y, cv, scoring, params):
        """Evaluate a parameter set using cross-validation."""
        try:
            # Clone estimator
            estimator = clone(self.estimator)
            
            # Set parameters
            if hasattr(estimator, 'set_params'):
                estimator.set_params(**params)
            else:
                # Handle nested parameters (for pipelines)
                for param, value in params.items():
                    if '__' in param:
                        # Pipeline parameter
                        step, param_name = param.split('__', 1)
                        if hasattr(estimator.named_steps[step], 'set_params'):
                            estimator.named_steps[step].set_params(**{param_name: value})
                        else:
                            setattr(estimator.named_steps[step], param_name, value)
                    else:
                        setattr(estimator, param, value)
            
            # Cross-validation
            scores = cross_val_score(
                estimator, X, y, cv=cv, scoring=scoring,
                n_jobs=self.n_jobs, error_score='raise'
            )
            
            return np.mean(scores), np.std(scores)
        
        except Exception as e:
            # If evaluation fails, return very low score
            if self.verbose > 1:
                print(f"    Evaluation failed for {params}: {e}")
            return -np.inf, np.inf
    
    def _process_results(self, results):
        """Process search results to find best parameters."""
        # Sort by mean score (descending)
        sorted_results = sorted(results, key=lambda x: x['mean_score'], reverse=True)
        
        # Store best
        self.best_index_ = 0
        self.best_params_ = sorted_results[0]['params']
        self.best_score_ = sorted_results[0]['mean_score']
        
        # Create and fit best estimator
        self.best_estimator_ = clone(self.estimator)
        if hasattr(self.best_estimator_, 'set_params'):
            self.best_estimator_.set_params(**self.best_params_)
        else:
            # Handle pipeline parameters
            for param, value in self.best_params_.items():
                if '__' in param:
                    step, param_name = param.split('__', 1)
                    if hasattr(self.best_estimator_.named_steps[step], 'set_params'):
                        self.best_estimator_.named_steps[step].set_params(**{param_name: value})
                    else:
                        setattr(self.best_estimator_.named_steps[step], param_name, value)
                else:
                    setattr(self.best_estimator_, param, value)
        
        # Fit best estimator on full data
        self.best_estimator_.fit(self._fit_X, self._fit_y)
        
        # Store CV results
        self.cv_results_ = {
            'params': [r['params'] for r in results],
            'mean_scores': [r['mean_score'] for r in results],
            'std_scores': [r['std_score'] for r in results]
        }
        
        if self.verbose > 0:
            print(f"  Best score: {self.best_score_:.4f}")
            print(f"  Best params: {self.best_params_}")
    
    @available_if(lambda self: hasattr(self, 'best_estimator_'))
    def predict(self, X):
        """Predict using the best estimator found."""
        return self.best_estimator_.predict(X)
    
    @available_if(lambda self: hasattr(self, 'best_estimator_'))
    def predict_proba(self, X):
        """Predict probabilities using the best estimator found."""
        if hasattr(self.best_estimator_, 'predict_proba'):
            return self.best_estimator_.predict_proba(X)
        else:
            raise AttributeError("Best estimator does not have predict_proba method")
    
    @available_if(lambda self: hasattr(self, 'best_estimator_'))
    def score(self, X, y):
        """Score using the best estimator found."""
        return self.best_estimator_.score(X, y)
    
    @available_if(lambda self: hasattr(self, 'best_estimator_'))
    def transform(self, X):
        """Transform using the best estimator found (for pipelines)."""
        if hasattr(self.best_estimator_, 'transform'):
            return self.best_estimator_.transform(X)
        else:
            return X
    
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            'estimator': self.estimator,
            'param_distributions': self.param_distributions,
            'n_iter': self.n_iter,
            'cv': self.cv,
            'search_strategy': self.search_strategy,
            'n_jobs': self.n_jobs,
            'scoring': self.scoring,
            'n_generations': self.n_generations,
            'tournament_size': self.tournament_size,
            'elitism_count': self.elitism_count,
            'crossover_rate': self.crossover_rate,
            'mutation_rate': self.mutation_rate,
            'mutation_scale': self.mutation_scale,
            'early_stopping_rounds': self.early_stopping_rounds,
            'random_state': self.random_state,
            'verbose': self.verbose
        }
    
    def set_params(self, **params):
        """Set parameters for this estimator."""
        for param, value in params.items():
            if param in self.get_params():
                setattr(self, param, value)
            else:
                warnings.warn(f"Unknown parameter: {param}")
        return self
