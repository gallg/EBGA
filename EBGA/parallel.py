import multiprocessing as mp
import os

# Use 'spawn' start method so worker processes are fresh Python processes
# that re-import this module. This ensures the BLAS thread-pinning below
# runs before numpy is imported in each worker.
mp.set_start_method('spawn', force=True)

# Pin BLAS to single thread per process. Process-level parallelism via
# multiprocessing means BLAS threading would cause oversubscription.
for _var in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS',
             'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_var, '1')

import numpy as np

from EBGA.nn import Sequential
from EBGA.layers import Dense, Flatten
from EBGA.activations import _activation_class_to_name
from EBGA.losses import _loss_class_to_name, get_loss


_WORKER = None


class _WorkerState:
    """Per-worker state: network, data, loss, and batching."""

    def __init__(self, network_template, X, y, loss_name, batch_size, seed):
        layers = []
        for info in network_template:
            ltype = info['type']
            if ltype == 'Dense':
                act = info.get('activation')
                layer = Dense(
                    output_size=info['output_size'],
                    activation=act if act else None,
                    use_bias=info.get('use_bias', True),
                )
            elif ltype == 'Flatten':
                layer = Flatten()
            else:
                raise ValueError(f"Unknown layer type: {ltype}")
            layers.append(layer)

        self.net = Sequential(*layers)
        self.net.initialize(X.shape[1])
        self.X = X
        self.y = y

        self.loss = get_loss(loss_name) if isinstance(loss_name, str) else loss_name

        self.batch_size = batch_size
        self.rng = np.random.RandomState(seed)

        if batch_size is not None:
            self._make_batches()
        else:
            self.batches = None
            self.batch_index = None

    def _make_batches(self):
        n = self.X.shape[0]
        indices = np.arange(n)
        self.rng.shuffle(indices)
        self.batches = [
            (self.X[indices[i:i + self.batch_size]],
             self.y[indices[i:i + self.batch_size]])
            for i in range(0, n, self.batch_size)
        ]
        self.batch_index = [0]

    def evaluate(self, params):
        """Evaluate a single candidate on this worker's data (or current batch)."""
        if self.batches is not None:
            idx = self.batch_index[0] % len(self.batches)
            X_b, y_b = self.batches[idx]
            self.batch_index[0] += 1
        else:
            X_b, y_b = self.X, self.y

        current = self.net.get_all_parameters()
        self.net.set_all_parameters(params)
        y_pred = self.net.forward(X_b)
        if self.net.output_size == 1:
            y_pred = y_pred.flatten()
        loss = self.loss(y_pred, y_b)
        self.net.set_all_parameters(current)

        if np.any(np.abs(params) > 1e5):
            return float('inf')
        return loss


def _init_worker(network_template, X, y, loss_name, batch_size, seed):
    """Called once per worker via Pool(initializer=...)."""
    global _WORKER
    _WORKER = _WorkerState(network_template, X, y, loss_name, batch_size, seed)


def _worker_evaluate(params):
    """Evaluate one candidate on this worker's data."""
    global _WORKER
    return _WORKER.evaluate(params)


def _worker_evaluate_batch(candidates_batch):
    """Evaluate a batch of candidates sequentially on this worker."""
    return [_worker_evaluate(p) for p in candidates_batch]


class ParallelEvaluator:
    """
    Parallel evaluator for candidate-based evolutionary optimization.

    Each worker holds a full copy of the network and dataset. Candidates
    are distributed across workers via ``pool.map`` and evaluated in
    parallel. All candidates within a step are evaluated on the same data
    (full dataset or same batch), so loss values are directly comparable.

    Provides near-linear speedup in the candidate dimension:
    ``calibration_size`` candidates on ``n_jobs`` workers run
    ~``min(n_jobs, calibration_size)`` times faster.

    For out-of-memory datasets, set ``batch_size`` so each step uses a
    mini-batch instead of the full dataset. All workers share the same
    batch index, so all candidates within a step see the same batch.

    Parameters
    ----------
    network : Sequential
        Network architecture used as a template to reconstruct identical
        networks on each worker.
    X : ndarray of shape (n_samples, n_features)
        Input features.
    y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
        Target values.
    loss : str or Loss instance
        Loss function name (e.g. 'mse', 'mae', 'cross_entropy') or a
        Loss instance. If a string, resolved via ``get_loss``.
    n_jobs : int, default=1
        Number of worker processes. 1 disables parallelism.
    batch_size : int, optional
        If set, each step uses a mini-batch of this size instead of the
        full dataset. All candidates within a step are evaluated on the
        same batch. The batch index advances each step.
    random_state : int, optional
        Seed for worker-local random state (used for batch shuffling).

    Examples
    --------
    >>> from EBGA.nn import Sequential
    >>> from EBGA.layers import Dense
    >>> from EBGA.optimizer import CompactEvoOptimizer
    >>> from EBGA.parallel import ParallelEvaluator
    >>>
    >>> net = Sequential(Dense(10, 'relu'), Dense(1, 'linear'))
    >>> net.initialize(X.shape[1])
    >>>
    >>> evaluator = ParallelEvaluator(net, X, y, loss='mse', n_jobs=4)
    >>> opt = CompactEvoOptimizer(param_dim=net.parameter_count())
    >>> opt.initialize(net.get_all_parameters())
    >>>
    >>> with evaluator:
    ...     for i in range(1000):
    ...         opt.step(None, evaluate_map=evaluator.evaluate_map)
    >>>
    >>> net.set_all_parameters(opt.get_parameters())
    """

    def __init__(self, network, X, y, loss, n_jobs=1, batch_size=None,
                 random_state=None):
        self.n_jobs = n_jobs
        self.batch_size = batch_size
        self._pool = None
        self._evaluate_map = None

        self._network_template = _serialize_network(network)

        if isinstance(loss, str):
            self._loss_name = loss
        else:
            self._loss_name = _loss_class_to_name(type(loss))

        self._X = X
        self._y = y
        self._seed = random_state

        if n_jobs <= 1:
            self._evaluate_map = self._sequential_map
        else:
            self._init_pool()

    def _init_pool(self):
        """Create the worker pool."""
        rng = np.random.RandomState(self._seed)
        self._pool = mp.Pool(
            processes=self.n_jobs,
            initializer=_init_worker,
            initargs=(
                self._network_template,
                self._X,
                self._y,
                self._loss_name,
                self.batch_size,
                rng.randint(0, 2**31),
            ),
        )

    @property
    def evaluate_map(self):
        """
        Batch evaluator for ``optimizer.step(evaluate_map=...)``.

        Signature: ``evaluate_map(candidates) -> np.ndarray``

        Evaluates all candidates in parallel across workers using
        ``pool.map``. When ``n_jobs=1``, falls back to sequential evaluation.
        """
        return self._evaluate_map

    def _sequential_map(self, candidates):
        """Evaluate all candidates sequentially (n_jobs=1)."""
        state = _WorkerState(
            self._network_template, self._X, self._y,
            self._loss_name, self.batch_size, self._seed or 0,
        )
        return np.array([state.evaluate(p) for p in candidates])

    def _parallel_map(self, candidates):
        """Evaluate all candidates in parallel across workers."""
        # Split candidates into exactly n_jobs chunks to minimize IPC overhead.
        # Each worker evaluates its chunk sequentially.
        n = len(candidates)
        k = min(self.n_jobs, n)
        chunk_size = (n + k - 1) // k
        chunks = [candidates[i:i + chunk_size] for i in range(0, n, chunk_size)]
        results = self._pool.map(_worker_evaluate_batch, chunks)
        return np.concatenate(results)

    def __enter__(self):
        if self.n_jobs > 1 and self._pool is None:
            self._init_pool()
        if self.n_jobs > 1:
            self._evaluate_map = self._parallel_map
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Shut down the worker pool."""
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None
        self._evaluate_map = self._sequential_map


def _serialize_network(network):
    """Convert a Sequential network into a picklable template dict."""
    template = []
    for layer in network.layers:
        info = {'type': type(layer).__name__}
        if hasattr(layer, 'output_size'):
            info['output_size'] = layer.output_size
        if hasattr(layer, 'use_bias'):
            info['use_bias'] = layer.use_bias
        if hasattr(layer, 'activation'):
            act = layer.activation
            if act is None:
                info['activation'] = None
            elif isinstance(act, str):
                info['activation'] = act
            else:
                info['activation'] = _activation_class_to_name(type(act))
        template.append(info)
    return template
