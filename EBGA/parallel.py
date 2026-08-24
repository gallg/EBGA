import multiprocessing as mp
import os

import numpy as np

from EBGA.nn import Sequential
from EBGA.layers import Dense, Flatten
from EBGA.activations import _activation_class_to_name
from EBGA.losses import _loss_class_to_name, get_loss


# Minimum rows per worker. Below this, data-parallel sharding costs more in
# IPC overhead than it saves in compute, so the worker count is capped (down
# to 1, i.e. in-process sequential evaluation). This is what keeps small
# networks / small datasets from being slower under n_jobs > 1.
_MIN_SHARD = 256

# BLAS backends are pinned to a single thread inside worker processes so that
# process-level parallelism does not oversubscribe cores. These are set in the
# parent right before spawning (spawn children inherit the environment) and
# therefore apply to workers without capping BLAS threads in the parent, whose
# numpy is already imported by then.
_BLAS_THREAD_VARS = (
    'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS',
    'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
)

_WORKER = None


class _WorkerState:
    """Per-worker state: network, full dataset, and loss."""

    def __init__(self, network_template, X, y, loss_name, seed):
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
        self.output_size = self.net.output_size

        self.loss = get_loss(loss_name) if isinstance(loss_name, str) else loss_name

    def evaluate(self, candidates, X, y):
        """
        Evaluate every candidate on the given data slice.

        Returns ``(sums, n)`` where ``sums[i] = loss(candidate_i) * n`` and
        ``n = X.shape[0]``. The caller reduces across slices by summing
        ``sums`` and dividing by the total row count, which reproduces the
        exact mean loss over the union of slices (all EBGA losses are sample
        means).
        """
        n = X.shape[0]
        if n == 0:
            return np.zeros(len(candidates)), 0

        sums = np.empty(len(candidates))
        current = self.net.get_all_parameters()
        try:
            for i, params in enumerate(candidates):
                # Short-circuit exploded candidates before the forward pass.
                if np.any(np.abs(params) > 1e5):
                    sums[i] = np.inf
                    continue
                self.net.set_all_parameters(params)
                y_pred = self.net.forward(X)
                if self.output_size == 1:
                    y_pred = y_pred.flatten()
                sums[i] = float(self.loss(y_pred, y)) * n
        finally:
            self.net.set_all_parameters(current)
        return sums, n

    def evaluate_shard(self, candidates, shard_idx, n_shards):
        """Evaluate candidates on a contiguous shard of the full dataset."""
        n = self.X.shape[0]
        base = n // n_shards
        rem = n % n_shards
        if shard_idx < rem:
            start = shard_idx * (base + 1)
            end = start + base + 1
        else:
            start = rem * (base + 1) + (shard_idx - rem) * base
            end = start + base
        return self.evaluate(candidates, self.X[start:end], self.y[start:end])


def _init_worker(network_template, X, y, loss_name, seed):
    """Called once per worker via Pool(initializer=...)."""
    global _WORKER
    _WORKER = _WorkerState(network_template, X, y, loss_name, seed)


def _worker_eval_shard(args):
    """Evaluate all candidates on this worker's shard of the full dataset."""
    candidates, shard_idx, n_shards = args
    return _WORKER.evaluate_shard(candidates, shard_idx, n_shards)


def _worker_eval_chunk(args):
    """Evaluate all candidates on an explicit data chunk (mini-batch shard)."""
    candidates, Xc, yc = args
    return _WORKER.evaluate(candidates, Xc, yc)


def _split_chunks(X, y, k):
    """Split (X, y) into ``k`` contiguous non-overlapping chunks."""
    n = X.shape[0]
    base = n // k
    rem = n % k
    chunks = []
    start = 0
    for i in range(k):
        size = base + (1 if i < rem else 0)
        end = start + size
        chunks.append((X[start:end], y[start:end]))
        start = end
    return chunks


class ParallelEvaluator:
    """
    Data-parallel evaluator for candidate-based evolutionary optimization.

    Each step, the dataset (or the current mini-batch) is sharded across
    workers. Every worker evaluates **all** candidates on its shard and
    returns ``mean_loss * n_rows``; the parent reduces these to the exact
    mean loss over the full data. Because every candidate is scored on the
    same data, losses are directly comparable across candidates.

    Scaling is in the **data dimension**, so speedup is not capped by the
    population size (``calibration_size``): more workers keep helping as long
    as there is data to shard. The number of workers actually used is capped
    to ``min(n_jobs, max(1, step_size // 256))`` -- when the data is small
    the evaluator falls back to in-process sequential evaluation (zero IPC),
    so small networks and small datasets are not slowed by parallel
    overhead.

    Memory: each worker holds a copy of the full dataset, so peak memory
    scales with the effective worker count. For truly out-of-core datasets
    use ``batch_size`` so each step scores a mini-batch instead of the full
    data.

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
        Loss instance. If a string, resolved via ``get_loss``. Must be a
        per-sample mean (all EBGA losses are).
    n_jobs : int, default=1
        Number of worker processes. 1 (or any value that yields a single
        effective worker) disables parallelism and runs in-process.
    batch_size : int, optional
        If set, each step scores a mini-batch of this size instead of the
        full dataset. The mini-batch advances once per step, shared by all
        candidates. The mini-batch is itself sharded across workers.
    random_state : int, optional
        Seed for mini-batch shuffling.

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
        self.n_jobs = max(1, int(n_jobs))
        self.batch_size = batch_size
        self._pool = None
        self._network_template = _serialize_network(network)

        if isinstance(loss, str):
            self._loss_name = loss
        else:
            self._loss_name = _loss_class_to_name(type(loss))

        self._X = np.asarray(X)
        self._y = np.asarray(y)
        self._seed = random_state

        # Cap the worker count to what the data can usefully shard. Below one
        # full shard per worker we fall back to in-process evaluation.
        step_size = self._X.shape[0] if batch_size is None else batch_size
        self._n_eff = min(self.n_jobs, max(1, step_size // _MIN_SHARD))

        # Mini-batch state, owned by the parent so the schedule is identical
        # regardless of how many workers are used.
        self._batch_rng = np.random.RandomState(self._seed)
        self._batch_order = None
        self._batch_pos = 0

        # In-process state used when running sequentially (n_eff == 1).
        self._seq_state = None

        if self._n_eff <= 1:
            self._evaluate_map = self._sequential_map
        else:
            self._evaluate_map = self._parallel_map

    def _ensure_seq_state(self):
        if self._seq_state is None:
            self._seq_state = _WorkerState(
                self._network_template, self._X, self._y,
                self._loss_name, self._seed or 0,
            )
        return self._seq_state

    def _next_batch(self):
        """Draw the mini-batch for the current step (shared by all candidates)."""
        n = self._X.shape[0]
        bs = self.batch_size
        if self._batch_order is None or self._batch_pos + bs > n:
            self._batch_order = self._batch_rng.permutation(n)
            self._batch_pos = 0
        idx = self._batch_order[self._batch_pos:self._batch_pos + bs]
        self._batch_pos += bs
        return self._X[idx], self._y[idx]

    @property
    def evaluate_map(self):
        """
        Batch evaluator for ``optimizer.step(evaluate_map=...)``.

        Signature: ``evaluate_map(candidates) -> np.ndarray`` of loss values.
        When the effective worker count is 1, runs in-process (no IPC).
        """
        return self._evaluate_map

    def _sequential_map(self, candidates):
        """Evaluate all candidates in-process (no worker pool)."""
        state = self._ensure_seq_state()
        if self.batch_size is None:
            sums, n = state.evaluate(candidates, self._X, self._y)
        else:
            Xb, yb = self._next_batch()
            sums, n = state.evaluate(candidates, Xb, yb)
        return sums / n

    def _parallel_map(self, candidates):
        """Evaluate all candidates by sharding the data across workers."""
        if self._pool is None:
            self._init_pool()

        if self.batch_size is None:
            k = self._n_eff
            tasks = [(candidates, i, k) for i in range(k)]
            results = self._pool.map(_worker_eval_shard, tasks)
        else:
            Xb, yb = self._next_batch()
            k = self._n_eff
            chunks = _split_chunks(Xb, yb, k)
            tasks = [(candidates, Xc, yc) for Xc, yc in chunks]
            results = self._pool.map(_worker_eval_chunk, tasks)

        sums = np.zeros(len(candidates))
        total_n = 0
        for s, n in results:
            sums += s
            total_n += n
        return sums / total_n

    def _init_pool(self):
        """Create the worker pool using a private 'spawn' context."""
        for _var in _BLAS_THREAD_VARS:
            os.environ.setdefault(_var, '1')
        ctx = mp.get_context('spawn')
        self._pool = ctx.Pool(
            processes=self._n_eff,
            initializer=_init_worker,
            initargs=(
                self._network_template,
                self._X,
                self._y,
                self._loss_name,
                self._seed or 0,
            ),
        )

    def __enter__(self):
        if self._n_eff > 1 and self._pool is None:
            self._init_pool()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Shut down the worker pool."""
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None


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
