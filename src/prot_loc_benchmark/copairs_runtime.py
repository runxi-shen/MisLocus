"""Configurable copairs execution and independent per-call null simulations."""
from contextlib import contextmanager
from tempfile import TemporaryDirectory

from copairs import compute, map as copairs_map
from copairs.map import multilabel
from threadpoolctl import threadpool_limits


def positive_threads(value):
    """Argparse-compatible positive integer, also checked at the API boundary."""
    if isinstance(value, bool) or str(value) != str(int(value)) or int(value) < 1:
        raise ValueError("thread counts must be positive integers")
    return int(value)


@contextmanager
def bounded_copairs(workers=16, blas_threads=1):
    workers = positive_threads(16 if workers is None else workers)
    blas_threads = positive_threads(blas_threads)
    original = compute.ThreadPool
    # ponytail: process-wide hook; run concurrent scoring jobs in separate processes.
    compute.ThreadPool = lambda processes=None, *args, **kwargs: original(
        min(workers, processes or workers), *args, **kwargs
    )
    try:
        with threadpool_limits(limits=blas_threads, user_api="blas"):
            yield workers
    finally:
        compute.ThreadPool = original


def average_precision(*args, max_workers=16, blas_threads=1, **kwargs):
    with bounded_copairs(max_workers, blas_threads):
        return copairs_map.average_precision(*args, **kwargs)


def average_precision_multilabel(*args, max_workers=16, blas_threads=1, **kwargs):
    with bounded_copairs(max_workers, blas_threads):
        return multilabel.average_precision(*args, **kwargs)


def mean_average_precision(*args, max_workers=16, blas_threads=1, **kwargs):
    # A shared cache can reuse a different configuration's derived random seed.
    with bounded_copairs(max_workers, blas_threads) as workers, TemporaryDirectory(prefix="copairs-") as cache:
        return copairs_map.mean_average_precision(*args, max_workers=workers, cache_dir=cache, **kwargs)
