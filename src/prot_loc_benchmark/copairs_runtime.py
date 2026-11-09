"""Independent per-call copairs null simulations."""
from tempfile import TemporaryDirectory
from copairs import map as copairs_map

def mean_average_precision(*args, **kwargs):
    with TemporaryDirectory(prefix='copairs-') as cache:
        return copairs_map.mean_average_precision(*args, cache_dir=cache, **kwargs)
