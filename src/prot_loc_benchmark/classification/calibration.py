"""Small, content-bound NC+PC calibration artifacts; no historical stage required."""
from importlib.metadata import version
import os
from pathlib import Path
import platform
import subprocess

import polars as pl
import xgboost

from prot_loc_benchmark.config import BATCH_CONTROLS, MAX_IMBALANCE_RATIO, MIN_CELL_COUNT, NULL_PERCENTILE, REPO_ROOT
from prot_loc_benchmark.provenance import read_json_with_hash, save_json, sha256
from .metrics import compute_null_threshold, validate_thresholds


def calibration_context(features, representation, batch, channels, protocol, params, device):
    sources = list(Path(__file__).parent.glob('*.py')) + [REPO_ROOT / 'src/prot_loc_benchmark/config.py', REPO_ROOT / 'scripts/09_classify.py']
    gpu = None
    if device != 'cpu':
        gpu = subprocess.check_output(['nvidia-smi', '--id=' + os.environ['CUDA_VISIBLE_DEVICES'],
            '--query-gpu=uuid,name,driver_version', '--format=csv,noheader'], text=True, timeout=10).strip()
    return dict(features_sha256=sha256(features), representation=representation, batch=batch,
        channel_features=channels, protocol=protocol, xgb_params=params, device=device, gpu=gpu,
        control_alleles=BATCH_CONTROLS[batch], min_cells=MIN_CELL_COUNT, max_imbalance=MAX_IMBALANCE_RATIO,
        percentile=NULL_PERCENTILE, interpolation='nearest', xgboost_build=xgboost.build_info(),
        runtime={p: version(p) for p in ('numpy', 'polars', 'xgboost', 'scikit-learn')},
        python=platform.python_version(), architecture=platform.machine(),
        code={p.name: sha256(p) for p in sources})


def save_calibration(directory: Path, context: dict) -> dict[str, float]:
    path = directory / 'calibration.json'
    if path.exists():
        raise FileExistsError(path)
    metrics = pl.read_csv(directory / 'metrics.csv')
    if not metrics['category'].is_in(['NC', 'PC']).fill_null(False).all():
        raise ValueError('Calibration must contain only NC+PC classifiers')
    thresholds = compute_null_threshold(metrics)
    validate_thresholds(thresholds, list(context['channel_features']))
    save_json(path, dict(schema_version=1, status='complete', context=context, thresholds=thresholds,
        outputs={p.name: sha256(p) for p in directory.iterdir() if p.is_file()}))
    return thresholds


def load_calibration(directory: Path, context: dict) -> tuple[dict[str, float], pl.DataFrame]:
    receipt, _ = read_json_with_hash(directory / 'calibration.json')
    if receipt.get('schema_version') != 1 or receipt.get('status') != 'complete' or receipt['context'] != context:
        raise ValueError('Missing, incomplete or mismatched control calibration context')
    outputs = receipt['outputs']
    if 'metrics.csv' not in outputs:
        raise ValueError('Calibration does not bind control metrics')
    for name, digest in outputs.items():
        if Path(name).name != name or not (directory / name).is_file() or sha256(directory / name) != digest:
            raise ValueError(f'Changed or missing calibration output: {name}')
    metrics = pl.read_csv(directory / 'metrics.csv')
    if not metrics['category'].is_in(['NC', 'PC']).fill_null(False).all():
        raise ValueError('Calibration must contain only NC+PC classifiers')
    thresholds = compute_null_threshold(metrics)
    validate_thresholds(thresholds, list(context['channel_features']))
    if thresholds != receipt['thresholds']:
        raise ValueError('Calibration thresholds do not match control metrics')
    return thresholds, metrics


def load_completed_summary(directory: Path, representation: str, batch: str) -> pl.DataFrame:
    receipt, _ = read_json_with_hash(directory / 'completion.json')
    context = receipt['context']
    if (receipt.get('status') != 'complete' or context['representation'] != representation
            or context['batch'] != batch or context['protocol'] != 't4'):
        raise ValueError('Mismatched completed T4 classification')
    controls = Path(receipt['calibration_dir'])
    if sha256(controls / 'calibration.json') != receipt['calibration_sha256']:
        raise ValueError('Changed control calibration')
    load_calibration(controls, context)
    path = directory / 'metrics_summary.csv'
    if sha256(path) != receipt['summary_sha256']:
        raise ValueError('Changed classification summary')
    return pl.read_csv(path, schema_overrides={'auroc_std': pl.Float64})
