#!/usr/bin/env python3
"""Evaluate explicit feature artifacts: supervised XGB, unsupervised PA or HPA."""
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', choices=['xgb', 'pa', 'hpa'])
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--batch', help='Required when batch column is absent; PA/XGB score a single batch')
    parser.add_argument('--channels', nargs='+', help='Explicit subset of the declared channels; default: all declared channels')
    parser.add_argument('--hpa-annotation', type=Path, default=ROOT / 'annotations/hpa_gene_localization_table.parquet')
    parser.add_argument('--hpa-reliability', type=float, choices=[.5, 1., 2., 3.], default=1.)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--reporting-model', help='Optional equivalent representation alias for downstream reporting (for example morphem -> vit); specification identity is preserved')
    parser.add_argument('--backend', choices=['cpu', 'gpu'], default='cpu', help='XGB fitting backend; GPU never falls back to CPU')
    parser.add_argument('--gpu-uuid', help='Full allocated GPU UUID for XGB GPU fitting')
    parser.add_argument('--control-plate-policy', choices=['complete', 'available'], default='complete',
                        help='XGB controls: require all T1/T2/T3 or use available non-T4 plates. Experimental training remains T1/T2/T3.')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.workers < 1:
        raise ValueError('workers must be positive')
    if args.task != 'xgb' and (args.backend != 'cpu' or args.gpu_uuid is not None or args.control_plate_policy != 'complete'):
        raise ValueError('Backend, GPU UUID and control-plate policy options apply only to XGB')
    # Set defaults before importing numerical packages; this command is also
    # usable directly, without the internal audit supervisor.
    os.environ['POLARS_MAX_THREADS'] = str(args.workers)
    for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        os.environ[name] = '1'
    from prot_loc_benchmark.workflows.features import load_artifact, save_json, sha256
    import duckdb
    old_connection = duckdb.default_connection()
    duckdb.set_default_connection(duckdb.connect(config={'threads': args.workers}))
    old_connection.close()
    source_paths = [*sorted((ROOT / 'src').rglob('*.py')), ROOT / 'scripts/09c_classify_PA.py', Path(__file__)]
    source_paths += [p for p in [ROOT / 'pyproject.toml', ROOT / 'pixi.lock'] if p.is_file()]
    copairs_root = Path(importlib.util.find_spec('copairs').origin).parent
    numerical_paths = sorted(copairs_root.rglob('*.py'))
    source_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in source_paths}
    dependency_hashes = {str(p.relative_to(copairs_root)): sha256(p) for p in numerical_paths}
    input_paths = [args.features, args.spec] + ([args.hpa_annotation] if args.task == 'hpa' else [])
    input_hashes = {str(p.resolve()): sha256(p) for p in input_paths}
    frame, specification, validation = load_artifact(args.features, args.spec, batch=args.batch)
    if validation['features_sha256'] != input_hashes[str(args.features.resolve())] or validation['specification_sha256'] != input_hashes[str(args.spec.resolve())]:
        raise ValueError('Inputs changed during feature admission')
    channels = specification['channels']
    from prot_loc_benchmark.config import canonical_representation
    reporting_model = args.reporting_model or specification['model']
    if canonical_representation(reporting_model) != canonical_representation(specification['model']):
        raise ValueError('Reporting model must identify the same representation as the feature specification')
    if args.channels:
        if len(args.channels) != len(set(args.channels)) or not set(args.channels) <= set(channels):
            raise ValueError('Requested channels must be distinct members of the feature specification')
        channels = {name: channels[name] for name in args.channels}
    from prot_loc_benchmark.workflows import evaluations
    if args.task == 'pa':
        details = evaluations.score_pa(frame, channels, args.output, workers=args.workers, model=reporting_model)
    elif args.task == 'xgb':
        details = evaluations.score_xgb(frame, channels, args.output, model=reporting_model,
            backend=args.backend, gpu_uuid=args.gpu_uuid, control_plate_policy=args.control_plate_policy)
    else:
        details = evaluations.score_hpa(frame, channels, args.hpa_annotation, args.output,
                                        reliability=args.hpa_reliability, workers=args.workers)
    if source_hashes != {str(p.relative_to(ROOT)): sha256(p) for p in source_paths} or dependency_hashes != {str(p.relative_to(copairs_root)): sha256(p) for p in numerical_paths}:
        raise ValueError('Numerical sources changed during evaluation')
    if input_hashes != {str(p.resolve()): sha256(p) for p in input_paths}:
        raise ValueError('Inputs or annotations changed during evaluation')
    receipt = {'schema_version': 1, 'status': 'complete', 'task': args.task,
               'model': reporting_model, 'declared_feature_model': specification['model'], 'feature_validation': validation,
               'channels': channels, 'batches': validation['batches'], 'details': details,
               'feature_specification': specification,
               'thread_settings': {'polars': args.workers, 'copairs': args.workers, 'blas': 1, 'xgboost_per_classifier': 1},
               'inputs_sha256': input_hashes, 'source_sha256': source_hashes,
               'copairs_python_sources_sha256': dependency_hashes,
               'runtime': {package: importlib.metadata.version(package) for package in ['numpy', 'polars', 'xgboost', 'copairs', 'scipy', 'scikit-learn', 'duckdb']},
               'outputs_sha256': {p.name: sha256(p) for p in sorted(args.output.iterdir()) if p.is_file()}}
    save_json(args.output / 'evaluation.json', receipt)
    print(json.dumps({'task': args.task, 'output': str(args.output), 'details': details}, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
