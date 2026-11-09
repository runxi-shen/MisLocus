#!/usr/bin/env python3
"""Generate hit-rate and clinical/predictor tables from explicit calibrated scores."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=['xgb', 'pa'], required=True)
    parser.add_argument('--input', nargs=3, action='append', required=True, metavar=('MODEL', 'BATCH', 'FILE'))
    parser.add_argument('--setting', nargs=2, action='append', required=True, metavar=('MODEL', 'CHANNEL'))
    parser.add_argument('--annotations', type=Path, required=True, help='Public allele_inventory.parquet or equivalent raw-column annotation table')
    parser.add_argument('--predictors', nargs='+', default=['clinvar'], choices=['clinvar', 'alphamissense', 'esm1b', 'revel', 'mutpred'])
    parser.add_argument('--biorep-pairs', type=Path, help='JSON object: pair name -> [batch1,batch2]; defaults to the six released batches')
    parser.add_argument('--all-rule', choices=['any_complete_pair', 'all_retained_batches'], default='any_complete_pair',
                        help='ALL across one successful complete pair or every retained batch across complete pairs')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    os.environ['POLARS_MAX_THREADS'] = '2'
    for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        os.environ[name] = '1'
    import polars as pl
    from prot_loc_benchmark.config import BIOREP_PAIRS, canonical_representation
    from prot_loc_benchmark.workflows.features import sha256, save_json
    from prot_loc_benchmark.workflows.clinical import clinical_tables
    sources = [*sorted((ROOT / 'src').rglob('*.py')), Path(__file__)]
    source_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in sources}
    input_hashes = {str(args.annotations.resolve()): sha256(args.annotations)}
    pairs = BIOREP_PAIRS
    if args.biorep_pairs:
        input_hashes[str(args.biorep_pairs.resolve())] = sha256(args.biorep_pairs)
        pairs = json.loads(args.biorep_pairs.read_text())
    if not isinstance(pairs, dict) or not pairs or any(not isinstance(k, str) or not k.strip() for k in pairs):
        raise ValueError('Biological pairs must be a nonempty named JSON object')
    if any(not isinstance(v, (list, tuple)) or len(v) != 2 or any(not isinstance(b, str) or not b.strip() or b != b.strip() for b in v) for v in pairs.values()):
        raise ValueError('Each biological pair must contain two explicit nonblank batch names')
    frames, declarations = [], []
    for model, batch, name in args.input:
        path = Path(name)
        model = canonical_representation(model)
        input_hashes[str(path.resolve())] = sha256(path)
        if path.suffix not in ['.csv', '.parquet']:
            raise ValueError('Scores must be CSV or Parquet tables')
        frame = pl.read_csv(path) if path.suffix == '.csv' else pl.read_parquet(path)
        for column, value in [('representation', model), ('batch', batch)]:
            if column in frame.columns and (frame[column].null_count() or set(frame[column]) != {value}):
                raise ValueError(f'Score {column} does not match its explicit declaration')
            frame = frame.with_columns(pl.lit(value).alias(column))
        frames.append(frame)
        declarations.append({'model': model, 'batch': batch, 'file': str(path.resolve())})
    settings = [(canonical_representation(model), channel) for model, channel in args.setting]
    details = clinical_tables(frames, task=args.task, settings=settings, biorep_pairs=pairs,
                              annotations=args.annotations, selected_predictors=args.predictors, output=args.output, all_rule=args.all_rule)
    if input_hashes != {name: sha256(name) for name in input_hashes}:
        raise ValueError('Inputs changed during reporting')
    if source_hashes != {str(p.relative_to(ROOT)): sha256(p) for p in sources}:
        raise ValueError('Sources changed during reporting')
    save_json(args.output / 'report.json', {'schema_version': 1, 'status': 'complete', 'details': details,
                                           'input_declarations': declarations, 'inputs_sha256': input_hashes,
                                           'source_sha256': source_hashes,
                                           'runtime': {name: importlib.metadata.version(name) for name in ['polars', 'numpy', 'scipy']},
                                           'outputs_sha256': {p.name: sha256(p) for p in sorted(args.output.iterdir()) if p.is_file()}})
    print(json.dumps({'output': str(args.output), 'task': args.task, 'settings': settings,
                      'binary_tests': details['binary_tests'], 'continuous_tests': details['continuous_tests']}, indent=2))


if __name__ == '__main__':
    main()
