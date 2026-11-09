#!/usr/bin/env python3
"""Export readable CSV and LaTeX tables from a completed explicit-score report."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def tex(value):
    if value == '':
        return '--'
    escapes = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#',
               '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}
    return ''.join(escapes.get(c, c) for c in value)


def number(value, field):
    if value == '' or field in ['representation', 'channel', 'predictor', 'cohort', 'tier']:
        return tex(value)
    try:
        v = float(value)
    except ValueError:
        return tex(value)
    if not math.isfinite(v):
        raise ValueError('Nonfinite table statistic')
    if field in ['n_alleles', 'n_hits', 'labelled_alleles', 'n_pathogenic', 'n_benign']:
        return str(int(v))
    return f'{v:.3g}' if field.endswith(('_p', '_q')) else f'{v:.3f}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True, help='Completed22_report_clinical.py output directory')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    receipt_path = args.report / 'report.json'
    receipt = json.loads(receipt_path.read_text())
    if receipt.get('status') != 'complete':
        raise ValueError('A completed clinical report is required')
    schemas = {'hit-rate-table.csv': ['representation', 'channel', 'cohort', 'n_alleles', 'n_hits', 'hit_fraction'],
        'binary-statistics.csv': ['representation', 'channel', 'predictor', 'cohort', 'labelled_alleles', 'precision', 'recall', 'F1', 'fisher_p', 'fisher_q'],
        'continuous-statistics.csv': ['representation', 'channel', 'predictor', 'tier', 'n_pathogenic', 'n_benign', 'pathogenic_mean', 'benign_mean', 'MWU_p', 'MWU_q']}
    tables = []
    for name, fields in schemas.items():
        path = args.report / name
        if sha(path) != receipt['outputs_sha256'][name]:
            raise ValueError('Changed report table: ' + name)
        with path.open() as stream:
            reader = csv.DictReader(stream)
            if not set(fields) <= set(reader.fieldnames):
                raise ValueError('Missing expected table fields: ' + name)
            rows = list(reader)
        tables.append((name, fields, rows, sha(path)))
    args.output.mkdir(parents=True)
    for name, fields, rows, digest in tables:
        shutil.copyfile(args.report / name, args.output / name)
        lines = ['% Derived display only; complete precision and denominators remain in the CSV.',
            '% ' + receipt['details']['cohort_policy'], r'\begin{tabular}{' + 'l' * len(fields) + '}',
            ' & '.join(tex(field) for field in fields) + r' \\', r'\hline']
        lines += [' & '.join(number(row[field], field) for field in fields) + r' \\' for row in rows]
        lines.append(r'\end{tabular}')
        (args.output / (Path(name).stem + '.tex')).write_text('\n'.join(lines) + '\n')
    (args.output / 'export.json').write_text(json.dumps({'schema_version': 1, 'status': 'complete',
        'report_receipt_sha256': sha(receipt_path), 'source_code_sha256': sha(__file__),
        'display_policy': 'Three decimal places; p/q three significant digits; missing statistics --; no inherited bolding or rank claims.',
        'outputs_sha256': {p.name: sha(p) for p in args.output.iterdir() if p.is_file()}}, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'tables': len(tables)}))


if __name__ == '__main__':
    main()
