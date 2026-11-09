#!/usr/bin/env python3
"""Pool declared T4 WT inputs without normalization for HPA evaluation."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', nargs=3, action='append', required=True, metavar=('BATCH', 'FEATURES', 'SPEC'))
    parser.add_argument('--model', required=True)
    parser.add_argument('--channels', nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if len({batch for batch, features, spec in args.input}) != len(args.input):
        parser.error('Each input batch must be declared once')
    from prot_loc_benchmark.workflows.features import pool_hpa
    pool_hpa(args.input, args.model, args.channels, args.output)


if __name__ == '__main__':
    main()
