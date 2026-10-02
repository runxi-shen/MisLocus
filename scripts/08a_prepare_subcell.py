#!/usr/bin/env python3
"""SubCell-specific inspection/preflight AFTER 01_prepare_cell_crops.py extraction.

Checks all six batches, canonical cell/array-row alignment, allele vocabulary,
T1/T2/T3/T4 splits and the fixed T3 selection. Does not extract or resize crops:
SubCellPreprocessor handles model-specific geometry/normalization at consumption.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from prot_loc_benchmark.representations.subcell_manifest import build_manifest, release_inventory, release_tables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['inspect', 'preflight'])
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--crops', type=Path, help='Completed shared extraction root from 01')
    parser.add_argument('--output', type=Path, help='New SubCell preflight directory')
    args = parser.parse_args()
    if args.action == 'inspect':
        inventory = release_inventory(args.release)
        frame, classes = release_tables(args.release)
        print(inventory['remote'], inventory['revision'])
        print('Cells:', frame.groupby('split').size().to_dict())
        print('Training classes:', len(classes))
        print('Missing T3:', sorted(set(classes) - set(frame.loc[frame.split == 'val', 'Metadata_gene_allele'])))
        print('Metadata-only check; crop alignment and payload integrity NOT certified.')
    else:
        if args.crops is None or args.output is None:
            parser.error('preflight requires --crops and --output')
        build_manifest(args.release, args.crops, args.output)


if __name__ == '__main__':
    main()
