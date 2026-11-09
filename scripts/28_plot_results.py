#!/usr/bin/env python3
"""Plot completed public workflow outputs and export the exact plotted data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--clinical-report', type=Path)
    choice.add_argument('--evaluation', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    from prot_loc_benchmark.workflows.features import sha256, save_json
    directory = args.clinical_report or args.evaluation
    receipt_name = 'report.json' if args.clinical_report else 'evaluation.json'
    receipt_path = directory / receipt_name
    receipt = json.loads(receipt_path.read_text())
    if receipt['status'] != 'complete':
        raise ValueError('Input workflow must have completed')
    inputs = {str(receipt_path.resolve()): sha256(receipt_path)}

    def load(name):
        path = directory / name
        digest = sha256(path)
        if receipt['outputs_sha256'].get(name) != digest:
            raise ValueError('Output differs from its completed receipt: ' + name)
        inputs[str(path.resolve())] = digest
        return pl.read_parquet(path) if path.suffix == '.parquet' else pl.read_csv(path)

    args.output.mkdir(parents=True)
    descriptions = []

    def save(fig, name, data, description):
        frame = data if isinstance(data, pl.DataFrame) else pl.DataFrame(data)
        frame.write_csv(args.output / (name + '-data.csv'))
        fig.tight_layout()
        for extension in ['png', 'svg']:
            fig.savefig(args.output / (name + '.' + extension), dpi=160, bbox_inches='tight')
        plt.close(fig)
        descriptions.append({'name': name, 'description': description, 'rows': frame.height,
                             'plotted_data': name + '-data.csv'})

    if args.clinical_report:
        table = load('hit-rate-table.csv')
        required = {'representation', 'channel', 'cohort', 'n_alleles', 'n_hits', 'hit_fraction'}
        if not required <= set(table.columns):
            raise ValueError('Clinical hit-rate table lacks expected columns')
        for cohort in table['cohort'].unique(maintain_order=True):
            part = table.filter(pl.col('cohort') == cohort)
            labels = [r['representation'] + '/' + r['channel'] for r in part.iter_rows(named=True)]
            rates = part['hit_fraction'].to_numpy().astype(float)
            valid = np.isfinite(rates)
            if ((rates[valid] < 0) | (rates[valid] > 1)).any():
                raise ValueError('Invalid hit rates')
            fig, ax = plt.subplots(figsize=(max(6, len(labels) * .65), 4))
            ax.bar(np.arange(len(labels)), rates * 100, color='#386b88')
            ax.set_xticks(np.arange(len(labels)), labels, rotation=45, ha='right')
            for i, row in enumerate(part.iter_rows(named=True)):
                ax.annotate(f"{row['n_hits']}/{row['n_alleles']}" if valid[i] else 'unscored', (i, rates[i] * 100 if valid[i] else 0), xytext=(0, 4),
                            textcoords='offset points', ha='center', fontsize=8)
            ax.set(ylabel='Imaging hits (%)', title=cohort.replace('_', ' '), ylim=(0, max(5, float(rates[valid].max() * 100) + 12) if valid.any() else 5))
            save(fig, f'hit-rates-{len(descriptions):02d}', part,
                 'Exact cohort hit rates and hit/eligible counts; no uncertainty intervals inferred.')
        binary = load('binary-statistics.csv')
        labels = [r['representation'] + '/' + r['channel'] + '\n' + r['predictor'] + '/' + r['cohort']
                  for r in binary.iter_rows(named=True)]
        fig, ax = plt.subplots(figsize=(max(7, len(labels) * .28), 4))
        ax.bar(np.arange(len(labels)), binary['F1'].cast(pl.Float64).to_numpy(), color='#537b43')
        ax.set_xticks(np.arange(len(labels)), labels, rotation=90, fontsize=6)
        ax.set(ylabel='F1', ylim=(0, 1), title='Clinical/predictor agreement on declared cohorts')
        save(fig, 'clinical-f1', binary, 'F1 from saved confusion counts; significance and missing-label populations remain in the source table.')
    elif receipt['task'] == 'hpa':
        table = load('display.csv')
        locations = table['hpa_location'].unique(maintain_order=True).to_list()
        channels = table['channel'].unique(maintain_order=True).to_list()
        lookup = {(r['channel'], r['hpa_location']): r['display_mAP_hpa_norm'] for r in table.iter_rows(named=True)}
        values = np.array([[lookup.get((channel, location), np.nan) for location in locations] for channel in channels])
        fig, ax = plt.subplots(figsize=(max(7, len(locations) * .4), max(2, len(channels) * .6)))
        image = ax.imshow(values, aspect='auto', vmin=0, vmax=max(.1, float(np.nanmax(values))), cmap='viridis')
        ax.set_xticks(np.arange(len(locations)), locations, rotation=90)
        ax.set_yticks(np.arange(len(channels)), channels)
        fig.colorbar(image, ax=ax, label='Display normalized mAP (zero floor)')
        save(fig, 'hpa-location-retrieval', table, 'Locations with at least three measured genes; signed scores and pre-display BH family remain in evaluation outputs.')
    else:
        scores, controls = load('scores.parquet'), load('controls.parquet')
        xgb = receipt['task'] == 'xgb'
        score = 'auroc_mean' if xgb else 'mAP_vs_ref_norm'
        control_score = 'auroc' if xgb else score
        threshold = 'null_threshold' if xgb else 'null_threshold_p95'
        data = []
        for index, channel in enumerate(scores['channel'].unique(maintain_order=True)):
            variant, null = scores.filter(pl.col('channel') == channel), controls.filter(pl.col('channel') == channel)
            cutoff, = variant[threshold].unique()
            fig, ax = plt.subplots(figsize=(6, 4))
            bins = np.linspace(0 if xgb else -1, 1, 31)
            ax.hist(variant[score].to_numpy(), bins=bins, alpha=.6, label=f'Variants, n={variant.height}')
            ax.hist(null[control_score].to_numpy(), bins=bins, alpha=.6, label=f'Controls, n={null.height}')
            ax.axvline(cutoff, color='black', linestyle='--', label='Control p95')
            ax.set(xlabel=score, ylabel='Scored comparisons', title=channel)
            ax.legend()
            values = [{'population': name, 'score': r[field], 'channel': channel, 'control_p95': cutoff}
                      for name, part, field in [('variant', variant, score), ('control', null, control_score)]
                      for r in part.iter_rows(named=True)]
            save(fig, f'score-distribution-{index:02d}', values,
                 'Saved observed score distributions and declared control p95. PA final hits also require BH q<0.05.')
    if inputs != {path: sha256(path) for path in inputs}:
        raise ValueError('Input changed while plotting')
    save_json(args.output / 'figures.json', {'schema_version': 1, 'status': 'complete',
        'inputs_sha256': inputs, 'source_sha256': sha256(__file__), 'figures': descriptions,
        'scope': 'Plots of completed explicit public outputs, not a claim of original manuscript asset recovery.',
        'outputs_sha256': {p.name: sha256(p) for p in args.output.iterdir() if p.is_file()}})
    print(json.dumps({'output': str(args.output), 'figures': len(descriptions)}))


if __name__ == '__main__':
    main()
