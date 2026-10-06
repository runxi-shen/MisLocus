"""Matched controls are required; smoke checks do not certify GPU equivalence."""
import importlib.util
import json
import os
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import polars as pl

from prot_loc_benchmark.classification import metrics, train
from prot_loc_benchmark.classification.calibration import save_calibration, load_calibration


class CalibrationChecks(unittest.TestCase):
    def test_nearest_quantile_and_missing_or_invalid_controls(self):
        frame = pl.DataFrame({'channel': ['x'] * 4, 'auroc': [.1, .3, .6, .9]})
        self.assertEqual(metrics.compute_null_threshold(frame, 50), {'x': .6})
        for bad in [frame.head(0), frame.with_columns(auroc=pl.lit(float('nan'))),
                    frame.with_columns(auroc=pl.lit(1.1)), frame.with_columns(auroc=pl.lit(float('inf')))]:
            with self.assertRaises(ValueError):
                metrics.compute_null_threshold(bad)
        exp = pl.DataFrame({'pair_id': ['a'], 'gene': ['G'], 'allele_var': ['G_v'],
                            'channel': ['x'], 'imbalance_ratio': [1.], 'auroc': [.8],
                            'auprc': [.8], 'balanced_accuracy': [.8]})
        with self.assertRaises(ValueError):
            metrics.aggregate_allele_metrics(exp, {}, min_classifiers=1)
        result = metrics.aggregate_allele_metrics(exp, {'x': .8}, min_classifiers=1)
        self.assertFalse(result['is_hit'][0])
        self.assertIsNone(result['auroc_std'][0])

    def test_calibration_binds_context_and_control_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            controls = pl.DataFrame({'category': ['NC', 'PC'], 'channel': ['x', 'x'], 'auroc': [.6, .8]})
            controls.write_csv(directory / 'metrics.csv')
            context = {'channel_features': {'x': ['f']}, 'device': 'cpu', 'protocol': 't4',
                       'features_sha256': 'input', 'xgb_params': {'n_jobs': 1}}
            self.assertEqual(save_calibration(directory, context), {'x': .8})
            self.assertEqual(load_calibration(directory, context)[0], {'x': .8})
            for key, value in [('device', 'cuda:0'), ('protocol', 'full'), ('features_sha256', 'other'),
                               ('channel_features', {'x': ['g']}), ('xgb_params', {'n_jobs': 2})]:
                with self.assertRaises(ValueError):
                    load_calibration(directory, {**context, key: value})
            controls.with_columns(auroc=pl.lit(.5)).write_csv(directory / 'metrics.csv')
            with self.assertRaises(ValueError):
                load_calibration(directory, context)

    def test_device_fails_closed_and_cpu_is_repeatable(self):
        with patch.dict(os.environ, {'MISLOCUS_CLASSIFIER_BACKEND': 'gpu', 'CUDA_VISIBLE_DEVICES': ''}):
            with self.assertRaises(ValueError):
                train.select_device()
        with patch.dict(os.environ, {'MISLOCUS_CLASSIFIER_BACKEND': 'typo'}):
            with self.assertRaises(ValueError):
                train.select_device()
        data = pl.DataFrame({'f': np.linspace(-1, 1, 40), 'Label': [0] * 20 + [1] * 20})
        a = train.train_and_predict(data, data, ['f'], xgb_params={'n_jobs': 1})
        b = train.train_and_predict(data, data, ['f'], xgb_params={'n_jobs': 1})
        np.testing.assert_array_equal(a[0], b[0])
        with patch.object(train, 'XGBClassifier') as classifier:
            classifier.return_value.get_booster.return_value.save_config.return_value = json.dumps(
                {'learner': {'generic_param': {'device': 'cpu'}}})
            with self.assertRaisesRegex(RuntimeError, 'fallback'):
                train.train_and_predict(data, data, ['f'], device='cuda:0')

    def test_failed_cli_does_not_rebind_existing_outputs(self):
        path = Path(__file__).resolve().parents[1] / 'scripts/09_classify.py'
        spec = importlib.util.spec_from_file_location('failed_cli', path)
        cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / 'vit' / 'batch').mkdir(parents=True)
            with patch.object(cli, 'CLASSIFICATION_OUTPUT_DIR', root), patch.object(
                    cli, 'classify_batch', side_effect=RuntimeError('fit failed')), patch.object(
                    sys, 'argv', ['script', '--batch', 'batch', '--representation', 'vit']), patch(
                    'prot_loc_benchmark.provenance.record') as record, self.assertRaises(RuntimeError):
                cli.main()
            record.assert_not_called()

    def test_cli_controls_first_then_experiment(self):
        path = Path(__file__).resolve().parents[1] / 'scripts/09_classify.py'
        spec = importlib.util.spec_from_file_location('classify', path)
        cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
        batch = '2025_03_17_Batch_15'; rep = 'subcell_finetuned_mae'
        rows = []
        # Three same-allele NC wells give three pairwise controls; one real ref/variant pair.
        for plate in range(1, 5):
            for well, allele, gene, node, category in [('A01', 'RHEB', 'RHEB', 'disease_wt', 'NC'),
                    ('A02', 'RHEB', 'RHEB', 'disease_wt', 'NC'), ('A03', 'RHEB', 'RHEB', 'disease_wt', 'NC'),
                    ('B01', 'G', 'G', 'disease_wt', 'Exp'), ('B02', 'G_v', 'G', 'allele', 'Exp')]:
                for i in range(60):
                    rows.append({'Metadata_Plate': f'plate_T{plate}', 'Metadata_plate_map_name': 'map',
                                 'Metadata_well_position': well, 'Metadata_gene_allele': allele,
                                 'Metadata_symbol': gene, 'Metadata_node_type': node, 'Metadata_Control': category,
                                 'Metadata_ObjectNumber': i, 'SubCell_0': (i % 11) / 10 + (allele == 'G_v')})
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'MISLOCUS_CLASSIFIER_BACKEND': 'cpu'}):
            root = Path(tmp); inputs = root / 'input' / rep / batch; inputs.mkdir(parents=True)
            pl.DataFrame(rows).write_parquet(inputs / 'features.parquet')
            with patch.object(cli, 'INTERIM_DIR', root / 'input'), patch.object(cli, 'CLASSIFICATION_OUTPUT_DIR', root / 'output'):
                with self.assertRaisesRegex(ValueError, 'full NC\\+PC'):
                    cli.classify_batch(batch, rep, scope='exp', test_split='t4', workers=1, threads=1)
                cli.classify_batch(batch, rep, scope='control', test_split='t4', workers=1, threads=1)
                base = root / 'output' / (rep + '_t4') / batch
                controls = base / 'controls'
                cli.classify_batch(batch, rep, scope='exp', test_split='t4', workers=1, threads=1,
                                   calibration_dir=controls)
                summary = metrics.load_single_fold_metrics(rep, batch, classification_dir=root / 'output')
                self.assertEqual(summary.height, 1)
                self.assertTrue(summary['is_hit'][0])
                info = pl.read_csv(base / 'classifier_info.csv')
                self.assertTrue(info['test_plates'].str.ends_with('T4').all())
                self.assertEqual(info['train_plates'][0], 'plate_T1,plate_T2,plate_T3')
                before = (base / 'metrics_summary.csv').read_bytes()
                with self.assertRaises(FileExistsError):
                    cli.classify_batch(batch, rep, scope='exp', test_split='t4', workers=1, threads=1,
                                       calibration_dir=controls)
                self.assertEqual(before, (base / 'metrics_summary.csv').read_bytes())
                (controls / 'metrics.csv').write_text('changed')
                with self.assertRaises(ValueError):
                    metrics.load_single_fold_metrics(rep, batch, classification_dir=root / 'output')


if __name__ == '__main__':
    unittest.main()
