"""Execution safeguards, without adding a new evaluation split or hit policy."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import polars as pl

from prot_loc_benchmark.classification import train, executor
from prot_loc_benchmark.classification.calibration import save_calibration, load_calibration
from prot_loc_benchmark import copairs_runtime as runtime


def classify_cli():
    path = Path(__file__).resolve().parents[1] / 'scripts/09_classify.py'
    spec = importlib.util.spec_from_file_location('execution_cli', path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    return cli


class DeviceChecks(unittest.TestCase):
    def test_cpu_default_and_explicit_gpu_allocation(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(train.select_device(), 'cpu')
        for visible in ['', '0', 'GPU-a,GPU-b']:
            with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': visible}):
                with self.assertRaisesRegex(ValueError, 'allocation'):
                    train.select_device('gpu')
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': 'GPU-01234567-89ab-cdef-0123-456789abcdef'}):
            with patch('xgboost.build_info', return_value={'USE_CUDA': False}):
                with self.assertRaisesRegex(ValueError, 'CUDA support'):
                    train.select_device('gpu')
            with patch('xgboost.build_info', return_value={'USE_CUDA': True}):
                self.assertEqual(train.select_device('gpu'), 'cuda:0')
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            train.select_device('typo')

    def test_cpu_predictions_repeat_and_gpu_fallback_raises(self):
        data = pl.DataFrame({'f': np.linspace(-1, 1, 40), 'Label': [0] * 20 + [1] * 20})
        a = train.train_and_predict(data, data, ['f'], xgb_params={'n_jobs': 1})
        b = train.train_and_predict(data, data, ['f'], xgb_params={'n_jobs': 1})
        np.testing.assert_array_equal(a[0], b[0])
        with patch.object(train, 'XGBClassifier') as classifier:
            classifier.return_value.get_booster.return_value.save_config.return_value = json.dumps(
                {'learner': {'generic_param': {'device': 'cpu'}}})
            with self.assertRaisesRegex(RuntimeError, 'fallback'):
                train.train_and_predict(data, data, ['f'], device='cuda:0')

    def test_executor_uses_one_selected_device_without_mutating_tasks(self):
        task = {'fold': SimpleNamespace(fold_id=3)}
        with tempfile.TemporaryDirectory() as tmp, patch.object(executor, '_run_classifier', return_value=None) as run:
            with self.assertRaisesRegex(ValueError, 'partial calibrated'):
                executor.run_classifier_tasks([task], 'cuda:0', Path(tmp), workers=1, xgb_params={'n_jobs': 2})
            run.assert_called_once_with(task, 'cuda:0', {'n_jobs': 2})
            self.assertNotIn('device', task)
        for workers in [0, -1, True, 1.5]:
            with self.assertRaisesRegex(ValueError, 'workers'):
                executor.run_classifier_tasks([], 'cpu', Path('/unused'), workers=workers)


class CalibrationChecks(unittest.TestCase):
    def test_context_and_control_bytes_must_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            controls = pl.DataFrame({'category': ['NC', 'PC'], 'channel': ['x', 'x'], 'auroc': [.6, .8]})
            controls.write_csv(directory / 'metrics.csv')
            context = {'channel_features': {'x': ['f']}, 'device': 'cpu', 'protocol': 'single_rep:lopo',
                       'features_sha256': 'input', 'xgb_params': {'n_jobs': 1}, 'code': {'train.py': 'source'},
                       'xgboost_library_sha256': 'build'}
            self.assertEqual(save_calibration(directory, context), {'x': .8})
            self.assertEqual(load_calibration(directory, context)[0], {'x': .8})
            for key, value in [('device', 'cuda:0'), ('protocol', 'multi_rep:lopo'),
                    ('features_sha256', 'other'), ('channel_features', {'x': ['g']}),
                    ('xgb_params', {'n_jobs': 2}), ('code', {'train.py': 'other'}),
                    ('xgboost_library_sha256', 'other')]:
                with self.subTest(key=key), self.assertRaises(ValueError):
                    load_calibration(directory, {**context, key: value})
            with self.assertRaises(FileExistsError):
                save_calibration(directory, context)
            controls.with_columns(auroc=pl.lit(.5)).write_csv(directory / 'metrics.csv')
            with self.assertRaisesRegex(ValueError, 'Changed'):
                load_calibration(directory, context)

    def test_altered_threshold_and_noncontrol_metrics_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            pl.DataFrame({'category': ['Exp'], 'channel': ['x'], 'auroc': [.8]}).write_csv(directory / 'metrics.csv')
            context = {'channel_features': {'x': ['f']}}
            with self.assertRaisesRegex(ValueError, 'NC\\+PC'):
                save_calibration(directory, context)
            pl.DataFrame({'category': ['NC'], 'channel': ['x'], 'auroc': [.8]}).write_csv(directory / 'metrics.csv')
            save_calibration(directory, context)
            receipt = directory / 'calibration.json'
            record = json.loads(receipt.read_text()); record['thresholds']['x'] = .5
            receipt.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, 'thresholds'):
                load_calibration(directory, context)

    def test_native_lopo_combined_and_reused_controls_match(self):
        cli = classify_cli()
        batch = '2025_03_17_Batch_15'; rep = 'subcell_finetuned_mae'
        rows = [dict(Metadata_Plate=f'plate_T{plate}', Metadata_plate_map_name='map',
                     Metadata_well_position=well, Metadata_gene_allele=allele, Metadata_symbol=gene,
                     Metadata_node_type=node, Metadata_Control=category, Metadata_ObjectNumber=i,
                     SubCell_0=(i % 11) / 10 + (allele == 'G_v'))
                for plate in range(1, 5)
                for well, allele, gene, node, category in [
                    ('A01', 'RHEB', 'RHEB', 'disease_wt', 'NC'),
                    ('A02', 'RHEB', 'RHEB', 'disease_wt', 'NC'),
                    ('A03', 'RHEB', 'RHEB', 'disease_wt', 'NC'),
                    ('B01', 'G', 'G', 'disease_wt', 'Exp'),
                    ('B02', 'G_v', 'G', 'allele', 'Exp')]
                for i in range(60)]
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'MISLOCUS_CLASSIFIER_BACKEND': 'cpu'}):
            root = Path(tmp); inputs = root / 'input' / rep / batch; inputs.mkdir(parents=True)
            pl.DataFrame(rows).write_parquet(inputs / 'features.parquet')
            with patch.object(cli, 'INTERIM_DIR', root / 'input'), patch.object(cli, 'CLASSIFICATION_OUTPUT_DIR', root / 'combined'):
                cli.classify_batch(batch, rep, workers=1, threads=1)
            with patch.object(cli, 'INTERIM_DIR', root / 'input'), patch.object(cli, 'CLASSIFICATION_OUTPUT_DIR', root / 'separate'):
                with self.assertRaisesRegex(ValueError, 'full NC\\+PC'):
                    cli.classify_batch(batch, rep, scope='exp', workers=1, threads=1)
                cli.classify_batch(batch, rep, scope='control', workers=1, threads=1)
                controls = root / 'separate' / rep / batch / 'controls'
                original_controls = (controls / 'metrics.csv').read_bytes()
                cli.classify_batch(batch, rep, scope='exp', workers=1, threads=1, calibration_dir=controls)
                combined = root / 'combined' / rep / batch
                separate = root / 'separate' / rep / batch
                left = pl.read_csv(combined / 'metrics_summary.csv').sort('pair_id', 'channel')
                right = pl.read_csv(separate / 'metrics_summary.csv').sort('pair_id', 'channel')
                self.assertEqual(left.to_dicts(), right.to_dicts())
                combined_metrics = pl.read_csv(combined / 'metrics.csv')
                self.assertEqual(set(combined_metrics['category']), {'NC', 'Exp'})
                self.assertEqual(combined_metrics.height, 16)
                experiment_info = pl.read_csv(separate / 'classifier_info.csv')
                self.assertEqual(set(experiment_info['fold_id']), {0, 1, 2, 3})
                for row in experiment_info.iter_rows(named=True):
                    self.assertEqual(len(row['train_plates'].split(',')), 3)
                    self.assertNotIn(row['test_plates'], row['train_plates'].split(','))
                self.assertEqual((controls / 'metrics.csv').read_bytes(), original_controls)
                before = (separate / 'metrics_summary.csv').read_bytes()
                with self.assertRaises(FileExistsError):
                    cli.classify_batch(batch, rep, scope='exp', workers=1, threads=1, calibration_dir=controls)
                self.assertEqual((separate / 'metrics_summary.csv').read_bytes(), before)
            # Reject mismatched reuse before fitting or creating new output files.
            with patch.object(cli, 'INTERIM_DIR', root / 'input'), patch.object(cli, 'CLASSIFICATION_OUTPUT_DIR', root / 'mismatch'), \
                    patch.object(cli, 'run_classifier_tasks') as fit:
                with self.assertRaisesRegex(ValueError, 'context'):
                    cli.classify_batch(batch, rep, scope='exp', workers=1, threads=2, calibration_dir=controls)
                fit.assert_not_called()
                self.assertFalse((root / 'mismatch').exists())

    def test_failed_cli_does_not_rebind_existing_provenance(self):
        cli = classify_cli()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / 'vit' / 'batch').mkdir(parents=True)
            with patch.object(cli, 'CLASSIFICATION_OUTPUT_DIR', root), patch.object(cli, 'classify_batch', side_effect=RuntimeError('fit failed')), \
                    patch.object(cli.sys, 'argv', ['script', '--batch', 'batch', '--representation', 'vit']), \
                    patch('prot_loc_benchmark.provenance.record') as record:
                with self.assertRaises(RuntimeError):
                    cli.main()
                record.assert_not_called()


class CopairsExecutionChecks(unittest.TestCase):
    def test_limits_apply_and_restore_on_failure(self):
        original_pool = runtime.compute.ThreadPool
        original_connect = runtime.duckdb.connect
        connection = runtime.duckdb.default_connection()
        original_threads = connection.execute("SELECT current_setting('threads')").fetchone()[0]
        with self.assertRaisesRegex(RuntimeError, 'synthetic'):
            with runtime.bounded_copairs(workers=2, blas_threads=1):
                self.assertEqual(connection.execute("SELECT current_setting('threads')").fetchone()[0], 2)
                pool = runtime.compute.ThreadPool(8)
                try:
                    self.assertEqual(pool._processes, 2)
                finally:
                    pool.close(); pool.join()
                other = runtime.duckdb.connect(config={'threads': 8})
                try:
                    self.assertEqual(other.execute("SELECT current_setting('threads')").fetchone()[0], 2)
                finally:
                    other.close()
                raise RuntimeError('synthetic')
        self.assertIs(runtime.compute.ThreadPool, original_pool)
        self.assertIs(runtime.duckdb.connect, original_connect)
        self.assertEqual(connection.execute("SELECT current_setting('threads')").fetchone()[0], original_threads)

    def test_invalid_counts_and_wrapper_argument_forwarding(self):
        for value in [0, -1, True, '1.5']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                runtime.positive_threads(value)
        for function, target in [(runtime.average_precision, runtime.copairs_map),
                                 (runtime.average_precision_multilabel, runtime.multilabel)]:
            with patch.object(target, 'average_precision', return_value='result') as native:
                self.assertEqual(function('data', max_workers=1, blas_threads=1, seed=42), 'result')
                native.assert_called_once_with('data', seed=42)


if __name__ == '__main__':
    unittest.main()
