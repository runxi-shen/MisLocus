"""Calibration and incomplete-run errors, independent of new execution APIs."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import polars as pl
from prot_loc_benchmark.classification import metrics, executor

class XGBThresholdChecks(unittest.TestCase):
    def test_nearest_quantile_and_missing_or_invalid_controls(self):
        frame = pl.DataFrame({'channel': ['x'] * 4, 'auroc': [.1, .3, .6, .9]})
        self.assertEqual(metrics.compute_null_threshold(frame, 50), {'x': .6})
        for bad in [frame.head(0), frame.with_columns(auroc=pl.lit(float('nan'))),
                    frame.with_columns(auroc=pl.lit(1.1)), frame.with_columns(auroc=pl.lit(float('inf'))),
                    frame.with_columns(auroc=pl.Series([.1, .3, .6, float('inf')]))]:
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

    def test_legacy_reader_does_not_invent_calibration_or_duplicate_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); folder = root / 'vit' / 'batch'; folder.mkdir(parents=True)
            pl.DataFrame(dict(classifier_id=['c'], pair_id=['G__G_v'], gene=['G'], allele_var=['G_v'],
                category=['Exp'], channel=['GFP'], auroc=[.8], auprc=[.8], balanced_accuracy=[.8], imbalance_ratio=[1.])).write_csv(folder/'metrics.csv')
            pl.DataFrame(dict(classifier_id=['c', 'c'], test_plates=['P_T4', 'P_T4'])).write_csv(folder/'classifier_info.csv')
            result = metrics.load_single_fold_metrics('vit', 'batch', classification_dir=root)
            self.assertEqual(result.height, 1)
            for column in ['auroc_std', 'null_threshold', 'is_hit']:
                self.assertIsNone(result[column][0])

    def test_unavailable_classifier_stops_the_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            task = dict(fold=SimpleNamespace(fold_id=0), train_df=pl.DataFrame(), test_df=pl.DataFrame(), ch_features=[])
            with patch.object(executor, 'train_and_predict', return_value=None):
                with self.assertRaisesRegex(ValueError, 'partial calibrated'):
                    executor.run_classifier_tasks([task], device='cpu', output_dir=Path(directory))
