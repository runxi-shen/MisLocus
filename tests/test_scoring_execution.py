"""Explicit device allocation and unchanged CPU fits; no GPU equivalence claim."""
import json, os, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import polars as pl
from prot_loc_benchmark.classification import train, executor

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
