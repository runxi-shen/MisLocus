"""Keep historical extraction usable until its explicit CLI retirement."""
import runpy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class LegacyExportChecks(unittest.TestCase):
    def test_all_selectors_use_archived_run_config(self):
        root = Path(__file__).resolve().parents[1]
        load = runpy.run_path(str(root / 'scripts/08d_extract_subcell_finetune_embeddings.py'))['load_finetuned_model']
        with tempfile.TemporaryDirectory() as temp, patch.dict(load.__globals__, INTERIM_DIR=Path(temp)), \
                patch('torch.load', return_value={}), patch('models.get_models.get_model_dict', side_effect=RuntimeError('dispatched')) as build:
            for family in ('mae', 'vit'):
                for seed in (None, 42, 43, 44):
                    run = Path(temp) / 'subcell_finetune/models' / (family + (f'_s{seed}' if seed else ''))
                    run.mkdir(parents=True)
                    (run / 'best_model_ap.ckpt').touch()
                    (run / 'config.yaml').write_text('model: {archived: true}\ntrain: {}\n')
                    with self.assertRaisesRegex(RuntimeError, 'dispatched'):
                        load(family, 'cpu', seed)
                    build.assert_called_with({'archived': True})
                    (run / 'config.yaml').unlink()
                    with self.assertRaises(FileNotFoundError):
                        load(family, 'cpu', seed)
