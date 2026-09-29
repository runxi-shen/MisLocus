"""Exercise interrupted versus finished production resume through real CPU Lightning.

Tiny synthetic architecture/data only; missing real data/config bindings prevent
these test checkpoints from being accepted by production embedding extraction.
"""
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback, EarlyStopping

from test_subcell_allele_v2 import make_cohort, tiny_components
from prot_loc_benchmark import provenance
from prot_loc_benchmark.preprocessing.subcell import SubCellPreprocessor
from prot_loc_benchmark.representations.subcell_finetune import fixed_validation
from prot_loc_benchmark.representations.subcell_run import AlleleCheckpoint, require_resumable, verify_selection
from prot_loc_benchmark.representations.subcell_training import AlleleDataModule, SubCellAlleleModule


class EndAttempt(Callback):
    def __init__(self, *, interrupt=False):
        self.interrupt = interrupt

    def on_train_epoch_end(self, trainer, module):
        if self.interrupt and trainer.current_epoch == 9:
            raise RuntimeError('Synthetic interruption after saved pass 10')
        if not self.interrupt and trainer.current_epoch == 19:
            trainer.should_stop = True


class Plateau(SubCellAlleleModule):
    def on_validation_epoch_end(self):
        self.validation_outputs.clear()
        self.log('val/macro_ap', torch.tensor(.5, dtype=torch.float64))


class InterruptAtPass(Callback):
    def __init__(self, sampling_pass):
        self.sampling_pass = sampling_pass

    def on_train_epoch_end(self, trainer, module):
        if trainer.current_epoch + 1 == self.sampling_pass:
            raise RuntimeError('Synthetic interruption before completion marker')


class ProductionResumeChecks(unittest.TestCase):
    def test_ledger_failure_cannot_publish_fit_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            module = SubCellAlleleModule(tiny_components('vit'), ['a', 'b'], {}, root)
            module.trainer = SimpleNamespace(is_global_zero=True, global_step=1)
            module.attempt_dir = root / 'attempts/test'
            module.attempt_dir.mkdir(parents=True)
            ledger = root / 'ledger.json'
            ledger.write_text('invalid json')
            with patch.object(provenance, 'PROVENANCE_LOG', ledger):
                with self.assertRaises(json.JSONDecodeError):
                    module.on_fit_end()
                self.assertFalse((module.attempt_dir / 'completed.json').exists())
                ledger.unlink()
                module.on_fit_end()
                self.assertEqual(json.loads((module.attempt_dir / 'completed.json').read_text())['status'],
                                 'fit_completed')
                self.assertEqual(len(json.loads(ledger.read_text())['runs']), 1)

    def test_plateau_saves_latest_state_and_terminal_interruption_cannot_resume(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame, classes = make_cohort(root)
            identity = {'kind': 'production', 'test_only': 'plateau-resume'}

            def fit(output, *, interrupt=None, resume=None):
                output.mkdir(exist_ok=True)
                L.seed_everything(42, workers=True)
                module = Plateau(tiny_components('vit'), sorted(classes), identity, output, augment=False)
                module.preprocess = SubCellPreprocessor(scale_factor=1, output_size=32)
                trainer = L.Trainer(accelerator='cpu', devices=1, max_epochs=100,
                                    default_root_dir=output, check_val_every_n_epoch=10,
                                    use_distributed_sampler=False, precision='32-true', gradient_clip_val=1.,
                                    num_sanity_val_steps=0, logger=False, enable_progress_bar=False,
                                    enable_model_summary=False,
                                    callbacks=[InterruptAtPass(interrupt), AlleleCheckpoint(output),
                                               EarlyStopping('val/macro_ap', mode='max', patience=5,
                                                             min_delta=.001, check_on_train_epoch_end=False)])
                data = AlleleDataModule(frame, classes, fixed_validation(frame), 42, workers=0)
                if interrupt is None:
                    trainer.fit(module, datamodule=data, ckpt_path=resume)
                else:
                    with self.assertRaisesRegex(RuntimeError, 'Synthetic interruption'):
                        trainer.fit(module, datamodule=data, ckpt_path=resume)
                return module.state_dict()

            output = root / 'interrupted'
            with patch.object(provenance, 'PROVENANCE_LOG', root / 'ledger.json'):
                expected = fit(root / 'full')
                fit(output, interrupt=20)
                last = output / 'models/last.ckpt'
                saved = torch.load(last, map_location='cpu', weights_only=False)
                self.assertEqual(saved['epoch'] + 1, 20)
                self.assertEqual(saved['global_step'], 20)
                self.assertEqual(saved['lr_schedulers'][0]['last_epoch'], 20)
                stopping = next(v for v in saved['callbacks'].values() if 'stopped_epoch' in v)
                self.assertEqual(stopping['wait_count'], 1)
                require_resumable(output, saved)
                best = output / 'models/best_model_ap.ckpt'
                best_hash = provenance.sha256(best)
                selection_hash = provenance.sha256(output / 'selection.json')
                actual = fit(output, interrupt=60, resume=last)
                for name, tensor in expected.items():
                    self.assertTrue(torch.equal(tensor, actual[name]), name)
                self.assertFalse(list(output.glob('attempts/*/completed.json')))
                saved = torch.load(last, map_location='cpu', weights_only=False)
                self.assertEqual(saved['epoch'] + 1, 60)
                stopping = next(v for v in saved['callbacks'].values() if 'stopped_epoch' in v)
                self.assertEqual(stopping['wait_count'], 5)
                self.assertEqual(stopping['stopped_epoch'], 59)
                with self.assertRaisesRegex(ValueError, 'already early-stopped'):
                    require_resumable(output, saved)
                self.assertEqual(provenance.sha256(best), best_hash)
                self.assertEqual(provenance.sha256(output / 'selection.json'), selection_hash)
                selected = verify_selection(best, torch.load(best, map_location='cpu', weights_only=False),
                                            output / 'selection.json')
                self.assertEqual(selected['pass'], 10)

    def test_interruption_resumes_but_finished_run_is_unchanged_on_rejection(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame, classes = make_cohort(root)
            output = root / 'run'
            output.mkdir()
            identity = {'kind': 'production', 'test_only': 'terminal-resume-guard'}

            def fit(*, interrupt=False, resume=None):
                L.seed_everything(42, workers=True)
                module = SubCellAlleleModule(tiny_components('vit'), sorted(classes), identity, output, augment=False)
                module.preprocess = SubCellPreprocessor(scale_factor=1, input_size=128, output_size=32)
                trainer = L.Trainer(accelerator='cpu', devices=1, max_epochs=100,
                                    default_root_dir=output, check_val_every_n_epoch=10,
                                    use_distributed_sampler=False, precision='32-true', gradient_clip_val=1.,
                                    num_sanity_val_steps=0, logger=False, enable_progress_bar=False,
                                    enable_model_summary=False,
                                    callbacks=[EndAttempt(interrupt=interrupt), AlleleCheckpoint(os.path.relpath(output)),
                                               EarlyStopping('val/macro_ap', mode='max', patience=5,
                                                             min_delta=.001, check_on_train_epoch_end=False)])
                data = AlleleDataModule(frame, classes, fixed_validation(frame), 42, workers=0)
                trainer.fit(module, datamodule=data, ckpt_path=resume)
                return trainer.global_step

            with patch.object(provenance, 'PROVENANCE_LOG', root / 'ledger.json'):
                with self.assertRaisesRegex(RuntimeError, 'Synthetic interruption'):
                    fit(interrupt=True)
                last = output / 'models/last.ckpt'
                self.assertTrue(last.exists())
                self.assertFalse(list(output.glob('attempts/*/completed.json')))
                self.assertEqual(fit(resume=last), 20)
                self.assertEqual(len(list(output.glob('attempts/*/completed.json'))), 1)
                best = output / 'models/best_model_ap.ckpt'
                saved = torch.load(best, map_location='cpu', weights_only=False)
                verify_selection(best, saved, output / 'selection.json')
                before = {str(p.relative_to(output)): provenance.sha256(p) for p in output.rglob('*') if p.is_file()}
                with self.assertRaisesRegex(ValueError, 'already completed'):
                    fit(resume=last)
                self.assertEqual(before, {str(p.relative_to(output)): provenance.sha256(p)
                                          for p in output.rglob('*') if p.is_file()})


if __name__ == '__main__':
    unittest.main()
