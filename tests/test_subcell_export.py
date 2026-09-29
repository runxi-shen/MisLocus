"""Real crop/preprocessing/Parquet/receipt flow; only the model backend is tiny."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import yaml

from test_subcell_provenance import stage_fixture
from prot_loc_benchmark import cell_crops, provenance
from prot_loc_benchmark.config import ALL_PUBLIC_BATCHES
from prot_loc_benchmark.representations import subcell_extract as export


class TinyEncoder(torch.nn.Module):
    def forward(self, images, **kwargs):
        return SimpleNamespace(last_hidden_state=images.mean((2, 3))[:, None, :])


class TinyPool(torch.nn.Module):
    def forward(self, encoded):
        return encoded[:, 0].repeat(1, 384), None


class ExportChecks(unittest.TestCase):
    def test_external_ledger_completion_last_and_diagnostic_isolation(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release, crops, cohort = stage_fixture(root)
            weights = root / 'weights.pth'
            torch.save({}, weights)
            verification = root / 'crop-check.json'
            provenance.save_json(verification, cell_crops.verify_crops(crops))
            ledger = root / 'control/provenance.json'
            original = provenance.PROVENANCE_LOG

            def run(output, extra=()):
                args = ['extract', '--preflight', str(cohort), '--family', 'vit', '--split', 'all',
                        '--frozen-weights', str(weights), '--weights-sha256', provenance.sha256(weights),
                        '--device', 'cpu', '--batch-size', '5', '--output', str(output),
                        '--provenance-log', str(ledger), '--crop-verification', str(verification), *extra]
                with patch('sys.argv', args), patch.object(export, 'get_model_dict',
                        side_effect=lambda config: {'vit_model': TinyEncoder(), 'pool_model': TinyPool()}), \
                        patch.object(provenance, 'PROVENANCE_LOG', original), \
                        patch('yaml.safe_load', return_value={'pretrained_sha256': provenance.sha256(weights)}):
                    export.main(frozen=True)

            output = root / 'success'
            run(output)
            receipt = json.loads((output / 'extraction.json').read_text())
            self.assertEqual(receipt['artifact_kind'], 'raw_embeddings')
            self.assertEqual(receipt['status'], 'complete')
            self.assertEqual(receipt['crop_verification_sha256'], provenance.sha256(verification))
            self.assertEqual(receipt['provenance_log'], str(ledger))
            self.assertEqual(len(json.loads(ledger.read_text())['runs']), 1)
            for batch in ALL_PUBLIC_BATCHES:
                file = output / batch / 'embeddings.parquet'
                frame = pd.read_parquet(file)
                self.assertEqual(len(frame), 24)
                self.assertEqual(frame.filter(like='SubCell_').shape, (24, 1536))
                self.assertTrue(np.isfinite(frame.filter(like='SubCell_').to_numpy()).all())
                self.assertEqual(receipt['outputs'][batch]['sha256'], provenance.sha256(file))
                self.assertEqual(frame.Metadata_Split.value_counts().to_dict(), {'train': 18, 'val': 4, 'test': 2})
            digest = provenance.sha256(output / 'extraction.json')
            with self.assertRaises(FileExistsError):
                run(output)
            self.assertEqual(provenance.sha256(output / 'extraction.json'), digest)
            pilot = root / 'pilot'
            run(pilot, ['--pilot-cells-per-split', '1'])
            self.assertFalse((pilot / 'extraction.json').exists())
            pilot_receipt = json.loads((pilot / 'pilot.json').read_text())
            self.assertEqual(pilot_receipt['artifact_kind'], 'diagnostic_embeddings')
            self.assertTrue(all(info['cells'] == 3 for info in pilot_receipt['outputs'].values()))
            # Live/source-archive disagreement must fail before publishing features.
            drifted = root / 'source-drift'
            with patch.object(provenance, 'code_fingerprint', return_value='0' * 64), \
                    self.assertRaisesRegex(ValueError, 'Source changed since capture'):
                run(drifted)
            self.assertFalse((drifted / 'extraction.json').exists())
            self.assertFalse(list(drifted.glob('*/embeddings.parquet')))
            # Ledger I/O failure must not publish a successful completion receipt.
            ledger.write_text('invalid json')
            failed = root / 'failed'
            with self.assertRaises(json.JSONDecodeError):
                run(failed)
            self.assertFalse((failed / 'extraction.json').exists())
            with self.assertRaisesRegex(ValueError, 'protected'):
                run(root / 'forbidden', ['--provenance-log', str(release / 'ledger.json')])

    def test_frozen_family_and_relocated_checkpoint_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / 'original-run'
            original.mkdir()
            alias = root / 'alias'
            alias.symlink_to(original, target_is_directory=True)
            evidence = {'release_root': str(root / 'release'), 'crops_root': str(root / 'crops')}
            identity = {'kind': 'production', 'config': {'family': 'vit', 'output': str(original)}, 'data': evidence}
            checkpoint = {'allele_v2': {'identity': identity}, 'state_dict': {}}
            common = ['extract', '--preflight', str(root / 'cohort'), '--device', 'cpu']
            with patch.object(export, 'load_preflight', return_value=(None, None, None, evidence)), \
                    patch.object(export, 'InferenceWrapper', side_effect=RuntimeError('reached model')):
                for family, other in (('mae', 'vit'), ('vit', 'mae')):
                    config = provenance.REPO_ROOT / f'configs/subcell_finetune_{other}_s42.yaml'
                    wrong = yaml.safe_load(config.read_text())['pretrained_sha256']
                    argv = common + ['--family', family, '--frozen-weights', str(root / 'weights'),
                                     '--weights-sha256', wrong, '--output', str(root / 'export')]
                    with patch('sys.argv', argv), self.assertRaisesRegex(ValueError, 'Frozen weights'):
                        export.main(frozen=True)
                selected = root / 'receipts/selection.json'
                for path in (root / 'archive/best.ckpt', root / 'relocated/models/best.ckpt'):
                    argv = common + ['--family', 'vit', '--checkpoint', str(path), '--selection', str(selected)]
                    with patch.object(torch, 'load', return_value=checkpoint), \
                            patch.object(export, 'verify_selection', return_value={'sha256': 'fixture', 'pass': 10}):
                        with patch('sys.argv', argv + ['--output', str(root / 'exports/new')]), \
                                self.assertRaisesRegex(RuntimeError, 'reached model'):
                            export.main()
                        for protected in (original, alias, path.parent, selected.parent):
                            for flags in (['--output', str(protected / 'new')],
                                          ['--output', str(root / 'exports/new'), '--provenance-log', str(protected / 'ledger.json')]):
                                with patch('sys.argv', argv + flags), self.assertRaisesRegex(ValueError, 'protected'):
                                    export.main()


if __name__ == '__main__':
    unittest.main()
