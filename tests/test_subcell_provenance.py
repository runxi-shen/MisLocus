"""Staging/selection/provenance regression checks, using tiny disposable releases."""
import contextlib
import io
import json
import os
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping

from test_subcell_allele_v2 import make_cohort
from prot_loc_benchmark import cell_crops, provenance
from prot_loc_benchmark.config import ALL_PUBLIC_BATCHES, REPO_ROOT, SUBCELL_CHANNEL_FILES
from prot_loc_benchmark.representations.subcell_manifest import (
    build_manifest, load_preflight, save_json, sha256,
)
from prot_loc_benchmark.representations.subcell_run import (
    AlleleCheckpoint, capture_source, require_resumable, runtime_info, verify_selection, verify_source,
)


def stage_fixture(root):
    """Exercise real tar extraction + metadata/header preflight, mock only git inventory."""
    payload, release, crops, cohort = [root / name for name in ('payload', 'release', 'crops', 'cohort')]
    payload.mkdir()
    frame, _ = make_cohort(payload, count=2)
    frame.loc[frame.cell_idx == 11, 'Metadata_Plate'] = frame.loc[frame.cell_idx == 11, 'Metadata_Plate'].str.replace('T3', 'T4')
    frame.loc[frame.cell_idx == 11, 'Metadata_CellID'] = frame.loc[frame.cell_idx == 11, 'Metadata_CellID'].str.replace('T3', 'T4')
    for allele, group in frame.groupby('Metadata_gene_allele'):
        meta = pd.read_parquet(payload / allele / 'metadata.parquet')
        meta['Metadata_Plate'] = group.Metadata_Plate.to_numpy()
        meta['Metadata_CellID'] = group.Metadata_CellID.to_numpy()
        meta.to_parquet(payload / allele / 'metadata.parquet', index=False)
    (release / 'manifest').mkdir(parents=True)
    inventory = {'remote': 'synthetic-only', 'revision': 'test-fixture', 'files': {}}
    for batch in ALL_PUBLIC_BATCHES:
        manifest = release / f'manifest/manifest_Batch_{batch.rsplit("_", 1)[1]}.parquet'
        frame[[c for c in frame if c.startswith('Metadata_')]].iloc[::-1].to_parquet(manifest, index=False)
        shard = release / 'single_cell_crops' / batch / 'shard-00.tar.gz'
        shard.parent.mkdir(parents=True)
        with tarfile.open(shard, 'w:gz') as archive:
            for allele in sorted(frame.Metadata_gene_allele.unique()):
                for name in [*SUBCELL_CHANNEL_FILES, 'metadata.parquet']:
                    archive.add(payload / allele / name, arcname=f'{allele}/{name}')
        plate_metadata = release / 'representations/cellprofiler' / batch / 'features.parquet'
        plate_metadata.parent.mkdir(parents=True)
        plates = frame[['Metadata_Plate']].drop_duplicates().copy()
        plates['Metadata_plate_map_name'] = 'synthetic_P1'
        plates.to_parquet(plate_metadata, index=False)
        for path in (manifest, shard, plate_metadata):
            inventory['files'][str(path.relative_to(release))] = {'sha256': sha256(path), 'size': path.stat().st_size}
    raw_inventory = {**inventory, 'files': {p: info for p, info in inventory['files'].items()
                                           if p.startswith(('manifest/', 'single_cell_crops/'))}}
    before = {p: sha256(release / p) for p in inventory['files']}
    with patch.object(cell_crops, 'release_inventory', return_value=raw_inventory), patch(
            'prot_loc_benchmark.representations.subcell_manifest.release_inventory', return_value=inventory), patch.object(
            provenance, 'PROVENANCE_LOG', root / 'ledger.json'):
        cell_crops.extract(release, crops)
        build_manifest(release, crops, cohort)
        # The earlier full-inventory receipt remains reusable without re-extraction.
        receipt_path = crops / 'extraction.json'
        receipt = json.loads(receipt_path.read_text())
        receipt['release'] = inventory
        save_json(receipt_path, receipt)
        legacy = root / 'legacy-cohort'
        build_manifest(release, crops, legacy)
        for name in ('manifest.parquet', 'class_index.json', 'validation_ids.json'):
            assert sha256(cohort / name) == sha256(legacy / name)
        receipt['release'] = raw_inventory
        save_json(receipt_path, receipt)
    assert before == {p: sha256(release / p) for p in inventory['files']}, 'Release checkout was modified'
    return release, crops, cohort


class ProvenanceChecks(unittest.TestCase):
    def test_snapshot_does_not_require_uncommitted_plan(self):
        from prot_loc_benchmark import provenance as snapshots
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('pyproject.toml', 'pixi.lock'):
                (root / name).write_text('test')
            with patch.object(snapshots, 'REPO_ROOT', root):
                without_plan = snapshots.code_fingerprint()
                plan = root / 'docs/plans/subcell-allele-rybg-finetuning.md'
                plan.parent.mkdir(parents=True)
                plan.write_text('locally retained contract')
                self.assertNotEqual(snapshots.code_fingerprint(), without_plan)
                self.assertIn(plan, snapshots.source_files())

    def test_frozen_and_adapted_cli_are_separate(self):
        from prot_loc_benchmark.representations.subcell_extract import main
        common = ['extract', '--preflight', 'unused', '--family', 'mae', '--output', 'unused']
        for frozen, wrong_source in ((True, ['--checkpoint', 'unused']),
                                     (False, ['--frozen-weights', 'unused', '--weights-sha256', 'unused'])):
            with patch('sys.argv', common + wrong_source), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as failure:
                    main(frozen=frozen)
            self.assertEqual(failure.exception.code, 2)

    def test_preflight_output_cannot_enter_either_input_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release, crops = root / 'release', root / 'crops'
            release.mkdir()
            crops.mkdir()
            for protected in (release, crops):
                alias = root / (protected.name + '-alias')
                alias.symlink_to(protected, target_is_directory=True)
                for output in (protected, protected / 'nested/cohort', alias / 'cohort'):
                    with self.subTest(output=output), patch(
                            'prot_loc_benchmark.representations.subcell_manifest.release_inventory') as inventory:
                        before = set(root.rglob('*'))
                        with self.assertRaisesRegex(ValueError, 'Preflight must not write'):
                            build_manifest(release, crops, output)
                        inventory.assert_not_called()
                        self.assertEqual(set(root.rglob('*')), before)

    def test_preflight_rejects_receipt_or_source_edits_during_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release, crops, cohort = stage_fixture(root)
            inventory = json.loads((cohort / 'preflight.json').read_text())['release']
            receipt_path = crops / 'extraction.json'
            original_receipt = receipt_path.read_bytes()
            source = root / 'editable-source'
            source.mkdir()
            read = pd.read_parquet
            for change in ('receipt', 'source'):
                for name in ('pixi.lock', 'pyproject.toml'):
                    (source / name).write_text('before')

                def read_then_edit(*args, **kwargs):
                    result = read(*args, **kwargs)
                    if change == 'receipt':
                        receipt_path.write_text('{"release": {"revision": "other"}, "files": {}}')
                    else:
                        (source / 'pixi.lock').write_text('after')
                    return result

                output = root / ('changed-' + change)
                with self.subTest(change=change), patch.object(provenance, 'REPO_ROOT', source), patch.object(
                        provenance.subprocess, 'check_output', return_value='fixture'), patch.object(
                        pd, 'read_parquet', side_effect=read_then_edit), patch(
                        'prot_loc_benchmark.representations.subcell_manifest.release_inventory', return_value=inventory):
                    with self.assertRaisesRegex(ValueError, 'Extraction receipt changed|recorded code identity'):
                        build_manifest(release, crops, output)
                    self.assertFalse((output / 'preflight.json').exists())
                receipt_path.write_bytes(original_receipt)

    def test_separate_extraction_preflight_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release, crops, cohort = stage_fixture(root)
            frame, classes, validation, evidence = load_preflight(cohort)
            self.assertEqual(len(frame), 144)
            self.assertEqual(len(classes), 2)
            self.assertEqual(set(frame.Metadata_plate_map_name), {'synthetic_P1'})
            self.assertEqual(set(frame.batch_id), set(ALL_PUBLIC_BATCHES))
            self.assertEqual(len(validation), 24)
            self.assertEqual(set(frame.loc[frame.cell_id.isin(validation), 'split']), {'val'})
            self.assertEqual(sha256(cohort / 'source.tar.gz'), evidence['source_archive_sha256'])
            self.assertFalse((release / 'extraction.json').exists())
            with self.assertRaisesRegex(ValueError, 'Hugging Face mirror'):
                build_manifest(release, crops, release / 'forbidden-output')
            path = Path(frame.base_path.iloc[0]) / 'dna.npy'
            stat = path.stat()
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))
            with self.assertRaisesRegex(ValueError, 'changed after preflight'):
                load_preflight(cohort)

    def test_source_snapshot_and_authoritative_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture_source(root)
            source = json.loads((root / 'source.json').read_text())
            self.assertEqual(source['archive_sha256'], sha256(root / 'source.tar.gz'))
            verify_source(root, source['code_sha256'])
            with self.assertRaisesRegex(ValueError, 'recorded code identity'):
                verify_source(root, '0' * 64)
            with tarfile.open(root / 'source.tar.gz') as archive:
                names = archive.getnames()
                self.assertIn('scripts/08d_extract_subcell_finetune_embeddings.py', names)
                self.assertIn('src/prot_loc_benchmark/representations/subcell_training.py', names)
                self.assertNotIn('data', names)
            identity = {'config': {'output': str(root)}}
            selector = AlleleCheckpoint(root)
            selector.best_model_score = torch.tensor(.123456789, dtype=torch.float64)
            checkpoint = {'global_step': 970, 'epoch': 9, 'allele_v2': {'identity': identity},
                          'callbacks': {selector.state_key: selector.state_dict()}}
            path = root / 'best.ckpt'
            torch.save(checkpoint, path)
            receipt = {'identity': identity, 'sha256': sha256(path), 'pass': 10, 'global_step': 970,
                       'macro_ap': .123456789, 'metric_dtype': 'float64'}
            save_json(root / 'selection.json', receipt)
            verify_selection(path, checkpoint)
            archived_receipt = root / 'archived-selection.json'
            for change in ({'macro_ap': .987654321}, {'metric_dtype': 'float32'}):
                save_json(archived_receipt, {**receipt, **change})
                with self.assertRaisesRegex(ValueError, 'native checkpoint selector'):
                    verify_selection(path, checkpoint, archived_receipt)
            for native_score in (None, torch.tensor(.123456789), torch.tensor(float('nan'), dtype=torch.float64)):
                altered = {**checkpoint, 'callbacks': {selector.state_key: {'best_model_score': native_score}}}
                with self.assertRaisesRegex(ValueError, 'native checkpoint selector'):
                    verify_selection(path, altered)
            save_json(archived_receipt, receipt)
            verify_selection(path, checkpoint, archived_receipt)
            path.write_bytes(path.read_bytes() + b'changed')
            with self.assertRaisesRegex(ValueError, 'authoritative'):
                verify_selection(path, checkpoint)

    def test_terminal_production_resume_and_backend_metadata(self):
        from test_subcell_allele_v2 import tiny_components
        from prot_loc_benchmark.representations.subcell_training import SubCellAlleleModule
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identity = {'kind': 'production'}
            checkpoint = {'global_step': 970, 'allele_v2': {'identity': identity, 'next_pass': 10,
                          'steps_per_pass': 97, 'rng_states': []}, 'callbacks': {}}
            module = SubCellAlleleModule(tiny_components('vit'), ['a', 'b'], identity, root, augment=False)
            module.on_load_checkpoint(checkpoint)  # An interrupted nonterminal pass can resume.
            marker = root / 'attempts/first/completed.json'
            marker.parent.mkdir(parents=True)
            marker.write_text('completed sentinel')
            with self.assertRaisesRegex(ValueError, 'already completed'):
                module.on_load_checkpoint(checkpoint)
            self.assertEqual(marker.read_text(), 'completed sentinel')
            marker.unlink()
            checkpoint['allele_v2']['next_pass'] = 100
            checkpoint['global_step'] = 9700
            with self.assertRaisesRegex(ValueError, '100-pass horizon'):
                module.on_load_checkpoint(checkpoint)
            checkpoint['allele_v2']['next_pass'] = 60
            checkpoint['global_step'] = 5820
            stopping = EarlyStopping('val/macro_ap', patience=5, mode='max')
            for stopped_epoch, wait_count in ((59, 1), (0, 5)):
                state = {**stopping.state_dict(), 'stopped_epoch': stopped_epoch, 'wait_count': wait_count}
                checkpoint['callbacks'] = {stopping.state_key: state}
                with self.assertRaisesRegex(ValueError, 'already early-stopped'):
                    module.on_load_checkpoint(checkpoint)
            checkpoint['allele_v2']['identity'] = {'kind': 'release_validation'}
            require_resumable(root, checkpoint)  # Diagnostic stop/resume gates remain possible.
        runtime = runtime_info()
        self.assertEqual(runtime['cudnn_deterministic'], torch.backends.cudnn.deterministic)
        self.assertEqual(runtime['deterministic_algorithms'], torch.are_deterministic_algorithms_enabled())
        self.assertEqual(runtime['deterministic_warn_only'], torch.is_deterministic_algorithms_warn_only_enabled())
        self.assertEqual(runtime['cublas_workspace_config'], os.environ.get('CUBLAS_WORKSPACE_CONFIG'))

    def test_native_selector_ties_small_improvements_and_early_stopping(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = AlleleCheckpoint(directory)
            trainer = SimpleNamespace(strategy=SimpleNamespace(reduce_boolean_decision=lambda value: value))
            checkpoint.best_k_models = {'best': torch.tensor(.123456789, dtype=torch.float64)}
            checkpoint.kth_value = checkpoint.best_k_models['best']
            checkpoint.kth_best_model_path = 'best'
            self.assertFalse(checkpoint.check_monitor_top_k(trainer, torch.tensor(.123456789, dtype=torch.float64)))
            self.assertTrue(checkpoint.check_monitor_top_k(trainer, torch.tensor(.123456790, dtype=torch.float64)))
            stopping = EarlyStopping('val/macro_ap', mode='max', min_delta=.001, patience=5)
            self.assertFalse(stopping._evaluate_stopping_criteria(torch.tensor(.5, dtype=torch.float64))[0])
            for value in (.5001, .5002, .5003):
                self.assertFalse(stopping._evaluate_stopping_criteria(torch.tensor(value, dtype=torch.float64))[0])
            resumed = EarlyStopping('val/macro_ap', mode='max', min_delta=.001, patience=5)
            resumed.load_state_dict(stopping.state_dict())
            self.assertFalse(resumed._evaluate_stopping_criteria(torch.tensor(.5004, dtype=torch.float64))[0])
            self.assertTrue(resumed._evaluate_stopping_criteria(torch.tensor(.5005, dtype=torch.float64))[0])


if __name__ == '__main__':
    unittest.main()
