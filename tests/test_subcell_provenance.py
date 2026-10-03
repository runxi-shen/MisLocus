"""Staging/selection/provenance regression checks, using tiny disposable releases."""
import contextlib
import io
import json
import os
import runpy
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from test_subcell_allele_v2 import make_cohort
from prot_loc_benchmark import cell_crops, provenance
from prot_loc_benchmark.config import ALL_PUBLIC_BATCHES, REPO_ROOT, SUBCELL_CHANNEL_FILES
from prot_loc_benchmark.representations.subcell_manifest import (
    build_manifest, load_preflight, save_json, sha256,
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
    def test_snapshot_includes_optional_local_contract(self):
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
            output = io.StringIO()
            command = str(REPO_ROOT / 'scripts/08a_prepare_subcell.py')
            with patch('sys.argv', [command, 'inspect', '--release', str(release)]), patch(
                    'prot_loc_benchmark.representations.subcell_manifest.release_inventory',
                    return_value=evidence['release']), contextlib.redirect_stdout(output):
                runpy.run_path(command, run_name='__main__')
            self.assertIn('Training classes: 2', output.getvalue())
            self.assertIn('payload integrity NOT certified', output.getvalue())
            with self.assertRaisesRegex(ValueError, 'Hugging Face mirror'):
                build_manifest(release, crops, release / 'forbidden-output')
            path = Path(frame.base_path.iloc[0]) / 'dna.npy'
            stat = path.stat()
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))
            with self.assertRaisesRegex(ValueError, 'changed after preflight'):
                load_preflight(cohort)


if __name__ == '__main__':
    unittest.main()
