"""Recognize the released batch/CellID encoding only with explicit proof."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import polars as pl

from prot_loc_benchmark.workflows.features import (
    QUALIFIED_ID, SOURCE_QUALIFIED_ID, load_artifact, validate_frame, validate_identity, write_artifact,
)
from test_feature_contracts import example_frame, example_spec


def released_frame():
    frame = example_frame().with_columns(pl.concat_str([
        pl.col('Metadata_Plate'), pl.col('Metadata_well_position'),
        pl.col('Metadata_ImageNumber').cast(pl.String), pl.col('Metadata_ObjectNumber').cast(pl.String)
    ], separator='_').alias('Metadata_CellID'))
    return frame.with_columns(pl.concat_str(['Metadata_Batch', 'Metadata_CellID'], separator='/').alias(QUALIFIED_ID),
                              pl.concat_str(['Metadata_Plate', 'Metadata_well_position', pl.col('Metadata_Site').cast(pl.String)], separator='_').alias('Metadata_SiteID'))


class ReleasedIdentityEncoding(unittest.TestCase):
    def test_opt_in_preserves_source_and_matches_physical_hashes_without_changing_values(self):
        frame = released_frame()
        with self.assertRaisesRegex(ValueError, 'Stored batch-qualified'):
            validate_identity(frame)
        result = validate_identity(frame, stored_id_encoding='batch_slash_cell_id')
        expected = validate_identity(frame.drop(QUALIFIED_ID))
        self.assertEqual(result[QUALIFIED_ID].to_list(), expected[QUALIFIED_ID].to_list())
        self.assertEqual(result[SOURCE_QUALIFIED_ID].to_list(), frame[QUALIFIED_ID].to_list())
        self.assertTrue(result.select(c for c in frame.columns if c != QUALIFIED_ID).equals(frame.drop(QUALIFIED_ID)))
        self.assertTrue(validate_identity(result, stored_id_encoding='batch_slash_cell_id').equals(result))

    def test_wrong_source_prefix_physical_cellid_and_canonical_hash_remain_errors(self):
        frame = released_frame()
        for broken in [frame.with_columns(pl.lit('wrong/source').alias(QUALIFIED_ID)),
                       frame.with_columns(pl.lit('fabricated').alias('Metadata_CellID')),
                       frame.with_columns(pl.lit('wrong_site').alias('Metadata_SiteID'))]:
            with self.assertRaises(ValueError):
                validate_identity(broken, stored_id_encoding='batch_slash_cell_id')
        result = validate_identity(frame, stored_id_encoding='batch_slash_cell_id')
        with self.assertRaisesRegex(ValueError, 'Stored batch-qualified'):
            validate_identity(result.with_columns(pl.lit('bad_hash').alias(QUALIFIED_ID)), stored_id_encoding='batch_slash_cell_id')
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            validate_identity(pl.concat([frame, frame.head(1)]), stored_id_encoding='batch_slash_cell_id')

    def test_source_is_not_invented_and_unknown_policy_is_rejected(self):
        frame = released_frame()
        with self.assertRaisesRegex(ValueError, 'require Metadata_CellID'):
            validate_identity(frame.drop('Metadata_CellID'), stored_id_encoding='batch_slash_cell_id')
        with self.assertRaisesRegex(ValueError, 'Unknown stored ID encoding'):
            validate_identity(frame, stored_id_encoding='guess')
        declared = example_spec()
        declared['identity'] = {'stored_id_encoding': 'guess'}
        with self.assertRaisesRegex(ValueError, 'identity.stored_id_encoding'):
            validate_frame(frame, declared)

    def test_explicit_specification_round_trip_retains_source_identity(self):
        declared = example_spec()
        declared['identity'] = {'stored_id_encoding': 'batch_slash_cell_id'}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'artifact'
            write_artifact(released_frame(), declared, root)
            frame, spec, report = load_artifact(root / 'features.parquet', root / 'feature-spec.json')
            self.assertEqual(spec, declared)
            self.assertEqual(report['preserved_source_id_column'], SOURCE_QUALIFIED_ID)
            self.assertEqual(frame.height, 16)
            self.assertEqual(frame[SOURCE_QUALIFIED_ID].to_list(), released_frame()[QUALIFIED_ID].to_list())

    def test_public_spec_and_validate_cli_bind_source_encoding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = root / 'raw.parquet'
            released_frame().write_parquet(frame)
            columns = root / 'columns.json'
            columns.write_text(json.dumps(['embedding_0', 'embedding_1']))
            spec = root / 'spec.json'
            report = root / 'validation.json'
            script = Path(__file__).resolve().parents[1] / 'scripts/20_feature_artifacts.py'
            command = [sys.executable, '-I', '-B', str(script)]
            r = subprocess.run([*command, 'spec', '--features', str(frame), '--model', 'new-model',
                                '--columns-json', str(columns), '--preprocessing', 'none',
                                '--stored-id-encoding', 'batch_slash_cell_id', '--output', str(spec)], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            r = subprocess.run([*command, 'validate', '--features', str(frame), '--spec', str(spec), '--report', str(report)], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(json.loads(report.read_text())['accepted_source_id_encoding'], 'batch_slash_cell_id')
