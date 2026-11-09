"""A public MorphEM specification can feed canonical downstream reporting."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import polars as pl

from test_explicit_feature_evaluations import paired_cells, specification


class ReportingModelAlias(unittest.TestCase):
    def test_public_alias_keeps_input_spec_and_binds_canonical_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = paired_cells()
            features = root / 'features.parquet'
            frame.write_parquet(features)
            declaration = specification()
            declaration['model'] = 'morphem'
            spec = root / 'feature-spec.json'
            spec.write_text(json.dumps(declaration))
            script = Path(__file__).resolve().parents[1] / 'scripts/21_evaluate_features.py'
            command = [sys.executable, '-I', '-B', str(script), 'xgb', '--features', str(features), '--spec', str(spec)]
            output = root / 'canonical'
            run = subprocess.run([*command, '--reporting-model', 'vit', '--output', str(output)], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            receipt = json.loads((output / 'evaluation.json').read_text())
            self.assertEqual(receipt['model'], 'vit')
            self.assertEqual(receipt['declared_feature_model'], 'morphem')
            self.assertEqual(receipt['feature_specification'], declaration)
            self.assertEqual(pl.read_parquet(output / 'scores.parquet')['representation'].unique().to_list(), ['vit'])
            bad = root / 'wrong-model'
            rejected = subprocess.run([*command, '--reporting-model', 'cellprofiler', '--output', str(bad)], capture_output=True, text=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn('same representation', rejected.stderr)
            self.assertFalse(bad.exists())
