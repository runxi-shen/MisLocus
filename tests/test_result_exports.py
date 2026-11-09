"""Real report/export/plot CLI integration, including rejection of changed data."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import polars as pl
from test_clinical_workflow import scores

ROOT = Path(__file__).resolve().parents[1]


class ResultExportChecks(unittest.TestCase):
    def test_report_export_and_plot_preserve_saved_values_and_reject_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); annotation = root / 'annotations.parquet'
            pl.DataFrame({'gene_variant': ['P1', 'P2', 'B1', 'B2'],
                          'clinvar_clnsig_clean': ['Pathogenic'] * 2 + ['Benign'] * 2,
                          'clinvar_clnsig_clean_pp_strict': ['Pathogenic'] * 2 + ['Benign'] * 2}).write_parquet(annotation)
            pairs = root / 'pairs.json'; pairs.write_text(json.dumps({'pair': ['A', 'B']}))
            report = root / 'report'
            args = ['--task', 'xgb', '--setting', 'model', 'EMBED', '--annotations', str(annotation),
                    '--biorep-pairs', str(pairs), '--output', str(report)]
            for batch in ['A', 'B']:
                path = root / (batch + '.parquet')
                scores().filter(pl.col('batch') == batch).write_parquet(path)
                args += ['--input', 'model', batch, str(path)]
            commands = [
                ['22_report_clinical.py', *args],
                ['25_export_tables.py', '--report', str(report), '--output', str(root / 'tables')],
                ['28_plot_results.py', '--clinical-report', str(report), '--output', str(root / 'plots')],
            ]
            for script, *arguments in commands:
                result = subprocess.run([sys.executable, '-I', '-B', str(ROOT / 'scripts' / script), *arguments],
                                        capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual((root / 'tables/binary-statistics.csv').read_bytes(), (report / 'binary-statistics.csv').read_bytes())
            figures = json.loads((root / 'plots/figures.json').read_text())
            self.assertEqual(figures['status'], 'complete')
            self.assertTrue(any(item['name'] == 'clinical-f1' for item in figures['figures']))
            part = pl.read_csv(root / 'plots/clinical-f1-data.csv')
            self.assertEqual(part['F1'].to_list(), [1., 1.])
            with (report / 'binary-statistics.csv').open('a') as stream:
                stream.write('\n')
            for script, flag in [('25_export_tables.py', '--report'), ('28_plot_results.py', '--clinical-report')]:
                result = subprocess.run([sys.executable, '-I', '-B', str(ROOT / 'scripts' / script), flag, str(report),
                                         '--output', str(root / (script + '-bad'))], capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
