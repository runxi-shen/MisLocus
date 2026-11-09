"""Real small workflow execution, hash-checked resume and allocation boundaries."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import polars as pl
from prot_loc_benchmark.workflows.features import QUALIFIED_ID, validate_frame
from prot_loc_benchmark.workflows.runner import ROOT, build_plan, bounded_execute, sha
from test_explicit_feature_evaluations import paired_cells, row, specification


class AnalysisRunnerChecks(unittest.TestCase):
    def test_released_plan_accounts_for_all_scoring_jobs_and_control_exceptions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = build_plan(ROOT / 'configs/analysis/inputs.json', ROOT / 'configs/analysis/settings.json',
                              root / 'data', root / 'output')
            self.assertEqual(plan['scoring_jobs'], {'xgb': 54, 'pa': 54, 'hpa': 7})
            available = [job for job in plan['jobs'] if job['kind'] == 'score' and job.get('task') == 'xgb'
                         and job['control_policy'] == 'available']
            self.assertEqual(len(available), 5)
            self.assertFalse((root / 'data').exists())
            self.assertFalse((root / 'output').exists())
            with self.assertRaises(ValueError):
                build_plan(ROOT / 'configs/analysis/inputs.json', ROOT / 'configs/analysis/settings.json',
                           root / 'data', root / 'output', backend='gpu', gpu_uuid='0')
            with self.assertRaises(ValueError):
                build_plan(ROOT / 'configs/analysis/inputs.json', ROOT / 'configs/analysis/settings.json',
                           root / 'data', root / 'output', cpus=9)

    def fixture(self, root):
        data = root / 'data'; data.mkdir(); spec = root / 'spec.json'
        spec.write_text(json.dumps(specification()))
        entries = []
        vectors = {'A': [1, .01, .1], 'B': [1, .05, .1], 'C': [.01, 1, .1],
                   'D': [.05, 1, .1], 'E': [.02, .03, 1]}
        for batch in ['batchA', 'batchB']:
            frame = paired_cells().drop(QUALIFIED_ID)
            extra = pl.DataFrame([row('map_T4', f'B{i+1:02}', gene, gene, 'disease_wt', 'Exp', 500, vector)
                                  for i, (gene, vector) in enumerate(vectors.items())])
            frame = pl.concat([frame, extra], how='vertical_relaxed').with_columns(Metadata_Batch=pl.lit(batch))
            frame = validate_frame(frame, specification())[0]
            path = data / (batch + '.parquet'); frame.write_parquet(path)
            entries.append(dict(representation='toy', batch=batch, path=path.name, size=path.stat().st_size,
                                sha256=sha(path), spec=str(spec), spec_sha256=sha(spec)))
        annotation = data / 'clinical.parquet'
        pl.DataFrame({'gene_variant': ['G_variant', 'unused'], 'clinvar_clnsig_clean': ['Pathogenic', 'Benign'],
                      'clinvar_clnsig_clean_pp_strict': ['Pathogenic', 'Benign']}).write_parquet(annotation)
        hpa = root / 'hpa.parquet'
        labels = {'A': ['L1'], 'B': ['L1', 'L2'], 'C': ['L2'], 'D': ['L2'], 'E': ['L3']}
        pl.DataFrame([{'Gene': str(i), 'Gene name': gene,
                       **{loc: 1. if loc in labels[gene] else None for loc in ['L1', 'L2', 'L3']}}
                      for i, gene in enumerate(vectors)]).write_parquet(hpa)
        settings = json.loads((ROOT / 'configs/analysis/settings.json').read_text())
        settings.update(settings=[['toy', 'EMBED']], batches=['batchA', 'batchB'],
                        biorep_pairs={'pair': ['batchA', 'batchB']}, predictors=['clinvar'])
        settings['xgb']['available_control_exceptions'] = []
        settings_path = root / 'settings.json'; settings_path.write_text(json.dumps(settings))
        manifest = dict(schema_version=1, repository='fixture/no-network', revision=settings['dataset_revision'],
                        features=entries, clinical_annotation=dict(path=annotation.name, size=annotation.stat().st_size, sha256=sha(annotation)),
                        hpa_annotation=dict(path=str(hpa), sha256=sha(hpa)))
        manifest_path = root / 'inputs.json'; manifest_path.write_text(json.dumps(manifest))
        return manifest_path, settings_path, data

    def test_real_small_workflow_partial_resume_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); manifest, settings, data = self.fixture(root); output = root / 'output'
            command = [sys.executable, '-I', '-B', str(ROOT / 'scripts/23_run_analysis.py'), 'run',
                       '--manifest', str(manifest), '--settings', str(settings), '--data-root', str(data),
                       '--output', str(output), '--workers', '2', '--cpus', '2', '--memory-gib', '8', '--seconds', '120']
            first = subprocess.run([*command, '--stop-after', '1'], capture_output=True, text=True, timeout=30)
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            self.assertEqual(json.loads((output / 'status.json').read_text())['status'], 'partial')
            run = subprocess.run([*command, '--resume'], capture_output=True, text=True, timeout=180)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            state = json.loads((output / 'status.json').read_text())
            self.assertEqual(state['status'], 'complete')
            self.assertEqual(state['fresh_scoring_jobs'], 5)
            self.assertEqual(state['reused_stages'], 1)
            self.assertFalse(state['scientific_qualification'])
            pool = json.loads((output / 'jobs/pool-toy/completed.json').read_text())
            pooled = pl.read_parquet(output / pool['output'] / 'features.parquet')
            self.assertEqual(set(pooled['Metadata_Batch']), {'batchA', 'batchB'})
            self.assertTrue(pooled['Metadata_Plate'].str.ends_with('T4').all())
            self.assertEqual(set(pooled['Metadata_node_type']), {'disease_wt'})
            # Resume verifies every saved byte instead of counting stale completion files.
            again = subprocess.run([*command, '--resume'], capture_output=True, text=True, timeout=90)
            self.assertEqual(again.returncode, 0, again.stdout + again.stderr)
            resumed = json.loads((output / 'status.json').read_text())
            self.assertEqual(resumed['fresh_scoring_jobs'], 0)
            self.assertEqual(resumed['reused_stages'], resumed['planned_stages'])
            source = data / 'batchA.parquet'
            with source.open('ab') as stream:
                stream.write(b'changed')
            bad = subprocess.run([*command, '--resume'], capture_output=True, text=True, timeout=30)
            self.assertNotEqual(bad.returncode, 0)
            failed = json.loads((output / 'status.json').read_text())
            self.assertEqual(failed['status'], 'failed_preserved')
            self.assertEqual(failed['failed_stage'], 'input-toy-batchA')

    def test_wall_limit_terminates_worker_and_preserves_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'attempt'
            binding = dict(workers=1, cpus=1, memory_gib=1, seconds=1, gpu_uuid=None)
            result = bounded_execute([sys.executable, '-I', '-B', '-c', 'import time; time.sleep(30)'], output, binding)
            self.assertEqual(result['status'], 'failed_preserved')
            self.assertEqual(result['termination_reason'], 'wall_limit')
            self.assertTrue((output / 'resource.json').is_file())
