"""Small full-command integration; production limits belong to the executor."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import polars as pl
from prot_loc_benchmark.workflows.features import QUALIFIED_ID, validate_frame, sha256 as sha
from prot_loc_benchmark.workflows.reproduce import ROOT, plan_steps, execute
from test_explicit_feature_evaluations import paired_cells, row, specification

def fixture(root):
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


def executor(root):
    trace = root / 'executor-trace.jsonl'
    forward = root / 'executor.py'
    forward.write_text('import json, subprocess, sys\nfrom pathlib import Path\n'
        + 'command = sys.argv[sys.argv.index("--") + 1:]\n'
        + 'with Path(' + repr(str(trace)) + ').open("a") as stream:\n'
        + '    stream.write(json.dumps(command) + "\\n")\n'
        + 'raise SystemExit(subprocess.run(command).returncode)\n')
    config = root / 'executor.json'
    config.write_text(json.dumps({'cpu': [sys.executable, '-I', '-B', str(forward), '--job', '{job}', '--']}))
    return config, trace


class ReproductionWorkflowChecks(unittest.TestCase):
    def test_released_plan_has_all_115_scores_and_no_side_effects(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            steps = plan_steps(ROOT / 'configs/analysis/inputs.json', ROOT / 'configs/analysis/settings.json',
                               root / 'data', root / 'output')
            names = [step['name'] for step in steps]
            self.assertEqual(sum(name.startswith('xgb-') for name in names), 54)
            self.assertEqual(sum(name.startswith('pa-') for name in names), 54)
            self.assertEqual(sum(name.startswith('hpa-') and not name.startswith('hpa-plot-') for name in names), 7)
            self.assertFalse((root / 'data').exists())
            self.assertFalse((root / 'output').exists())

    def test_failure_stops_and_preserves_log_with_no_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); config, trace = executor(root)
            steps = [dict(name='failed', executor='cpu', command=[sys.executable, '-I', '-B', '-c', 'raise SystemExit(7)']),
                     dict(name='unreached', executor='cpu', command=[sys.executable, '-c', 'raise SystemExit(0)'])]
            with self.assertRaises(subprocess.CalledProcessError):
                execute(steps, root / 'output', config)
            self.assertEqual(len(trace.read_text().splitlines()), 1)
            self.assertEqual(json.loads((root / 'output/failure.json').read_text())['exit_status'], 7)
            self.assertFalse((root / 'output/completion.json').exists())

    def test_small_complete_workflow_uses_executor_and_rejects_changed_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); manifest, settings, data = fixture(root); config, trace = executor(root)
            output = root / 'output'
            command = [sys.executable, '-I', '-B', str(ROOT / 'scripts/23_run_analysis.py'), 'run',
                '--manifest', str(manifest), '--settings', str(settings), '--data-root', str(data),
                '--output', str(output), '--executor-config', str(config), '--cpus', '2', '--workers', '2',
                '--memory-gib', '16', '--seconds', '120']
            run = subprocess.run(command, capture_output=True, text=True, timeout=180)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            complete = json.loads((output / 'completion.json').read_text())
            self.assertEqual(complete['status'], 'complete')
            calls = [json.loads(line) for line in trace.read_text().splitlines()]
            self.assertEqual(len(calls), complete['stages'])
            self.assertEqual(sum(any(arg.endswith('21_evaluate_features.py') for arg in call) for call in calls), 5)
            self.assertFalse(json.loads((output / 'admission.json').read_text())['downloaded'])
            pooled = pl.read_parquet(output / 'pooled/toy/features.parquet')
            self.assertEqual(set(pooled['Metadata_Batch']), {'batchA', 'batchB'})
            self.assertTrue(pooled['Metadata_Plate'].str.ends_with('T4').all())
            self.assertEqual(set(pooled['Metadata_node_type']), {'disease_wt'})
            self.assertTrue((output / 'figures/xgb/figures.json').is_file())
            rerun = subprocess.run(command, capture_output=True, text=True, timeout=20)
            self.assertNotEqual(rerun.returncode, 0)
            self.assertEqual(len(trace.read_text().splitlines()), len(calls))
            source = output / 'scores/xgb/toy/EMBED/batchA'
            with (source / 'controls.parquet').open('ab') as stream:
                stream.write(b'changed')
            report = subprocess.run([sys.executable, '-I', '-B', str(ROOT / 'scripts/22_report_clinical.py'),
                '--task', 'xgb', '--completed-input', 'toy', 'batchA', str(source), '--setting', 'toy', 'EMBED',
                '--annotations', str(data / 'clinical.parquet'), '--output', str(root / 'changed-report')],
                capture_output=True, text=True, timeout=30)
            self.assertNotEqual(report.returncode, 0)
            self.assertIn('Changed completed evaluator output', report.stderr)
            self.assertFalse((root / 'changed-report').exists())
