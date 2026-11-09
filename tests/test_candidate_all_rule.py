"""Candidate ALL semantics differ from the public default for multiple pairs."""
import itertools
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import polars as pl
from prot_loc_benchmark.benchmark.hits import hit_cohorts


class CandidateAllRule(unittest.TestCase):
    def test_public_clinical_cli_records_candidate_rule_and_changes_only_all_calls(self):
        script = Path(__file__).resolve().parents[1] / 'scripts/22_report_clinical.py'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotation = root / 'labels.parquet'
            pl.DataFrame([dict(gene_variant=a, clinvar_clnsig_clean=label, clinvar_clnsig_clean_pp_strict=label)
                          for a, label in [('P1', 'Pathogenic'), ('P2', 'Pathogenic'), ('B1', 'Benign'), ('B2', 'Benign')]]).write_parquet(annotation)
            pairs = root / 'pairs.json'
            pairs.write_text(json.dumps({'first': ['A', 'B'], 'second': ['C', 'D']}))
            command = [sys.executable, '-I', '-B', str(script), '--task', 'xgb', '--setting', 'r', 'EMBED',
                       '--annotations', str(annotation), '--biorep-pairs', str(pairs)]
            for batch in 'ABCD':
                source = root / (batch + '.parquet')
                pl.DataFrame([dict(allele_var=a, channel='EMBED', auroc_mean=.8 if hit else .2, null_threshold=.5)
                              for a, hit in [('P1', batch in 'AB'), ('P2', True), ('B1', False), ('B2', False)]]).write_parquet(source)
                command += ['--input', 'r', batch, str(source)]
            outputs = {}
            for rule in ['any_complete_pair', 'all_retained_batches']:
                output = root / rule
                run = subprocess.run([*command, '--all-rule', rule, '--output', str(output)], capture_output=True, text=True)
                self.assertEqual(run.returncode, 0, run.stderr)
                self.assertEqual(json.loads((output / 'report.json').read_text())['details']['all_rule'], rule)
                with (output / 'binary-statistics.csv').open() as stream:
                    outputs[rule] = {r['cohort']: r for r in csv.DictReader(stream)}
            self.assertEqual(outputs['any_complete_pair']['complete_pair_any'], outputs['all_retained_batches']['complete_pair_any'])
            self.assertEqual(float(outputs['any_complete_pair']['complete_pair_all']['F1']), 1.)
            self.assertAlmostEqual(float(outputs['all_retained_batches']['complete_pair_all']['F1']), 2/3)

    def test_exhaustive_multiple_pairs_both_tasks(self):
        for task in ['pa', 'xgb']:
            for flags in itertools.product([False, True], repeat=4):
                rows = [dict(representation='r', channel='GFP', batch=batch,
                             Metadata_gene_allele='v', mAP_vs_ref_norm=.8 if hit else .4,
                             null_threshold_p95=.5, corrected_p_value_vs_ref=.01)
                        for batch, hit in zip('ABCDE', [*flags, False])]
                frame = pl.DataFrame(rows)
                if task == 'xgb':
                    frame = frame.rename({'Metadata_gene_allele': 'allele_var',
                        'mAP_vs_ref_norm': 'auroc_mean', 'null_threshold_p95': 'null_threshold'})
                args = (frame, task, [('r', 'GFP')], {'first': ('A', 'B'), 'second': ('C', 'D'), 'third': ('E', 'F')})
                default = hit_cohorts(*args)['allele_summary']
                candidate = hit_cohorts(*args, all_rule='all_retained_batches')['allele_summary']
                a = {r['cohort']: r for r in default.iter_rows(named=True)}
                b = {r['cohort']: r for r in candidate.iter_rows(named=True)}
                with self.subTest(task=task, flags=flags):
                    self.assertEqual(a['complete_pair_all']['hit'], (flags[0] and flags[1]) or (flags[2] and flags[3]))
                    self.assertEqual(b['complete_pair_all']['hit'], all(flags))
                    self.assertEqual(b['complete_pair_all']['n_batches'], 4)
                    self.assertEqual(a['complete_pair_all']['mean_score'], b['complete_pair_all']['mean_score'])
                    for cohort in ['any_available', 'complete_pair_any']:
                        self.assertEqual(a[cohort], b[cohort])

    def test_invalid_policy_fails_before_scoring(self):
        with self.assertRaisesRegex(ValueError, 'all_rule'):
            hit_cohorts([], 'pa', [('r', 'GFP')], all_rule='unknown')
