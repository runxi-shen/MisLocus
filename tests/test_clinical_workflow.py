"""Exact small clinical tables, consensus conflicts and explicit missing support."""
from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import polars as pl

from prot_loc_benchmark.workflows.clinical import annotation_labels, clinical_tables


def scores(models=('model',), batches=('A', 'B')):
    rows = []
    for model in models:
        for batch in batches:
            for allele, value in [('P1', .9), ('P2', .8), ('B1', .2), ('B2', .1)]:
                rows.append({'representation': model, 'channel': 'EMBED', 'batch': batch, 'allele_var': allele,
                             'auroc_mean': value, 'null_threshold': .5, 'is_hit': value > .5})
    return pl.DataFrame(rows)


class ClinicalWorkflowChecks(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.annotation = self.directory / 'annotations.parquet'
        pl.DataFrame([{'gene_variant': name, 'clinvar_clnsig_clean': label, 'clinvar_clnsig_clean_pp_strict': label}
                      for name, label in [('P1', 'Pathogenic'), ('P2', 'Pathogenic'), ('B1', 'Benign'), ('B2', 'Benign')]]).write_parquet(self.annotation)

    def report(self, frame, name='result', settings=None, pairs=None):
        return clinical_tables([frame], task='xgb', settings=settings or [('model', 'EMBED')],
                               biorep_pairs=pairs or {'pair': ('A', 'B')}, annotations=self.annotation,
                               selected_predictors=['clinvar'], output=self.directory / name)

    def test_perfect_four_variant_table_has_exact_fisher_and_MWU_probabilities(self):
        self.report(scores())
        binary = list(csv.DictReader((self.directory / 'result/binary-statistics.csv').open()))
        self.assertEqual(len(binary), 2)
        for row in binary:
            self.assertEqual([int(row[c]) for c in ['TP', 'FP', 'FN', 'TN']], [2, 0, 0, 2])
            self.assertEqual(float(row['F1']), 1.)
            self.assertAlmostEqual(float(row['fisher_p']), 1/6, places=14)
            self.assertAlmostEqual(float(row['fisher_q']), 1/6, places=14)
        continuous = list(csv.DictReader((self.directory / 'result/continuous-statistics.csv').open()))
        for row in continuous:
            self.assertEqual(float(row['U']), 4.)
            self.assertAlmostEqual(float(row['MWU_p']), 1/3, places=14)
            self.assertEqual(float(row['rank_biserial_P_minus_B']), 1.)

    def test_shared_support_removes_incomplete_pairs_without_creating_misses(self):
        frame = scores(('model', 'second')).filter(~((pl.col('representation') == 'second') & (pl.col('batch') == 'B') & (pl.col('allele_var') == 'P1')))
        self.report(frame, settings=[('model', 'EMBED'), ('second', 'EMBED')])
        data = pl.read_parquet(self.directory / 'result/clinical-alleles.parquet')
        complete = data.filter(pl.col('cohort') == 'complete_pair_any')
        self.assertNotIn('P1', set(complete['allele']))
        self.assertEqual(complete.height, 6)
        rows = list(csv.DictReader((self.directory / 'result/binary-statistics.csv').open()))
        for row in rows:
            self.assertEqual(int(row['TP']), 1)
            self.assertEqual(int(row['FN']), 0)
            self.assertEqual(row['fisher_q'], '')
            self.assertEqual(row['family_complete'], 'False')

    def test_zero_hits_are_reported_and_not_dropped_from_a_family(self):
        frame = scores().with_columns(auroc_mean=pl.lit(.1), is_hit=pl.lit(False))
        self.report(frame)
        rows = list(csv.DictReader((self.directory / 'result/binary-statistics.csv').open()))
        for row in rows:
            self.assertEqual(float(row['F1']), 0.)
            self.assertEqual(float(row['fisher_p']), 1.)
            self.assertEqual(float(row['fisher_q']), 1.)
            self.assertEqual(row['precision'], '')

    def test_both_batch_rule_requires_a_successful_pair_not_all_retained_batches(self):
        frame = scores(batches=('A', 'B', 'C', 'D')).with_columns(
            pl.when((pl.col('allele_var') == 'P1') & pl.col('batch').is_in(['C', 'D']))
            .then(pl.lit(.1)).otherwise(pl.col('auroc_mean')).alias('auroc_mean'))
        frame = frame.with_columns((pl.col('auroc_mean') > pl.col('null_threshold')).alias('is_hit'))
        self.report(frame, pairs={'first': ('A', 'B'), 'second': ('C', 'D')})
        alleles = pl.read_parquet(self.directory / 'result/allele_summary.parquet')
        p1 = alleles.filter((pl.col('allele') == 'P1') & (pl.col('cohort') == 'complete_pair_all'))
        self.assertEqual(p1['n_batches'][0], 4)
        self.assertTrue(p1['hit'][0])

    def test_raw_conflicts_are_resolved_before_label_binning_including_nulls(self):
        rows = [
            {'gene_variant': 'G1', 'clinvar_clnsig_clean': 'Pathogenic', 'clinvar_clnsig_clean_pp_strict': 'Pathogenic', 'REVEL_score': '.8', 'AlphaMissense_pred': 'P'},
            {'gene_variant': 'G1', 'clinvar_clnsig_clean': 'Pathogenic', 'clinvar_clnsig_clean_pp_strict': 'Likely pathogenic', 'REVEL_score': '.9', 'AlphaMissense_pred': 'LP'},
            {'gene_variant': 'G2', 'clinvar_clnsig_clean': 'Pathogenic', 'clinvar_clnsig_clean_pp_strict': 'Pathogenic', 'REVEL_score': '.8', 'AlphaMissense_pred': 'P'},
            {'gene_variant': 'G2', 'clinvar_clnsig_clean': None, 'clinvar_clnsig_clean_pp_strict': 'Pathogenic', 'REVEL_score': '.8', 'AlphaMissense_pred': 'P'},
        ]
        path = self.directory / 'conflicts.parquet'
        pl.DataFrame(rows).write_parquet(path)
        labels, conflicts = annotation_labels(path, ['clinvar', 'revel', 'alphamissense'])
        g1 = labels.filter(pl.col('allele') == 'G1')
        self.assertEqual(g1.filter(pl.col('predictor') == 'clinvar')['coarse_label'][0], 'Pathogenic')
        self.assertIsNone(g1.filter(pl.col('predictor') == 'clinvar')['strict_label'][0])
        self.assertIsNone(g1.filter(pl.col('predictor') == 'revel')['coarse_label'][0])
        self.assertIsNone(g1.filter(pl.col('predictor') == 'alphamissense')['coarse_label'][0])
        self.assertIsNone(labels.filter((pl.col('allele') == 'G2') & (pl.col('predictor') == 'clinvar'))['coarse_label'][0])
        self.assertEqual(conflicts.height, 4)

    def test_incoherent_optional_reporting_flag_is_not_silently_replaced(self):
        with self.assertRaisesRegex(ValueError, 'reporting_hit disagrees'):
            self.report(scores().with_columns(reporting_hit=pl.lit(False)))

    def test_strict_only_labels_contribute_independently(self):
        annotations = pl.read_parquet(self.annotation).with_columns(
            pl.lit(None, dtype=pl.String).alias('clinvar_clnsig_clean'))
        annotations.write_parquet(self.annotation)
        self.report(scores())
        rows = list(csv.DictReader((self.directory / 'result/continuous-statistics.csv').open()))
        strict = next(row for row in rows if row['tier'] == 'strict')
        coarse = next(row for row in rows if row['tier'] == 'coarse')
        self.assertEqual((strict['n_pathogenic'], strict['n_benign']), ('2', '2'))
        self.assertAlmostEqual(float(strict['MWU_p']), 1/3)
        self.assertEqual(coarse['estimable'], 'False')


if __name__ == '__main__':
    unittest.main()
