"""Clinical evaluation fixes exercised through the original numerical helpers."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import polars as pl

from prot_loc_benchmark.benchmark import clinvar


class ClinVarEvaluationChecks(unittest.TestCase):
    def test_annotation_consensus_is_independent_for_each_raw_column(self):
        rows = [
            ('G_v', 'Pathogenic', 'Pathogenic'), ('G_v', 'Pathogenic', 'Uncertain_significance'),
            ('H_v', 'Pathogenic', 'Benign'), ('H_v', 'Benign', 'Benign'),
            ('I_v', None, 'Pathogenic'), ('I_v', 'Pathogenic', 'Pathogenic'),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'annotations.parquet'
            pl.DataFrame(rows, schema=['gene_variant', 'clinvar_clnsig_clean',
                                      'clinvar_clnsig_clean_pp_strict'], orient='row').write_parquet(path)
            with patch.object(clinvar, 'ALLELE_COLLECTION_PATH', path):
                result = {r['gene_variant']: r for r in clinvar.load_clinvar_annotations().iter_rows(named=True)}
        self.assertEqual(result['G_v']['clinvar_clnsig_clean'], 'Pathogenic')
        self.assertIsNone(result['G_v']['clinvar_clnsig_clean_pp_strict'])
        self.assertIsNone(result['H_v']['clinvar_clnsig_clean'])
        self.assertEqual(result['H_v']['clinvar_clnsig_clean_pp_strict'], 'Benign')
        self.assertIsNone(result['I_v']['clinvar_clnsig_clean'])
        self.assertEqual(result['I_v']['clinvar_clnsig_clean_pp_strict'], 'Pathogenic')

    def test_join_retains_rows_with_a_strict_label_when_coarse_is_unavailable(self):
        data = pl.DataFrame({'allele_var': ['G_v', 'H_v', 'I_v'], 'auroc_avg': [.8, .6, .4]})
        annotations = pl.DataFrame({'gene_variant': ['G_v', 'H_v', 'I_v'],
            'clinvar_clnsig_clean': ['Pathogenic', None, None],
            'clinvar_clnsig_clean_pp_strict': [None, 'Benign', None]})
        self.assertEqual(clinvar.join_clinvar(data, annotations)['allele_var'].to_list(), ['G_v', 'H_v'])

    def statistical_rows(self):
        return pl.DataFrame([dict(representation=f'r{setting}', channel='GFP', allele_var=f'G_v{i}',
            auroc_avg=.95 - i * .1, clinvar_clnsig_clean='Pathogenic' if i < 4 else 'Benign',
            clinvar_clnsig_clean_pp_strict='Pathogenic' if i < 2 else 'Benign' if i >= 6 else None)
            for setting in range(9) for i in range(8)])

    def test_bh_is_separate_for_coarse_and_strict_nine_setting_families(self):
        result = clinvar.run_wilcoxon_tests(self.statistical_rows())
        self.assertEqual(result.height, 18)
        # With separated ranks, two-sided exact U has p=2/C(8,4) and
        # p=2/C(4,2). Nine identical p-values retain those values under BH.
        # Pooling all 18 would incorrectly double the coarse adjusted p.
        for row in result.iter_rows(named=True):
            expected = 2 / 6 if row['comparison'].endswith('_strict') else 2 / 70
            self.assertAlmostEqual(row['pvalue'], expected)
            self.assertAlmostEqual(row['pvalue_bh'], expected)

    def test_incomplete_setting_or_strict_family_is_rejected(self):
        data = self.statistical_rows()
        with self.assertRaisesRegex(ValueError, 'nine selected'):
            clinvar.run_wilcoxon_tests(data.filter(pl.col('representation') != 'r8'))
        broken = data.with_columns(pl.when(pl.col('representation') == 'r8').then(None)
            .otherwise(pl.col('clinvar_clnsig_clean_pp_strict')).alias('clinvar_clnsig_clean_pp_strict'))
        with self.assertRaisesRegex(ValueError, 'nine estimable'):
            clinvar.run_wilcoxon_tests(broken)
