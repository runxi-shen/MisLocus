"""Small independent numerical examples and unsupported-population checks."""
from __future__ import annotations

import math
import os
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import polars as pl

from prot_loc_benchmark.workflows.evaluations import require_metadata, score_hpa, score_pa, score_xgb
from prot_loc_benchmark.workflows.features import validate_frame


def specification():
    return {'schema_version': 1, 'model': 'toy', 'features': ['f0', 'f1', 'f2'],
            'channels': {'EMBED': ['f0', 'f1', 'f2']}, 'preprocessing': {'kind': 'none'}}


def row(plate, well, allele, gene, node, role, index, vector):
    return {'Metadata_Batch': 'toy-batch', 'Metadata_Plate': plate, 'Metadata_well_position': well,
            'Metadata_Site': 1, 'Metadata_ImageNumber': index % 2 + 1, 'Metadata_ObjectNumber': index // 2 + 1,
            'Metadata_gene_allele': allele, 'Metadata_symbol': gene, 'Metadata_node_type': node,
            'Metadata_Control': role, 'Metadata_plate_map_name': 'map',
            **{f'f{i}': float(v) for i, v in enumerate(vector)}}


def paired_cells():
    rows = []
    for technical in range(1, 5):
        for index in range(13):
            plate = f'map_T{technical}'
            rows.append(row(plate, 'A01', 'G', 'G', 'disease_wt', 'Exp', index, [1, .05, .02]))
            rows.append(row(plate, 'A02', 'G_variant', 'G', 'allele', 'Exp', index, [.05, 1, .02]))
            for well in ['A10', 'A11', 'A12']:
                rows.append(row(plate, well, 'CONTROL', 'CONTROL', 'neutral_wt', 'NC', index, [1, 1, 1]))
    return validate_frame(pl.DataFrame(rows), specification())[0]


class ExplicitFeatureEvaluationChecks(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def test_native_xgb_separates_variant_and_calibrates_identical_controls_at_half(self):
        frame = paired_cells()
        output = self.directory / 'xgb'
        details = score_xgb(frame, specification()['channels'], output, model='toy')
        metrics = pl.read_parquet(output / 'classifier-metrics.parquet')
        controls = metrics.filter(pl.col('is_control'))
        self.assertEqual(controls.height, 3)
        self.assertEqual(controls['auroc'].to_list(), [.5] * 3)
        self.assertEqual(details['thresholds'], {'EMBED': .5})
        scores = pl.read_parquet(output / 'scores.parquet')
        self.assertEqual(scores.height, 1)
        self.assertEqual(scores['auroc_mean'][0], 1.)
        self.assertTrue(scores['reporting_hit'][0])
        predictions = pl.read_parquet(output / 'predictions.parquet')
        self.assertTrue(predictions['Metadata_Plate'].str.ends_with('T4').all())
        self.assertEqual(predictions.height, 4 * 26)

    def test_native_pa_perfect_variant_retrieval_and_single_anchor_control_families(self):
        output = self.directory / 'pa'
        details = score_pa(paired_cells(), specification()['channels'], output, workers=2, model='toy')
        scores = pl.read_parquet(output / 'scores.parquet')
        self.assertEqual(scores.height, 1)
        self.assertEqual(scores['mAP_vs_ref'][0], 1.)
        self.assertEqual(scores['mAP_vs_ref_norm'][0], 1.)
        controls = pl.read_parquet(output / 'controls.parquet')
        self.assertEqual(controls.height, 3)
        self.assertTrue(controls['p_value_vs_ref'].equals(controls['corrected_p_value_vs_ref']))
        ap = pl.read_parquet(output / 'query-ap.parquet')
        self.assertEqual(set(ap['population']), {'experimental', 'control'})
        self.assertEqual(set(ap.filter(pl.col('population') == 'experimental')['Metadata_gene_allele']), {'G_variant'})
        self.assertTrue(ap.filter(pl.col('population') == 'control')['Metadata_gene_allele'].str.starts_with('loo__').all())
        support = pl.read_csv(output / 'query-support.csv')
        self.assertEqual(set(support['population']), {'experimental', 'control'})
        self.assertTrue(support.filter(pl.col('query_supported'))['has_different_plate_positive'].all())
        self.assertTrue(support.filter(pl.col('query_supported'))['has_matching_reference'].all())
        variant = ap.filter(pl.col('Metadata_gene_allele') == 'G_variant')
        self.assertEqual(variant.height, 2)
        self.assertEqual(variant['n_pos_pairs'].to_list(), [6, 6])
        self.assertEqual(variant['n_total_pairs'].to_list(), [8, 8])
        self.assertEqual(details['retained_rows'], 1)

    def test_available_control_policy_retains_two_training_plates_without_changing_experimental_split(self):
        frame = paired_cells().filter(pl.col('Metadata_Control') == 'Exp')
        controls = []
        for technical in [1, 2, 4]:
            for well in ['A10', 'A11', 'A12']:
                for index in range(18):
                    controls.append(row(f'map_T{technical}', well, 'CONTROL', 'CONTROL', 'neutral_wt', 'NC', index, [1, 1, 1]))
        controls = validate_frame(pl.DataFrame(controls), specification())[0]
        frame = pl.concat([frame, controls])
        with patch('prot_loc_benchmark.classification.executor._run_classifier') as fit:
            with self.assertRaisesRegex(ValueError, 'supported NC/PC control folds'):
                score_xgb(frame, specification()['channels'], self.directory / 'strict')
            fit.assert_not_called()
        output = self.directory / 'available'
        score_xgb(frame, specification()['channels'], output, control_plate_policy='available')
        metrics = pl.read_parquet(output / 'classifier-metrics.parquet')
        control = metrics.filter(pl.col('is_control'))
        self.assertEqual(control.height, 3)
        self.assertEqual(control['auroc'].to_list(), [.5] * 3)
        self.assertTrue((control['train_plates'] == 'map_T1,map_T2').all())
        experimental = metrics.filter(~pl.col('is_control'))
        self.assertEqual(experimental['train_plates'].to_list(), ['map_T1,map_T2,map_T3'])
        self.assertEqual(pl.read_parquet(output / 'scores.parquet')['auroc_mean'].to_list(), [1.])

    def test_gpu_requires_explicit_uuid_and_refuses_ordinal_before_fitting(self):
        for options in [{'backend': 'gpu'}, {'backend': 'cpu', 'gpu_uuid': 'GPU-invalid'}]:
            with self.assertRaisesRegex(ValueError, 'explicit full GPU UUID'):
                score_xgb(paired_cells(), specification()['channels'], self.directory / 'invalid', **options)
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ''}):
            with self.assertRaisesRegex(ValueError, 'full GPU UUID'):
                score_xgb(paired_cells(), specification()['channels'], self.directory / 'ordinal', backend='gpu', gpu_uuid='0')

    def hpa_case(self):
        vectors = {'A': [1, .01, .1], 'B': [1, .05, .1], 'C': [.01, 1, .1],
                   'D': [.05, 1, .1], 'E': [.02, .03, 1]}
        labels = {'A': ['L1'], 'B': ['L1', 'L2'], 'C': ['L2'], 'D': ['L2'], 'E': ['L3']}
        rows = [row('map_T4', f'A{i+1:02}', gene, gene, 'disease_wt', 'Exp', 0, values)
                for i, (gene, values) in enumerate(vectors.items())]
        frame = validate_frame(pl.DataFrame(rows), specification())[0]
        annotation = self.directory / 'hpa.parquet'
        pl.DataFrame([{'Gene': f'ENSG{i}', 'Gene name': gene,
                       **{loc: 1. if loc in labels[gene] else None for loc in ['L1', 'L2', 'L3']}}
                      for i, gene in enumerate(vectors)]).write_parquet(annotation)
        return frame, annotation, vectors, labels

    def test_hpa_multilabel_ap_matches_pairwise_hand_calculation_before_display_filter(self):
        frame, annotation, vectors, labels = self.hpa_case()
        output = self.directory / 'hpa'
        score_hpa(frame, specification()['channels'], annotation, output)
        ap = pl.read_parquet(output / 'EMBED-query-ap.parquet')
        for query in ap.iter_rows(named=True):
            gene, location = query['Metadata_symbol'], query['Metadata_hpa_locations']
            positive = {other for other in vectors if other != gene and location in labels[other]}
            negative = {other for other in vectors if set(labels[gene]).isdisjoint(labels[other])}
            def cosine(other):
                a, b = vectors[gene], vectors[other]
                return math.fsum(x * y for x, y in zip(a, b)) / math.sqrt(math.fsum(x*x for x in a) * math.fsum(x*x for x in b))
            ordered = sorted(positive | negative, key=cosine, reverse=True)
            seen = 0
            precisions = []
            for rank, other in enumerate(ordered, 1):
                if other in positive:
                    seen += 1
                    precisions.append(seen / rank)
            self.assertAlmostEqual(query['average_precision'], math.fsum(precisions) / len(positive), places=12)
            self.assertEqual(query['n_pos_pairs'], len(positive))
            self.assertEqual(query['n_total_pairs'], len(positive) + len(negative))
        scores = pl.read_parquet(output / 'scores.parquet')
        self.assertEqual(set(scores['hpa_location']), {'L1', 'L2'})
        self.assertEqual(set(pl.read_csv(output / 'display.csv')['hpa_location']), {'L2'})
        p = scores['p_value_hpa'].to_list()
        q = scores['corrected_p_value_hpa'].to_list()
        order = sorted(range(2), key=p.__getitem__)
        self.assertAlmostEqual(q[order[0]], min(1., 2*p[order[0]], p[order[1]]), places=12)
        self.assertAlmostEqual(q[order[1]], p[order[1]], places=12)

    def test_hpa_positive_queries_without_disjoint_negatives_fail(self):
        frame, annotation, _, _ = self.hpa_case()
        all_shared = pl.read_parquet(annotation).with_columns(L1=pl.lit(1.))
        all_shared.write_parquet(annotation)
        with self.assertRaisesRegex(ValueError, 'no disjoint negatives'):
            score_hpa(frame, specification()['channels'], annotation, self.directory / 'invalid-hpa')

    def test_native_float32_norm_overflow_fails(self):
        frame, annotation, _, _ = self.hpa_case()
        frame = frame.with_columns(pl.lit(1e30).alias('f0'))
        with self.assertRaisesRegex(ValueError, 'finite nonzero reference profiles'):
            score_hpa(frame, specification()['channels'], annotation, self.directory / 'overflow')

    def test_contradictory_plate_and_allele_metadata_fail(self):
        frame = paired_cells()
        with self.assertRaisesRegex(ValueError, 'one platemap'):
            require_metadata(frame.with_columns(pl.when(pl.col('Metadata_well_position') == 'A02').then(pl.lit('other')).otherwise(pl.col('Metadata_plate_map_name')).alias('Metadata_plate_map_name')), 'pa')
        with self.assertRaisesRegex(ValueError, 'one gene'):
            require_metadata(frame.with_columns(pl.when(pl.col('Metadata_Plate') == 'map_T1').then(pl.lit('different')).otherwise(pl.col('Metadata_symbol')).alias('Metadata_symbol')), 'xgb')

    def test_unqualified_tiny_extraction_is_not_scored_as_a_benchmark(self):
        one = paired_cells().head(1)
        with self.assertRaisesRegex(ValueError, '50-cell minimum'):
            score_pa(one, specification()['channels'], self.directory / 'unsupported-pa')
        with self.assertRaisesRegex(ValueError, 'requires supported NC/PC'):
            score_xgb(one, specification()['channels'], self.directory / 'unsupported-xgb')

    def test_mixed_variant_roles_and_multiple_reference_ids_are_ambiguous(self):
        frame = paired_cells()
        mixed = frame.with_columns(pl.when((pl.col('Metadata_gene_allele') == 'G_variant') & (pl.col('Metadata_Plate') == 'map_T1'))
                                   .then(pl.lit('cPC')).otherwise(pl.col('Metadata_Control')).alias('Metadata_Control'))
        # Both native builders would create the same pair ID. Rejection must
        # occur before fitting, while legitimate per-row role reuse is retained.
        # Add enough cPC cells to meet the native pair minimum as well as Exp.
        extra = mixed.filter((pl.col('Metadata_gene_allele') == 'G_variant') & (pl.col('Metadata_Plate') == 'map_T1'))
        extra = extra.with_columns((pl.col('Metadata_ObjectNumber') + 100).alias('Metadata_ObjectNumber'))
        extra = extra.drop('Metadata_BatchQualifiedCellID')
        extra_exp = mixed.filter((pl.col('Metadata_gene_allele') == 'G_variant') & (pl.col('Metadata_Plate') == 'map_T2')).drop('Metadata_BatchQualifiedCellID')
        extra_exp = extra_exp.with_columns((pl.col('Metadata_ObjectNumber') + 100).alias('Metadata_ObjectNumber'))
        mixed = pl.concat([mixed.drop('Metadata_BatchQualifiedCellID'), extra, extra.with_columns((pl.col('Metadata_ObjectNumber') + 100).alias('Metadata_ObjectNumber')), extra.with_columns((pl.col('Metadata_ObjectNumber') + 200).alias('Metadata_ObjectNumber')), extra_exp])
        # cPC also needs a reference; clone the WT role onto another well.
        cpc_ref = mixed.filter(pl.col('Metadata_node_type') == 'disease_wt').with_columns(Metadata_Control=pl.lit('cPC'), Metadata_well_position=pl.lit('A03'))
        mixed = validate_frame(pl.concat([mixed, cpc_ref]), specification())[0]
        with self.assertRaisesRegex(ValueError, 'duplicate classification pair IDs'):
            score_xgb(mixed, specification()['channels'], self.directory / 'duplicate-pairs')
        references = frame.with_columns(pl.when((pl.col('Metadata_node_type') == 'disease_wt') & (pl.col('Metadata_Plate') == 'map_T1'))
                                        .then(pl.lit('G_other_reference')).otherwise(pl.col('Metadata_gene_allele')).alias('Metadata_gene_allele'))
        with self.assertRaisesRegex(ValueError, 'Multiple reference allele IDs'):
            require_metadata(references, 'xgb')

    def test_legitimate_same_name_reference_and_variant_rows_are_preserved(self):
        frame = paired_cells().with_columns(pl.when(pl.col('Metadata_gene_allele') == 'G_variant')
                                           .then(pl.lit('G')).otherwise(pl.col('Metadata_gene_allele')).alias('Metadata_gene_allele'))
        require_metadata(frame, 'pa')
        require_metadata(frame, 'xgb')
        self.assertEqual(frame.filter(pl.col('Metadata_gene_allele') == 'G')['Metadata_node_type'].n_unique(), 2)

    def test_loo_ids_cannot_be_duplicated_by_cross_plate_control_categories(self):
        frame = paired_cells().drop('Metadata_BatchQualifiedCellID')
        controls = frame.filter(pl.col('Metadata_Control') == 'NC').with_columns((pl.col('Metadata_ObjectNumber') + 100).alias('Metadata_ObjectNumber'))
        frame = pl.concat([frame, controls]).with_columns(pl.when((pl.col('Metadata_gene_allele') == 'CONTROL') & pl.col('Metadata_Plate').is_in(['map_T3', 'map_T4']))
                                                        .then(pl.lit('PC')).otherwise(pl.col('Metadata_Control')).alias('Metadata_Control'))
        frame = validate_frame(frame, specification())[0]
        with self.assertRaisesRegex(ValueError, 'duplicate LOO anchor IDs'):
            score_pa(frame, specification()['channels'], self.directory / 'duplicate-loo')


if __name__ == '__main__':
    unittest.main()
