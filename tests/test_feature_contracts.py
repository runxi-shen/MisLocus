"""Feature/identity contracts, independent of optional encoder workflows."""
import itertools
import unittest
import polars as pl
from prot_loc_benchmark.workflows.features import QUALIFIED_ID, validate_frame, validate_identity

def example_spec():
    return {'schema_version': 1, 'model': 'new-model', 'features': ['embedding_0', 'embedding_1'], 'channels': {'EMBED': ['embedding_0', 'embedding_1']}, 'preprocessing': {'kind': 'none'}}

def example_frame():
    rows = []
    for batch, platemap, technical in itertools.product(['batch-A', 'batch-B'], ['map1', 'map2'], range(1, 5)):
        rows.append({'Metadata_Batch': batch, 'Metadata_Plate': f'{platemap}_T{technical}', 'Metadata_well_position': 'A01', 'Metadata_Site': 1, 'Metadata_ImageNumber': 10, 'Metadata_ObjectNumber': 1, 'Metadata_plate_map_name': platemap, 'Metadata_gene_allele': 'GENE_Variant', 'Metadata_symbol': 'GENE', 'Metadata_node_type': 'allele', 'Metadata_Control': 'Exp', 'embedding_0': float(technical), 'embedding_1': float(technical * 2)})
    return pl.DataFrame(rows)

class FeatureContractChecks(unittest.TestCase):

    def test_batch_qualified_ids_and_full_coverage(self):
        frame = example_frame()
        result, report = validate_frame(frame, example_spec())
        assert report['rows'] == 16 and report['physical_plates'] == 16
        assert result[QUALIFIED_ID].n_unique() == 16
        assert result.select(frame.columns).equals(frame)
        assert not report['evaluation_metadata']['pa']['missing_columns']

    def test_nonfinite_features_fail_without_dropping_rows(self):
        for value in [float('nan'), float('inf'), None]:
            with self.subTest(value=value):
                frame = example_frame().with_columns(embedding_0=pl.lit(value, dtype=pl.Float64))
                with self.assertRaisesRegex(ValueError, 'null, NaN, or infinite'):
                    validate_frame(frame, example_spec())
                assert frame.height == 16

    def test_supplied_semantic_metadata_is_validated(self):
        for column, value in [('Metadata_symbol', None), ('Metadata_Control', ''), ('Metadata_gene_allele', ' GENE_Variant'), ('Metadata_plate_map_name', ' '), ('Metadata_CellID', '')]:
            with self.subTest(column=column, value=value):
                frame = example_frame().with_columns(pl.lit(value, dtype=pl.String).alias(column))
                with self.assertRaises(ValueError):
                    validate_frame(frame, example_spec())

    def test_missing_roles_are_reported_as_unsupported_not_invented(self):
        frame = example_frame().drop('Metadata_Control', 'Metadata_plate_map_name')
        result, report = validate_frame(frame, example_spec())
        assert result.height == frame.height
        assert report['evaluation_metadata']['pa']['missing_columns'] == ['Metadata_Control', 'Metadata_plate_map_name']

    def test_legitimate_transfection_controls_remain_visible(self):
        frame = example_frame().with_columns(Metadata_Control=pl.lit('TC'))
        result, report = validate_frame(frame, example_spec())
        assert result.height == 16
        assert report['role_counts'] == [{'Metadata_Control': 'TC', 'len': 16}]

    def test_corrupt_ids_fractional_identity_and_conflicting_well_fail(self):
        frame, _ = validate_frame(example_frame(), example_spec())
        with self.assertRaisesRegex(ValueError, 'Stored batch-qualified'):
            validate_identity(frame.with_columns(pl.lit('fabricated').alias(QUALIFIED_ID)))
        with self.assertRaisesRegex(ValueError, 'nonnegative integers'):
            validate_identity(example_frame().with_columns(Metadata_Site=pl.lit(1.5)))
        with self.assertRaisesRegex(ValueError, 'conflicts'):
            validate_identity(example_frame().with_columns(Metadata_Well=pl.lit('B02')))

    def test_labels_cannot_be_selected_as_features(self):
        spec = example_spec()
        spec['features'] = ['Label']
        spec['channels'] = {'EMBED': ['Label']}
        with self.assertRaisesRegex(ValueError, 'exclude metadata/labels'):
            validate_frame(example_frame().with_columns(Label=pl.lit(1)), spec)
