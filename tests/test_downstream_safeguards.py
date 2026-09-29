"""Synthetic safety/lineage checks. No production data, checkpoints or GPU jobs."""

import importlib.util
import unittest

import polars as pl

from prot_loc_benchmark.identity import CELL_ID, identify_cells


def load_script(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Safeguards(unittest.TestCase):
    def test_identity_repeated_object_numbers_missing_and_conflicting_ids(self):
        frame = pl.DataFrame(
            {
                "Metadata_Plate": ["P_T4"] * 2,
                "Metadata_Well": ["A01"] * 2,
                "Metadata_ImageNumber": [1, 2],
                "Metadata_ObjectNumber": [7, 7],
            }
        )
        result = identify_cells(frame, "batch")
        self.assertEqual(result[CELL_ID].n_unique(), 2)
        sites = frame.with_columns(pl.lit(1).alias("Metadata_ImageNumber"), pl.Series("Metadata_Site", [1, 2]))
        self.assertEqual(identify_cells(sites, "batch")[CELL_ID].n_unique(), 2)
        self.assertEqual(
            identify_cells(frame.reverse(), "batch").sort(CELL_ID).to_dicts(), result.sort(CELL_ID).to_dicts()
        )
        for invalid in (
            frame.drop("Metadata_ImageNumber"),
            frame.with_columns(pl.lit(1).alias("Metadata_ImageNumber")),
            result.with_columns(pl.lit("wrong").alias(CELL_ID)),
        ):
            with self.assertRaises(ValueError):
                identify_cells(invalid, "batch")
        with self.assertRaisesRegex(ValueError, "identity"):
            identify_cells(frame, "batch", canonical=True)


if __name__ == "__main__":
    unittest.main()
