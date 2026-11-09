"""PA support and control-failure checks on small synthetic pools."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import polars as pl

from test_pa_query_pool import pa
from prot_loc_benchmark.benchmark.hits import hit_cohorts

BATCH = "2024_01_23_Batch_7"


def pool():
    return pl.DataFrame({
        "Metadata_gene_allele": ["G_V"] * 3 + ["G"] * 3 + ["G_W"] * 2,
        "Metadata_symbol": ["G"] * 8,
        "Metadata_node_type": ["allele"] * 3 + ["disease_wt"] * 3 + ["allele"] * 2,
        "Metadata_Plate": ["P1T1", "P1T2", "P1T4"] * 2 + ["P1T1", "P1T4"],
        "Metadata_well_position": ["A01"] * 3 + ["A02"] * 3 + ["A03"] * 2,
        "Metadata_ImageNumber": ["F1"] * 8,
        "x": [1., .8, .6, .1, .2, .3, .4, .5],
        "y": [0., .2, .4, .9, .8, .7, .6, .5],
    })


def run(frame, split="t4"):
    return pa._run_map(frame, ["x", "y"], "Metadata_node_type == 'disease_wt'",
                       null_size=32, threshold=.05, seed=42, label="vs_ref", max_workers=1,
                       neg_sameby=["Metadata_Plate", "Metadata_symbol"], test_split=split)


class PASupportChecks(unittest.TestCase):
    def test_supported_native_scores_match_direct_copairs(self):
        frame = pool()
        meta = pa.assign_reference_index(frame.to_pandas(), "Metadata_node_type == 'disease_wt'",
                                         reference_col=pa.REFERENCE_COL, default_value=-1)
        features = meta[["x", "y"]].to_numpy().astype(np.float32)
        ap = pa.average_precision(meta.drop(columns=["x", "y"]), features,
                                  pos_sameby=["Metadata_gene_allele", pa.REFERENCE_COL],
                                  pos_diffby=["Metadata_Plate"],
                                  neg_sameby=["Metadata_Plate", "Metadata_symbol"],
                                  neg_diffby=["Metadata_node_type", pa.REFERENCE_COL])
        for split in [None, "t4"]:
            queries = ap.loc[ap[pa.REFERENCE_COL] == -1]
            if split == "t4":
                queries = queries.loc[queries["Metadata_Plate"].str.endswith("T4")]
            expected = pa.mean_average_precision(queries, sameby=["Metadata_gene_allele"],
                                                 null_size=32, threshold=.05, seed=42, max_workers=1)
            actual = run(frame, split).set_index("Metadata_gene_allele").sort_index()
            expected = expected.set_index("Metadata_gene_allele").sort_index()
            for old, new in [("mean_average_precision", "mAP_vs_ref"),
                             ("mean_normalized_average_precision", "mAP_vs_ref_norm"),
                             ("p_value", "p_value_vs_ref"),
                             ("corrected_p_value", "corrected_p_value_vs_ref"),
                             ("below_p", "below_p_vs_ref"),
                             ("below_corrected_p", "below_corrected_p_vs_ref")]:
                pd.testing.assert_series_equal(actual[new], expected[old], check_names=False, check_exact=True)

    def test_missing_reference_excludes_query_not_positive_partner(self):
        frame = pool().filter(~((pl.col("Metadata_node_type") == "disease_wt") &
                                (pl.col("Metadata_Plate") == "P1T2")))
        with patch.object(pa, "mean_average_precision", wraps=pa.mean_average_precision) as mean_ap:
            run(frame, split=None)
        queries = mean_ap.call_args.args[0]
        self.assertNotIn("P1T2", set(queries["Metadata_Plate"]))
        t4_v = queries.loc[(queries["Metadata_Plate"] == "P1T4") &
                           (queries["Metadata_gene_allele"] == "G_V")]
        self.assertEqual(t4_v["n_pos_pairs"].tolist(), [2])  # T2 partner is retained.
        self.assertEqual(t4_v["n_total_pairs"].tolist(), [3])
        pd.testing.assert_frame_equal(run(frame), run(pool()), check_exact=True)  # T4 scores unchanged.
        extra = pool().filter((pl.col("Metadata_gene_allele") == "G_V") &
                              (pl.col("Metadata_Plate") == "P1T4")).with_columns(Metadata_Plate=pl.lit("P2T4"))
        with patch.object(pa, "mean_average_precision", wraps=pa.mean_average_precision) as mean_ap:
            run(pl.concat([pool(), extra]))
        queries = mean_ap.call_args.args[0]
        self.assertEqual(queries["Metadata_Plate"].tolist(), ["P1T4", "P1T4"])
        supported = queries.loc[queries["Metadata_gene_allele"] == "G_V"]
        self.assertEqual(supported["n_pos_pairs"].tolist(), [3])  # Excluded P2T4 still supplies a positive.
        self.assertEqual(supported["n_total_pairs"].tolist(), [4])

    def test_no_eligible_queries_returns_empty_without_copairs(self):
        for frame in [pool().head(0), pool().filter(~pl.col("Metadata_Plate").str.ends_with("T4")),
                      pool().filter(pl.col("Metadata_node_type") == "allele"),
                      pool().with_columns(Metadata_symbol=pl.when(pl.col("Metadata_node_type") == "disease_wt")
                                          .then(pl.lit("H")).otherwise(pl.col("Metadata_symbol")))]:
            with self.subTest(rows=frame.height), patch.object(pa, "average_precision") as ap:
                self.assertTrue(run(frame).empty)
                ap.assert_not_called()

    def test_invalid_vectors_are_rejected_before_copairs(self):
        for x, y in [(float("nan"), 1.), (float("inf"), 1.), (0., 0.), (1e30, 1e30)]:
            # Even a non-T4 variant must be valid: it remains a positive partner.
            frame = pool().with_row_index("i").with_columns(
                x=pl.when(pl.col("i") == 0).then(x).otherwise(pl.col("x")),
                y=pl.when(pl.col("i") == 0).then(y).otherwise(pl.col("y")),
            ).drop("i")
            with self.subTest(x=x, y=y), np.errstate(over="ignore", invalid="ignore"), \
                    patch.object(pa, "average_precision") as ap:
                with self.assertRaisesRegex(ValueError, "finite.*norm"):
                    run(frame)
                ap.assert_not_called()

    def test_query_requires_positive_and_negative_support(self):
        frame = pool().filter(~((pl.col("Metadata_gene_allele") == "G_V") &
                                (pl.col("Metadata_Plate") != "P1T4")))
        with patch.object(pa, "mean_average_precision", wraps=pa.mean_average_precision) as mean_ap:
            with self.assertRaisesRegex(ValueError, "positive.*negative"):
                run(frame)
            mean_ap.assert_not_called()

    def test_invalid_ap_or_negative_support_stops_aggregation(self):
        real_ap = pa.average_precision
        for column, value in [("average_precision", float("nan")),
                              ("normalized_average_precision", float("inf")),
                              ("n_total_pairs", 2)]:  # The G_V query has two positives.
            def corrupt(*args, **kwargs):
                scores = real_ap(*args, **kwargs)
                scores.loc[2, column] = value
                return scores

            with self.subTest(column=column), patch.object(pa, "average_precision", side_effect=corrupt), \
                    patch.object(pa, "mean_average_precision") as mean_ap:
                with self.assertRaisesRegex(ValueError, "positive.*negative"):
                    run(pool())
                mean_ap.assert_not_called()




if __name__ == "__main__":
    unittest.main()
