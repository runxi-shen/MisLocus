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

    def test_single_loo_error_cannot_produce_partial_calibration(self):
        controls = pl.DataFrame([
            {"Metadata_gene_allele": "G", "Metadata_symbol": "G", "Metadata_node_type": "disease_wt",
             "Metadata_Control": "NC", "Metadata_plate_map_name": "P1", "Metadata_Plate": f"P1T{t}",
             "Metadata_well_position": well, "Metadata_ImageNumber": well,
             "x": x, "y": 1. - x}
            for t in [1, 4] for well, x in [("A01", .1), ("A02", .5), ("A03", .9)]
        ])
        real_map = pa._compute_map_vs_ref
        calls = []

        def fail_second(*args, **kwargs):
            calls.append(args[1])
            if len(calls) == 2:
                raise RuntimeError("synthetic control failure")
            return real_map(*args, **kwargs)

        with patch.object(pa, "_compute_map_vs_ref", side_effect=fail_second):
            with self.assertRaisesRegex(RuntimeError, "synthetic control failure"):
                pa._compute_control_null(controls, {"EMBED": ["x", "y"]}, 20, 0, "site", True,
                                         32, .05, 42, 1, "t4", min_cells=1)
        self.assertEqual(len(calls), 2)
        full = pa._compute_control_null(controls, {"EMBED": ["x", "y"]}, 20, 0, "site", True,
                                        32, .05, 42, 1, "t4", min_cells=1)
        self.assertEqual(full.height, 3)
        self.assertTrue(full["mAP_vs_ref_norm"].is_finite().all())
        # One well has no T4 pseudo-query: exclude that run explicitly, without
        # confusing it with a failed calculation in either remaining well.
        missing_t4 = controls.filter(~((pl.col("Metadata_well_position") == "A01") &
                                      (pl.col("Metadata_Plate") == "P1T4")))
        remaining = pa._compute_control_null(missing_t4, {"EMBED": ["x", "y"]}, 20, 0, "site", True,
                                             32, .05, 42, 1, "t4", min_cells=1)
        self.assertEqual(set(remaining["well_var"]), {"A02", "A03"})

    def test_top_level_control_error_vs_explicit_empty_controls(self):
        for fails in [True, False]:
            with self.subTest(fails=fails), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                with patch.object(pa, "PHENOTYPIC_ACTIVITY_DIR", root), \
                        patch.object(pa, "_load_dl", return_value=pool()), \
                        patch.object(pa, "_resolve_alleles", return_value=["G_V", "G_W"]), \
                        patch.object(pa, "get_feature_channels", return_value={"EMBED": ["x", "y"]}), \
                        patch.object(pa, "_compute_control_null", return_value=pl.DataFrame(),
                                     side_effect=RuntimeError("control failed") if fails else None), \
                        patch.object(pa, "record") as record, \
                        patch.object(pa.sys, "argv", ["09c_classify_PA.py", "--batch", BATCH,
                                     "--representation", "vit", "--test-split", "t4", "--null-size", "32",
                                     "--max-workers", "1"]):
                    output = root / "vit_t4" / BATCH
                    if fails:
                        output.mkdir(parents=True)
                        previous = output / "mAP_results.parquet"
                        previous.write_bytes(b"previous result must not be rebound to this failed run")
                        with self.assertRaisesRegex(RuntimeError, "control failed"):
                            pa.main()
                        record.assert_not_called()
                        self.assertEqual(list(root.rglob("*.parquet")), [previous])
                        self.assertEqual(previous.read_bytes(), b"previous result must not be rebound to this failed run")
                    else:
                        pa.main()
                        record.assert_called_once_with(output_dirs=[output])
                        scores = pl.read_parquet(output / "mAP_results.parquet")
                        self.assertEqual(scores["is_hit"].null_count(), scores.height)
                        report = hit_cohorts(scores, "pa", [("vit", "EMBED")])
                        self.assertTrue(report["shared_mask"].is_empty())
                        self.assertEqual(report["batch_audit"]["reporting_hit"].null_count(), scores.height)


if __name__ == "__main__":
    unittest.main()
