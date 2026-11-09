"""Synthetic copairs checks: T4 selects queries, never the comparison pool."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import polars as pl
from copairs.matching import find_pairs

spec = importlib.util.spec_from_file_location(
    "pa_query_pool", Path(__file__).resolve().parents[1] / "scripts/09c_classify_PA.py"
)
pa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pa)


class PAQueryPoolChecks(unittest.TestCase):
    def test_t4_selects_queries_after_full_pool_ap(self):
        pool = pl.DataFrame({
            "Metadata_gene_allele": ["G_V"] * 5 + ["G", "G", "G_W", "G_W", "H", "G_V"],
            "Metadata_symbol": ["G"] * 9 + ["H", "G"],
            "Metadata_node_type": ["allele"] * 5 + ["disease_wt"] * 2
                                  + ["allele"] * 2 + ["disease_wt", "allele"],
            "Metadata_Plate": ["P1T1", "P1T2", "P1T3", "P1T4", "P2T4",
                               "P1T4", "P2T4", "P1T4", "P1T1", "P1T4", "P1T4"],
            "x": [1., .8, .6, .9, .7, 0., .1, .2, .1, .3, .95],
            "y": [0., .2, .4, .1, .3, 1., .9, .8, .9, .7, .05],
        })
        # Keep this ordering test fully supported in both split modes.
        pool = pool.vstack(pl.DataFrame({
            "Metadata_gene_allele": ["G"] * 3, "Metadata_symbol": ["G"] * 3,
            "Metadata_node_type": ["disease_wt"] * 3,
            "Metadata_Plate": ["P1T1", "P1T2", "P1T3"],
            "x": [.1, .2, .3], "y": [.9, .8, .7],
        }))
        observed = []
        real_ap = pa.average_precision

        def capture_ap(meta, features, **kwargs):
            result = real_ap(meta, features, **kwargs)
            observed.append(result.copy())
            positive = {tuple(pair) for pair in find_pairs(
                meta, sameby=kwargs["pos_sameby"], diffby=kwargs["pos_diffby"]
            )}
            negative = {tuple(pair) for pair in find_pairs(
                meta, sameby=kwargs["neg_sameby"], diffby=kwargs["neg_diffby"]
            )}
            partners = lambda pairs, i: {b if a == i else a for a, b in pairs if i in (a, b)}
            self.assertEqual(partners(positive, 3), {0, 1, 2, 4})
            self.assertEqual(partners(negative, 3), {5})
            self.assertEqual(partners(positive, 4), {0, 1, 2, 3, 10})
            self.assertFalse(any(a in {5, 6, 9, 11, 12, 13} and b in {5, 6, 9, 11, 12, 13} for a, b in positive))
            return result

        for split, indices in [("t4", [3, 4, 7, 10]), (None, [0, 1, 2, 3, 4, 7, 8, 10])]:
            with self.subTest(split=split), patch.object(pa, "average_precision", side_effect=capture_ap), patch.object(
                pa, "mean_average_precision", wraps=pa.mean_average_precision
            ) as mean_ap:
                result = pa._run_map(
                    pool, ["x", "y"], "Metadata_node_type == 'disease_wt'",
                    null_size=32, threshold=.05, seed=42, label="vs_ref", max_workers=1,
                    neg_sameby=["Metadata_Plate", "Metadata_symbol"], test_split=split,
                )
                queries = mean_ap.call_args.args[0]
                self.assertEqual(queries.index.tolist(), indices)
                pd.testing.assert_frame_equal(queries, observed[-1].loc[indices])
                self.assertEqual(mean_ap.call_args.kwargs["sameby"], ["Metadata_gene_allele"])
                self.assertEqual(mean_ap.call_args.kwargs["null_size"], 32)
                expected = queries.groupby("Metadata_gene_allele")["normalized_average_precision"].mean()
                np.testing.assert_allclose(result.set_index("Metadata_gene_allele")["mAP_vs_ref_norm"], expected)
        pd.testing.assert_frame_equal(observed[0], observed[1])
        # A literal T4-only pool loses G_W's only positive; G_V retains
        # cross-platemap partners. Multiple T4 plates do not imply support for every allele.
        t4 = observed[0].loc[observed[0]["Metadata_Plate"].str.endswith("T4")].reset_index(drop=True)
        pairs = find_pairs(t4, sameby=["Metadata_gene_allele", pa.REFERENCE_COL], diffby=["Metadata_Plate"])
        supported = {t4.iloc[int(i)]["Metadata_gene_allele"] for i in pairs.ravel()}
        self.assertEqual(supported, {"G_V"})

    def test_loo_controls_keep_cross_plate_partners_and_t4_queries(self):
        rows = []
        for t in range(1, 5):
            for well, x in [("A01", 1.), ("A02", .1), ("A03", .2)]:
                for site in (1, 2):
                    rows.append({
                        "Metadata_gene_allele": "G", "Metadata_symbol": "G",
                        "Metadata_node_type": "disease_wt", "Metadata_Control": "NC",
                        "Metadata_plate_map_name": "P1", "Metadata_Plate": f"P1T{t}",
                        "Metadata_well_position": well, "Metadata_ImageNumber": f"{well}_{site}",
                        "x": x, "y": 1. - x + site / 100,
                    })
        cells = pl.DataFrame(rows)
        allele, pool = pa._build_pseudo_loo_pool(cells, "G", "P1", "A01", ["A02", "A03"])
        with patch.object(pa, "mean_average_precision", wraps=pa.mean_average_precision) as mean_ap:
            result = pa._compute_map_vs_ref(
                pool, [allele], ["x", "y"], cells_per_site=20, neg_per_plate=0,
                sample_level="site", aggregate=True, null_size=32, threshold=.05,
                seed=42, max_workers=1, test_split="t4",
            )
        queries = mean_ap.call_args.args[0]
        self.assertEqual(queries["Metadata_Plate"].tolist(), ["P1T4", "P1T4"])
        self.assertEqual(queries["n_pos_pairs"].tolist(), [6, 6])
        self.assertEqual(queries["n_total_pairs"].tolist(), [10, 10])
        self.assertEqual(result["Metadata_gene_allele"].to_list(), [allele])
        t4 = pool.filter(pl.col("Metadata_Plate") == "P1T4").to_pandas()
        t4 = pa.assign_reference_index(t4, "Metadata_node_type == 'disease_wt'", reference_col=pa.REFERENCE_COL)
        pairs = find_pairs(t4, sameby=["Metadata_gene_allele", pa.REFERENCE_COL], diffby=["Metadata_Plate"])
        self.assertEqual(len(pairs), 0)


if __name__ == "__main__":
    unittest.main()
