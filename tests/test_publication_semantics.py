"""Publication numerical semantics and configurable execution, on small inputs."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import polars as pl
from copairs import compute
from copairs.map import mean_average_precision as native_map
from scipy.stats import false_discovery_control

from prot_loc_benchmark.benchmark import clinvar
from prot_loc_benchmark import copairs_runtime
from test_hf_feature_names import hpa
from test_pa_support import BATCH, pa, pool, run


class PublicationSemanticsChecks(unittest.TestCase):
    def test_worker_setting_limits_similarity_and_null_pools(self):
        original = compute.ThreadPool
        with patch.object(compute, "ThreadPool", wraps=original) as pools:
            run(pool())
        self.assertTrue(pools.call_args_list)
        self.assertEqual({call.args[0] for call in pools.call_args_list}, {1})
        self.assertIs(compute.ThreadPool, original)

    def test_map_cache_does_not_depend_on_previous_calls(self):
        supported = pd.DataFrame({"allele": ["A", "B"], "n_pos_pairs": [1, 2],
                                  "n_total_pairs": [2, 4], "average_precision": [.6, .7],
                                  "normalized_average_precision": [.1, .2]})
        extra = pd.DataFrame({"allele": ["C"], "n_pos_pairs": [1], "n_total_pairs": [1],
                              "average_precision": [1.], "normalized_average_precision": [0.]})
        kwargs = dict(sameby=["allele"], null_size=512, threshold=.05, seed=42,
                      max_workers=1, progress_bar=False)
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"HOME": tmp}):
            expected = native_map(supported, cache_dir=Path(tmp) / "independent", **kwargs)
            pa.mean_average_precision(pd.concat([supported, extra]), **kwargs)
            actual = pa.mean_average_precision(supported, **kwargs)
            pd.testing.assert_frame_equal(actual, expected, check_exact=True)

    def test_cli_runtime_settings_and_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            for flags, workers, blas in [([], 16, 1), (["--max-workers", "2", "--blas-threads", "3"], 2, 3)]:
                with patch.object(pa, "PHENOTYPIC_ACTIVITY_DIR", Path(tmp)), \
                        patch.object(pa, "run_phenotypic_activity") as score, \
                        patch.object(pa.sys, "argv", ["pa", "--batch", BATCH, "--representation", "vit", *flags]):
                    pa.main()
                    self.assertEqual(score.call_args.kwargs["max_workers"], workers)
                    self.assertEqual(score.call_args.kwargs["blas_threads"], blas)
            for flag in ["--max-workers", "--blas-threads"]:
                with patch.object(pa, "run_phenotypic_activity") as score, \
                        patch.object(pa.sys, "argv", ["pa", "--batch", BATCH, "--representation", "vit", flag, "0"]):
                    with self.assertRaises(SystemExit):
                        pa.main()
                    score.assert_not_called()

    def test_blas_setting_reaches_both_hpa_paths_and_restores_on_failure(self):
        frame = pl.DataFrame({"Metadata_symbol": ["A", "B", "C", "D"],
                              "Metadata_node_type": ["disease_wt"] * 4,
                              "x": [1., .9, .1, .2], "y": [.1, .2, 1., .9]})
        labels = {"A": ["X"], "B": ["X"], "C": ["Y"], "D": ["Y"]}
        with patch.object(copairs_runtime, "threadpool_limits") as limits, \
                patch.object(pa, "_load_hpa_labels", return_value=labels):
            self.assertEqual(hpa._run_hpa_map(frame, ["x", "y"], labels, 32, .05, 42, 1, 3).height, 2)
            self.assertEqual(pa._compute_map_hpa(frame, ["x", "y"], 20, "site", True, 1., "gene", 32, .05, 42, 1, 3).height, 4)
            self.assertEqual({c.kwargs["limits"] for c in limits.call_args_list}, {3})
        original = compute.ThreadPool
        with self.assertRaisesRegex(RuntimeError, "failure"), copairs_runtime.bounded_copairs(1, 1):
            raise RuntimeError("failure")
        self.assertIs(compute.ThreadPool, original)

    def test_consensus_is_per_annotation_column(self):
        coarse, strict = "clinvar_clnsig_clean", "clinvar_clnsig_clean_pp_strict"
        source = pl.DataFrame({"gene_variant": ["G_V", "G_V", "H_V", "H_V"],
                               coarse: ["Pathogenic"] * 2 + ["Benign", "Benign"],
                               strict: ["Pathogenic", "Likely pathogenic", "Benign", "Benign"]})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "annotations.parquet"
            for frame in [source, source.reverse()]:
                frame.write_parquet(path)
                with patch.object(clinvar, "ALLELE_COLLECTION_PATH", path):
                    result = clinvar.load_clinvar_annotations().sort("gene_variant")
                self.assertEqual(result[coarse].to_list(), ["Pathogenic", "Benign"])
                self.assertEqual(result[strict].to_list(), [None, "Benign"])

    def test_bh_is_separate_for_each_nine_setting_comparison(self):
        rng = np.random.default_rng(42)
        data = pl.DataFrame([
            {"representation": f"r{rep}", "channel": "ALL", "allele_var": f"G_{i}",
             "auroc_avg": float(rng.normal() + rep / 5 * (i < 10)),
             "clinvar_clnsig_clean": "Pathogenic" if i < 10 else "Benign",
             "clinvar_clnsig_clean_pp_strict": "Pathogenic" if i < 5 else "Likely pathogenic" if i < 10 else "Benign"}
            for rep in range(9) for i in range(20)
        ])
        result = clinvar.run_wilcoxon_tests(data)
        self.assertEqual(result.height, 18)
        for frame in result.partition_by("comparison"):
            self.assertEqual(frame.height, 9)
            np.testing.assert_allclose(frame["pvalue_bh"], false_discovery_control(frame["pvalue"].to_numpy()), rtol=0, atol=1e-15)
        changed = clinvar.run_wilcoxon_tests(data.with_columns(
            clinvar_clnsig_clean_pp_strict=pl.col("clinvar_clnsig_clean")))
        coarse = pl.col("comparison") == "Pathogenic_vs_Benign"
        self.assertTrue(result.filter(coarse).equals(changed.filter(coarse)))
        for invalid in [data.filter(pl.col("representation") != "r0"),
                        data.with_columns(clinvar_clnsig_clean_pp_strict=pl.lit(None, dtype=pl.String))]:
            with self.assertRaisesRegex(ValueError, "nine"):
                clinvar.run_wilcoxon_tests(invalid)


if __name__ == "__main__":
    unittest.main()
