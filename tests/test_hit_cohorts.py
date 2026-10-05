"""Small numerical/CLI contracts; not historical-data reproduction."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import polars as pl

from prot_loc_benchmark.benchmark.hits import hit_cohorts
from prot_loc_benchmark.config import BIOREP_PAIRS

PAIRS = {"first": ("A", "B"), "second": ("C", "D"), "third": ("E", "F")}
SETTINGS = [("r1", "GFP"), ("r2", "GFP")]


def pa_row(rep, batch, allele, score=.8, threshold=.5, p=.01):
    return dict(representation=rep, channel="GFP", batch=batch,
                Metadata_gene_allele=allele, mAP_vs_ref_norm=score,
                null_threshold_p95=threshold, corrected_p_value_vs_ref=p)


class HitCohortChecks(unittest.TestCase):
    def test_shared_support_and_distinct_denominators(self):
        rows = [pa_row(rep, batch, "paired", .4 if (rep, batch) == ("r1", "B") else .8)
                for rep, _ in SETTINGS for batch in ["A", "B"]]
        rows += [pa_row(rep, "A", "single") for rep, _ in SETTINGS]
        rows += [pa_row("r1", "A", "invalid", float("nan")),
                 pa_row("r2", "A", "invalid"), pa_row("r1", "A", "missing")]
        out = hit_cohorts(pl.DataFrame(rows), "pa", SETTINGS, PAIRS)
        self.assertEqual(out["shared_mask"].rows(), [("A", "paired"), ("A", "single"), ("B", "paired")])
        audit = out["batch_audit"]
        invalid = audit.filter((pl.col("allele") == "invalid") & (pl.col("representation") == "r1"))
        self.assertEqual(invalid["exclusion_reason"].item(), "nonfinite_score")
        self.assertIsNone(invalid["reporting_hit"].item())
        missing = audit.filter((pl.col("allele") == "missing") & (pl.col("representation") == "r2"))
        self.assertFalse(missing["observed"].item())
        self.assertEqual(missing["exclusion_reason"].item(), "missing_setting_row")
        for row in out["denominators"].iter_rows(named=True):
            self.assertEqual(row["n_alleles"], 2 if row["cohort"] == "any_available" else 1)
            expected_hits = 0 if (row["cohort"], row["representation"]) == ("complete_pair_all", "r1") else row["n_alleles"]
            self.assertEqual(row["n_hits"], expected_hits)
        # A requested setting with no rows must not silently disappear.
        empty = hit_cohorts(pl.DataFrame(rows), "pa", SETTINGS + [("absent", "GFP")], PAIRS)
        self.assertTrue(empty["shared_mask"].is_empty())
        self.assertTrue(empty["denominators"]["n_alleles"].eq(0).all())
        self.assertEqual(empty["denominators"]["hit_fraction"].null_count(), 9)

    def test_strict_joint_pa_rule_and_negative_scores(self):
        rows = [pa_row("r1", "A", "tie", .5), pa_row("r1", "A", "p_boundary", p=.05),
                pa_row("r1", "A", "negative", -.4, -.5),
                pa_row("r1", "A", "bad_p", p=float("inf")),
                pa_row("r1", "A", "no_threshold", threshold=None)]
        frame = pl.DataFrame(rows).with_columns(is_hit=pl.col("mAP_vs_ref_norm") > pl.col("null_threshold_p95"))
        out = hit_cohorts(frame, "pa", SETTINGS[:1], PAIRS)["batch_audit"]
        calls = dict(out.select("allele", "reporting_hit").rows())
        self.assertEqual(calls, dict(tie=False, p_boundary=False, negative=True, bad_p=None, no_threshold=None))
        self.assertTrue(out.filter(pl.col("allele") == "p_boundary")["is_hit"].item())
        self.assertEqual(out.filter(pl.col("allele") == "negative")["score"].item(), -.4)
        with self.assertRaisesRegex(ValueError, "disagrees"):
            hit_cohorts(frame.with_columns(is_hit=pl.lit(False)), "pa", SETTINGS[:1], PAIRS)

    def test_multiple_pairs_and_incomplete_extra_batch(self):
        rows = [pa_row("r1", b, "v", .4 if b in {"C", "E"} else .8) for b in "ABCDE"]
        out = hit_cohorts(pl.DataFrame(rows), "pa", SETTINGS[:1], PAIRS)["allele_summary"]
        calls = {r["cohort"]: r for r in out.iter_rows(named=True)}
        self.assertEqual(calls["any_available"]["n_batches"], 5)
        self.assertEqual(calls["complete_pair_all"]["n_batches"], 4)
        self.assertAlmostEqual(calls["complete_pair_all"]["mean_score"], .7)
        self.assertFalse(calls["complete_pair_all"]["hit"])
        self.assertTrue(calls["complete_pair_any"]["hit"])

    def test_invalid_input_and_empty_support(self):
        frame = pl.DataFrame([pa_row("r1", "A", "v")])
        for bad, settings, pairs in [
            (pl.concat([frame, frame]), SETTINGS[:1], PAIRS),
            (frame.drop("corrected_p_value_vs_ref"), SETTINGS[:1], PAIRS),
            (frame.with_columns(batch=pl.lit("unknown")), SETTINGS[:1], PAIRS),
            (frame.with_columns(representation=pl.lit("")), SETTINGS[:1], PAIRS),
            (frame, SETTINGS[:1] * 2, PAIRS),
            (frame, SETTINGS[:1], {"bad": ("A", "A")}),
        ]:
            with self.subTest(settings=settings, pairs=pairs), self.assertRaises(ValueError):
                hit_cohorts(bad, "pa", settings, pairs)
        for p in [None, -.1, 1.1, float("nan")]:
            bad = frame.with_columns(corrected_p_value_vs_ref=pl.lit(p, dtype=pl.Float64),
                                     is_hit=pl.lit(None, dtype=pl.String))
            out = hit_cohorts(bad, "pa", SETTINGS[:1], PAIRS)
            self.assertTrue(out["allele_summary"].is_empty())
            self.assertEqual(out["batch_audit"]["exclusion_reason"].item(), "invalid_corrected_p")
        out = hit_cohorts(frame.head(0), "pa", SETTINGS[:1], PAIRS)
        self.assertEqual(out["denominators"].height, 3)
        self.assertTrue(out["batch_audit"].is_empty())

    def test_xgb_rule_and_placeholder_disagreement(self):
        frame = pl.DataFrame(dict(representation=["r1"]*3, channel=["GFP"]*3, batch=["A"]*3,
                                  allele_var=["tie", "hit", "invalid"], auroc_mean=[.5, .6, float("inf")],
                                  null_threshold=[.5]*3))
        out = hit_cohorts(frame, "xgb", SETTINGS[:1], PAIRS)["batch_audit"]
        self.assertEqual(dict(out.select("allele", "reporting_hit").rows()), dict(tie=False, hit=True, invalid=None))
        with self.assertRaisesRegex(ValueError, "disagrees"):
            hit_cohorts(frame.with_columns(is_hit=pl.lit(False)), "xgb", SETTINGS[:1], PAIRS)
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            hit_cohorts(frame.with_columns(auroc_mean=pl.lit(1.2)), "xgb", SETTINGS[:1], PAIRS)

    def test_per_input_schema_before_union(self):
        first = pl.DataFrame([pa_row("r1", "A", "v")]).with_columns(
            is_hit=pl.lit(True), below_corrected_p_vs_ref=pl.lit(True)
        )
        second = pl.DataFrame([pa_row("r1", "B", "v")])
        out = hit_cohorts([first, second], "pa", SETTINGS[:1], PAIRS)
        self.assertTrue(out["allele_summary"]["hit"].all())
        self.assertEqual(out["batch_audit"]["is_hit"].to_list(), [True, None])
        for column in ["corrected_p_value_vs_ref", "null_threshold_p95", "mAP_vs_ref_norm"]:
            with self.subTest(column=column), self.assertRaisesRegex(ValueError, "Missing score columns"):
                hit_cohorts([first, second.drop(column)], "pa", SETTINGS[:1], PAIRS)
        with self.assertRaisesRegex(ValueError, "disagrees"):
            hit_cohorts([first, second.with_columns(is_hit=pl.lit(None, dtype=pl.Boolean))],
                        "pa", SETTINGS[:1], PAIRS)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            hit_cohorts([first, first], "pa", SETTINGS[:1], PAIRS)

    def test_real_cli_and_no_overwrite(self):
        script = Path(__file__).resolve().parents[1] / "scripts/10c_report_hits.py"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "report"
            command = [sys.executable, str(script), "--task", "xgb", "--setting", "vit", "GFP",
                       "--output-dir", str(out)]
            for i, batch in enumerate(BIOREP_PAIRS["pair_78"]):
                frame = pl.DataFrame(dict(allele_var=["v"], channel=["GFP"],
                                          auroc_mean=[.8 if i == 0 else .5], null_threshold=[.5]))
                if i == 0:
                    frame = frame.with_columns(is_hit=pl.lit(True))
                path = root / f"scores-{i}.csv"
                frame.write_csv(path)
                command += ["--input", "morphem", batch, str(path)]
            subprocess.run(command, check=True, capture_output=True, env=os.environ)
            summary = pl.read_parquet(out / "allele_summary.parquet")
            self.assertEqual(summary["hit"].to_list(), [True, False, True])
            self.assertEqual(summary["representation"].unique().to_list(), ["vit"])
            report = json.loads((out / "report.json").read_text())
            self.assertEqual(report["inputs"][0]["sha256"], hashlib.sha256((root / "scores-0.csv").read_bytes()).hexdigest())
            before = {p.name: p.read_bytes() for p in out.iterdir()}
            retry = subprocess.run(command, capture_output=True)
            self.assertNotEqual(retry.returncode, 0)
            self.assertEqual(before, {p.name: p.read_bytes() for p in out.iterdir()})
            self.assertEqual(len(before), 6)
            # Required columns cannot be rescued by another file's schema.
            missing = root / "missing-threshold.csv"
            pl.read_csv(root / "scores-1.csv").drop("null_threshold").write_csv(missing)
            bad_command = command.copy()
            bad_command[bad_command.index(str(out))] = str(root / "bad-schema")
            bad_command[-1] = str(missing)
            bad = subprocess.run(bad_command, capture_output=True)
            self.assertNotEqual(bad.returncode, 0)
            self.assertIn(b"Missing score columns", bad.stderr)
            self.assertFalse((root / "bad-schema/report.json").exists())
            # PA Parquet uses the same real CLI; a producer p95 hit alone is not
            # a reporting hit at the exact BH boundary.
            pa = root / "pa.parquet"
            batch = BIOREP_PAIRS["pair_78"][0]
            pl.DataFrame([pa_row("vit", batch, "v", p=.05)]).with_columns(
                is_hit=pl.lit(True)
            ).write_parquet(pa)
            pa_command = [sys.executable, str(script), "--task", "pa", "--setting", "vit", "GFP",
                          "--input", "vit", batch, str(pa), "--output-dir", str(root / "pa-report")]
            subprocess.run(pa_command, check=True, capture_output=True)
            audit = pl.read_parquet(root / "pa-report/batch_audit.parquet")
            self.assertTrue(audit["is_hit"].item())
            self.assertFalse(audit["reporting_hit"].item())
            pa_command[-1] = str(root / "bad-identity")
            pa_command[pa_command.index("--input") + 2] = BIOREP_PAIRS["pair_78"][1]
            self.assertNotEqual(subprocess.run(pa_command, capture_output=True).returncode, 0)
            self.assertFalse((root / "bad-identity/report.json").exists())


if __name__ == "__main__":
    unittest.main()
