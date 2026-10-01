"""Shared loaders + statistical helpers for ClinVar-style benchmarks.

Used by ``scripts/10_benchmark_clinvar.py`` for metrics loading, bio-rep
averaging, ClinVar joining, and Wilcoxon Pathogenic-vs-Benign tests.
Other benchmark scripts that compare per-allele AUROC against an external
label set should reuse these helpers.
"""

from __future__ import annotations

import logging

import numpy as np
import polars as pl
from scipy.stats import false_discovery_control, mannwhitneyu

from prot_loc_benchmark.classification.metrics import load_single_fold_metrics
from prot_loc_benchmark.config import (
    ALLELE_COLLECTION_PATH,
    CLASSIFICATION_OUTPUT_DIR,
    canonical_representation,
)

log = logging.getLogger(__name__)


# ============================================================================
# DATA LOADING
# ============================================================================


def load_metrics(
    representations: list[str],
    biorep_pairs: dict[str, tuple[str, str]],
    benchmark_channels: dict[str, list[str]],
    fold_mode: str = "full",
    test_plate_suffix: str = "T4",
) -> pl.DataFrame:
    """Load per-(rep, batch) classification metrics for ClinVar-style benchmarks.

    Args:
        representations: Reps to load.
        biorep_pairs: Mapping from pair name → (batch_a, batch_b).
        benchmark_channels: Per-rep list of channels to keep.
        fold_mode:
            ``"full"`` reads ``metrics_summary.csv`` (4-fold mean — original
            behavior).  ``"t4-only"`` reads ``metrics.csv`` +
            ``classifier_info.csv`` and keeps only the fold whose held-out
            plate ends with ``test_plate_suffix`` (default ``T4``), giving
            a clean train=T1+T2+T3 / test=T4 evaluation.
        test_plate_suffix: Suffix used to identify the held-out plate when
            ``fold_mode="t4-only"``. B13+ plates are ``..._T4`` and B7/B8
            plates are ``...P4T4``; both end with the literal ``T4`` so the
            default works for all batches.

    Returns:
        Long-format DataFrame: ``representation, pair_name, batch, pair_id,
        gene, allele_var, channel, auroc_mean, auroc_std, auprc_mean, ...``
        (same schema regardless of ``fold_mode``).
    """
    if fold_mode not in ("full", "t4-only"):
        raise ValueError(f"Unknown fold_mode: {fold_mode!r} (expected 'full' or 't4-only')")

    frames = []
    for rep in dict.fromkeys(map(canonical_representation, representations)):
        channels = benchmark_channels[rep]
        for pair_name, (batch_a, batch_b) in biorep_pairs.items():
            for batch in (batch_a, batch_b):
                if fold_mode == "full":
                    path = CLASSIFICATION_OUTPUT_DIR / rep / batch / "metrics_summary.csv"
                    if not path.exists():
                        log.warning("Missing: %s", path)
                        continue
                    df = pl.read_csv(path)
                else:
                    df = load_single_fold_metrics(
                        rep, batch, test_plate_suffix=test_plate_suffix
                    )
                    if df.is_empty():
                        continue

                df = (
                    df.filter(~pl.col("pair_id").str.starts_with("ctrl__"))
                    .filter(pl.col("channel").is_in(channels))
                    .with_columns(
                        representation=pl.lit(rep),
                        pair_name=pl.lit(pair_name),
                        batch=pl.lit(batch),
                    )
                )
                frames.append(df)
                log.info(
                    "Loaded %s/%s [%s]: %d alleles × %d channels",
                    rep, batch, fold_mode,
                    df["allele_var"].n_unique(),
                    df["channel"].n_unique(),
                )
    if not frames:
        raise ValueError(
            f"No metrics found for representations={representations} (fold_mode={fold_mode})"
        )
    return pl.concat(frames)


# ============================================================================
# BIO-REP AVERAGING
# ============================================================================


def average_across_bioreps(metrics: pl.DataFrame) -> pl.DataFrame:
    """Average AUROC across two batches within each bio-rep pair.

    Only keeps alleles present in BOTH batches of a pair.
    Pools alleles from different pairs after averaging.
    """
    averaged = (
        metrics.group_by(["representation", "pair_name", "allele_var", "gene", "channel"])
        .agg(
            auroc_avg=pl.col("auroc_mean").mean(),
            auroc_std_avg=pl.col("auroc_std").mean(),
            auprc_avg=pl.col("auprc_mean").mean(),
            n_batches=pl.col("batch").n_unique(),
        )
        .filter(pl.col("n_batches") == 2)
        .drop("n_batches", "pair_name")
    )
    log.info(
        "After bio-rep averaging: %d alleles × %d rep × %d channels",
        averaged["allele_var"].n_unique(),
        averaged["representation"].n_unique(),
        averaged["channel"].n_unique(),
    )
    return averaged


# ============================================================================
# CLINVAR ANNOTATION
# ============================================================================


def load_clinvar_annotations() -> pl.DataFrame:
    """Load ClinVar annotations from the merged allele collection.

    Returns DataFrame with columns: gene_variant, clinvar_clnsig_clean,
    clinvar_clnsig_clean_pp_strict (deduplicated on gene_variant).
    """
    df = pl.read_parquet(
        ALLELE_COLLECTION_PATH,
        columns=["gene_variant", "clinvar_clnsig_clean", "clinvar_clnsig_clean_pp_strict"],
    ).unique(subset=["gene_variant"])
    log.info("Allele collection: %d unique variants", len(df))
    return df


def join_clinvar(averaged: pl.DataFrame, clinvar: pl.DataFrame) -> pl.DataFrame:
    """Join averaged metrics with ClinVar annotations.

    Drops rows without ClinVar annotation.
    """
    n_before = averaged["allele_var"].n_unique()
    joined = averaged.join(
        clinvar,
        left_on="allele_var",
        right_on="gene_variant",
        how="left",
    )
    annotated = joined.filter(pl.col("clinvar_clnsig_clean").is_not_null())
    n_after = annotated["allele_var"].n_unique()
    log.info(
        "ClinVar join: %d → %d alleles (%d dropped, no annotation)",
        n_before, n_after, n_before - n_after,
    )
    return annotated


# ============================================================================
# STATISTICAL TESTING
# ============================================================================


def run_wilcoxon_tests(data: pl.DataFrame) -> pl.DataFrame:
    """Run Wilcoxon rank-sum tests: Pathogenic vs Benign per channel × representation.

    Tests on both ClinVar groupings:
    1. clinvar_clnsig_clean: Pathogenic vs Benign (5-cat)
    2. clinvar_clnsig_clean_pp_strict: Pathogenic vs Benign (7-cat, strict split)

    Returns DataFrame with test results and BH-corrected p-values.
    """
    results = []
    for rep in sorted(data["representation"].unique().to_list()):
        rep_data = data.filter(pl.col("representation") == rep)
        for channel in sorted(rep_data["channel"].unique().to_list()):
            ch_data = rep_data.filter(pl.col("channel") == channel)

            for comparison, col, group_a, group_b in [
                ("Pathogenic_vs_Benign", "clinvar_clnsig_clean", "Pathogenic", "Benign"),
                ("Pathogenic_vs_Benign_strict", "clinvar_clnsig_clean_pp_strict", "Pathogenic", "Benign"),
            ]:
                vals_a = ch_data.filter(
                    (pl.col(col) == group_a) & pl.col("auroc_avg").is_not_nan()
                )["auroc_avg"].to_numpy()
                vals_b = ch_data.filter(
                    (pl.col(col) == group_b) & pl.col("auroc_avg").is_not_nan()
                )["auroc_avg"].to_numpy()

                if len(vals_a) < 2 or len(vals_b) < 2:
                    log.warning(
                        "Skipping %s/%s/%s: n=%d vs n=%d",
                        rep, channel, comparison, len(vals_a), len(vals_b),
                    )
                    continue

                stat, pval = mannwhitneyu(vals_a, vals_b, alternative="two-sided")
                results.append(
                    {
                        "representation": rep,
                        "channel": channel,
                        "comparison": comparison,
                        "clinvar_col": col,
                        "group_a": group_a,
                        "group_b": group_b,
                        "n_a": len(vals_a),
                        "n_b": len(vals_b),
                        "median_a": float(np.median(vals_a)),
                        "median_b": float(np.median(vals_b)),
                        "mean_a": float(np.mean(vals_a)),
                        "mean_b": float(np.mean(vals_b)),
                        "U_statistic": float(stat),
                        "pvalue": float(pval),
                    }
                )

    result_df = pl.DataFrame(results)
    if len(result_df) > 0:
        pvals = result_df["pvalue"].to_numpy()
        pvals_bh = false_discovery_control(pvals, method="bh")
        result_df = result_df.with_columns(pvalue_bh=pl.Series(pvals_bh))
    return result_df
