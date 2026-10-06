"""Classification metrics: per-classifier and allele-level aggregation."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)

from prot_loc_benchmark.config import (
    CLASSIFICATION_OUTPUT_DIR,
    MAX_IMBALANCE_RATIO,
    MIN_CLASSIFIERS,
    NULL_PERCENTILE,
    canonical_representation,
)

logger = logging.getLogger(__name__)


def compute_classifier_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute all metrics for one classifier.

    Returns dict with: auroc, auprc, macro_f1, sensitivity,
    specificity, balanced_accuracy.
    """
    # Handle edge cases
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return {
            "auroc": float("nan"),
            "auprc": float("nan"),
            "macro_f1": float("nan"),
            "sensitivity": float("nan"),
            "specificity": float("nan"),
            "balanced_accuracy": float("nan"),
        }

    auroc = roc_auc_score(labels, predictions)
    auprc = average_precision_score(labels, predictions)

    # Binary predictions at threshold
    y_pred = (predictions >= threshold).astype(int)
    macro_f1 = f1_score(labels, y_pred, average="macro")
    bal_acc = balanced_accuracy_score(labels, y_pred)

    # Sensitivity (recall for positive class) and specificity
    tp = int(((y_pred == 1) & (labels == 1)).sum())
    fn = int(((y_pred == 0) & (labels == 1)).sum())
    tn = int(((y_pred == 0) & (labels == 0)).sum())
    fp = int(((y_pred == 1) & (labels == 0)).sum())

    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)

    return {
        "auroc": auroc,
        "auprc": auprc,
        "macro_f1": macro_f1,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": bal_acc,
    }


def compute_null_threshold(
    control_metrics: pl.DataFrame,
    percentile: int = NULL_PERCENTILE,
) -> dict[str, float]:
    """Compute per-channel hit threshold from control null distribution.

    Takes the Nth percentile of AUROC across all control classifiers,
    grouped by feature channel.

    Returns dict mapping channel name to AUROC threshold.
    """
    if control_metrics.is_empty():
        raise ValueError("No control metrics: run controls before calling hits")

    thresholds: dict[str, float] = {}
    q = percentile / 100.0

    if not 0 <= percentile <= 100:
        raise ValueError("percentile must be between 0 and 100")
    valid = control_metrics.filter(pl.col("auroc").is_finite())
    if control_metrics["auroc"].is_infinite().any() or valid.is_empty() or not valid["auroc"].is_between(0, 1).all():
        raise ValueError("Calibration requires finite AUROCs in [0, 1]")
    n_dropped = control_metrics.height - valid.height
    if n_dropped > 0:
        logger.info("Dropped %d control classifiers with NaN AUROC", n_dropped)

    for row in (
        valid.group_by("channel")
        .agg(pl.col("auroc").quantile(q, "nearest").alias("threshold"))
        .iter_rows(named=True)
    ):
        thresholds[row["channel"]] = row["threshold"]
        logger.info(
            "Null threshold for %s: %.4f (p%d)",
            row["channel"],
            row["threshold"],
            percentile,
        )

    return thresholds


def validate_thresholds(thresholds: dict[str, float], channels: list[str]) -> None:
    if any(not isinstance(thresholds.get(c), (int, float)) or not np.isfinite(thresholds[c])
           or not 0 <= thresholds[c] <= 1 for c in channels):
        raise ValueError("Missing or invalid channel calibration")


def aggregate_allele_metrics(
    metrics_df: pl.DataFrame,
    null_thresholds: dict[str, float],
    max_imbalance: float = MAX_IMBALANCE_RATIO,
    min_classifiers: int = MIN_CLASSIFIERS,
) -> pl.DataFrame:
    """Aggregate per-classifier metrics to allele-level summary.

    Steps:
    1. Filter out classifiers with imbalance_ratio > max_imbalance
    2. Group by (pair_id, gene, allele_var, channel)
    3. Require at least min_classifiers per group
    4. Compute mean AUROC and other summary stats
    5. Call hits: mean_auroc > null_threshold[channel]
    """
    validate_thresholds(null_thresholds, metrics_df["channel"].unique().to_list())
    filtered = metrics_df.filter((pl.col("imbalance_ratio") <= max_imbalance) & pl.col("auroc").is_finite())

    if filtered.is_empty():
        logger.warning("All classifiers filtered out by imbalance threshold")
        return pl.DataFrame()

    # Aggregate per (pair_id, channel)
    agg = (
        filtered.group_by("pair_id", "gene", "allele_var", "channel")
        .agg(
            pl.col("auroc").mean().alias("auroc_mean"),
            pl.col("auroc").std().alias("auroc_std"),
            pl.col("auprc").mean().alias("auprc_mean"),
            pl.col("balanced_accuracy").mean().alias("balanced_accuracy_mean"),
            pl.len().alias("n_classifiers"),
        )
        .filter(pl.col("n_classifiers") >= min_classifiers)
    )

    if agg.is_empty():
        logger.warning("No alleles with >= %d classifiers", min_classifiers)
        return pl.DataFrame()

    # Add null threshold and hit call
    agg = agg.with_columns(
        pl.col("channel")
        .replace_strict(null_thresholds, return_dtype=pl.Float64)
        .alias("null_threshold"),
    )
    agg = agg.with_columns(
        (pl.col("auroc_mean") > pl.col("null_threshold")).alias("is_hit"),
    )

    return agg.sort("gene", "allele_var", "channel")


def load_single_fold_metrics(
    representation: str,
    batch: str,
    test_plate_suffix: str = "T4",
    classification_dir: Path | None = None,
    max_imbalance: float = MAX_IMBALANCE_RATIO,
) -> pl.DataFrame:
    """Load classification metrics filtered to one held-out plate (default: T4).

    Reads ``metrics.csv`` (per-fold rows) + ``classifier_info.csv`` (which
    carries ``test_plates`` per fold), keeps only classifiers whose held-out
    plate name ends with ``test_plate_suffix``, and reshapes to the same
    column schema as ``metrics_summary.csv`` so it is a drop-in replacement
    for the 4-fold-mean view.

    Note: plate-naming differs by batch — B13/B14/B15/B16 use ``..._T4``
    (underscore separator), B7/B8 use ``...P1T4`` (no underscore). Both
    end with the literal substring ``T4``, so the default suffix matches
    all batches uniformly.

    For T4-only mode, training set is T1+T2+T3 and test set is T4 — the
    cleanest evaluation against DL encoders that were trained on T1+T2 with
    T3 as validation (T4 is fully held out from the encoder).

    Each (pair, channel) maps to exactly one fold here, so ``auroc_mean`` is
    the single-fold AUROC. Completed T4 runs validate their matched calibration.
    Legacy files remain score-only: uncertainty, threshold and hit are unavailable,
    never fabricated as 0 / 0.5 / False.
    """
    representation = canonical_representation(representation)
    root = classification_dir or CLASSIFICATION_OUTPUT_DIR
    direct = root / f"{representation}_t4" / batch
    if test_plate_suffix == "T4" and direct.exists():
        from .calibration import load_completed_summary
        return load_completed_summary(direct, representation, batch, max_imbalance)
    base = root / representation / batch
    info_path = base / "classifier_info.csv"
    metrics_path = base / "metrics.csv"
    if not info_path.exists() or not metrics_path.exists():
        logger.warning("Missing metrics.csv or classifier_info.csv for %s/%s", representation, batch)
        return pl.DataFrame()

    info = pl.read_csv(info_path)
    metrics = pl.read_csv(metrics_path)

    keep_ids = info.filter(
        pl.col("test_plates").str.ends_with(test_plate_suffix)
    ).select("classifier_id").unique()
    if keep_ids.is_empty():
        logger.warning(
            "No classifiers with test_plates ending in %r for %s/%s",
            test_plate_suffix, representation, batch,
        )
        return pl.DataFrame()

    one_fold = (
        metrics.join(keep_ids, on="classifier_id", how="inner")
        .filter(pl.col("category").is_in(["Exp", "cPC"]))
        .filter(pl.col("imbalance_ratio") <= max_imbalance)
    )
    if one_fold.is_empty():
        return pl.DataFrame()

    return one_fold.select([
        "pair_id",
        "gene",
        "allele_var",
        "channel",
        pl.col("auroc").alias("auroc_mean"),
        pl.lit(None, dtype=pl.Float64).alias("auroc_std"),
        pl.col("auprc").alias("auprc_mean"),
        pl.col("balanced_accuracy").alias("balanced_accuracy_mean"),
        pl.lit(1).alias("n_classifiers"),
        pl.lit(None, dtype=pl.Float64).alias("null_threshold"),
        pl.lit(None, dtype=pl.Boolean).alias("is_hit"),
    ])
