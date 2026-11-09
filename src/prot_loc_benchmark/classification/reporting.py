"""Classification result reporting helpers.

Wide-format summary CSV + AUROC distribution plots. Extracted from
scripts/09_classify.py so multiple classification scripts can share.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from prot_loc_benchmark.config import NULL_PERCENTILE
from .metrics import compute_null_threshold

logger = logging.getLogger(__name__)


def write_wide_summary(
    summary: pl.DataFrame,
    output_dir: Path,
    batch_id: str,
) -> None:
    """Write wide-format classification summary: one row per allele, columns per channel.

    Produces a CSV matching the reference format:
    gene_allele, Gene, AUROC_{channel}, ..., Altered_p{N}_{channel}, ...
    """
    pct = NULL_PERCENTILE

    wide = summary.pivot(
        on="channel",
        index=["gene", "allele_var"],
        values=["auroc_mean", "is_hit", "null_threshold"],
    )

    rename_map = {}
    for col in wide.columns:
        if col.startswith("auroc_mean_"):
            ch = col.replace("auroc_mean_", "")
            rename_map[col] = f"AUROC_{ch}"
        elif col.startswith("is_hit_"):
            ch = col.replace("is_hit_", "")
            rename_map[col] = f"Altered_p{pct}_{ch}"
        elif col.startswith("null_threshold_"):
            ch = col.replace("null_threshold_", "")
            rename_map[col] = f"Null_p{pct}_{ch}"

    wide = wide.rename(rename_map)
    wide = wide.with_columns(pl.lit(batch_id).alias("Batch"))

    auroc_cols = sorted([c for c in wide.columns if c.startswith("AUROC_")])
    hit_cols = sorted([c for c in wide.columns if c.startswith("Altered_")])
    null_cols = sorted([c for c in wide.columns if c.startswith("Null_")])
    col_order = ["gene", "allele_var", "Batch"] + auroc_cols + hit_cols + null_cols
    col_order = [c for c in col_order if c in wide.columns]
    wide = wide.select(col_order)

    path = output_dir / "classification_summary.csv"
    wide.write_csv(str(path))
    logger.info("Wrote classification_summary.csv: %d alleles", wide.height)


def plot_auroc_distributions(
    control_metrics: pl.DataFrame,
    exp_metrics: pl.DataFrame,
    output_dir: Path,
    batch_id: str,
) -> None:
    """Plot AUROC distributions per channel: control null vs experimental."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning("matplotlib not available, skipping distribution plot")
        return

    ctrl = control_metrics.filter(~pl.col("auroc").is_nan())
    exp = exp_metrics.filter(~pl.col("auroc").is_nan())

    channels = sorted(ctrl["channel"].unique().to_list())
    n_ch = len(channels)
    if n_ch == 0:
        logger.warning(
            "No control channels with valid AUROC values, skipping distribution plot"
        )
        return

    thresholds = compute_null_threshold(control_metrics)

    fig, axes = plt.subplots(1, n_ch, figsize=(4 * n_ch, 4), sharey=False)
    if n_ch == 1:
        axes = [axes]

    for ax, ch in zip(axes, channels):
        ctrl_auroc = ctrl.filter(pl.col("channel") == ch)["auroc"].to_numpy()
        exp_auroc = exp.filter(pl.col("channel") == ch)["auroc"].to_numpy()

        bins = np.linspace(0.3, 1.0, 40)
        ax.hist(ctrl_auroc, bins=bins, alpha=0.6,
                label=f"Control (n={len(ctrl_auroc)})",
                color="steelblue", density=True)
        ax.hist(exp_auroc, bins=bins, alpha=0.6,
                label=f"Exp+cPC (n={len(exp_auroc)})",
                color="coral", density=True)

        if len(ctrl_auroc) > 0:
            pval = thresholds[ch]
            ax.axvline(pval, color="navy", linestyle="--", linewidth=1.5,
                       label=f"p{NULL_PERCENTILE}={pval:.3f}")

        ax.set_title(ch)
        ax.set_xlabel("AUROC")
        ax.legend(fontsize=7)

    axes[0].set_ylabel("Density")
    fig.suptitle(f"AUROC Distribution — {batch_id}", fontsize=13, y=1.02)
    fig.tight_layout()

    path = output_dir / "auroc_distribution.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote auroc_distribution.png")
