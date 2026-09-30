#!/usr/bin/env python3
"""Run XGBoost classification for one batch and representation.

Trains binary classifiers to distinguish variant allele cells from
reference allele cells using leave-one-out cross-validation.

Usage:
    # Full run (all alleles + controls)
    pixi run python scripts/09_classify.py \\
        --batch 2025_01_27_Batch_13 --representation cellprofiler

    # Quick test on a single allele
    pixi run python scripts/09_classify.py \\
        --batch 2025_01_27_Batch_13 --representation cellprofiler \\
        --scope CCM2_Ile432Thr

    # Controls only (null distribution)
    pixi run python scripts/09_classify.py \\
        --batch 2025_01_27_Batch_13 --representation cellprofiler \\
        --scope control

    # With GPU acceleration
    pixi run -e gpu python scripts/09_classify.py \\
        --batch 2025_01_27_Batch_13 --representation cellprofiler --gpu

"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import polars as pl

from prot_loc_benchmark.classification import (
    aggregate_allele_metrics,
    build_control_pairs,
    build_cpc_pairs,
    build_experimental_pairs,
    compute_null_threshold,
    filter_pairs_by_scope,
    generate_folds,
    get_feature_channels,
    get_pair_data,
    plot_auroc_distributions,
    run_classifier_tasks,
    select_device,
    split_fold,
    write_wide_summary,
)
from prot_loc_benchmark.config import (
    BATCH_LAYOUT,
    CLASSIFICATION_OUTPUT_DIR,
    INTERIM_DIR,
    MIN_CELL_COUNT,
    REP_FEATURE_FILES,
    canonical_representation,
)

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def _parse_template_number(plate: str) -> int | None:
    """Extract template number (1-4) from plate barcode."""
    import re
    # Use [0-9] instead of \d — conda-forge Python 3.12 has a broken \d
    m = re.search(r"T([0-9]+)$", plate)
    return int(m.group(1)) if m else None


def classify_batch(
    batch_id: str,
    representation: str,
    scope: str = "all",
    use_gpu: bool = False,
    unseen_only: bool = False,
    channels: list[str] | None = None,
) -> None:
    """Run full classification pipeline for one batch + representation."""
    t0 = time.time()

    # ── Resolve paths and config ─────────────────────────────────────
    representation = canonical_representation(representation)
    layout = BATCH_LAYOUT.get(batch_id)
    if layout is None:
        logger.error("Unknown batch %s (not in BATCH_LAYOUT)", batch_id)
        sys.exit(1)

    feature_file = REP_FEATURE_FILES.get(representation)
    if feature_file is None:
        logger.error("Unknown representation: %s", representation)
        sys.exit(1)

    features_path = INTERIM_DIR / representation / batch_id / feature_file
    if not features_path.exists():
        logger.error("Features not found: %s", features_path)
        sys.exit(1)

    suffix = "_unseen" if unseen_only else ""
    output_dir = CLASSIFICATION_OUTPUT_DIR / (representation + suffix) / batch_id
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("Classification: batch=%s rep=%s layout=%s scope=%s unseen_only=%s",
                batch_id, representation, layout, scope, unseen_only)
    logger.info("=" * 70)

    # ── Device selection ─────────────────────────────────────────────
    if use_gpu:
        os.environ["MISLOCUS_CLASSIFIER_BACKEND"] = "gpu"
    device = select_device()
    logger.info("Device: %s", device)

    # ── Load features ────────────────────────────────────────────────
    logger.info("Loading features from %s", features_path)
    lf = pl.scan_parquet(str(features_path))
    schema = lf.collect_schema()
    all_cols = schema.names()
    meta_cols = [c for c in all_cols if c.startswith("Metadata_")]
    feat_cols = [c for c in all_cols if not c.startswith("Metadata_")]
    logger.info("Schema: %d metadata, %d feature columns", len(meta_cols), len(feat_cols))

    # Eagerly collect the full dataframe once to avoid per-pair parquet re-scans.
    # ~727K rows x 1060 cols ≈ 3 GB in memory — acceptable for this pipeline.
    logger.info("Collecting full dataframe into memory...")
    t_load = time.time()
    df_full = lf.collect()
    logger.info("Loaded %d rows in %.1fs", df_full.height, time.time() - t_load)

    # Filter to unseen plates (T3+T4) for strict DL evaluation
    if unseen_only:
        df_full = df_full.with_columns(
            pl.col("Metadata_Plate").map_elements(
                _parse_template_number, return_dtype=pl.Int64
            ).alias("_template")
        )
        n_before = df_full.height
        df_full = df_full.filter(pl.col("_template").is_in([3, 4])).drop("_template")
        logger.info(
            "Unseen-only filter: %d → %d cells (T3+T4 plates only)",
            n_before, df_full.height,
        )

    # Wrap as lazy for pair building (which only needs group_by counts)
    lf = df_full.lazy()

    # ── Build pairs ──────────────────────────────────────────────────
    logger.info("Building classification pairs...")
    experimental_pairs = build_experimental_pairs(lf, min_cells=MIN_CELL_COUNT)
    cpc_pairs = build_cpc_pairs(lf, min_cells=MIN_CELL_COUNT)
    control_pairs = build_control_pairs(lf, control_types=["NC", "PC"], min_cells=MIN_CELL_COUNT)
    all_pairs = experimental_pairs + cpc_pairs + control_pairs

    # Filter by scope
    all_pairs = filter_pairs_by_scope(all_pairs, scope)
    if not all_pairs:
        logger.error("No pairs to classify after scope filter")
        sys.exit(1)

    # ── Feature channels ─────────────────────────────────────────────
    channel_features = get_feature_channels(feat_cols, representation)
    if channels:
        missing = [c for c in channels if c not in channel_features]
        if missing:
            logger.error(
                "Requested --channels %s not available for representation %r "
                "(available: %s)",
                missing, representation, sorted(channel_features),
            )
            sys.exit(1)
        channel_features = {c: channel_features[c] for c in channels}
        logger.info("Channel filter: restricting to %s", list(channel_features))

    # ── Classification loop ──────────────────────────────────────────
    n_pairs = len(all_pairs)
    n_skipped = 0

    # Build all classification tasks upfront for parallel execution
    tasks: list[dict] = []
    for pair in all_pairs:
        pair_df = get_pair_data(df_full, pair)
        if pair_df.is_empty():
            n_skipped += 1
            continue

        folds = generate_folds(pair_df, layout)
        if not folds:
            n_skipped += 1
            continue

        for channel, ch_features in channel_features.items():
            for fold in folds:
                train_df, test_df = split_fold(pair_df, fold, layout)
                if train_df.height < MIN_CELL_COUNT or test_df.height < 10:
                    continue

                tasks.append({
                    "pair": pair,
                    "channel": channel,
                    "ch_features": ch_features,
                    "fold": fold,
                    "train_df": train_df,
                    "test_df": test_df,
                })

    logger.info(
        "Prepared %d classifier tasks from %d pairs (%d skipped)",
        len(tasks), n_pairs, n_skipped,
    )

    metrics_rows, importance_rows, info_rows, n_classifiers = run_classifier_tasks(
        tasks, device=device, output_dir=output_dir,
    )

    # ── Write output files ───────────────────────────────────────────
    logger.info(
        "Classification complete: %d classifiers, %d skipped pairs",
        n_classifiers,
        n_skipped,
    )

    if not metrics_rows:
        logger.warning("No classifiers produced results")
        return

    metrics_df = pl.DataFrame(metrics_rows)
    metrics_df.write_csv(str(output_dir / "metrics.csv"))
    logger.info("Wrote metrics.csv (%d rows)", len(metrics_rows))

    pl.DataFrame(importance_rows).write_csv(str(output_dir / "feat_importance.csv"))
    logger.info("Wrote feat_importance.csv (%d rows)", len(importance_rows))

    pl.DataFrame(info_rows).write_csv(str(output_dir / "classifier_info.csv"))
    logger.info("Wrote classifier_info.csv (%d rows)", len(info_rows))

    # ── Null threshold + allele summary ──────────────────────────────
    # Control null distribution: NC + PC only
    control_metrics = metrics_df.filter(pl.col("category").is_in(["NC", "PC"]))
    # Experimental: Exp + cPC (both get ref-vs-var classification)
    exp_metrics = metrics_df.filter(pl.col("category").is_in(["Exp", "cPC"]))

    null_thresholds = compute_null_threshold(control_metrics)

    # Log category breakdown
    for cat, grp in metrics_df.group_by("category"):
        logger.info("  Category %s: %d classifiers", cat[0], grp.height)

    if not exp_metrics.is_empty():
        summary = aggregate_allele_metrics(exp_metrics, null_thresholds)
        if not summary.is_empty():
            summary.write_csv(str(output_dir / "metrics_summary.csv"))
            n_hits = int(summary["is_hit"].sum()) if "is_hit" in summary.columns else 0
            logger.info(
                "Wrote metrics_summary.csv: %d alleles, %d hits",
                summary.height,
                n_hits,
            )

            # ── Wide-format summary (one row per allele, columns per channel) ─
            write_wide_summary(summary, output_dir, batch_id)
        else:
            logger.info("No alleles passed aggregation filters")
    else:
        logger.info("No experimental metrics to aggregate (scope may be control-only)")

    # ── AUROC distribution plot ──────────────────────────────────────
    if not control_metrics.is_empty() and not exp_metrics.is_empty():
        plot_auroc_distributions(control_metrics, exp_metrics, output_dir, batch_id)

    elapsed = time.time() - t0
    logger.info("=" * 70)
    logger.info("Done in %.1f min", elapsed / 60)
    logger.info("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="XGBoost classification for variant mislocalization prediction."
    )
    parser.add_argument(
        "--batch",
        required=True,
        help="Batch ID (e.g., 2025_01_27_Batch_13)",
    )
    parser.add_argument(
        "--representation",
        required=True,
        type=canonical_representation,
        choices=sorted(REP_FEATURE_FILES),
        help="Feature representation to classify (morphem is an alias for vit)",
    )
    parser.add_argument(
        "--scope",
        default="all",
        help=(
            "Which pairs to classify: 'all' (default), 'exp', 'control', "
            "or comma-separated allele names (e.g., 'CCM2_Ile432Thr,KRAS_Gly12Val')"
        ),
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU-accelerated XGBoost",
    )
    parser.add_argument(
        "--unseen-only",
        action="store_true",
        help=(
            "Restrict to cells from plates unseen during DL model training "
            "(T3+T4 only). For strict no-leakage evaluation of DL embeddings. "
            "Results saved to {rep}_unseen/ subdirectory."
        ),
    )
    parser.add_argument(
        "--channels",
        default=None,
        help=(
            "Comma-separated subset of feature channels to classify. If omitted, "
            "all channels discovered by get_feature_channels() are used. Example: "
            "--channels combined (for Cytoself, skips global/spectrum and saves "
            "~2/3 of runtime)."
        ),
    )
    args = parser.parse_args()

    suffix = "_unseen" if args.unseen_only else ""
    output_dir = CLASSIFICATION_OUTPUT_DIR / (args.representation + suffix) / args.batch

    channels = (
        [c.strip() for c in args.channels.split(",") if c.strip()]
        if args.channels else None
    )

    try:
        classify_batch(
            batch_id=args.batch,
            representation=args.representation,
            scope=args.scope,
            use_gpu=args.gpu,
            unseen_only=args.unseen_only,
            channels=channels,
        )
    finally:
        if output_dir.exists():
            from prot_loc_benchmark.provenance import record
            record(output_dirs=[output_dir])


if __name__ == "__main__":
    main()
