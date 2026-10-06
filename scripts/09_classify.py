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
import sys
import time
from pathlib import Path

import polars as pl

from prot_loc_benchmark.classification import (
    aggregate_allele_metrics,
    build_control_pairs,
    build_cpc_pairs,
    build_experimental_pairs,
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
    XGBOOST_PARAMS,
    REP_FEATURE_FILES,
    canonical_representation,
)

from prot_loc_benchmark.classification.calibration import calibration_context, load_calibration, save_calibration
from prot_loc_benchmark.provenance import save_json, sha256

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


def _execute(tasks, directory, device, workers, params):
    metrics, importance, info, _ = run_classifier_tasks(
        tasks, device=device, output_dir=directory, workers=workers, xgb_params=params)
    if not metrics:
        raise ValueError("No classifiers produced results")
    for name, rows in [("metrics", metrics), ("feat_importance", importance), ("classifier_info", info)]:
        pl.DataFrame(rows).write_csv(directory / f"{name}.csv")
    return pl.DataFrame(metrics)


def classify_batch(
    batch_id: str,
    representation: str,
    scope: str = "all",
    use_gpu: bool = False,
    unseen_only: bool = False,
    channels: list[str] | None = None,
    test_split: str | None = None,
    workers: int = 2,
    threads: int = 1,
    calibration_dir: Path | None = None,
) -> None:
    """Run full classification pipeline for one batch + representation."""
    t0 = time.time()
    if workers < 1 or threads < 1:
        raise ValueError("workers and threads must be positive")
    if test_split not in (None, "t4") or (unseen_only and test_split):
        raise ValueError("T4 evaluation cannot be combined with unseen-only")
    if scope == "control" and calibration_dir is not None:
        raise ValueError("A control run cannot reuse calibration")

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

    if test_split and layout != "single_rep":
        raise ValueError("T4 evaluation requires single_rep layout")
    suffix = "_t4" if test_split else ("_unseen" if unseen_only else "")
    output_dir = CLASSIFICATION_OUTPUT_DIR / (representation + suffix) / batch_id
    control_dir = output_dir / "controls"
    if scope != "control" and output_dir.exists() and any(p.name != "controls" for p in output_dir.iterdir()):
        raise FileExistsError(output_dir)
    if scope == "control" and control_dir.exists():
        raise FileExistsError(control_dir)

    logger.info("=" * 70)
    logger.info("Classification: batch=%s rep=%s layout=%s scope=%s unseen_only=%s",
                batch_id, representation, layout, scope, unseen_only)
    logger.info("=" * 70)

    # ── Device selection ─────────────────────────────────────────────
    device = select_device("gpu" if use_gpu else None)
    logger.info("Device: %s", device)

    # ── Load features ────────────────────────────────────────────────
    logger.info("Loading features from %s", features_path)
    input_sha256 = sha256(features_path)
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
    if not df_full.select(pl.all_horizontal(pl.col(feat_cols).is_finite().fill_null(False)).all()).item():
        raise ValueError("Classification requires finite features")
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
    if calibration_dir is None and {p.pair_id for p in all_pairs if p.is_control} != {p.pair_id for p in control_pairs}:
        raise ValueError("Calibration requires the full NC+PC scope or --calibration-dir")
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

        folds = generate_folds(pair_df, layout, test_split=test_split)
        if not folds:
            n_skipped += 1
            continue

        for channel, ch_features in channel_features.items():
            for fold in folds:
                train_df, test_df = split_fold(pair_df, fold, layout)
                if train_df.height < MIN_CELL_COUNT or test_df.height < 10:
                    continue
                train_counts = train_df["Label"].value_counts()["count"]
                if len(train_counts) != 2 or test_df["Label"].n_unique() != 2 or train_counts.max() / train_counts.min() > 100:
                    logger.info("Skipping unsupported fold %s for %s", fold.fold_id, pair.pair_id)
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

    params = {**XGBOOST_PARAMS, "n_jobs": threads}
    context = calibration_context(features_path, representation, batch_id, channel_features,
                                  test_split or ("unseen_lopo" if unseen_only else "lopo"), params, device)
    if context["features_sha256"] != input_sha256:
        raise ValueError("Features changed during loading")
    if calibration_dir is not None:
        control_dir = Path(calibration_dir).resolve()
        null_thresholds, control_metrics = load_calibration(control_dir, context)
    else:
        control_tasks = [t for t in tasks if t["pair"].is_control]
        if not control_tasks:
            raise ValueError("Experimental runs require --calibration-dir or same-run NC+PC controls")
        control_dir.mkdir(parents=True, exist_ok=False)
        control_metrics = _execute(control_tasks, control_dir, device, workers, params)
        null_thresholds = save_calibration(control_dir, context)
    if scope == "control":
        return
    # A controls-only parent is allowed; never reuse any experimental output.
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "started").touch(exist_ok=False)
    exp_tasks = [t for t in tasks if not t["pair"].is_control]
    exp_metrics = _execute(exp_tasks, output_dir, device, workers, params)

    summary = aggregate_allele_metrics(exp_metrics, null_thresholds, **({"min_classifiers": 1} if test_split else {}))
    if summary.is_empty():
        raise ValueError("No alleles passed aggregation filters")
    summary.write_csv(str(output_dir / "metrics_summary.csv"))
    logger.info("Wrote metrics_summary.csv: %d alleles, %d hits", summary.height, summary["is_hit"].sum())
    write_wide_summary(summary, output_dir, batch_id)
    plot_auroc_distributions(control_metrics, exp_metrics, output_dir, batch_id)

    save_json(output_dir / "completion.json", dict(status="complete", context=context,
        calibration_dir=str(control_dir.resolve()), calibration_sha256=sha256(control_dir / "calibration.json"),
        summary_sha256=sha256(output_dir / "metrics_summary.csv")))
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
    parser.add_argument("--test-split", choices=["t4"], help="Train T1/T2/T3, test T4; separate _t4 outputs")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--calibration-dir", type=Path, help="Completed controls from matching input/settings/backend")
    args = parser.parse_args()

    suffix = "_t4" if args.test_split else ("_unseen" if args.unseen_only else "")
    output_dir = CLASSIFICATION_OUTPUT_DIR / (args.representation + suffix) / args.batch

    channels = (
        [c.strip() for c in args.channels.split(",") if c.strip()]
        if args.channels else None
    )

    classify_batch(
        batch_id=args.batch, representation=args.representation, scope=args.scope,
        use_gpu=args.gpu, unseen_only=args.unseen_only, channels=channels,
        test_split=args.test_split, workers=args.workers, threads=args.threads,
        calibration_dir=args.calibration_dir,
    )
    from prot_loc_benchmark.provenance import record
    if output_dir.exists():
        record(output_dirs=[output_dir / "controls" if args.scope == "control" else output_dir])


if __name__ == "__main__":
    main()
