#!/usr/bin/env python3
"""Aggregate per-representation ClinVar benchmark summaries into cross-rep comparison.

Reads existing {dataset_dir}/{rep}/summary/ outputs produced by 10_benchmark_clinvar.py
and generates combined heatmaps, CSVs, and correlation plots in
{dataset_dir}/summary_across_reps/.

Usage:
    # Auto-discover all representations under full_dataset/
    pixi run python scripts/11_summarize_across_reps.py \
        --dataset-dir data/processed/benchmark/clinvar/full_dataset

    # Explicit list of representations
    pixi run python scripts/11_summarize_across_reps.py \
        --dataset-dir data/processed/benchmark/clinvar/full_dataset \
        --representations cellprofiler cytoself subcell_portable_rbg_vit
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prot_loc_benchmark.config import canonical_representation
from prot_loc_benchmark.predictor_annotations import PREDICTORS
from prot_loc_benchmark.viz.benchmark import (
    plot_auroc_correlation,
    plot_summary_heatmap,
)

log = logging.getLogger(__name__)


def discover_representations(dataset_dir: Path) -> list[str]:
    """Find all representation dirs that have summary/wilcoxon_results.csv."""
    reps = []
    for child in sorted(dataset_dir.iterdir()):
        if child.is_dir() and (child / "summary" / "wilcoxon_results.csv").exists():
            reps.append(child.name)
    return reps


def load_per_rep_summaries(
    dataset_dir: Path,
    representations: list[str],
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Load and concatenate per-rep summary CSVs."""
    avg_frames, clinvar_frames, wilcox_frames = [], [], []
    for rep in dict.fromkeys(map(canonical_representation, representations)):
        summary_dir = dataset_dir / rep / "summary"
        if not summary_dir.exists():
            log.warning("No summary dir for %s, skipping", rep)
            continue

        for fname, frames in [
            ("averaged_metrics.csv", avg_frames),
            ("averaged_metrics_clinvar.csv", clinvar_frames),
            ("wilcoxon_results.csv", wilcox_frames),
        ]:
            path = summary_dir / fname
            if not path.exists():
                log.warning("Missing %s for %s", fname, rep)
                continue
            df = pl.read_csv(path)
            # Empty CSVs (zero data rows) cause polars to infer every column
            # as String; concatenating them with populated frames raises
            # SchemaError. Skip them — typically a rep that was registered
            # but produced no classifier outputs in this dataset (e.g. seed
            # variants without _t4 runs).
            if df.height == 0:
                log.warning("Empty %s for %s, skipping", fname, rep)
                continue
            frames.append(df)

    return (
        pl.concat(avg_frames) if avg_frames else pl.DataFrame(),
        pl.concat(clinvar_frames) if clinvar_frames else pl.DataFrame(),
        pl.concat(wilcox_frames) if wilcox_frames else pl.DataFrame(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate per-rep ClinVar benchmarks into cross-rep summary."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Dataset directory (e.g., data/processed/benchmark/clinvar/full_dataset)",
    )
    parser.add_argument(
        "--representations",
        type=canonical_representation,
        nargs="+",
        default=None,
        help="Representations to include (default: auto-discover from dataset-dir)",
    )
    parser.add_argument(
        "--pa",
        action="store_true",
        help=(
            "Phenotypic-activity mode: plot heatmap with means instead of "
            "medians (default: auto-detect from dataset-dir name containing "
            "'clinvar_PA')."
        ),
    )
    parser.add_argument(
        "--predictor",
        choices=sorted(PREDICTORS),
        default=None,
        help=(
            "Predictor whose labels are in averaged_metrics_clinvar.csv. "
            "Used to pick the label column + palette for the cross-rep "
            "correlation scatter plots. Default: auto-detect from the "
            "dataset-dir name (falls back to clinvar)."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    dataset_dir = args.dataset_dir
    if not dataset_dir.exists():
        log.error("Dataset dir does not exist: %s", dataset_dir)
        sys.exit(1)

    representations = list(dict.fromkeys(args.representations or discover_representations(dataset_dir)))
    if len(representations) < 2:
        log.error(
            "Need ≥2 representations for cross-rep summary, found %d: %s",
            len(representations), representations,
        )
        sys.exit(1)
    log.info("Representations: %s", representations)

    # Load per-rep summaries
    averaged, annotated, stat_results = load_per_rep_summaries(dataset_dir, representations)
    log.info(
        "Loaded: %d averaged rows, %d annotated rows, %d stat rows",
        len(averaged), len(annotated), len(stat_results),
    )

    # Write cross-rep outputs
    cross_dir = dataset_dir / "summary_across_reps"
    cross_dir.mkdir(parents=True, exist_ok=True)

    averaged.write_csv(cross_dir / "averaged_metrics.csv")
    annotated.write_csv(cross_dir / "averaged_metrics_clinvar.csv")
    stat_results.write_csv(cross_dir / "wilcoxon_results.csv")
    is_pa = args.pa or "clinvar_PA" in str(dataset_dir) or "_PA" in dataset_dir.name

    # Pick the label column / palette from the active predictor. When the
    # column expected by the chosen predictor isn't present (most common when
    # a per-rep CSV came from the legacy clinvar pipeline that wrote
    # ``clinvar_clnsig_clean`` instead of ``clinvar_label``), fall back to
    # the legacy column name.
    predictor_key = args.predictor
    if predictor_key is None:
        for cand in PREDICTORS:
            if cand in str(dataset_dir):
                predictor_key = cand
                break
        if predictor_key is None:
            predictor_key = "clinvar"
    cfg = PREDICTORS[predictor_key]
    score_label = "norm-mAP" if is_pa else "AUROC"
    heatmap_title = (
        f"{cfg.title} benchmark — {cfg.tests[0].group_a} vs "
        f"{cfg.tests[0].group_b} {score_label}"
    )
    plot_summary_heatmap(
        stat_results, cross_dir,
        rep_order=representations,
        stat="mean" if is_pa else "median",
        title=heatmap_title,
    )

    # Cross-representation correlation plots vs cellprofiler GFP (the morphology baseline)
    reps_set = set(representations)
    label_alleles = ["F9_Glu73Val", "F9_Glu73Lys"]

    label_col = cfg.label_col
    if len(annotated) > 0 and label_col not in annotated.columns:
        legacy_col = "clinvar_clnsig_clean"
        if legacy_col in annotated.columns:
            log.info(
                "Predictor %s expects column %s but only %s is present — "
                "using legacy column for correlation plots.",
                predictor_key, label_col, legacy_col,
            )
            label_col = legacy_col
    palette = cfg.palette[cfg.label_col]
    pathogenic_label = cfg.tests[0].group_a
    benign_label = cfg.tests[0].group_b

    correlation_kwargs = dict(
        label_col=label_col,
        palette=palette,
        pathogenic_label=pathogenic_label,
        benign_label=benign_label,
    )

    # PA data uses channels suffixed with `_vs_ref` (e.g. GFP_vs_ref,
    # combined_vs_ref, EMBED_vs_ref); classification AUROC data uses bare
    # channel names. Append the suffix when running on a PA dataset so the
    # filter inside plot_auroc_correlation matches actual rows.
    pa_suffix = "_vs_ref" if is_pa else ""

    for anchor_rep in ("cellprofiler",):
        if anchor_rep not in reps_set:
            continue

        if "cytoself" in reps_set:
            plot_auroc_correlation(
                annotated,
                rep_a=anchor_rep, channel_a="GFP" + pa_suffix,
                rep_b="cytoself", channel_b="combined" + pa_suffix,
                output_dir=cross_dir,
                label_alleles=label_alleles,
                **correlation_kwargs,
            )

        # One correlation plot per SubCell portable variant
        for rep in representations:
            if rep.startswith("subcell_portable_"):
                plot_auroc_correlation(
                    annotated,
                    rep_a=anchor_rep, channel_a="GFP" + pa_suffix,
                    rep_b=rep, channel_b="EMBED" + pa_suffix,
                    output_dir=cross_dir,
                    label_alleles=label_alleles,
                    **correlation_kwargs,
                )

    log.info("Done. Cross-rep summary in %s", cross_dir)

    from prot_loc_benchmark.provenance import record
    record(output_dirs=[cross_dir])


if __name__ == "__main__":
    main()
