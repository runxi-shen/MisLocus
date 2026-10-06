"""Benchmark 1: ClinVar Pathogenic vs Benign AUROC comparison.

Averages per-allele AUROC across biological replicate batch pairs,
annotates with ClinVar clinical significance, and compares AUROC
distributions (Pathogenic vs Benign) via Wilcoxon rank-sum test.

Outputs per-representation results only. For cross-representation
comparison, run 11_summarize_across_reps.py after benchmarking all reps.

``--fold-mode`` controls which classification fold is used:
- ``full`` (default): mean across all 4 LOPO folds  → ``clinvar/full_dataset/``
- ``t4-only``: single T4-test fold (T1+T2+T3 train) → ``clinvar/single_fold/``
The two modes are complementary: ``full`` uses every cell as both train and
test (mixed encoder-seen / encoder-unseen), ``t4-only`` is the clean
encoder-unseen evaluation that mirrors the DL train/val/test split.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import polars as pl

from prot_loc_benchmark.benchmark.clinvar import (
    average_across_bioreps,
    join_clinvar,
    load_clinvar_annotations,
    load_metrics,
    run_wilcoxon_tests,
)
from prot_loc_benchmark.config import (
    BENCHMARK_CHANNELS,
    BIOREP_PAIRS,
    CLASSIFICATION_PA_DIR,
    CLINVAR_BENCHMARK_DIR,
    CLINVAR_SINGLE_FOLD_DIR,
    canonical_representation,
)

from prot_loc_benchmark.viz.benchmark import (
    plot_clinvar_violin,
    plot_summary_heatmap,
)
from prot_loc_benchmark.viz.palettes import CLINVAR_ORDER, CLINVAR_PALETTE

log = logging.getLogger(__name__)


# ============================================================================
# DATA LOADING (PA-mode only — main load_metrics moved to benchmark.clinvar)
# ============================================================================


def load_pa_metrics(
    representations: list[str],
    biorep_pairs: dict[str, tuple[str, str]],
) -> pl.DataFrame:
    """Load mAP_results.parquet from classification_PA for all reps and batch pairs.

    Reshapes normalized mAP vs reference (same-gene disease_wt) scores into
    the long format expected by average_across_bioreps. mAP_vs_NC is ignored
    if present (legacy column).
    """
    frames = []
    for rep in dict.fromkeys(map(canonical_representation, representations)):
        for pair_name, (batch_a, batch_b) in biorep_pairs.items():
            for batch in (batch_a, batch_b):
                path = CLASSIFICATION_PA_DIR / rep / batch / "mAP_results.parquet"
                if not path.exists():
                    log.warning("Missing PA: %s", path)
                    continue
                df = pl.read_parquet(str(path))

                # Use normalized mAP vs same-gene reference (scale-independent).
                has_channel = "channel" in df.columns
                for label in ("vs_ref",):
                    norm_col = f"mAP_{label}_norm"
                    if norm_col not in df.columns:
                        continue
                    channel_expr = (
                        (pl.col("channel") + "_" + pl.lit(label))
                        if has_channel
                        else pl.lit(f"mAP_{label}")
                    )
                    sub = df.select([
                        pl.col("Metadata_gene_allele").alias("allele_var"),
                        pl.col("Metadata_gene_allele").str.split("_").list.first().alias("gene"),
                        pl.col(norm_col).alias("auroc_mean"),
                        pl.lit(0.0).alias("auroc_std"),
                        pl.lit(0.0).alias("auprc_mean"),
                        channel_expr.alias("channel"),
                        pl.lit(rep).alias("representation"),
                        pl.lit(pair_name).alias("pair_name"),
                        pl.lit(batch).alias("batch"),
                    ]).drop_nulls(subset=["auroc_mean"])
                    frames.append(sub)
                n_ch = df["channel"].n_unique() if has_channel else 1
                log.info("Loaded PA %s/%s: %d alleles × %d channels", rep, batch, df["Metadata_gene_allele"].n_unique(), n_ch)

    if not frames:
        raise ValueError("No PA metrics found for the given representations/batches")
    return pl.concat(frames)


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark ClinVar pathogenic vs benign AUROC comparison."
    )
    parser.add_argument(
        "--representations",
        type=canonical_representation,
        nargs="+",
        default=["cellprofiler", "cytoself", "cytoself_unseen"],
        help="Representations to benchmark (default: cellprofiler cytoself cytoself_unseen)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Default depends on --fold-mode: "
            "full → CLINVAR_BENCHMARK_DIR (clinvar/full_dataset/); "
            "t4-only → CLINVAR_SINGLE_FOLD_DIR (clinvar/single_fold/)."
        ),
    )
    parser.add_argument(
        "--exclude-genes",
        nargs="+",
        default=None,
        help="Gene symbols to exclude (e.g., --exclude-genes BRCA1 BRCA2)",
    )
    parser.add_argument(
        "--pa",
        action="store_true",
        help="Use phenotypic activity (mAP) scores from 09c_classify_PA.py instead of AUROC",
    )
    parser.add_argument(
        "--fold-mode",
        choices=["full", "t4-only"],
        default="full",
        help=(
            "full: 4-fold mean from metrics_summary.csv (original behavior). "
            "t4-only: re-derive single-fold AUROC from metrics.csv where the "
            "held-out plate ends with T4, giving a clean train=T1+T2+T3 / "
            "test=T4 evaluation. Ignored if --pa is set."
        ),
    )
    parser.add_argument("--setting", action="append", nargs=2, metavar=("REP", "CHANNEL"),
                        help="Select each of nine test settings; PA channel names end in _vs_ref")
    args = parser.parse_args()
    if args.setting:
        args.setting = [(canonical_representation(rep), channel) for rep, channel in args.setting]
        if len(args.setting) != 9 or len(set(args.setting)) != 9:
            parser.error("--setting requires nine distinct representation/channel settings")
        args.representations = [rep for rep, _ in args.setting]
    args.representations = list(dict.fromkeys(args.representations))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.output_dir:
        output_dir = args.output_dir
    elif args.fold_mode == "t4-only" and not args.pa:
        output_dir = CLINVAR_SINGLE_FOLD_DIR
    else:
        output_dir = CLINVAR_BENCHMARK_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load metrics
    if args.pa:
        log.info("Loading phenotypic activity (mAP) metrics...")
        metrics = load_pa_metrics(args.representations, BIOREP_PAIRS)
    else:
        log.info("Loading classification metrics (fold_mode=%s)...", args.fold_mode)
        metrics = load_metrics(
            args.representations, BIOREP_PAIRS, BENCHMARK_CHANNELS,
            fold_mode=args.fold_mode,
        )

    if args.setting:
        settings = pl.DataFrame(args.setting, schema=["representation", "channel"], orient="row")
        metrics = metrics.join(settings, on=["representation", "channel"], how="semi")

    # Step 2: Average across bio-reps
    log.info("Averaging across biological replicates...")
    averaged = average_across_bioreps(metrics)

    # Step 2b: Exclude genes if requested
    if args.exclude_genes:
        n_before = averaged.select("allele_var").unique().height
        averaged = averaged.filter(~pl.col("gene").is_in(args.exclude_genes))
        n_after = averaged.select("allele_var").unique().height
        log.info("Excluded genes %s: %d → %d alleles", args.exclude_genes, n_before, n_after)

    # Step 3: Join ClinVar
    log.info("Loading ClinVar annotations...")
    clinvar = load_clinvar_annotations()
    annotated = join_clinvar(averaged, clinvar)

    # Step 4: Statistical tests
    log.info("Running Wilcoxon rank-sum tests...")
    stat_results = run_wilcoxon_tests(annotated)
    log.info("Test results:\n%s", stat_results)

    # Step 5: Per-representation outputs (violins + per-rep summary)
    log.info("Generating per-representation outputs...")

    # Plot parameters depend on score type
    if args.pa:
        violin_kwargs = dict(ylabel="Norm. mAP (bio-rep avg)", ylim=(-0.15, 0.8), yref=0.0)
    else:
        violin_kwargs = dict(ylabel="AUROC (bio-rep avg)", ylim=(0.3, 1.12), yref=0.5)

    for rep in args.representations:
        rep_dir = output_dir / rep
        rep_dir.mkdir(parents=True, exist_ok=True)
        rep_summary_dir = rep_dir / "summary"
        rep_summary_dir.mkdir(parents=True, exist_ok=True)

        for clinvar_col, palette_key, order_key in [
            ("clinvar_clnsig_clean", "clinvar_clnsig_clean", "clinvar_clnsig_clean"),
            ("clinvar_clnsig_clean_pp_strict", "clinvar_clnsig_clean_pp_strict", "clinvar_clnsig_clean_pp_strict"),
        ]:
            plot_clinvar_violin(
                annotated, clinvar_col,
                CLINVAR_PALETTE[palette_key], CLINVAR_ORDER[order_key],
                rep, rep_dir, stat_results,
                **violin_kwargs,
            )

        # Per-rep summary CSVs + heatmap
        rep_averaged = averaged.filter(pl.col("representation") == rep)
        rep_annotated = annotated.filter(pl.col("representation") == rep)
        rep_stats = stat_results.filter(pl.col("representation") == rep)
        rep_averaged.write_csv(rep_summary_dir / "averaged_metrics.csv")
        rep_annotated.write_csv(rep_summary_dir / "averaged_metrics_clinvar.csv")
        rep_stats.write_csv(rep_summary_dir / "wilcoxon_results.csv")
        plot_summary_heatmap(rep_stats, rep_summary_dir, stat="mean" if args.pa else "median")

    log.info("Done. Outputs in %s", output_dir)

    from prot_loc_benchmark.provenance import record
    prov_dirs = [output_dir / rep / "summary" for rep in args.representations]
    record(output_dirs=prov_dirs)


if __name__ == "__main__":
    main()
