"""Cross-batch HPA phenotypic consistency benchmark (per-organelle mAP).

For each representation:
  1. Load reference (disease_wt) cell profiles from all batches and pool them.
     For CellProfiler, intersect feature columns across batches (per-batch
     feature selection differs) so the pooled matrix has no null columns.
  2. Aggregate to a single median profile per gene (one row per gene across
     all pooled cells/batches).
  3. Attach HPA organelle labels above --hpa-threshold (default 1.0) to each
     gene as a multilabel list. Genes without any label are dropped.
  4. For each feature channel (GFP/DNA/AGP/Mito/Morph/ALL for CellProfiler,
     embedding groups for DL reps) run copairs multilabel average_precision:
         positive pairs: two different genes sharing ≥1 HPA location
         negative pairs: two different genes sharing no HPA location
  5. Aggregate to mAP per HPA location with mean_average_precision using
     sameby=["Metadata_hpa_locations"] — one mAP per organelle, measuring
     how well genes annotated to that organelle cluster together.

Outputs per representation:
    {rep}/summary/ap_scores_pooled.parquet      — per-(location, channel) rows
    {rep}/summary/per_channel_summary_pooled.csv — mean/median/%sig per channel
    {rep}/distribution.png                      — norm mAP violin per channel
    {rep}/embedding_{rep}_{channel}.png         — PCA grid colored by
        True/False for every HPA location (ordered by frequency), with
        per-panel mAP_norm and BH-corrected significance indicator

Cross-representation:
    summary/ap_scores_pooled.parquet
    summary/per_channel_summary_pooled.csv
    summary/per_channel_summary_pooled_min3genes.csv  — same metrics restricted
        to organelles with ≥3 HPA-annotated genes
    summary/cross_rep_heatmap.png

Inputs:
    CellProfiler: data/interim/cellprofiler/{batch}/features.parquet
    DL reps:      data/interim/{rep}/{batch}/embeddings.parquet
    HPA table:    annotations/hpa_gene_localization_table.parquet
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from copairs.map import mean_average_precision
from copairs.map.multilabel import average_precision as average_precision_multilabel

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from prot_loc_benchmark.classification.channels import get_feature_channels
from prot_loc_benchmark.config import (
    BIOREP_PAIRS,
    HPA_BENCHMARK_DIR,
    HPA_GENE_LOCALIZATION_PATH,
    INTERIM_DIR,
    REP_FEATURE_FILES,
    canonical_representation,
    load_hpa_labels,
)
from prot_loc_benchmark.viz.benchmark import significance_stars
from prot_loc_benchmark.provenance import record

HPA_LABELS_COL = "Metadata_hpa_locations"

log = logging.getLogger(__name__)


# ============================================================================
# DATA LOADING
# ============================================================================


def _load_cellprofiler(batch: str) -> pl.DataFrame:
    """Load the post-preprocessing CellProfiler features for one batch.

    Reads data/interim/cellprofiler/{batch}/features.parquet, which is produced
    by 06_preprocess_profiles.py (RobustMAD-normalized, blocklisted,
    feature-selected). Returns an empty DataFrame if the file is missing.
    """
    path = INTERIM_DIR / "cellprofiler" / batch / "features.parquet"
    if not path.exists():
        log.warning("Missing: %s", path)
        return pl.DataFrame()
    return pl.read_parquet(str(path))


def _load_dl(rep: str, batch: str) -> pl.DataFrame:
    """Load a DL representation's embeddings for one batch.

    Dispatches on REP_FEATURE_FILES to pick the right filename per rep
    (features.parquet, embeddings.parquet, etc.). Returns an empty DataFrame
    if the batch file is missing; rejects unknown representations.
    """
    rep = canonical_representation(rep)
    fname = REP_FEATURE_FILES.get(rep)
    if fname is None:
        raise ValueError(f"Unknown representation: {rep}")
    path = INTERIM_DIR / rep / batch / fname
    if not path.exists():
        log.warning("Missing: %s", path)
        return pl.DataFrame()
    return pl.read_parquet(str(path))


def _load_reference_cells_pooled(
    rep: str, batches: list[str], test_split: str | None = None,
) -> pl.DataFrame:
    """Load reference (disease_wt) cells for one rep, pooled across batches.

    For each batch, filters to `Metadata_node_type == "disease_wt"`, then
    concatenates across batches. Feature columns are restricted to the
    intersection present in ALL batches (CellProfiler's post-selection feature
    set differs per batch; using the intersection prevents null-filled
    columns after a diagonal concat). Returns a DataFrame with
    Metadata_symbol, common feature columns, and _source_batch.

    When ``test_split == "t4"``, also restricts to cells on T4 plates
    (plate barcode ending in ``T4``). Batches with no T4 plates are
    silently dropped — that's the expected outcome for multi_rep
    layouts.
    """
    frames_raw: list[tuple[str, pl.DataFrame]] = []
    for batch in batches:
        df = _load_cellprofiler(batch) if rep == "cellprofiler" else _load_dl(rep, batch)
        if df.is_empty() or "Metadata_node_type" not in df.columns:
            continue
        ref_df = df.filter(pl.col("Metadata_node_type") == "disease_wt")
        if test_split == "t4":
            ref_df = ref_df.filter(pl.col("Metadata_Plate").str.ends_with("T4"))
        if ref_df.is_empty():
            if test_split == "t4":
                log.info("Skipping %s/%s: no T4 reference cells", rep, batch)
            continue
        frames_raw.append((batch, ref_df))
        log.info("Loaded %s/%s: %d reference cells", rep, batch, ref_df.height)

    if not frames_raw:
        return pl.DataFrame()

    # Intersection of feature columns across all batches
    per_batch_feats = [
        {c for c in df.columns if not c.startswith("Metadata_") and c != "_label"}
        for _, df in frames_raw
    ]
    common_feats = sorted(set.intersection(*per_batch_feats))
    log.info(
        "Common feature columns across %d batches: %d (per-batch range %d–%d)",
        len(frames_raw), len(common_feats),
        min(len(s) for s in per_batch_feats),
        max(len(s) for s in per_batch_feats),
    )

    frames = []
    for batch, df in frames_raw:
        keep = ["Metadata_symbol"] + common_feats
        sub = df.select(keep).with_columns(pl.lit(batch).alias("_source_batch"))
        frames.append(sub)
    return pl.concat(frames, how="diagonal")


# ============================================================================
# AGGREGATION
# ============================================================================


def _median_per_gene(df: pl.DataFrame, feat_cols: list[str]) -> pl.DataFrame:
    """Collapse all replicates of each gene to a single median profile.

    Groups by Metadata_symbol (across batches, plates, cells) and computes the
    median of each feature column. First-value for remaining metadata columns.
    Output has one row per gene.
    """
    meta_cols = [c for c in df.columns if c.startswith("Metadata_") and c != "Metadata_symbol"]
    agg_exprs = [pl.col(c).median().alias(c) for c in feat_cols]
    agg_exprs += [pl.col(c).first().alias(c) for c in meta_cols]
    return df.group_by("Metadata_symbol").agg(agg_exprs)


# ============================================================================
# COPAIRS HPA mAP
# ============================================================================


def _run_hpa_map(
    pool: pl.DataFrame,
    feat_cols: list[str],
    gene_to_labels: dict[str, list[str]],
    null_size: int,
    threshold: float,
    seed: int,
    max_workers: int | None,
) -> pl.DataFrame:
    """Run copairs multilabel AP on gene-consensus profiles and aggregate per organelle.

    Attaches gene_to_labels as a list-valued column, drops genes with no labels,
    then computes average_precision_multilabel (positive = same organelle,
    different gene; negative = no shared organelle, different gene) and
    mean_average_precision grouped by HPA location. Returns one row per
    organelle location with mAP_hpa, mAP_hpa_norm, p_value_hpa,
    corrected_p_value_hpa, below_p_hpa, below_corrected_p_hpa.
    """
    pool_pd = pool.to_pandas()
    pool_pd[HPA_LABELS_COL] = pool_pd["Metadata_symbol"].map(lambda g: gene_to_labels.get(g, []))
    pool_pd = pool_pd[pool_pd[HPA_LABELS_COL].map(len) > 0].reset_index(drop=True)
    if pool_pd.empty:
        return pl.DataFrame()

    print(pool_pd)
    
    non_feat = [c for c in pool_pd.columns if c not in feat_cols]
    feats_np = pool_pd[feat_cols].to_numpy().astype(np.float32)

    
    try:
        ap = average_precision_multilabel(
            pool_pd[non_feat],
            feats_np,
            pos_sameby=[HPA_LABELS_COL],
            pos_diffby=["Metadata_symbol"],
            neg_sameby=[],
            neg_diffby=["Metadata_symbol", HPA_LABELS_COL],
            multilabel_col=HPA_LABELS_COL,
            progress_bar=False,
        )
    except Exception as e:
        log.warning("multilabel AP failed: %s", e)
        return pl.DataFrame()

    map_scores = mean_average_precision(
        ap,
        sameby=[HPA_LABELS_COL],
        null_size=null_size,
        threshold=threshold,
        seed=seed,
        max_workers=max_workers,
        progress_bar=False,
    )
    map_scores = map_scores.rename(columns={
        HPA_LABELS_COL: "hpa_location",
        "mean_average_precision": "mAP_hpa",
        "mean_normalized_average_precision": "mAP_hpa_norm",
        "p_value": "p_value_hpa",
        "corrected_p_value": "corrected_p_value_hpa",
        "below_p": "below_p_hpa",
        "below_corrected_p": "below_corrected_p_hpa",
    }).drop(columns=["indices"], errors="ignore")
    return pl.from_pandas(map_scores)


# ============================================================================
# PLOTS
# ============================================================================


def plot_distribution(results: pl.DataFrame, rep: str, output_dir: Path) -> None:
    """Violin + strip plot of per-organelle norm mAP for one representation.

    One violin per feature channel, x-axis = channels, y-axis = mAP_hpa_norm.
    Each point is one HPA organelle. Reference line at 0 (null expectation).
    Saved as `{output_dir}/distribution.png`.
    """
    rep_df = results.filter(pl.col("representation") == rep).to_pandas()
    if rep_df.empty:
        return
    channels = sorted(rep_df["channel"].unique())
    fig, ax = plt.subplots(figsize=(max(4.0, 0.8 * len(channels) + 2), 4.0))
    sns.violinplot(
        data=rep_df, x="channel", y="mAP_hpa_norm",
        order=channels, inner="box", cut=0, linewidth=0.8, ax=ax,
    )
    sns.stripplot(
        data=rep_df, x="channel", y="mAP_hpa_norm",
        order=channels, color="black", alpha=0.35, size=2.5, jitter=0.15, ax=ax,
    )
    ax.axhline(0, color="grey", ls="--", lw=0.7, alpha=0.5)
    ax.set_ylabel("Norm. mAP_hpa (per organelle)")
    ax.set_xlabel("")
    ax.set_title(f"{rep} — HPA phenotypic consistency (pooled)", fontsize=11)
    ax.tick_params(axis="x", rotation=30, labelsize=9)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
    fig.tight_layout()
    fig.savefig(output_dir / "distribution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved distribution.png for %s", rep)


def plot_embedding(
    pool: pl.DataFrame,
    feat_cols: list[str],
    gene_to_labels: dict[str, list[str]],
    rep: str,
    channel: str,
    output_dir: Path,
    location_scores: dict[str, tuple[float, bool]] | None = None,
    n_top: int | None = None,
) -> None:
    """PCA projection of gene-consensus profiles, faceted by HPA locations.

    Computes one 2D PCA (PC1 vs PC2) on the pool and re-uses the same (x, y)
    coordinates across one subplot per location. Panels are ordered by
    frequency (most common first). Pass ``n_top=N`` to keep only the N most
    common; default (``None``) plots every location present in the pool.
    In each subplot, genes annotated to that location are colored red and
    others grey.

    If `location_scores` is provided (location → (mAP_norm, is_significant)),
    each panel title shows `mAP_norm=±X.XXX*` with `*` marking a
    BH-corrected-significant location. Significance also controls title color.
    """
    from sklearn.decomposition import PCA

    pool_pd = pool.to_pandas()
    pool_pd = pool_pd[pool_pd["Metadata_symbol"].isin(gene_to_labels)].reset_index(drop=True)
    if pool_pd.empty:
        return
    X = pool_pd[feat_cols].to_numpy().astype(np.float32)
    if X.shape[0] < 5:
        return

    pca = PCA(n_components=2, random_state=0).fit(X)
    emb = pca.transform(X)
    pc1_var, pc2_var = pca.explained_variance_ratio_[:2]
    pool_pd["_x"], pool_pd["_y"] = emb[:, 0], emb[:, 1]

    # Order locations by frequency; keep top-N only if requested
    loc_counter = Counter()
    for g in pool_pd["Metadata_symbol"]:
        for loc in gene_to_labels[g]:
            loc_counter[loc] += 1
    top_locs = [loc for loc, _ in loc_counter.most_common(n_top)]

    # Shortened label for title (organelle_GO_sub_major → organelle)
    def short(loc: str) -> str:
        return loc.split("_")[0]

    # Grid
    ncols = 5
    nrows = (len(top_locs) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.5 * nrows), sharex=True, sharey=True)
    axes = np.atleast_2d(axes).ravel()

    for i, loc in enumerate(top_locs):
        ax = axes[i]
        mask = pool_pd["Metadata_symbol"].map(lambda g: loc in gene_to_labels[g])
        false_df = pool_pd[~mask]
        true_df = pool_pd[mask]
        ax.scatter(false_df["_x"], false_df["_y"], c="lightgrey", s=18, alpha=0.5, edgecolor="none")
        ax.scatter(true_df["_x"], true_df["_y"], c="crimson", s=24, alpha=0.9, edgecolor="white", linewidth=0.4)

        # Compose title: organelle — n — mAP norm — significance star
        title = f"{short(loc)}  (n={mask.sum()})"
        if location_scores and loc in location_scores:
            score, sig = location_scores[loc]
            star = "*" if sig else ""
            title += f"\nmAP_norm={score:+.3f}{star}"
            color = "black" if sig else "dimgrey"
        else:
            color = "black"
        ax.set_title(title, fontsize=9, color=color)
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused axes
    for j in range(len(top_locs), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"{rep} / {channel} — reference genes PCA (PC1 {pc1_var:.1%}, PC2 {pc2_var:.1%}), "
        "colored by HPA location",
        fontsize=12, y=1.00,
    )
    fig.tight_layout()
    out_path = output_dir / f"embedding_{rep}_{channel}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out_path.name)


def plot_cross_rep_heatmap(summary: pl.DataFrame, output_dir: Path) -> None:
    """Side-by-side heatmaps of HPA consistency summary metrics.

    One subplot per metric (mean norm mAP, median norm mAP, % significant)
    because they have different numeric ranges. Rows = `{rep} / {channel}`.
    Saved as `{output_dir}/cross_rep_heatmap.png`.
    """
    df = summary.with_columns(
        label=pl.col("representation") + " / " + pl.col("channel"),
    ).to_pandas()
    if df.empty:
        return

    metrics = [
        ("mAP_hpa_norm_mean", "Mean norm mAP", "RdBu_r", True),    # diverging, centered at 0
        ("mAP_hpa_norm_median", "Median norm mAP", "RdBu_r", True),
        ("pct_sig", "% significant", "viridis", False),            # sequential, 0..100
    ]
    mat_full = df.set_index("label")[[m[0] for m in metrics]].astype(float)

    fig, axes = plt.subplots(
        1, len(metrics),
        figsize=(2.3 * len(metrics) + 1.5, max(3.0, len(mat_full) * 0.35 + 1.5)),
        sharey=True,
    )
    if len(metrics) == 1:
        axes = [axes]

    for ax, (col, title, cmap, diverging) in zip(axes, metrics):
        vals = mat_full[[col]]
        if diverging:
            absmax = max(0.05, float(vals.abs().max().max()))
            vmin, vmax, center = -absmax, absmax, 0
        else:
            vmin, vmax, center = float(vals.min().min()), float(vals.max().max()), None
        sns.heatmap(
            vals, annot=True, fmt=".3f", cmap=cmap,
            vmin=vmin, vmax=vmax, center=center,
            linewidths=0.5, linecolor="white",
            cbar_kws={"label": title, "shrink": 0.7},
            annot_kws={"fontsize": 8}, ax=ax,
        )
        ax.set_xticklabels([title], rotation=0, fontsize=9)
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.tick_params(axis="y", rotation=0, labelsize=9)

    fig.suptitle("HPA consistency — pooled across batches", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "cross_rep_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved cross_rep_heatmap.png")


def plot_per_organelle_heatmap(
    results: pl.DataFrame,
    output_dir: Path,
    exclude_organelles: list[str] | None = None,
    filename: str = "per_organelle_heatmap.png",
    title_suffix: str = "",
    loc_counts: dict[str, int] | None = None,
    min_gene_count: int | None = None,
) -> None:
    """Heatmap of per-organelle norm mAP across representation × channel.

    Rows = `{representation} / {channel}`, columns = HPA organelle short name.
    Cell value = `mAP_hpa_norm` for that (rep, channel, organelle) triple.
    Columns sorted left-to-right by mean norm mAP across reps.

    `exclude_organelles` drops matching organelle short names (case-insensitive
    substring match). `min_gene_count` (requires `loc_counts`) drops organelles
    with fewer than N annotated genes. Use `filename`/`title_suffix` to save
    multiple variants side by side. Pass `loc_counts` (hpa_location → gene
    count) to annotate column labels with `(n=N)`.
    """
    df = results.with_columns(
        label=pl.col("representation") + " / " + pl.col("channel"),
    ).to_pandas()
    if df.empty:
        return

    # Shorten organelle labels (col format: "Organelle_GO_Sub_Major")
    df["organelle"] = df["hpa_location"].str.split("_").str[0]
    loc_counts_short = (
        {loc.split("_")[0]: n for loc, n in loc_counts.items()} if loc_counts else {}
    )

    if exclude_organelles:
        lowered = [e.lower() for e in exclude_organelles]
        keep = ~df["organelle"].str.lower().apply(
            lambda o: any(e in o for e in lowered)
        )
        df = df[keep]

    if min_gene_count is not None and loc_counts_short:
        df = df[df["organelle"].map(lambda o: loc_counts_short.get(o, 0) >= min_gene_count)]

    mat = df.pivot_table(
        index="label", columns="organelle", values="mAP_hpa_norm", aggfunc="mean",
    )
    pmat = df.pivot_table(
        index="label", columns="organelle", values="corrected_p_value_hpa", aggfunc="min",
    )
    # Sort columns by mean mAP_norm across reps (best-distinguished on the left)
    col_order = mat.mean(axis=0).sort_values(ascending=False).index
    mat = mat[col_order]
    pmat = pmat.reindex(index=mat.index, columns=col_order)

    annot = mat.copy().astype(object)
    for r in mat.index:
        for c in mat.columns:
            val = mat.loc[r, c]
            p = pmat.loc[r, c]
            stars = significance_stars(p) if pd.notna(p) else ""
            annot.loc[r, c] = f"{val:+.2f}{stars}" if pd.notna(val) else ""

    absmax = max(0.05, float(mat.abs().max().max()))
    fig, ax = plt.subplots(
        figsize=(max(6.0, 0.4 * len(mat.columns) + 3), max(3.0, len(mat) * 0.35 + 1.5)),
    )
    sns.heatmap(
        mat, annot=annot.values, fmt="", cmap="RdBu_r",
        vmin=-absmax, vmax=absmax, center=0,
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Norm mAP", "shrink": 0.7},
        annot_kws={"fontsize": 5}, ax=ax,
    )
    ax.set_title(
        f"HPA consistency — per-organelle norm mAP (pooled, gene consensus){title_suffix}\n"
        "(* p<0.05, ** p<0.01, *** p<0.001, BH-corrected)",
        fontsize=11, pad=28,
    )
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
    ax.tick_params(axis="y", rotation=0, labelsize=9)
    if loc_counts_short:
        top_ax = ax.secondary_xaxis("top")
        top_ax.set_xticks(np.arange(len(mat.columns)) + 0.5)
        top_ax.set_xticklabels(
            [f"n={loc_counts_short.get(c, 0)}" for c in mat.columns],
            fontsize=7, rotation=0,
        )
        top_ax.tick_params(axis="x", length=0, pad=2)
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", filename)


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-batch HPA phenotypic consistency benchmark (pooled consensus)."
    )
    parser.add_argument(
        "--representations",
        type=canonical_representation,
        choices=sorted(REP_FEATURE_FILES),
        help="Feature representations (morphem is an alias for vit)",
        nargs="+",
        default=["cellprofiler", "cytoself", "vit"],
    )
    parser.add_argument(
        "--batches",
        nargs="+",
        default=None,
        help="Batches to pool (default: all batches from BIOREP_PAIRS)",
    )
    parser.add_argument("--hpa-threshold", type=float, default=1.0)
    parser.add_argument("--null-size", type=int, default=10_000)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--test-split",
        choices=["none", "t4"],
        default="none",
        help=(
            "Restrict reference cells to a held-out plate subset before "
            "gene-consensus aggregation: 't4' keeps only T4 plates. "
            "Output dir gets a '_t4' suffix when active. Default: 'none'."
        ),
    )
    args = parser.parse_args()
    args.representations = list(dict.fromkeys(args.representations))
    test_split = None if args.test_split == "none" else args.test_split

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = HPA_BENCHMARK_DIR.parent / (
            f"{HPA_BENCHMARK_DIR.name}_t4" if test_split else HPA_BENCHMARK_DIR.name
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    if test_split:
        log.info("Test-split filter active: keeping only %s plates", test_split.upper())

    if args.batches:
        batches = args.batches
    else:
        batches = sorted({b for pair in BIOREP_PAIRS.values() for b in pair})
    log.info("Pooling batches: %s", batches)

    gene_to_labels = load_hpa_labels(args.hpa_threshold)
    log.info(
        "HPA labels @ threshold=%s: %d genes, %d unique organelles",
        args.hpa_threshold,
        len(gene_to_labels),
        len({o for labs in gene_to_labels.values() for o in labs}),
    )

    all_results: list[pl.DataFrame] = []
    pools_per_rep: dict[str, tuple[pl.DataFrame, dict[str, list[str]]]] = {}
    for rep in args.representations:
        log.info("=== %s ===", rep)
        ref_df = _load_reference_cells_pooled(rep, batches, test_split=test_split)
        if ref_df.is_empty():
            log.warning("%s: no reference cells loaded", rep)
            continue

        # Restrict to HPA-labeled genes BEFORE aggregation to save memory
        valid_genes = list(gene_to_labels.keys())
        ref_df = ref_df.filter(pl.col("Metadata_symbol").is_in(valid_genes))

        feat_cols = [
            c for c in ref_df.columns
            if not c.startswith("Metadata_") and c != "_label" and c != "_source_batch"
        ]

        # Median-aggregate to one profile per gene across all pooled batches+cells
        pool = _median_per_gene(ref_df, feat_cols)
        log.info(
            "%s: pooled consensus → %d genes from %d cells",
            rep, pool.height, ref_df.height,
        )

        channel_map = get_feature_channels(feat_cols, rep)
        log.info("Channels: %s", {k: len(v) for k, v in channel_map.items()})
        pools_per_rep[rep] = (pool, channel_map)

        for channel_name, ch_feats in channel_map.items():
            map_hpa = _run_hpa_map(
                pool, ch_feats, gene_to_labels,
                args.null_size, args.threshold, args.seed, args.max_workers,
            )
            if map_hpa.is_empty():
                continue
            map_hpa = map_hpa.with_columns(
                pl.lit(channel_name).alias("channel"),
                pl.lit(rep).alias("representation"),
            )
            n = map_hpa.height  # number of organelle locations
            n_sig = map_hpa["below_corrected_p_hpa"].sum()
            log.info(
                "  %s  n_locs=%d  mAP=%.3f  norm=%.3f  sig=%.1f%%",
                channel_name, n,
                map_hpa["mAP_hpa"].mean(),
                map_hpa["mAP_hpa_norm"].mean(),
                100.0 * n_sig / n if n else float("nan"),
            )
            all_results.append(map_hpa)

    if not all_results:
        log.error("No results produced")
        sys.exit(1)

    results = pl.concat(all_results, how="diagonal")

    # Per-organelle gene counts from the HPA universe (used for the n>=3
    # filtered summary and the existing per-organelle heatmap variant).
    loc_counts: dict[str, int] = Counter()
    for labs in gene_to_labels.values():
        for loc in labs:
            loc_counts[loc] += 1
    loc_counts_df = pl.DataFrame(
        {"hpa_location": list(loc_counts.keys()), "n_genes": list(loc_counts.values())},
        schema={"hpa_location": pl.Utf8, "n_genes": pl.Int64},
    )
    results = results.join(loc_counts_df, on="hpa_location", how="left")

    def _summarize(df: pl.DataFrame) -> pl.DataFrame:
        return (
            df.group_by(["representation", "channel"])
            .agg(
                n_locations=pl.col("hpa_location").n_unique(),
                n_significant=pl.col("below_corrected_p_hpa").cast(pl.Int64).sum(),
                mAP_hpa_mean=pl.col("mAP_hpa").mean(),
                mAP_hpa_median=pl.col("mAP_hpa").median(),
                mAP_hpa_norm_mean=pl.col("mAP_hpa_norm").mean(),
                mAP_hpa_norm_median=pl.col("mAP_hpa_norm").median(),
                pct_sig=(
                    pl.col("below_corrected_p_hpa").cast(pl.Float64).sum()
                    / pl.col("below_corrected_p_hpa").count()
                )
                * 100,
            )
            .sort(["representation", "channel"])
        )

    summary = _summarize(results)
    summary_min3 = _summarize(results.filter(pl.col("n_genes") >= 3))
    log.info("Per-channel summary:\n%s", summary)
    log.info("Per-channel summary (n_genes >= 3):\n%s", summary_min3)

    # Outputs
    summary_dir = output_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    results.write_parquet(str(summary_dir / "ap_scores_pooled.parquet"))
    summary.write_csv(summary_dir / "per_channel_summary_pooled.csv")
    summary_min3.write_csv(summary_dir / "per_channel_summary_pooled_min3genes.csv")

    plot_cross_rep_heatmap(summary, summary_dir)
    plot_per_organelle_heatmap(results, summary_dir, loc_counts=loc_counts)
    plot_per_organelle_heatmap(
        results, summary_dir,
        min_gene_count=3,
        filename="per_organelle_heatmap_min3genes.png",
        title_suffix=" — organelles with ≥3 genes",
        loc_counts=loc_counts,
    )

    prov_dirs = [summary_dir]
    for rep in args.representations:
        rep_dir = output_dir / rep
        rep_dir.mkdir(parents=True, exist_ok=True)
        rep_summary_dir = rep_dir / "summary"
        rep_summary_dir.mkdir(parents=True, exist_ok=True)
        rep_res = results.filter(pl.col("representation") == rep)
        rep_sum = summary.filter(pl.col("representation") == rep)
        rep_res.write_parquet(str(rep_summary_dir / "ap_scores_pooled.parquet"))
        rep_sum.write_csv(rep_summary_dir / "per_channel_summary_pooled.csv")
        plot_distribution(results, rep, rep_dir)

        # PCA of gene consensus profiles, colored by HPA locations — one per channel
        if rep in pools_per_rep:
            pool, channel_map = pools_per_rep[rep]
            for plot_channel in channel_map:
                ch_scores = rep_res.filter(pl.col("channel") == plot_channel)
                location_scores = {
                    row["hpa_location"]: (row["mAP_hpa_norm"], bool(row["below_corrected_p_hpa"]))
                    for row in ch_scores.iter_rows(named=True)
                }
                plot_embedding(
                    pool, channel_map[plot_channel], gene_to_labels,
                    rep, plot_channel, rep_dir,
                    location_scores=location_scores,
                )
        prov_dirs.append(rep_summary_dir)

    log.info("Done. Outputs in %s", output_dir)
    record(output_dirs=prov_dirs)


if __name__ == "__main__":
    main()
