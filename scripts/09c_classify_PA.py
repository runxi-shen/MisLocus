#!/usr/bin/env python3
"""Phenotypic activity assessment using copairs mAP.

Per allele:
  - mAP_vs_ref — variant profiles vs same-gene reference (disease_wt) profiles

Optional (--hpa):
  - mAP_hpa   — reference-gene consistency against HPA organelle annotation

Profiles are median-aggregated per FOV by default, restricted to a fixed
set of reference cells per plate. The variant × reference pool is built per
channel (GFP / DNA / AGP / Mito / Morph / ALL for CellProfiler, embedding
groups for DL reps) and fed into copairs.

Defaults reflect the workflow in use:
    --sample-level site --aggregate --cp-feature-file features

Usage:
    pixi run python scripts/09c_classify_PA.py \\
        --batch 2025_01_27_Batch_13 --representation cellprofiler

    pixi run python scripts/09c_classify_PA.py \\
        --batch 2025_01_27_Batch_13 --representation vit \\
        --hpa  # also compute HPA consistency on reference genes
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import numpy as np
import pandas as pd
import polars as pl
from copairs.map import average_precision, mean_average_precision
from copairs.map.multilabel import average_precision as average_precision_multilabel
from copairs.matching import assign_reference_index

REFERENCE_COL = "Metadata_reference_index"
HPA_LABELS_COL = "Metadata_hpa_locations"

from prot_loc_benchmark.classification.channels import get_feature_channels
from prot_loc_benchmark.config import (
    BATCH_LAYOUT,
    HPA_GENE_LOCALIZATION_PATH,
    INTERIM_DIR,
    MIN_CELL_COUNT,
    NULL_PERCENTILE,
    PROCESSED_DIR,
    REP_FEATURE_FILES,
    canonical_representation,
)
from prot_loc_benchmark.provenance import record

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

PHENOTYPIC_ACTIVITY_DIR = PROCESSED_DIR / "classification_PA"

# CellProfiler feature source. "normalized" = normalized.parquet (more features,
# pre-feature-selection, needs Metadata_Control joined from features.parquet).
# "features" = features.parquet (post-selection, already annotated).
_CP_FEATURE_FILES = {
    "normalized": "normalized.parquet",
    "features": "features.parquet",
}


# ── Data loading ─────────────────────────────────────────────────────────────


def _load_cellprofiler(batch_id: str, cp_feature_file: str = "normalized") -> pl.DataFrame:
    """Load CellProfiler features from either normalized.parquet or features.parquet."""
    filename = _CP_FEATURE_FILES[cp_feature_file]
    path = INTERIM_DIR / "cellprofiler" / batch_id / filename

    if not path.exists():
        logger.error("%s not found: %s", filename, path)
        sys.exit(1)

    df = pl.read_parquet(str(path))

    # normalized.parquet lacks Metadata_Control — join it from features.parquet.
    if cp_feature_file == "normalized":
        feat_path = INTERIM_DIR / "cellprofiler" / batch_id / "features.parquet"
        if not feat_path.exists():
            logger.error("features.parquet not found (needed for Metadata_Control): %s", feat_path)
            sys.exit(1)
        ctrl_df = (
            pl.scan_parquet(str(feat_path))
            .select([
                "Metadata_Plate",
                "Metadata_Well",
                "Metadata_ImageNumber",
                "Metadata_ObjectNumber",
                "Metadata_Control",
            ])
            .collect()
        )
        df = df.join(
            ctrl_df,
            on=["Metadata_Plate", "Metadata_Well", "Metadata_ImageNumber", "Metadata_ObjectNumber"],
            how="inner",
        )
        logger.info(
            "CellProfiler [%s]: %d cells after join with features.parquet",
            filename, df.height,
        )
    else:
        logger.info("CellProfiler [%s]: %d cells", filename, df.height)
    return df


def _load_dl(representation: str, batch_id: str) -> pl.DataFrame:
    """Load the configured feature file without additional preprocessing."""
    representation = canonical_representation(representation)
    feature_file = REP_FEATURE_FILES.get(representation)
    if feature_file is None:
        logger.error("Unknown representation: %s", representation)
        sys.exit(1)
    path = INTERIM_DIR / representation / batch_id / feature_file
    if not path.exists():
        logger.error("Embeddings not found: %s", path)
        sys.exit(1)
    df = pl.read_parquet(str(path))
    logger.info("Loaded %s: %d cells", representation, df.height)
    return df


# ── Sampling ─────────────────────────────────────────────────────────────────


def _sample_per_site(
    df: pl.DataFrame,
    group_cols: list[str],
    n: int,
    seed: int = 42,
) -> pl.DataFrame:
    """Sample up to N cells per (group_cols) group."""
    rng = np.random.default_rng(seed)
    parts = df.partition_by(group_cols, maintain_order=True)
    sampled = []
    for part in parts:
        if part.height <= n:
            sampled.append(part)
        else:
            idx = sorted(rng.choice(part.height, n, replace=False).tolist())
            sampled.append(part[idx])
    return pl.concat(sampled)


def _aggregate_per_group(
    df: pl.DataFrame,
    group_cols: list[str],
    feat_cols: list[str],
) -> pl.DataFrame:
    """Aggregate cells to median feature values per group.

    Returns one row per group with median features and first-row metadata.
    """
    meta_cols = [c for c in df.columns if c.startswith("Metadata_") and c not in group_cols]
    agg_exprs = [pl.col(c).median().alias(c) for c in feat_cols]
    # Keep first value of metadata columns (they're constant within group for the ones we care about)
    agg_exprs += [pl.col(c).first().alias(c) for c in meta_cols]
    # maintain_order=True is REQUIRED for reproducibility: copairs assigns
    # reference indices in row order and ties break by row order, so a
    # non-deterministic group_by produces non-deterministic mAP.
    return df.group_by(group_cols, maintain_order=True).agg(agg_exprs).sort(group_cols)


def _subsample_per_plate(df: pl.DataFrame, n: int, seed: int = 42) -> pl.DataFrame:
    """Sample up to N cells per plate (same subset used across all alleles)."""
    return _sample_per_site(df, ["Metadata_Plate"], n, seed)


# Column used as the sampling unit. "site" = one FOV (image), "well" = one well.
# DL embeddings use Metadata_well_position; CellProfiler uses Metadata_Well.
_SAMPLE_UNIT_COL = {"site": "Metadata_ImageNumber", "well": "Metadata_Well"}
_WELL_COL_ALIASES = ["Metadata_Well", "Metadata_well_position"]


def _resolve_well_col(df: pl.DataFrame) -> str:
    """Return whichever well column exists in the DataFrame."""
    for col in _WELL_COL_ALIASES:
        if col in df.columns:
            return col
    raise ValueError(f"No well column found. Tried: {_WELL_COL_ALIASES}")


# ── Scope filtering ──────────────────────────────────────────────────────────

_SCOPE_KEYWORDS = {"all", "exp", "cpc"}


def _resolve_alleles(df: pl.DataFrame, scope: str) -> list[str]:
    """Return allele names to include based on scope, mirroring 09_classify.py.

    Scope values:
    - "all"  : Exp + cPC alleles with >= MIN_CELL_COUNT cells
    - "exp"  : Exp alleles only
    - "cpc"  : cPC alleles only
    - comma-separated allele names: e.g. "CCM2_Ile432Thr,KRAS_Gly12Val"
    - mix    : e.g. "CCM2_Ile432Thr,exp"
    """
    # All Exp + cPC variant cells (same pool the classifier uses)
    var_df = df.filter(
        pl.col("Metadata_node_type") == "allele",
        pl.col("Metadata_Control").is_in(["Exp", "cPC"]),
    )

    # Drop alleles below MIN_CELL_COUNT (mirrors build_experimental_pairs min_cells filter)
    cell_counts = var_df.group_by("Metadata_gene_allele").len()
    valid_alleles = (
        cell_counts
        .filter(pl.col("len") >= MIN_CELL_COUNT)
        ["Metadata_gene_allele"]
        .to_list()
    )
    var_df = var_df.filter(pl.col("Metadata_gene_allele").is_in(valid_alleles))

    if scope == "all":
        return var_df["Metadata_gene_allele"].unique().to_list()

    tokens = {t.strip() for t in scope.split(",")}
    keywords = tokens & _SCOPE_KEYWORDS
    allele_names = tokens - _SCOPE_KEYWORDS

    result: set[str] = set()

    if "all" in keywords:
        result |= set(var_df["Metadata_gene_allele"].unique().to_list())
    if "exp" in keywords:
        result |= set(
            var_df.filter(pl.col("Metadata_Control") == "Exp")
            ["Metadata_gene_allele"].unique().to_list()
        )
    if "cpc" in keywords:
        result |= set(
            var_df.filter(pl.col("Metadata_Control") == "cPC")
            ["Metadata_gene_allele"].unique().to_list()
        )

    # Explicit allele names (keep only those that exist and pass cell count)
    valid_set = set(valid_alleles)
    for name in allele_names:
        if name in valid_set:
            result.add(name)
        else:
            logger.warning("Scope allele '%s' not found or below MIN_CELL_COUNT — skipped", name)

    return sorted(result)





# ── mAP computations ─────────────────────────────────────────────────────────


def _run_map(
    pool_pl: pl.DataFrame,
    feat_cols: list[str],
    reference_condition: str,
    null_size: int,
    threshold: float,
    seed: int,
    label: str,
    max_workers: int | None = None,
    neg_sameby: list[str] | None = None,
    test_split: str | None = None,
) -> pd.DataFrame:
    """Core mAP computation shared by both comparisons.

    Steps:
    1. assign_reference_index — controls/references get unique row indices,
       variants get -1. Prevents positive pairs among control cells.
    2. average_precision — per-cell AP scores.
    3. mean_average_precision — mAP per allele with permutation p-values
       and BH FDR correction.

    Positive pairs:  same allele + same reference_index (-1), different plates.
    Negative pairs:  differ in allele + reference_index, same neg_sameby group.
    Queries without matching references are excluded, not removed from the pool:
    they may still supply cross-plate positives for supported queries.

    ``test_split`` filters the per-row APs before per-allele aggregation.
    Currently supports ``"t4"`` (keep only rows on T4 plates). The pool
    used for AP computation is unchanged — only the queries that get
    aggregated into the per-allele mAP shrink. Permutation null is also
    computed on the filtered set, so p-values are noisier than the
    full-data run.
    """
    if neg_sameby is None:
        neg_sameby = ["Metadata_Plate"]
    if pool_pl.is_empty():
        return pd.DataFrame()

    pool = pool_pl.to_pandas()
    pool = assign_reference_index(
        pool,
        condition=reference_condition,
        reference_col=REFERENCE_COL,
        default_value=-1,
    )

    non_feat = [c for c in pool.columns if c not in feat_cols]
    feats_np = pool[feat_cols].to_numpy().astype(np.float32)

    query_mask = pool[REFERENCE_COL] == -1
    if test_split == "t4":
        query_mask &= pool["Metadata_Plate"].str.endswith("T4")
    reference_groups = pd.MultiIndex.from_frame(pool.loc[pool[REFERENCE_COL] != -1, neg_sameby])
    supported = pd.MultiIndex.from_frame(pool[neg_sameby]).isin(reference_groups)
    missing_reference = query_mask & ~supported
    logger.info(
        "Query support (%s): excluding %d/%d query profiles without matching references",
        label, int(missing_reference.sum()), int(query_mask.sum()),
    )
    query_mask &= supported
    if not query_mask.any():
        logger.info("%s: no eligible query profiles", label)
        return pd.DataFrame()
    norms = np.linalg.norm(feats_np, axis=1)
    if not np.isfinite(feats_np).all() or not np.isfinite(norms).all() or (norms <= 0).any():
        raise ValueError("Cosine retrieval requires finite profiles with positive finite norms")

    # Strict vs_ref pairing rules.
    # Positives: same variant allele, both with REFERENCE_COL=-1, on
    #            different plates (cross-plate variant replicates).
    # Negatives: same plate AND same gene (Metadata_symbol), with the
    #            partner explicitly a disease_wt cell. The diffby rule
    #            requires the partner to differ in BOTH Metadata_node_type
    #            (variant→reference) AND REFERENCE_COL (-1 for variants,
    #            unique non-negative idx per disease_wt cell from
    #            ``assign_reference_index``). The node_type constraint
    #            makes the variant↔reference intent explicit and
    #            independently excludes any variant↔variant negatives.
    ap_scores = average_precision(
        pool[non_feat],
        feats_np,
        pos_sameby=["Metadata_gene_allele", REFERENCE_COL],
        pos_diffby=["Metadata_Plate"],
        neg_sameby=neg_sameby,
        neg_diffby=["Metadata_node_type", REFERENCE_COL],
    )

    # Compute mAP + p-values + BH FDR (variant cells only)
    ap_variant = ap_scores[ap_scores[REFERENCE_COL] == -1]

    if test_split == "t4":
        mask = ap_variant["Metadata_Plate"].str.endswith("T4")
        n_before, n_after = len(ap_variant), int(mask.sum())
        logger.info(
            "T4 filter (%s): %d → %d variant rows (%d alleles → %d)",
            label, n_before, n_after,
            ap_variant["Metadata_gene_allele"].nunique(),
            ap_variant.loc[mask, "Metadata_gene_allele"].nunique(),
        )
        ap_variant = ap_variant.loc[mask]
    ap_variant = ap_variant.loc[query_mask.reindex(ap_variant.index)]
    unsupported = (ap_variant["n_pos_pairs"] <= 0) | (ap_variant["n_total_pairs"] <= ap_variant["n_pos_pairs"])
    if unsupported.any() or not np.isfinite(
        ap_variant[["average_precision", "normalized_average_precision"]].to_numpy()
    ).all():
        raise ValueError("Unsupported PA queries: finite AP and positive and negative partners are required")

    map_scores = mean_average_precision(
        ap_variant,
        sameby=["Metadata_gene_allele"],
        null_size=null_size,
        threshold=threshold,
        seed=seed,
        max_workers=max_workers,
    )

    # Prefix columns with comparison label to avoid collisions on merge
    map_scores = map_scores.rename(columns={
        "mean_average_precision": f"mAP_{label}",
        "mean_normalized_average_precision": f"mAP_{label}_norm",
        "p_value": f"p_value_{label}",
        "corrected_p_value": f"corrected_p_value_{label}",
        "below_p": f"below_p_{label}",
        "below_corrected_p": f"below_corrected_p_{label}",
    })
    map_scores = map_scores.drop(columns=["indices"], errors="ignore")
    return map_scores


def _compute_map_vs_ref(
    df_full: pl.DataFrame,
    alleles: list[str],
    feat_cols: list[str],
    cells_per_site: int,
    neg_per_plate: int,
    sample_level: str,
    aggregate: bool,
    null_size: int,
    threshold: float,
    seed: int,
    max_workers: int | None = None,
    test_split: str | None = None,
) -> pl.DataFrame:
    """mAP: variant cells vs reference (disease_wt) cells of the same gene.

    Strict vs_ref: only variant alleles whose gene has a *same-batch*
    ``disease_wt`` reference are kept. Without this pre-filter, alleles in
    orphan genes are still scored by copairs but with no negatives in the
    same-plate same-gene group → AP collapses to 1.0 (mAP_vs_ref_norm=0),
    a degenerate value that contaminates the output.
    """
    unit_col = _resolve_well_col(df_full) if sample_level == "well" else _SAMPLE_UNIT_COL[sample_level]
    var_df = df_full.filter(
        pl.col("Metadata_gene_allele").is_in(alleles) &
        (pl.col("Metadata_node_type") == "allele")
    )

    # Pre-filter: drop variants whose gene has no disease_wt in this batch.
    ref_genes_in_batch = set(
        df_full.filter(pl.col("Metadata_node_type") == "disease_wt")
        ["Metadata_symbol"].unique().to_list()
    )
    candidate_genes = set(var_df["Metadata_symbol"].unique().to_list())
    paired_genes = candidate_genes & ref_genes_in_batch
    orphan_genes = sorted(candidate_genes - paired_genes)
    if orphan_genes:
        n_dropped = (
            var_df.filter(pl.col("Metadata_symbol").is_in(orphan_genes))
            ["Metadata_gene_allele"].n_unique()
        )
        shown = orphan_genes if len(orphan_genes) <= 10 else orphan_genes[:10] + ["..."]
        logger.info(
            "Strict vs_ref: dropping %d allele(s) in %d orphan gene(s) [no same-batch disease_wt]: %s",
            n_dropped, len(orphan_genes), shown,
        )
    var_df = var_df.filter(pl.col("Metadata_symbol").is_in(list(paired_genes)))
    variant_genes = sorted(paired_genes)
    ref_df = df_full.filter(
        (pl.col("Metadata_node_type") == "disease_wt") &
        pl.col("Metadata_symbol").is_in(variant_genes)
    )

    if var_df.is_empty() or ref_df.is_empty():
        logger.warning("mAP_vs_ref: no variant or reference cells found — skipping")
        return pl.DataFrame()

    if aggregate:
        var_sampled = _aggregate_per_group(
            var_df, ["Metadata_gene_allele", "Metadata_Plate", unit_col], feat_cols
        )
        ref_sampled = _aggregate_per_group(
            ref_df, ["Metadata_symbol", "Metadata_Plate", unit_col], feat_cols
        )
    else:
        var_sampled = _sample_per_site(
            var_df, ["Metadata_gene_allele", "Metadata_Plate", unit_col],
            cells_per_site,
        )
        ref_sampled = _sample_per_site(
            ref_df, ["Metadata_symbol", "Metadata_Plate", unit_col],
            cells_per_site,
        )
    # Cap ref to neg_per_plate per plate — same subset used for all alleles.
    # neg_per_plate <= 0 disables the cap (default: disabled at site-level
    # aggregation, since FOV-aggregated rows are already pre-reduced).
    if neg_per_plate > 0:
        ref_sampled = _subsample_per_plate(ref_sampled, neg_per_plate, seed=seed)
    mode = "median" if aggregate else f"{cells_per_site}/site"
    cap_str = f"{neg_per_plate}/plate cap" if neg_per_plate > 0 else "no cap"
    logger.info(
        "mAP_vs_ref: %d variant profiles (%d alleles) + %d reference profiles (%s) [%s]",
        var_sampled.height, var_df["Metadata_gene_allele"].n_unique(),
        ref_sampled.height, cap_str, mode,
    )

    pool = pl.concat([var_sampled, ref_sampled], how="diagonal")
    result = _run_map(
        pool, feat_cols,
        reference_condition="Metadata_node_type == 'disease_wt'",
        null_size=null_size, threshold=threshold, seed=seed,
        label="vs_ref", max_workers=max_workers,
        neg_sameby=["Metadata_Plate", "Metadata_symbol"],
        test_split=test_split,
    )
    return pl.from_pandas(result)


# ── Empirical control null (NC + PC, mirrors XGBoost) ────────────────────────


def _ensure_well_position_col(df: pl.DataFrame) -> pl.DataFrame:
    """Alias Metadata_Well → Metadata_well_position if missing.

    build_control_pairs (and _build_pseudo_pair_pool) require
    Metadata_well_position; CellProfiler features.parquet stores the
    column as Metadata_Well, DL embeddings already use the *_position form.
    """
    if "Metadata_well_position" in df.columns:
        return df
    if "Metadata_Well" in df.columns:
        return df.with_columns(Metadata_well_position=pl.col("Metadata_Well"))
    raise ValueError(
        "DataFrame missing both Metadata_well_position and Metadata_Well"
    )


def _enumerate_loo_groups(
    df_full: pl.DataFrame,
    min_cells: int,
) -> list[tuple[str, str, list[str], str]]:
    """Enumerate (allele, platemap, [wells], category) groups for NC+PC controls.

    Same subset filter as XGBoost's build_control_pairs: ``Metadata_Control``
    in {NC, PC}, with at least ``min_cells`` cells per (allele, platemap, well,
    Metadata_Control) group. Only groups with ≥3 wells are returned (LOO
    construction needs ≥1 pseudo-variant + ≥2 pseudo-references).
    """
    ctrl = (
        df_full.lazy()
        .filter(pl.col("Metadata_Control").is_in(["NC", "PC"]))
        .group_by(
            "Metadata_gene_allele",
            "Metadata_plate_map_name",
            "Metadata_well_position",
            "Metadata_Control",
            maintain_order=True,
        )
        .agg(pl.len().alias("n"))
        .filter(pl.col("n") >= min_cells)
        .sort("Metadata_gene_allele", "Metadata_plate_map_name", "Metadata_Control",
              "Metadata_well_position")
        .collect()
    )
    groups: list[tuple[str, str, list[str], str]] = []
    for (allele, platemap, ctrl_type), grp in ctrl.group_by(
        "Metadata_gene_allele", "Metadata_plate_map_name", "Metadata_Control",
        maintain_order=True,
    ):
        wells = sorted(grp["Metadata_well_position"].to_list())
        if len(wells) >= 3:
            groups.append((str(allele), str(platemap), wells, str(ctrl_type)))
    return groups


def _build_pseudo_loo_pool(
    df_full: pl.DataFrame,
    allele: str,
    platemap: str,
    w_var: str,
    w_refs: list[str],
) -> tuple[str, pl.DataFrame]:
    """Leave-one-out pool: one well as pseudo-variant vs all OTHER wells of the same allele.

    Relabels:
      • w_var cells  → node_type='allele',     gene_allele=loo_id (unique)
      • w_refs cells → node_type='disease_wt', gene_allele=allele (kept)
      • both         → symbol=loo_id (keeps same-gene negatives within this LOO group)

    Per anchor: 27 cross-plate same-well-X positives + 27 same-plate other-3-wells negatives
    (vs. per-pair's 27+9). Symmetric pool, less single-well-pair tail dominance.
    """
    loo_id = f"loo__{allele}__{platemap}__{w_var}"
    base = (
        (pl.col("Metadata_gene_allele") == allele)
        & (pl.col("Metadata_plate_map_name") == platemap)
        & pl.col("Metadata_Control").is_in(["NC", "PC"])
    )
    var = df_full.filter(base & (pl.col("Metadata_well_position") == w_var)).with_columns(
        Metadata_node_type=pl.lit("allele"),
        Metadata_symbol=pl.lit(loo_id),
        Metadata_gene_allele=pl.lit(loo_id),
    )
    ref = df_full.filter(base & pl.col("Metadata_well_position").is_in(w_refs)).with_columns(
        Metadata_node_type=pl.lit("disease_wt"),
        Metadata_symbol=pl.lit(loo_id),
        Metadata_gene_allele=pl.lit(allele),
    )
    return loo_id, pl.concat([var, ref])


def _compute_control_null(
    df_full: pl.DataFrame,
    channel_map: dict[str, list[str]],
    cells_per_site: int,
    neg_per_plate: int,
    sample_level: str,
    aggregate: bool,
    null_size: int,
    threshold: float,
    seed: int,
    max_workers: int | None,
    test_split: str | None,
    min_cells: int = MIN_CELL_COUNT,
) -> pl.DataFrame:
    """Empirical control null using leave-one-out (LOO) construction.

    For each NC/PC control allele's set of wells on a platemap (4 in the 2×2
    technical-control quadrant), iterate each well as the pseudo-variant and
    use the OTHER same-allele wells as the pseudo-reference pool. Each
    (allele, platemap, well_var) → one mAP per channel.

    Per anchor pool: 27 pos + 27 neg (vs per-pair's 27 + 9), so AP saturates
    much less easily and the threshold is robust to single-well outliers.
    """
    groups = _enumerate_loo_groups(df_full, min_cells)
    n_runs_total = sum(len(wells) for _, _, wells, _ in groups) * len(channel_map)
    logger.info(
        "Control null (LOO): %d (allele × platemap) groups, %d total (well × channel) runs",
        len(groups), n_runs_total,
    )

    rows: list[pl.DataFrame] = []
    skipped = 0
    for allele, platemap, wells, category in groups:
        for w_var in wells:
            w_refs = [w for w in wells if w != w_var]
            loo_id, pool = _build_pseudo_loo_pool(df_full, allele, platemap, w_var, w_refs)
            if pool.height < 2:
                skipped += 1
                continue
            for ch_name, ch_feats in channel_map.items():
                # Empty queries are explicit exclusions. Unexpected failures
                # must abort rather than calibrate on a silently partial null.
                ch_ctrl = _compute_map_vs_ref(
                    pool, [loo_id], ch_feats,
                    cells_per_site, neg_per_plate, sample_level, aggregate,
                    null_size, threshold, seed, max_workers,
                    test_split=test_split,
                )
                if ch_ctrl.is_empty():
                    skipped += 1
                    continue
                rows.append(
                    ch_ctrl.with_columns(
                        channel=pl.lit(ch_name),
                        pair_id=pl.lit(loo_id),
                        allele=pl.lit(allele),
                        platemap=pl.lit(platemap),
                        well_var=pl.lit(w_var),
                        category=pl.lit(category),
                    )
                )
    if skipped:
        logger.info("Control null (LOO): skipped=%d (empty result)", skipped)
    if not rows:
        return pl.DataFrame()
    return pl.concat(rows, how="diagonal")


def _plot_map_distributions(
    variant_results: pl.DataFrame,
    control_results: pl.DataFrame,
    output_dir,
    batch_id: str,
    representation: str,
    null_percentile: float,
) -> None:
    """Per-channel histogram: variant mAP_vs_ref_norm vs LOO control null.

    Mirrors plot_auroc_distributions for XGBoost: one panel per channel,
    overlapping density histograms for control (LOO null) and variant
    (Exp + cPC) distributions, vertical line at the per-channel pNN
    threshold from the control distribution.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning("matplotlib not available, skipping mAP_distribution plot")
        return

    if "channel" not in variant_results.columns or "channel" not in control_results.columns:
        return
    channels = sorted(variant_results["channel"].unique().to_list())
    n_ch = len(channels)
    if n_ch == 0:
        return

    fig, axes = plt.subplots(1, n_ch, figsize=(4 * n_ch, 4), sharey=False)
    if n_ch == 1:
        axes = [axes]

    bins = np.linspace(-0.4, 1.0, 40)
    q = null_percentile / 100.0
    for ax, ch in zip(axes, channels):
        ctrl = (
            control_results.filter(pl.col("channel") == ch)["mAP_vs_ref_norm"]
            .drop_nulls().drop_nans().to_numpy()
        )
        exp = (
            variant_results.filter(pl.col("channel") == ch)["mAP_vs_ref_norm"]
            .drop_nulls().drop_nans().to_numpy()
        )
        ax.hist(
            ctrl, bins=bins, alpha=0.6, density=True,
            color="steelblue", label=f"Control LOO (n={len(ctrl)})",
        )
        ax.hist(
            exp, bins=bins, alpha=0.6, density=True,
            color="coral", label=f"Exp+cPC (n={len(exp)})",
        )
        if len(ctrl) > 0:
            thr = float(np.quantile(ctrl, q))
            ax.axvline(
                thr, color="navy", linestyle="--", linewidth=1.5,
                label=f"p{int(null_percentile)}={thr:.3f}",
            )
        ax.set_title(ch)
        ax.set_xlabel("mAP_vs_ref_norm")
        ax.legend(fontsize=7)

    axes[0].set_ylabel("Density")
    fig.suptitle(
        f"mAP_vs_ref_norm distribution — {representation} / {batch_id}",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()

    path = output_dir / "mAP_distribution.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", path)


# ── HPA phenotypic consistency (reference genes) ─────────────────────────────


def _load_hpa_labels(threshold: float) -> dict[str, list[str]]:
    """Load HPA gene → list-of-organelles labels at a reliability threshold.

    Scoring in the HPA table: Enhanced=3, Supported=2, Approved=1, Uncertain=0.5.
    Genes with no organelle above threshold are excluded.
    """
    hpa = pl.read_parquet(str(HPA_GENE_LOCALIZATION_PATH))
    loc_cols = [c for c in hpa.columns if c not in ("Gene", "Gene name")]
    labels: dict[str, list[str]] = {}
    for row in hpa.iter_rows(named=True):
        gene = row["Gene name"]
        if gene is None:
            continue
        organelles = [c for c in loc_cols if (row[c] or 0) >= threshold]
        if organelles:
            labels[gene] = organelles
    return labels


def _compute_map_hpa(
    df_full: pl.DataFrame,
    feat_cols: list[str],
    cells_per_site: int,
    sample_level: str,
    aggregate: bool,
    hpa_threshold: float,
    hpa_consensus: str,
    null_size: int,
    threshold: float,
    seed: int,
    max_workers: int | None = None,
) -> pl.DataFrame:
    """mAP: reference (disease_wt) genes sharing HPA organelle labels.

    Tests whether reference genes annotated to the same HPA organelle cluster
    together in the profile space. Uses copairs multilabel AP: each gene gets a
    list of organelles above `hpa_threshold`; positive pairs are two different
    genes sharing at least one organelle; negative pairs are two different
    genes sharing none.

    hpa_consensus controls aggregation level before running copairs:
      - "gene": 1 median profile per gene (smallest pool, no replicate structure)
      - "plate": 1 median profile per gene × plate (standard copairs pattern)
      - "fov": 1 median profile per gene × plate × site (most points, pseudo-replicated)
    """
    unit_col = _resolve_well_col(df_full) if sample_level == "well" else _SAMPLE_UNIT_COL[sample_level]
    gene_to_labels = _load_hpa_labels(hpa_threshold)
    if not gene_to_labels:
        logger.warning("HPA: no genes with organelle scores ≥ %s", hpa_threshold)
        return pl.DataFrame()

    ref_df = df_full.filter(
        (pl.col("Metadata_node_type") == "disease_wt") &
        pl.col("Metadata_symbol").is_in(list(gene_to_labels.keys()))
    )
    if ref_df.is_empty():
        logger.warning("HPA: no reference cells matched HPA genes")
        return pl.DataFrame()

    # Consensus aggregation: pick grouping columns based on consensus level
    consensus_groups = {
        "gene": ["Metadata_symbol"],
        "plate": ["Metadata_symbol", "Metadata_Plate"],
        "fov": ["Metadata_symbol", "Metadata_Plate", unit_col],
    }
    group_cols = consensus_groups[hpa_consensus]

    if aggregate or hpa_consensus in ("gene", "plate"):
        # Always aggregate (median) for gene/plate consensus; also when --aggregate
        pool = _aggregate_per_group(ref_df, group_cols, feat_cols)
    else:
        pool = _sample_per_site(ref_df, group_cols, cells_per_site)

    n_genes = pool["Metadata_symbol"].n_unique()
    logger.info(
        "HPA: %d profiles from %d reference genes (threshold=%s, consensus=%s)",
        pool.height, n_genes, hpa_threshold, hpa_consensus,
    )

    # Attach list-of-labels as a pandas object column (copairs multilabel expects lists)
    pool_pd = pool.to_pandas()
    pool_pd[HPA_LABELS_COL] = pool_pd["Metadata_symbol"].map(lambda g: gene_to_labels.get(g, []))
    # Drop any rows that accidentally ended up with no labels (safety)
    pool_pd = pool_pd[pool_pd[HPA_LABELS_COL].map(len) > 0].reset_index(drop=True)

    non_feat = [c for c in pool_pd.columns if c not in feat_cols]
    feats_np = pool_pd[feat_cols].to_numpy().astype(np.float32)

    # Pair rules: cross-gene (different genes = pos or neg by label overlap).
    # For plate/fov consensus, also require cross-plate positives to avoid
    # plate-confounded signal; gene-level consensus has no plate dimension.
    pos_diffby = ["Metadata_symbol"]
    if hpa_consensus in ("plate", "fov"):
        pos_diffby.append("Metadata_Plate")

    try:
        ap_scores = average_precision_multilabel(
            pool_pd[non_feat],
            feats_np,
            pos_sameby=[HPA_LABELS_COL],
            pos_diffby=pos_diffby,
            neg_sameby=[],
            neg_diffby=["Metadata_symbol", HPA_LABELS_COL],
            multilabel_col=HPA_LABELS_COL,
            progress_bar=False,
        )
    except Exception as e:
        logger.warning("HPA: multilabel AP failed: %s", e)
        return pl.DataFrame()

    # mAP per gene (aggregates across organelle labels and profiles)
    map_scores = mean_average_precision(
        ap_scores,
        sameby=["Metadata_symbol"],
        null_size=null_size,
        threshold=threshold,
        seed=seed,
        max_workers=max_workers,
        progress_bar=False,
    )
    map_scores = map_scores.rename(columns={
        "mean_average_precision": "mAP_hpa",
        "mean_normalized_average_precision": "mAP_hpa_norm",
        "p_value": "p_value_hpa",
        "corrected_p_value": "corrected_p_value_hpa",
        "below_p": "below_p_hpa",
        "below_corrected_p": "below_corrected_p_hpa",
    }).drop(columns=["indices"], errors="ignore")

    return pl.from_pandas(map_scores)


# ── Main pipeline ─────────────────────────────────────────────────────────────


def run_phenotypic_activity(
    batch_id: str,
    representation: str,
    scope: str = "all",
    cells_per_site: int = 20,
    neg_per_plate: int = 0,
    sample_level: str = "site",
    aggregate: bool = True,
    hpa: bool = False,
    hpa_threshold: float = 1.0,
    hpa_consensus: str = "plate",
    null_size: int = 10_000,
    threshold: float = 0.05,
    seed: int = 42,
    max_workers: int | None = None,
    cp_feature_file: str = "features",
    test_split: str | None = None,
    control_null: bool = True,
    null_percentile: float = NULL_PERCENTILE,
    ctrl_null_size: int = 1_000,
) -> None:
    """Run phenotypic activity assessment for one batch + representation.

    ``test_split`` (currently only ``"t4"`` supported) restricts per-allele
    aggregation to wells/cells on T4 plates. The full pool is still used for
    pair construction and per-row AP computation. Output is written to a
    ``{rep}_t4`` directory so it doesn't clobber the all-data run.

    ``control_null`` adds an empirical null based on the same NC+PC subset
    XGBoost uses, but with leave-one-out (LOO) construction: each well of a
    control allele is treated as a pseudo-variant against the OTHER same-allele
    wells on the same platemap. Per anchor: 27 cross-plate same-well positives
    + 27 same-plate other-well negatives — symmetric and robust to single-well
    tail dominance. Per-channel p95 of LOO mAP_vs_ref_norm becomes the hit
    threshold. Adds ``null_threshold_p95`` and ``is_hit`` columns to
    mAP_results.parquet, writes the raw LOO distribution to mAP_control.parquet,
    and writes a per-channel histogram (variant vs control) to
    mAP_distribution.png.
    """
    t0 = time.time()

    # ── Resolve paths and config ──────────────────────────────────────
    representation = canonical_representation(representation)
    if representation not in REP_FEATURE_FILES:
        raise ValueError(f"Unknown representation: {representation}")
    layout = BATCH_LAYOUT.get(batch_id)
    if layout is None:
        logger.error("Unknown batch %s (not in BATCH_LAYOUT)", batch_id)
        sys.exit(1)

    if test_split is not None and layout != "single_rep":
        logger.error(
            "T4 evaluation requires single_rep batch layout; %s is %s.",
            batch_id, layout,
        )
        sys.exit(1)

    if control_null and layout != "single_rep":
        logger.warning(
            "--control-null only supported on single_rep batches; %s is %s — disabling.",
            batch_id, layout,
        )
        control_null = False

    rep_suffix = f"_{test_split}" if test_split else ""
    output_dir = PHENOTYPIC_ACTIVITY_DIR / f"{representation}{rep_suffix}" / batch_id
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info(
        "Phenotypic activity: batch=%s rep=%s layout=%s scope=%s %s neg_per_plate=%d",
        batch_id, representation, layout, scope,
        f"aggregate=median/{sample_level}" if aggregate else f"cells_per_{sample_level}={cells_per_site}",
        neg_per_plate,
    )
    logger.info("=" * 70)

    # ── Load features ─────────────────────────────────────────────────
    logger.info("Loading features...")
    if representation == "cellprofiler":
        df_full = _load_cellprofiler(batch_id, cp_feature_file)
    else:
        df_full = _load_dl(representation, batch_id)

    df_full = _ensure_well_position_col(df_full)

    all_cols = df_full.columns
    feat_cols = [c for c in all_cols if not c.startswith("Metadata_") and c != "_label"]
    logger.info("Schema: %d feature columns, %d cells", len(feat_cols), df_full.height)

    # ── Resolve alleles (scope + min-cell-count filter) ───────────────
    alleles = _resolve_alleles(df_full, scope)
    if not alleles:
        logger.error("No alleles found after scope filter '%s'", scope)
        sys.exit(1)
    logger.info("%d alleles selected (scope='%s')", len(alleles), scope)

    # ── Split features by channel ────────────────────────────────────
    channel_map = get_feature_channels(feat_cols, representation)
    logger.info("Channels: %s", {k: len(v) for k, v in channel_map.items()})

    # ── Compute mAP per channel ──────────────────────────────────────
    all_channel_results: list[pl.DataFrame] = []
    for channel_name, ch_feats in channel_map.items():
        logger.info("─── Channel: %s (%d features) ───", channel_name, len(ch_feats))

        ch_results = _compute_map_vs_ref(
            df_full, alleles, ch_feats, cells_per_site, neg_per_plate, sample_level, aggregate, null_size, threshold, seed, max_workers,
            test_split=test_split,
        )

        if ch_results.is_empty():
            logger.warning("Channel %s: no mAP results", channel_name)
            continue

        ch_results = ch_results.with_columns(pl.lit(channel_name).alias("channel"))
        all_channel_results.append(ch_results)

        # Per-channel summary
        if "mAP_vs_ref" in ch_results.columns:
            n = ch_results["mAP_vs_ref"].drop_nulls().len()
            sig_col = "below_corrected_p_vs_ref"
            pct_sig = (100.0 * ch_results[sig_col].sum() / n) if (sig_col in ch_results.columns and n) else float("nan")
            logger.info(
                "  %s mAP_vs_ref  mean=%.3f  mean_norm=%.3f  significant=%.1f%%",
                channel_name,
                ch_results["mAP_vs_ref"].mean(),
                ch_results["mAP_vs_ref_norm"].mean() if "mAP_vs_ref_norm" in ch_results.columns else float("nan"),
                pct_sig,
            )

    if not all_channel_results:
        logger.error("No mAP results produced for any channel")
        sys.exit(1)

    results = pl.concat(all_channel_results, how="diagonal")
    results = results.with_columns([
        pl.lit(batch_id).alias("batch"),
        pl.lit(representation).alias("representation"),
    ])

    # ── Empirical control null (NC + PC, leave-one-out) ──────────────
    if control_null:
        logger.info("─── Control null (LOO NC+PC, p%d) ───", int(null_percentile))
        control_results = _compute_control_null(
            df_full, channel_map,
            cells_per_site, neg_per_plate, sample_level, aggregate,
            ctrl_null_size, threshold, seed, max_workers, test_split,
        )

        if control_results.is_empty():
            logger.warning("Control null: no mAP scores produced — skipping hit calls")
            results = results.with_columns(
                null_threshold_p95=pl.lit(None, dtype=pl.Float64),
                is_hit=pl.lit(None, dtype=pl.Boolean),
            )
        else:
            ctrl_p95 = (
                control_results
                .group_by("channel")
                .agg(
                    pl.col("mAP_vs_ref_norm")
                      .quantile(null_percentile / 100.0, "linear")
                      .alias("null_threshold_p95"),
                    pl.col("mAP_vs_ref_norm").count().alias("n_control_pairs"),
                )
            )
            for row in ctrl_p95.iter_rows(named=True):
                logger.info(
                    "  %s null_threshold_p95=%.4f  n_control_pairs=%d",
                    row["channel"], row["null_threshold_p95"], row["n_control_pairs"],
                )
            results = (
                results
                .join(ctrl_p95.drop("n_control_pairs"), on="channel", how="left")
                .with_columns(
                    is_hit=(pl.col("mAP_vs_ref_norm") > pl.col("null_threshold_p95")),
                )
            )
            ctrl_path = output_dir / "mAP_control.parquet"
            control_results.with_columns(
                batch=pl.lit(batch_id),
                representation=pl.lit(representation),
            ).write_parquet(str(ctrl_path))
            logger.info(
                "Wrote %s: %d (LOO well × channel) rows", ctrl_path, control_results.height,
            )
            n_hit = int(results["is_hit"].cast(pl.Int8).sum() or 0)
            logger.info(
                "Empirical hits (mAP_vs_ref_norm > p%d): %d / %d (alleles × channels)",
                int(null_percentile), n_hit, results.height,
            )

            # Per-batch histogram: variant vs LOO control distribution by channel
            try:
                _plot_map_distributions(
                    results, control_results, output_dir, batch_id, representation,
                    null_percentile,
                )
            except Exception as e:
                logger.warning("mAP_distribution plot failed: %s", e)

    # ── Write output ──────────────────────────────────────────────────
    out_path = output_dir / "mAP_results.parquet"
    results.write_parquet(str(out_path))
    n_channels = results["channel"].n_unique()
    n_alleles = results["Metadata_gene_allele"].n_unique()
    logger.info("Wrote %s: %d alleles × %d channels", out_path, n_alleles, n_channels)

    # ── HPA phenotypic consistency (reference genes only) ────────────
    if hpa:
        logger.info("─── HPA consistency (threshold=%.1f, consensus=%s) ───",
                    hpa_threshold, hpa_consensus)
        hpa_channel_results: list[pl.DataFrame] = []
        for channel_name, ch_feats in channel_map.items():
            map_hpa = _compute_map_hpa(
                df_full, ch_feats, cells_per_site, sample_level, aggregate,
                hpa_threshold, hpa_consensus, null_size, threshold, seed, max_workers,
            )
            if map_hpa.is_empty():
                continue
            map_hpa = map_hpa.with_columns(pl.lit(channel_name).alias("channel"))
            n_sig = map_hpa["below_corrected_p_hpa"].sum() if "below_corrected_p_hpa" in map_hpa.columns else 0
            n = map_hpa.height
            logger.info(
                "  %s mAP_hpa mean=%.3f  mean_norm=%.3f  significant=%.1f%%",
                channel_name,
                map_hpa["mAP_hpa"].mean(),
                map_hpa["mAP_hpa_norm"].mean(),
                100.0 * n_sig / n if n else float("nan"),
            )
            hpa_channel_results.append(map_hpa)

        if hpa_channel_results:
            hpa_results = pl.concat(hpa_channel_results, how="diagonal").with_columns([
                pl.lit(batch_id).alias("batch"),
                pl.lit(representation).alias("representation"),
                pl.lit(hpa_threshold).alias("hpa_threshold"),
                pl.lit(hpa_consensus).alias("hpa_consensus"),
            ])
            hpa_path = output_dir / f"hpa_consistency_t{hpa_threshold}_{hpa_consensus}.parquet"
            hpa_results.write_parquet(str(hpa_path))
            logger.info(
                "Wrote %s: %d genes × %d channels",
                hpa_path, hpa_results["Metadata_symbol"].n_unique(), hpa_results["channel"].n_unique(),
            )

    elapsed = time.time() - t0
    logger.info("=" * 70)
    logger.info("Done in %.1f min", elapsed / 60)
    logger.info("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phenotypic activity assessment using copairs mAP (single-cell)."
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
        help="Feature representation to use (morphem is an alias for vit)",
    )
    parser.add_argument(
        "--scope",
        default="all",
        help=(
            "Which pairs to include: 'all' (default), 'exp', "
            "or comma-separated allele names (e.g., 'CCM2_Ile432Thr,KRAS_Gly12Val')"
        ),
    )
    parser.add_argument(
        "--cells-per-site",
        type=int,
        default=20,
        help="Cells sampled per site (FOV) per allele/group (default: 20)",
    )
    parser.add_argument(
        "--null-size",
        type=int,
        default=10_000,
        help="Null distribution size for mAP p-value calculation (default: 10000)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="p-value threshold for significance (default: 0.05)",
    )
    parser.add_argument(
        "--neg-per-plate",
        type=int,
        default=0,
        help=(
            "Max reference rows per plate (same subset for all alleles). "
            "<=0 disables the cap. Default: 0 (no cap) — appropriate for "
            "site-level aggregation. Set to a positive value (e.g. 135) "
            "for cell-level mode where capping at a fixed cell count makes sense."
        ),
    )
    parser.add_argument(
        "--sample-level",
        choices=["site", "well"],
        default="site",
        help="Sampling unit for --cells-per-site: 'site' (FOV/ImageNumber) or 'well'. Default: site",
    )
    parser.add_argument(
        "--aggregate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use median feature values per FOV/well (default True). Pass --no-aggregate to sample single cells.",
    )
    parser.add_argument(
        "--hpa",
        action="store_true",
        help="Also compute HPA phenotypic-consistency mAP on reference (disease_wt) genes",
    )
    parser.add_argument(
        "--hpa-threshold",
        type=float,
        default=1.0,
        help=(
            "HPA reliability score cutoff for including an organelle as a label "
            "(Enhanced=3, Supported=2, Approved=1, Uncertain=0.5). Default: 1.0"
        ),
    )
    parser.add_argument(
        "--hpa-consensus",
        choices=["gene", "plate", "fov"],
        default="plate",
        help=(
            "Aggregation level before HPA consistency mAP: "
            "'gene' (1 profile per gene), 'plate' (per gene × plate, recommended), "
            "'fov' (per gene × plate × site). Default: plate"
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Number of threads for p-value computation (default: all available)",
    )
    parser.add_argument(
        "--cp-feature-file",
        choices=["normalized", "features"],
        default="features",
        help=(
            "CellProfiler feature source: 'normalized' (normalized.parquet, "
            "pre-feature-selection, ~2946 feats) or 'features' (features.parquet, "
            "post-selection, ~1060 feats). Default: features"
        ),
    )
    parser.add_argument(
        "--test-split",
        choices=["none", "t4"],
        default="none",
        help=(
            "Held-out test-split filter applied AFTER per-row AP computation: "
            "'t4' aggregates only over wells/cells on T4 plates (full pool "
            "still used for pair construction). Outputs land at "
            "data/processed/classification_PA/{rep}_t4/{batch}/. "
            "Default: 'none' (all-data behavior)."
        ),
    )
    parser.add_argument(
        "--control-null",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Compute empirical null from NC+PC controls using leave-one-out: "
            "each well of a control allele is treated as pseudo-variant against "
            "the OTHER same-allele wells on the same platemap (metric: "
            "mAP_vs_ref_norm). Adds null_threshold_p95 + is_hit columns to "
            "mAP_results.parquet, writes mAP_control.parquet + "
            "mAP_distribution.png. Single_rep batches only. Default: enabled."
        ),
    )
    parser.add_argument(
        "--null-percentile",
        type=float,
        default=NULL_PERCENTILE,
        help=(
            f"Percentile of control mAP_vs_ref_norm used as the empirical "
            f"hit threshold (default: {NULL_PERCENTILE}, matches XGBoost)."
        ),
    )
    parser.add_argument(
        "--ctrl-null-size",
        type=int,
        default=1_000,
        help=(
            "Permutation-null size for control mAP runs. Smaller than "
            "--null-size since we only use the norm (mean), not p-values. "
            "Default: 1000."
        ),
    )
    args = parser.parse_args()

    test_split = None if args.test_split == "none" else args.test_split
    rep_suffix = f"_{test_split}" if test_split else ""
    output_dir = PHENOTYPIC_ACTIVITY_DIR / f"{args.representation}{rep_suffix}" / args.batch
    run_phenotypic_activity(
        batch_id=args.batch,
        representation=args.representation,
        scope=args.scope,
        cells_per_site=args.cells_per_site,
        neg_per_plate=args.neg_per_plate,
        sample_level=args.sample_level,
        aggregate=args.aggregate,
        hpa=args.hpa,
        hpa_threshold=args.hpa_threshold,
        hpa_consensus=args.hpa_consensus,
        null_size=args.null_size,
        threshold=args.threshold,
        max_workers=args.max_workers,
        cp_feature_file=args.cp_feature_file,
        test_split=test_split,
        control_null=args.control_null,
        null_percentile=args.null_percentile,
        ctrl_null_size=args.ctrl_null_size,
    )
    # A failed rerun must not attach its provenance to previous result files.
    if output_dir.exists():
        record(output_dirs=[output_dir])


if __name__ == "__main__":
    main()
