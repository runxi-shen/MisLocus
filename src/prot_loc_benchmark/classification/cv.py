"""Cross-validation fold generation based on plate layout."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations

import polars as pl

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CVFold:
    """One cross-validation fold with train/test plate or well assignments."""

    fold_id: int
    test_plates: list[str]
    test_wells: list[str]
    train_plates: list[str]
    train_wells: list[str]


def generate_folds_single_rep(df: pl.DataFrame) -> list[CVFold]:
    """Generate leave-one-plate-out CV folds for single_rep layout.

    Groups by Metadata_plate_map_name, then for each plate in the platemap
    group, uses that plate as test and all others as train.

    Returns up to 4 folds (one per technical replicate plate).
    """
    plate_info = df.select("Metadata_Plate", "Metadata_plate_map_name").unique()

    platemap_counts = plate_info.group_by("Metadata_plate_map_name").agg(
        pl.col("Metadata_Plate").alias("plates")
    )

    folds: list[CVFold] = []
    fold_id = 0

    for row in platemap_counts.iter_rows(named=True):
        plates = sorted(row["plates"])
        if len(plates) < 2:
            continue

        for i, test_plate in enumerate(plates):
            train_plates = [p for j, p in enumerate(plates) if j != i]
            test_wells = (
                df.filter(pl.col("Metadata_Plate") == test_plate)["Metadata_well_position"]
                .unique()
                .sort()
                .to_list()
            )
            train_wells = (
                df.filter(pl.col("Metadata_Plate").is_in(train_plates))["Metadata_well_position"]
                .unique()
                .sort()
                .to_list()
            )
            folds.append(
                CVFold(
                    fold_id=fold_id,
                    test_plates=[test_plate],
                    test_wells=test_wells,
                    train_plates=train_plates,
                    train_wells=train_wells,
                )
            )
            fold_id += 1

    return folds


def generate_folds_multi_rep(df: pl.DataFrame) -> list[CVFold]:
    """Generate C(4,2)=6 well-pair CV folds for multi_rep layout.

    For multi_rep, each allele has 4 wells on the same plate. We pair
    ref and var wells positionally, giving 4 well-pairs. Then we generate
    all C(4,2)=6 ways to split 4 pairs into 2 train + 2 test pairs.

    Each fold: train on 2 well-pairs (4 wells), test on 2 well-pairs (4 wells).

    Requires a ``Label`` column (added by ``get_pair_data``).
    """
    if "Label" not in df.columns:
        raise ValueError("DataFrame must have a 'Label' column (call get_pair_data first)")

    plates = df["Metadata_Plate"].unique().to_list()
    ref_wells = sorted(
        df.filter(pl.col("Label") == 1)["Metadata_well_position"]
        .unique()
        .to_list()
    )
    var_wells = sorted(
        df.filter(pl.col("Label") == 0)["Metadata_well_position"]
        .unique()
        .to_list()
    )

    # Pair wells positionally
    n_pairs = min(len(ref_wells), len(var_wells))
    if n_pairs < 2:
        logger.warning(
            "Not enough well pairs for multi_rep CV: %d ref, %d var wells",
            len(ref_wells),
            len(var_wells),
        )
        return []

    well_pairs = list(zip(ref_wells[:n_pairs], var_wells[:n_pairs]))
    pair_indices = list(range(n_pairs))

    # C(n_pairs, 2) test combinations: each selects 2 pairs for testing
    folds: list[CVFold] = []
    for fold_id, test_idx in enumerate(combinations(pair_indices, 2)):
        test_idx_set = set(test_idx)
        train_idx = [i for i in pair_indices if i not in test_idx_set]

        test_wells_flat = []
        for i in test_idx:
            test_wells_flat.extend(well_pairs[i])

        train_wells_flat = []
        for i in train_idx:
            train_wells_flat.extend(well_pairs[i])

        folds.append(
            CVFold(
                fold_id=fold_id,
                test_plates=plates,
                test_wells=sorted(test_wells_flat),
                train_plates=plates,
                train_wells=sorted(train_wells_flat),
            )
        )

    logger.debug(
        "multi_rep: %d well-pairs → C(%d,2)=%d folds",
        n_pairs, n_pairs, len(folds),
    )
    return folds


def generate_folds(df: pl.DataFrame, layout: str, test_split: str | None = None) -> list[CVFold]:
    """Generate CV folds based on plate layout.

    Args:
        df: DataFrame for one classification pair (must have Label column).
        layout: "single_rep" or "multi_rep".

    Returns:
        - single_rep: up to 4 folds (leave-one-plate-out)
        - multi_rep: C(4,2)=6 folds (choose-2 well-pairs for test)
    """
    if test_split not in (None, "t4"):
        raise ValueError(f"Unknown test split: {test_split}")
    if test_split and (layout != "single_rep" or df["Metadata_Plate"].null_count()
                       or not df["Metadata_Plate"].str.contains(r"T[1-4]$").all()):
        raise ValueError("T4 holdout requires single_rep and explicit T1/T2/T3/T4 plates")
    if layout == "single_rep":
        folds = generate_folds_single_rep(df)
    elif layout == "multi_rep":
        folds = generate_folds_multi_rep(df)
    else:
        raise ValueError(f"Unknown layout: {layout!r}")

    if test_split:
        folds = [f for f in folds if all(p.endswith("T4") for p in f.test_plates)
                 and len(f.train_plates) == 3 and {p[-2:] for p in f.train_plates} == {"T1", "T2", "T3"}]
    logger.debug("Generated %d CV folds for layout=%s", len(folds), layout)
    return folds


def split_fold(
    df: pl.DataFrame,
    fold: CVFold,
    layout: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split DataFrame into train/test for one fold.

    For single_rep: split by plate membership.
    For multi_rep: split by well membership (same plate).
    """
    if layout == "single_rep":
        train_df = df.filter(pl.col("Metadata_Plate").is_in(fold.train_plates))
        test_df = df.filter(pl.col("Metadata_Plate").is_in(fold.test_plates))
    else:
        # multi_rep: split by wells within same plate
        train_df = df.filter(
            pl.col("Metadata_well_position").is_in(fold.train_wells)
        )
        test_df = df.filter(
            pl.col("Metadata_well_position").is_in(fold.test_wells)
        )

    return train_df, test_df
