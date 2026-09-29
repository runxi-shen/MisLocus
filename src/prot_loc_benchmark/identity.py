"""Canonical downstream cell identity. Never identify a cell by well/object alone."""

import hashlib
import json

import polars as pl

CELL_ID = "Metadata_BatchQualifiedCellID"


def identify_cells(frame: pl.DataFrame, batch: str, *, canonical: bool = False) -> pl.DataFrame:
    required = ["Metadata_Plate", "Metadata_ImageNumber", "Metadata_ObjectNumber"]
    if "Metadata_Well" not in frame.columns and "Metadata_well_position" in frame.columns:
        frame = frame.with_columns(pl.col("Metadata_well_position").alias("Metadata_Well"))
    required.append("Metadata_Well")
    if canonical:
        required += [CELL_ID, "Metadata_CellID", "Metadata_Batch", "Metadata_Site"]
    missing = set(required) - set(frame.columns)
    if missing or frame.is_empty():
        raise ValueError(f"Missing cell identity columns or cells: {sorted(missing)}")
    if frame.select(pl.any_horizontal(pl.col(required).is_null()).any()).item():
        raise ValueError("Null cell identity coordinates")
    if (
        "Metadata_well_position" in frame.columns
        and not frame.select(
            (pl.col("Metadata_Well") == pl.col("Metadata_well_position")).fill_null(False).all()
        ).item()
    ):
        raise ValueError("Conflicting well identity aliases")
    if "Metadata_Batch" in frame.columns:
        if not frame.select((pl.col("Metadata_Batch") == batch).fill_null(False).all()).item():
            raise ValueError("Cell identity batch mismatch")
    else:
        frame = frame.with_columns(pl.lit(batch).alias("Metadata_Batch"))
    coordinates = ["Metadata_Plate", "Metadata_Well", "Metadata_ImageNumber", "Metadata_ObjectNumber"]
    if "Metadata_Site" in frame.columns:
        coordinates.append("Metadata_Site")
    if "Metadata_CellID" not in frame.columns:
        # Legacy inputs only: unambiguous, image-qualified tuple, not string concatenation.
        ids = [json.dumps(row, separators=(",", ":")) for row in frame.select(coordinates).iter_rows()]
        frame = frame.with_columns(pl.Series("Metadata_CellID", ids))
    expected = pl.concat_str([pl.lit(batch + "/"), pl.col("Metadata_CellID")])
    if CELL_ID not in frame.columns:
        frame = frame.with_columns(expected.alias(CELL_ID))
    elif not frame.select((pl.col(CELL_ID) == expected).fill_null(False).all()).item():
        raise ValueError("Batch-qualified identity disagrees with source cell ID")
    for name in (CELL_ID, "Metadata_CellID"):
        if frame[name].null_count() or frame[name].n_unique() != frame.height or (frame[name] == "").any():
            raise ValueError(f"Null, empty or duplicate cell identity: {name}")
    if frame.select(coordinates).unique().height != frame.height:
        raise ValueError("Multiple cell IDs refer to the same image/object coordinates")
    return frame


def ordered_id_hash(frame: pl.DataFrame) -> str:
    digest = hashlib.sha256()
    for value in frame[CELL_ID]:
        digest.update((value + "\n").encode())
    return digest.hexdigest()


def metadata(frame):
    return frame.select([c for c in frame.columns if c.startswith("Metadata_")])


def check_cell_subset(before: pl.DataFrame, after: pl.DataFrame) -> None:
    """Allow accounted-for removals/reordering, never identity or metadata corruption."""
    if after[CELL_ID].null_count() or after[CELL_ID].n_unique() != after.height:
        raise ValueError("Duplicate/null output cell identity")
    if not after.join(before.select(CELL_ID), on=CELL_ID, how="anti").is_empty():
        raise ValueError("Output contains cells absent from its parent")
    columns = [
        c for c in before.columns if c.startswith("Metadata_") and c not in ("Metadata_Control", "Metadata_node_type")
    ]
    if set(columns) - set(after.columns):
        raise ValueError("Output lost identity/metadata columns")
    expected = before.join(after.select(CELL_ID), on=CELL_ID, how="semi").select(columns).sort(CELL_ID)
    if not expected.equals(after.select(columns).sort(CELL_ID)):
        raise ValueError("Output changed source cell identity/metadata")


def save_feature_identity(before, frame, directory, name):
    from prot_loc_benchmark.provenance import save_json

    check_cell_subset(before, frame)
    features = [c for c in frame.columns if not c.startswith("Metadata_")]
    if (
        frame.height != before.height
        or not features
        or not frame.select(pl.all_horizontal(pl.col(features).is_finite().fill_null(False)).all()).item()
    ):
        raise ValueError("Unexpected cell loss, empty features or nonfinite processed output")
    metadata(frame).with_row_index("artifact_row").write_parquet(directory / f"{name}_cells.parquet")
    save_json(
        directory / f"{name}_schema.json",
        {
            "rows": frame.height,
            "features": features,
            "dtypes": {c: str(frame.schema[c]) for c in frame.columns},
            "ordered_cell_ids_sha256": ordered_id_hash(frame),
            "cohort_cell_ids_sha256": ordered_id_hash(frame.select(CELL_ID).sort(CELL_ID)),
        },
    )


def save_cell_filter(before, after, directory, name):
    check_cell_subset(before, after)
    removed = before.join(after.select(CELL_ID), on=CELL_ID, how="anti")
    metadata(removed).with_columns(pl.lit(name).alias("exclusion_reason")).write_parquet(
        directory / f"excluded_{name}.parquet"
    )
    return metadata(after)
