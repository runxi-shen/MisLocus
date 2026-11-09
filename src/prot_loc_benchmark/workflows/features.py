"""Feature artifacts with explicit columns, physical identities and preprocessing."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import polars as pl

IDENTITY_COLUMNS = [
    "Metadata_Batch", "Metadata_Plate", "Metadata_well_position", "Metadata_Site",
    "Metadata_ImageNumber", "Metadata_ObjectNumber",
]
QUALIFIED_ID = "Metadata_BatchQualifiedCellID"
SOURCE_QUALIFIED_ID = "Metadata_ReleasedBatchQualifiedCellID"
STORED_ID_ENCODINGS = {"sha256_physical", "batch_slash_cell_id"}
EVALUATION_COLUMNS = {
    "xgb": ["Metadata_gene_allele", "Metadata_symbol", "Metadata_node_type", "Metadata_Control", "Metadata_plate_map_name"],
    "pa": ["Metadata_gene_allele", "Metadata_symbol", "Metadata_node_type", "Metadata_Control", "Metadata_plate_map_name"],
    "hpa": ["Metadata_symbol", "Metadata_node_type"],
}


def sha256(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_json(path: str | Path, value: object) -> None:
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def validate_spec(spec: dict) -> None:
    """A model name does not implicitly choose feature columns or transformations."""
    if spec.get("schema_version") != 1 or not isinstance(spec.get("model"), str) or not spec["model"].strip():
        raise ValueError("Feature specification needs schema_version=1 and a nonblank model name")
    identity = spec.get("identity", {})
    if not isinstance(identity, dict) or set(identity) - {"stored_id_encoding"} or identity.get("stored_id_encoding", "sha256_physical") not in STORED_ID_ENCODINGS:
        raise ValueError("Declare identity.stored_id_encoding as sha256_physical or batch_slash_cell_id")
    features = spec.get("features")
    if not isinstance(features, list) or not features or any(not isinstance(c, str) or not c for c in features):
        raise ValueError("Declare the ordered, nonempty feature list")
    if len(features) != len(set(features)) or any(c.startswith("Metadata_") or c in {"_label", "Label", "label", "Prediction", "split_role"} for c in features):
        raise ValueError("Feature names must be unique and exclude metadata/labels")
    channels = spec.get("channels")
    if not isinstance(channels, dict) or not channels:
        raise ValueError("Declare channel names and their ordered feature lists")
    used = set()
    for name, columns in channels.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(columns, list) or not columns:
            raise ValueError("Channel names and feature lists must be nonempty")
        if Path(name).name != name or name in {".", ".."} or any(not isinstance(c, str) for c in columns):
            raise ValueError("Channel names must be simple names and feature names must be strings")
        if len(columns) != len(set(columns)) or not set(columns) <= set(features):
            raise ValueError(f"Invalid feature columns in channel {name!r}")
        used.update(columns)
    if used != set(features):
        raise ValueError("Every declared feature must belong to at least one channel")
    preprocessing = spec.get("preprocessing")
    if not isinstance(preprocessing, dict) or preprocessing.get("kind") not in {"released_cleaned", "none", "fitted"}:
        raise ValueError("Declare preprocessing.kind as released_cleaned, none, or fitted")
    if preprocessing["kind"] == "fitted" and not re.fullmatch(r"[0-9a-f]{64}", str(preprocessing.get("transform_receipt_sha256", ""))):
        raise ValueError("Fitted preprocessing requires a bound transform receipt")


def validate_identity(frame: pl.DataFrame, *, batch: str | None = None,
                      stored_id_encoding: str = "sha256_physical") -> pl.DataFrame:
    """Validate physical and supplied semantic metadata without dropping any cells."""
    if frame.is_empty():
        raise ValueError("Feature artifact has no cells")
    if stored_id_encoding not in STORED_ID_ENCODINGS:
        raise ValueError("Unknown stored ID encoding")
    original_rows = frame.height
    if "Metadata_well_position" not in frame.columns:
        if "Metadata_Well" not in frame.columns:
            raise ValueError("Missing well metadata: Metadata_well_position or Metadata_Well")
        frame = frame.with_columns(pl.col("Metadata_Well").alias("Metadata_well_position"))
    elif "Metadata_Well" in frame.columns and not frame["Metadata_Well"].equals(frame["Metadata_well_position"]):
        raise ValueError("Metadata_Well conflicts with Metadata_well_position")
    if "Metadata_Batch" not in frame.columns:
        if not batch or not batch.strip():
            raise ValueError("Supply batch explicitly when Metadata_Batch is absent")
        frame = frame.with_columns(pl.lit(batch).alias("Metadata_Batch"))
    elif batch is not None and not (frame["Metadata_Batch"] == batch).all():
        raise ValueError("Explicit batch conflicts with cell metadata")
    missing = set(IDENTITY_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing feature/identity columns: {sorted(missing)}")
    for column in IDENTITY_COLUMNS:
        if frame[column].null_count():
            raise ValueError(f"Null physical identity: {column}")
    for column in IDENTITY_COLUMNS[:3]:
        if frame[column].dtype != pl.String or not frame[column].str.contains(r"\S").all():
            raise ValueError(f"Physical identity must be a nonblank string: {column}")
        if not (frame[column] == frame[column].str.strip_chars()).all():
            raise ValueError(f"Physical identity has leading/trailing whitespace: {column}")
    for column in IDENTITY_COLUMNS[3:]:
        values = frame[column]
        if not values.dtype.is_integer() or not (values >= 0).all():
            raise ValueError(f"Physical site/image/object identifiers must be nonnegative integers: {column}")
    semantic = set(c for columns in EVALUATION_COLUMNS.values() for c in columns) | {"Metadata_CellID"}
    for column in sorted(semantic & set(frame.columns)):
        values = frame[column]
        if values.dtype != pl.String or values.null_count() or not values.str.contains(r"\S").all():
            raise ValueError(f"Supplied semantic metadata must be non-null, nonblank strings: {column}")
        if not (values == values.str.strip_chars()).all():
            raise ValueError(f"Semantic metadata has leading/trailing whitespace: {column}")
    if "Metadata_Control" in frame.columns and not frame["Metadata_Control"].is_in(["Exp", "cPC", "NC", "PC", "TC"]).all():
        raise ValueError("Metadata_Control must use the declared benchmark roles Exp/cPC/NC/PC/TC")
    if frame.select(IDENTITY_COLUMNS).is_duplicated().any():
        raise ValueError("Duplicate batch/plate/well/site/image/object identity")
    identities = [
        hashlib.sha256(json.dumps(list(row), separators=(",", ":")).encode()).hexdigest()
        for row in frame.select(IDENTITY_COLUMNS).iter_rows()
    ]
    if stored_id_encoding == "batch_slash_cell_id" and (SOURCE_QUALIFIED_ID in frame.columns or
            (QUALIFIED_ID in frame.columns and frame[QUALIFIED_ID].to_list() != identities)):
        if not {"Metadata_CellID", "Metadata_SiteID"} <= set(frame.columns):
            raise ValueError("Released batch/slash IDs require Metadata_CellID and Metadata_SiteID")
        physical_source = frame.select(pl.concat_str([
            pl.col("Metadata_Plate"), pl.col("Metadata_well_position"),
            pl.col("Metadata_ImageNumber").cast(pl.String), pl.col("Metadata_ObjectNumber").cast(pl.String)
        ], separator="_").alias("expected"))["expected"]
        if not frame["Metadata_CellID"].equals(physical_source):
            raise ValueError("Released Metadata_CellID contradicts plate/well/image/object identity")
        physical_site = frame.select(pl.concat_str([
            pl.col("Metadata_Plate"), pl.col("Metadata_well_position"), pl.col("Metadata_Site").cast(pl.String)
        ], separator="_").alias("expected"))["expected"]
        if not frame["Metadata_SiteID"].equals(physical_site):
            raise ValueError("Released Metadata_SiteID contradicts plate/well/site identity")
        released = frame.select(pl.concat_str(["Metadata_Batch", "Metadata_CellID"], separator="/").alias("expected"))["expected"]
        if SOURCE_QUALIFIED_ID in frame.columns:
            if not frame[SOURCE_QUALIFIED_ID].equals(released):
                raise ValueError("Preserved released source IDs disagree with physical identity")
        if QUALIFIED_ID in frame.columns and frame[QUALIFIED_ID].to_list() != identities:
            if not frame[QUALIFIED_ID].equals(released):
                raise ValueError("Stored batch-qualified IDs disagree with verified batch/slash source encoding")
            frame = frame.with_columns(pl.col(QUALIFIED_ID).alias(SOURCE_QUALIFIED_ID))
    elif QUALIFIED_ID in frame.columns and frame[QUALIFIED_ID].to_list() != identities:
        raise ValueError("Stored batch-qualified IDs disagree with the declared physical identity")
    if "Metadata_CellID" in frame.columns:
        if frame["Metadata_CellID"].null_count() or frame.select("Metadata_Batch", "Metadata_CellID").is_duplicated().any():
            raise ValueError("Source Metadata_CellID is null or not unique within batch")
    frame = frame.with_columns(pl.Series(QUALIFIED_ID, identities))
    if frame.height != original_rows:
        raise RuntimeError("Validation must not change cell coverage")
    return frame


def validate_frame(frame: pl.DataFrame, spec: dict, *, batch: str | None = None) -> tuple[pl.DataFrame, dict]:
    """Validate explicit features and metadata, preserving cell order and feature values."""
    validate_spec(spec)
    stored_encoding = spec.get("identity", {}).get("stored_id_encoding", "sha256_physical")
    frame = validate_identity(frame, batch=batch, stored_id_encoding=stored_encoding)
    missing = set(spec["features"]) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing declared feature columns: {sorted(missing)}")
    for column in spec["features"]:
        if not frame[column].dtype.is_numeric():
            raise ValueError(f"Feature is not numeric: {column}")
    finite = frame.select(pl.all_horizontal([
        pl.col(c).is_finite().fill_null(False) for c in spec["features"]
    ]).all()).item()
    if not finite:
        raise ValueError("Features contain null, NaN, or infinite values; no cells were dropped")
    identities = frame[QUALIFIED_ID].to_list()
    report = {
        "schema_version": 1, "rows": frame.height, "features": len(spec["features"]),
        "channels": {name: len(columns) for name, columns in spec["channels"].items()},
        "batches": sorted(frame["Metadata_Batch"].unique().to_list()),
        "physical_plates": frame.select("Metadata_Batch", "Metadata_Plate").unique().height,
        "physical_wells": frame.select("Metadata_Batch", "Metadata_Plate", "Metadata_well_position").unique().height,
        "fields_of_view": frame.select(IDENTITY_COLUMNS[:-1]).unique().height,
        "identity_columns": IDENTITY_COLUMNS, "identity_encoding": "sha256(compact JSON array in declared identity-column order)",
        "accepted_source_id_encoding": stored_encoding,
        "preserved_source_id_column": SOURCE_QUALIFIED_ID if SOURCE_QUALIFIED_ID in frame.columns else None,
        "ordered_cell_ids_sha256": hashlib.sha256(json.dumps(identities, separators=(",", ":")).encode()).hexdigest(),
        "cell_coverage": "all input rows retained, original order", "preprocessing_applied_by_validator": "none",
        "declared_preprocessing": spec["preprocessing"], "transform_fit_population_verified": False,
        "evaluation_metadata": {task: {"missing_columns": sorted(set(columns) - set(frame.columns)),
                                        "eligibility_and_control_support": "must be checked by the evaluation"}
                                for task, columns in EVALUATION_COLUMNS.items()},
    }
    for column, label in [("Metadata_symbol", "proteins"), ("Metadata_gene_allele", "alleles")]:
        if column in frame.columns:
            report[label] = frame[column].drop_nulls().n_unique()
    if "Metadata_Control" in frame.columns:
        report["role_counts"] = frame.group_by("Metadata_Control").len().sort("Metadata_Control").to_dicts()
    return frame, report


def load_artifact(features: str | Path, specification: str | Path, *, batch: str | None = None) -> tuple[pl.DataFrame, dict, dict]:
    spec = json.loads(Path(specification).read_text())
    validate_spec(spec)
    frame, report = validate_frame(pl.read_parquet(features), spec, batch=batch)
    report.update(features_sha256=sha256(features), specification_sha256=sha256(specification))
    return frame, spec, report


def write_artifact(frame: pl.DataFrame, specification: dict, output: str | Path, *, batch: str | None = None,
                   provenance: dict | None = None) -> dict:
    """Write a new artifact; existing directories and input data are never overwritten."""
    frame, report = validate_frame(frame, specification, batch=batch)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    frame.write_parquet(output / "features.parquet")
    save_json(output / "feature-spec.json", specification)
    save_json(output / "validation.json", report)
    receipt = {"schema_version": 1, "model": specification["model"], "provenance": provenance or {},
               "validator_source_sha256": sha256(__file__),
               "outputs_sha256": {p.name: sha256(p) for p in output.iterdir() if p.is_file()}}
    save_json(output / "artifact.json", receipt)
    return receipt


def admit_inputs(manifest_path, data_root, output):
    """Verify pre-downloaded payloads/specifications; perform no downloads or normalization."""
    root = Path(__file__).resolve().parents[3]
    manifest = json.loads(Path(manifest_path).read_text())
    if not re.fullmatch(r'[0-9a-f]{40}', manifest['revision']):
        raise ValueError('Declare an immutable dataset revision')
    verified = {}
    for entry in [*manifest['features'], manifest['clinical_annotation']]:
        path = (Path(data_root) / entry['path']).resolve()
        if not path.is_relative_to(Path(data_root).resolve()):
            raise ValueError('Payload escapes data root')
        if ('size' in entry and path.stat().st_size != entry['size']) or sha256(path) != entry['sha256']:
            raise ValueError('Payload hash/size mismatch: ' + str(path))
        verified[str(path)] = entry['sha256']
        if 'spec' in entry:
            spec = Path(entry['spec']); spec = spec if spec.is_absolute() else root / spec
            if sha256(spec) != entry['spec_sha256']:
                raise ValueError('Feature specification hash mismatch')
            verified[str(spec)] = entry['spec_sha256']
    entry = manifest['hpa_annotation']; annotation = Path(entry['path'])
    annotation = annotation if annotation.is_absolute() else root / annotation
    if sha256(annotation) != entry['sha256']:
        raise ValueError('Curated HPA annotation hash mismatch')
    verified[str(annotation)] = entry['sha256']
    save_json(output, dict(status='complete', dataset_revision=manifest['revision'],
        manifest_sha256=sha256(manifest_path), inputs_sha256=verified,
        downloaded=False, hpa_source='Separately pinned local curated annotation',
        scope='Pinned byte identities verified; not evidence of fresh downloads or numerical scoring.'))


def pool_hpa(inputs, model, channels, output):
    import polars as pl
    frames, specs, bindings = [], [], {}
    for batch, features, specification in inputs:
        admission = dict(path=str(features), spec=str(specification), batch=batch, sha256=sha256(features))
        frame, spec, _ = load_artifact(admission['path'], admission['spec'], batch=admission['batch'])
        frame = frame.filter((pl.col('Metadata_node_type') == 'disease_wt') & pl.col('Metadata_Plate').str.ends_with('T4'))
        frames.append(frame); specs.append(spec)
        bindings[admission['path']] = admission['sha256']
    common = sorted(set.intersection(*(set(spec['features']) for spec in specs)))
    channels = {channel: [name for name in common if all(name in spec['channels'][channel] for spec in specs)]
                for channel in channels}
    if any(not values for values in channels.values()):
        raise ValueError('HPA has an empty common channel feature intersection')
    metadata = [name for name in frames[0].columns if name.startswith('Metadata_') and all(name in f.columns for f in frames)]
    # Qualified identities have already been verified; source IDs, when present, remain metadata.
    assert QUALIFIED_ID in metadata
    selected = pl.concat([frame.select(*metadata, *common) for frame in frames], how='vertical_relaxed')
    common = sorted({name for values in channels.values() for name in values})
    selected = selected.select(*metadata, *common)
    preprocessing = specs[0]['preprocessing']
    if preprocessing['kind'] == 'fitted' or any(spec['preprocessing'] != preprocessing for spec in specs):
        raise ValueError('Pooling requires consistent unfitted or released-cleaned preprocessing declarations')
    declaration = dict(schema_version=1, model=model, features=common, channels=channels,
                       preprocessing=preprocessing)
    write_artifact(selected, declaration, output, provenance=dict(inputs_sha256=bindings,
        selection='T4 disease_wt cells from all declared batches; sorted common feature intersection; no normalization'))
