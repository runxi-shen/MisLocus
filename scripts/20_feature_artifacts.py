#!/usr/bin/env python3
"""Declare and validate explicit feature artifacts without altering their values."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prot_loc_benchmark.workflows.features import (
    IDENTITY_COLUMNS, QUALIFIED_ID, load_artifact, save_json, sha256, validate_spec,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    spec = commands.add_parser("spec", help="Declare ordered features/channels for an existing Parquet")
    spec.add_argument("--features", type=Path, required=True)
    spec.add_argument("--model", required=True)
    choice = spec.add_mutually_exclusive_group(required=True)
    choice.add_argument("--columns-json", type=Path, help="JSON ordered feature list or channel->ordered features mapping")
    choice.add_argument("--released-representation", help="Use a known published representation's channel naming")
    spec.add_argument("--preprocessing", choices=["released_cleaned", "none"], required=True)
    spec.add_argument("--stored-id-encoding", choices=["sha256_physical", "batch_slash_cell_id"], default="sha256_physical",
                      help="Explicit released-ID adapter; verifies batch/CellID and physical components, preserves source IDs")
    spec.add_argument("--output", type=Path, required=True)
    validate = commands.add_parser("validate", help="Check identities, finite features and metadata without changing the input")
    validate.add_argument("--features", type=Path, required=True)
    validate.add_argument("--spec", type=Path, required=True)
    validate.add_argument("--batch")
    validate.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.command == "spec":
        import polars as pl
        names = pl.scan_parquet(args.features).collect_schema().names()
        if args.released_representation:
            if args.preprocessing != "released_cleaned":
                raise ValueError("Published feature inference requires explicit released_cleaned mode")
            from prot_loc_benchmark.classification.channels import get_feature_channels
            features = [c for c in names if not c.startswith("Metadata_") and c != "_label"]
            channels = get_feature_channels(features, args.released_representation)
        else:
            declaration = json.loads(args.columns_json.read_text())
            channels = {"EMBED": declaration} if isinstance(declaration, list) else declaration
            if not isinstance(channels, dict):
                raise ValueError("columns-json must contain an ordered list or channel mapping")
            features = list(dict.fromkeys(c for columns in channels.values() for c in columns))
        declaration = {"schema_version": 1, "model": args.model, "features": features,
                       "channels": channels, "preprocessing": {"kind": args.preprocessing}}
        if args.stored_id_encoding != 'sha256_physical':
            declaration['identity'] = {'stored_id_encoding': args.stored_id_encoding}
        validate_spec(declaration)
        missing = set(features) - set(names)
        if missing:
            raise ValueError(f"Declared features absent from Parquet: {sorted(missing)}")
        save_json(args.output, declaration)
        print(json.dumps({"specification": str(args.output), "features": len(features), "channels": list(channels)}))
    elif args.command == "validate":
        _, _, report = load_artifact(args.features, args.spec, batch=args.batch)
        if args.report:
            save_json(args.report, report)
        print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
