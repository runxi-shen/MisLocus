#!/usr/bin/env python3
"""Report calibrated batch hits and matched cohorts without rerunning scoring."""

import argparse
import hashlib
import io
import json
from pathlib import Path

import polars as pl

from prot_loc_benchmark.benchmark.hits import hit_cohorts
from prot_loc_benchmark.config import canonical_representation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["xgb", "pa"], required=True)
    parser.add_argument("--input", nargs=3, action="append", required=True,
                        metavar=("REPRESENTATION", "BATCH", "FILE"),
                        help="Repeat for calibrated CSV/Parquet batch score tables")
    parser.add_argument("--setting", nargs=2, action="append", required=True,
                        metavar=("REPRESENTATION", "CHANNEL"),
                        help="Explicit settings whose finite support is intersected")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Must not already exist; failed runs may leave an empty directory")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    frames, inputs = [], []
    for rep, batch, name in args.input:
        rep = canonical_representation(rep)
        path = Path(name)
        if path.suffix not in {".csv", ".parquet"}:
            raise ValueError("Score inputs must be CSV or Parquet")
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        stream = io.BytesIO(payload)
        frame = pl.read_csv(stream) if path.suffix == ".csv" else pl.read_parquet(stream)
        for key, value in [("representation", rep), ("batch", batch)]:
            if key in frame.columns and (frame[key].null_count() or set(frame[key]) != {value}):
                raise ValueError(f"Input {key} does not match the declared value")
            frame = frame.with_columns(pl.lit(value).alias(key))
        frames.append(frame)
        inputs.append({"file": path.name, "sha256": digest, "representation": rep, "batch": batch})
    settings = [(canonical_representation(rep), channel) for rep, channel in args.setting]
    outputs = hit_cohorts(pl.concat(frames, how="diagonal_relaxed"), args.task, settings)
    for name, frame in outputs.items():
        frame.write_parquet(args.output_dir / f"{name}.parquet")
    outputs["denominators"].write_csv(args.output_dir / "denominators.csv")
    # Written last: identifies a completed report, not a producer/calibration receipt.
    (args.output_dir / "report.json").write_text(json.dumps({
        "task": args.task, "settings": settings, "inputs": inputs,
        "calibration": "Caller-supplied thresholds; matching backend/settings and control adequacy not verified here",
        "bh": "Existing producer corrected p-values reused; no recomputation",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
