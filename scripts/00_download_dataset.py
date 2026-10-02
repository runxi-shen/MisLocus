#!/usr/bin/env python3
"""Download the dataset bundle from Hugging Face.

Downloads ``anonymous-xyz96/MisLocus`` into ``data/`` at a pinned commit
and remaps it to the pipeline layout. Select another dataset with --hf-repo
or PROT_LOC_BENCHMARK_HF_REPO. Published cleaned features need no preprocessing.
Use --no-include-crops to download features without crop archives:

  representations/{rep}/{batch}/features.parquet
      → data/interim/{rep}/{batch}/features.parquet (morphem → vit)
  manifest/manifest_Batch_X.parquet
      → data/interim/crop_manifest/{full_batch}/manifest.parquet
  single_cell_crops/{batch}/shard-NN.tar.gz
      → extracted into data/interim/single_cell_crops/{batch}/

HF repo metadata (LICENSE, README.md, MisLocus_croissant.json,
``.gitattributes``) is removed from the download directory after remapping.

Uses ``huggingface_hub.snapshot_download`` with an immutable revision.
Identical feature re-imports are safe; differing destination bytes are rejected.
Use a fresh data directory when changing repositories or revisions. Remapped
files may be fetched again; --force re-fetches but does not bypass collisions.

SubCell encoder weights are downloaded by a separate script
(``scripts/00b_download_subcell_weights.py``) — only needed if you'll run
``08a`` / ``08c`` (SubCell extract / fine-tune).

Usage:
    .pixi/envs/default/bin/python scripts/00_download_dataset.py --no-include-crops

    # Just the small browseable sample subset (~1.2 GB, 8 alleles × 2
    # batches of single-cell crops + a sample manifest). Doesn't touch
    # the rest of the bundle. Files land at data/sample/.
    .pixi/envs/default/bin/python scripts/00_download_dataset.py --sample

    # One representation and batch (complete feature files, not a row sample).
    # Crops are excluded unless --include-crops is supplied.
    .pixi/envs/default/bin/python scripts/00_download_dataset.py \\
        --rep morphem --batch 2024_02_06_Batch_8

    # Multiple reps / batches (comma-separated or repeated flags).
    .pixi/envs/default/bin/python scripts/00_download_dataset.py --rep cytoself,cellprofiler

    # Another dataset with the same layout; replace its name and commit ID.
    .pixi/envs/default/bin/python scripts/00_download_dataset.py \\
        --hf-repo myorg/my-dataset --revision "<40-character-commit-sha>" --no-include-crops
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import sys
import tarfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from prot_loc_benchmark.config import (
    BIOREP_PAIRS,
    DATA_DIR,
    INTERIM_DIR,
    PUBLIC_TO_SOURCE_REP,
    REP_FEATURE_FILES,
    canonical_representation,
)
from prot_loc_benchmark.provenance import sha256

logger = logging.getLogger(__name__)

HF_REPO = "anonymous-xyz96/MisLocus"
DEFAULT_HF_REVISION = "74f63113a76b4a832285da308f3df2932266e456"
DEFAULT_HF_REPO = os.environ.get("PROT_LOC_BENCHMARK_HF_REPO", HF_REPO)


def _resolve_revision(hf_repo: str, revision: str | None) -> str:
    if revision is None:
        if hf_repo != HF_REPO:
            raise ValueError("An alternate HF repository requires --revision with its own commit SHA")
        revision = DEFAULT_HF_REVISION
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("--revision must be a full 40-character lowercase hexadecimal commit SHA")
    return revision

# Build {short_batch_num: full_batch_id}, e.g. "13" -> "2025_01_27_Batch_13".
# Used to remap HF's manifest_Batch_X.parquet to per-batch crop_manifest dirs.
_BATCH_NUM_TO_FULL_ID: dict[str, str] = {}
for _b1, _b2 in BIOREP_PAIRS.values():
    for _b in (_b1, _b2):
        _m = re.match(r".+_Batch_(\d+)$", _b)
        if _m:
            _BATCH_NUM_TO_FULL_ID[_m.group(1)] = _b


def _build_allow_patterns(
    reps: list[str] | None,
    batches: list[str] | None,
    include_crops: bool,
) -> list[str] | None:
    """Translate --rep / --batch filters into HF allow_patterns.

    Returns None when no filter is active (download everything). Otherwise
    returns a gitignore-style pattern list matching the HF repo layout.
    """
    if not reps and not batches and include_crops:
        return None  # full mirror

    patterns: list[str] = []
    if reps:
        source_to_public = {source: public for public, source in PUBLIC_TO_SOURCE_REP.items()}
        reps = list(dict.fromkeys(canonical_representation(rep) for rep in reps))
        for rep in reps:
            if rep not in REP_FEATURE_FILES:
                raise ValueError(f"Unknown representation: {rep}")
        reps = [source_to_public.get(rep, rep) for rep in reps]

    # Per-rep features.
    if reps and batches:
        for r in reps:
            for b in batches:
                patterns.append(f"representations/{r}/{b}/**")
    elif reps:
        for r in reps:
            patterns.append(f"representations/{r}/**")
    elif batches:
        for b in batches:
            patterns.append(f"representations/*/{b}/**")
    else:
        patterns.append("representations/**")

    # Per-batch crop manifest (small).
    if batches:
        for b in batches:
            m = re.match(r".+_Batch_(\d+)$", b)
            if m:
                patterns.append(f"manifest/manifest_Batch_{m.group(1)}.parquet")
    else:
        patterns.append("manifest/**")

    # Single-cell crop tarballs (huge). Off by default in subset mode.
    if include_crops:
        if batches:
            for b in batches:
                patterns.append(f"single_cell_crops/{b}/**")
        else:
            patterns.append("single_cell_crops/**")

    # Always include small metadata + repo-level files.
    patterns += ["metadata/**", "*.md", "LICENSE", "*.json", ".gitattributes"]
    return patterns


def _remap_to_pipeline_layout() -> None:
    """Move HF-layout files under DATA_DIR into the pipeline-expected layout.

    See module docstring for the full mapping table.

    Idempotent: silently no-ops on directories that don't exist (e.g. when
    crops were skipped by an allow_patterns filter).
    """
    # representations/ → interim/
    src_root = DATA_DIR / "representations"
    if src_root.is_dir():
        moves = []
        destinations = {}
        # Preflight alias collisions before moving any feature files.
        for src in sorted(src_root.rglob("*")):
            if not src.is_file():
                continue
            relative = src.relative_to(src_root)
            rep = canonical_representation(relative.parts[0])
            if rep not in REP_FEATURE_FILES:
                raise ValueError(f"Unknown representation: {relative.parts[0]}")
            dst = INTERIM_DIR / rep / Path(*relative.parts[1:])
            for existing in (dst, destinations.get(dst)):
                if existing is not None and existing.exists():
                    if not existing.is_file() or sha256(src) != sha256(existing):
                        raise FileExistsError(f"Conflicting feature files: {src} and {existing}")
            destinations[dst] = src
            moves.append((src, dst))
        for src, dst in moves:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                src.unlink()  # Identical bytes already admitted; keep the existing file.
            else:
                shutil.move(str(src), str(dst))
            logger.info("  [features] %s → %s", src.relative_to(DATA_DIR), dst.relative_to(DATA_DIR))
        _rmdir_if_empty_recursive(src_root)

    # manifest/manifest_Batch_X.parquet → interim/crop_manifest/{full}/manifest.parquet
    manifest_dir = DATA_DIR / "manifest"
    if manifest_dir.is_dir():
        for src in sorted(manifest_dir.glob("manifest_Batch_*.parquet")):
            m = re.match(r"manifest_Batch_(\d+)\.parquet", src.name)
            if not m:
                logger.warning("  [manifest] unrecognized name, leaving in place: %s", src)
                continue
            num = m.group(1)
            full = _BATCH_NUM_TO_FULL_ID.get(num)
            if full is None:
                logger.warning(
                    "  [manifest] no BIOREP_PAIRS entry for Batch_%s — leaving at %s", num, src
                )
                continue
            dst = INTERIM_DIR / "crop_manifest" / full / "manifest.parquet"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            logger.info("  [manifest] %s → %s", src.relative_to(DATA_DIR), dst.relative_to(DATA_DIR))
        _rmdir_if_empty_recursive(manifest_dir)

    # single_cell_crops/{batch}/shard-*.tar.gz → extract into interim/single_cell_crops/{batch}/
    crops_root = DATA_DIR / "single_cell_crops"
    if crops_root.is_dir():
        for src in sorted(crops_root.rglob("shard-*.tar.gz")):
            batch = src.parent.name
            dst_dir = INTERIM_DIR / "single_cell_crops" / batch
            dst_dir.mkdir(parents=True, exist_ok=True)
            logger.info("  [crops] extracting %s → %s", src.relative_to(DATA_DIR), dst_dir.relative_to(DATA_DIR))
            with tarfile.open(src, "r:gz") as tar:
                tar.extractall(dst_dir)
            src.unlink()
        _rmdir_if_empty_recursive(crops_root)

    _cleanup_hf_root_noise()


def _rmdir_if_empty_recursive(root: Path) -> None:
    """Remove ``root`` and any empty subdirs left behind after files were moved out."""
    if not root.exists():
        return
    for p in sorted(root.rglob("*"), reverse=True):
        if p.is_dir() and not any(p.iterdir()):
            p.rmdir()
    if root.is_dir() and not any(root.iterdir()):
        root.rmdir()


def _cleanup_hf_root_noise() -> None:
    """Remove HF repository metadata from the download directory."""
    for noise in ("LICENSE", "README.md", "MisLocus_croissant.json", ".gitattributes"):
        p = DATA_DIR / noise
        if p.is_file():
            p.unlink()
            logger.info("  [cleanup] removed HF artifact %s", p.relative_to(DATA_DIR))


def download_sample(hf_repo: str, force: bool = False, *, revision: str | None = None) -> None:
    """Download the small browseable sample subset (~1.2 GB) into ``data/sample/``.

    The HF repo ships a ``sample/`` directory with one tarball per
    (batch, allele) — a curated handful of alleles for two batches —
    plus ``manifest_sample.parquet``. Each tarball is extracted in place
    so the result is browseable as
    ``data/sample/{batch}/{allele}/{dna,gfp,agp,mito}.npy +
    metadata.parquet``. Independent of the main bundle layout — does not
    populate ``data/interim/`` or interfere with downstream pipeline
    consumers.
    """
    from huggingface_hub import snapshot_download

    revision = _resolve_revision(hf_repo, revision)
    logger.info("Downloading sample subset from %s at %s into %s ...", hf_repo, revision, DATA_DIR)
    snapshot_download(
        repo_id=hf_repo,
        repo_type="dataset",
        revision=revision,
        local_dir=str(DATA_DIR),
        force_download=force,
        allow_patterns=["sample/**", "*.md", "LICENSE", "*.json", ".gitattributes"],
    )

    sample_dir = DATA_DIR / "sample"
    if not sample_dir.is_dir():
        logger.warning("No sample/ directory after download — repo may not ship one.")
        _cleanup_hf_root_noise()
        return

    archives = sorted(sample_dir.rglob("*.tar.gz"))
    logger.info("Extracting %d sample tarballs in place ...", len(archives))
    for src in archives:
        dst_dir = src.parent  # data/sample/{batch}/
        logger.info("  [sample] extracting %s", src.relative_to(DATA_DIR))
        with tarfile.open(src, "r:gz") as tar:
            tar.extractall(dst_dir)
        src.unlink()

    _cleanup_hf_root_noise()


def download_dataset_bundle(
    hf_repo: str,
    *,
    revision: str | None = None,
    reps: list[str] | None = None,
    batches: list[str] | None = None,
    include_crops: bool = True,
    force: bool = False,
) -> None:
    """Mirror the HF dataset repo into ``data/`` and remap to pipeline layout.

    See module docstring for details.
    """
    from huggingface_hub import snapshot_download

    revision = _resolve_revision(hf_repo, revision)
    allow_patterns = _build_allow_patterns(reps, batches, include_crops)

    is_subset = allow_patterns is not None
    logger.info(
        "Snapshotting HF repo %s at %s into %s (mode=%s, crops=%s)",
        hf_repo, revision, DATA_DIR,
        "subset" if is_subset else "full",
        "yes" if include_crops else "no",
    )
    if allow_patterns:
        for p in allow_patterns:
            logger.info("  allow_pattern: %s", p)

    snapshot_download(
        repo_id=hf_repo,
        repo_type="dataset",
        revision=revision,
        local_dir=str(DATA_DIR),
        force_download=force,
        allow_patterns=allow_patterns,
    )

    logger.info("Remapping HF layout → pipeline layout under %s ...", DATA_DIR)
    _remap_to_pipeline_layout()


def _split_csv(values: list[str] | None) -> list[str] | None:
    """Accept argparse ``--flag a,b --flag c`` and flatten to ``[a, b, c]``."""
    if not values:
        return None
    out: list[str] = []
    for v in values:
        out.extend(p.strip() for p in v.split(",") if p.strip())
    return out or None


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--hf-repo",
        default=DEFAULT_HF_REPO,
        help=(
            "Hugging Face dataset repo id (default: env var "
            "PROT_LOC_BENCHMARK_HF_REPO or the repo baked into this script)."
        ),
    )
    parser.add_argument(
        "--revision",
        help=f"Full commit SHA (default for {HF_REPO}: {DEFAULT_HF_REVISION}); required for alternate repos.",
    )
    parser.add_argument(
        "--rep", "--representation", action="append", default=None,
        help=(
            "Restrict download to one or more representations "
            "(e.g. morphem, cytoself, cellprofiler). 'vit' also downloads "
            "public 'morphem'; both are admitted under interim/vit/. "
            "Comma-separated or repeat the flag. Default: all reps."
        ),
    )
    parser.add_argument(
        "--batch", action="append", default=None,
        help=(
            "Restrict download to one or more batch IDs "
            "(e.g. 2025_01_27_Batch_13). Comma-separated or repeat the flag. "
            "Default: all batches."
        ),
    )
    parser.add_argument(
        "--include-crops",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Include the single-cell crop tarball shards (~tens of GB per "
            "batch). Default: included in full-download mode, excluded in "
            "subset mode (--rep / --batch). Use --include-crops in subset "
            "mode to opt in, or --no-include-crops in full mode to opt out."
        ),
    )
    parser.add_argument(
        "--sample", action="store_true",
        help=(
            "Download only the small browseable sample subset (~1.2 GB) "
            "from sample/ in the HF repo, extracted under data/sample/. "
            "Mutually exclusive with --rep / --batch / --include-crops."
        ),
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download even if destination already exists.",
    )
    args = parser.parse_args()
    try:
        revision = _resolve_revision(args.hf_repo, args.revision)
    except ValueError as error:
        parser.error(str(error))

    if args.sample and (args.rep or args.batch or args.include_crops is not None):
        parser.error("--sample is mutually exclusive with --rep / --batch / --include-crops")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    reps = _split_csv(args.rep)
    batches = _split_csv(args.batch)
    is_subset = bool(reps or batches)

    # Resolve --include-crops tri-state default.
    if args.include_crops is None:
        include_crops = not is_subset
    else:
        include_crops = args.include_crops

    if args.sample:
        download_sample(args.hf_repo, force=args.force, revision=revision)
    else:
        download_dataset_bundle(
            args.hf_repo,
            revision=revision,
            reps=reps,
            batches=batches,
            include_crops=include_crops,
            force=args.force,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
