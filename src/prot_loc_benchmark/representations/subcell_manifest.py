"""Canonical release preflight. Reads metadata/NPY headers, never crop payloads.

Extraction is a separate, explicit operation; a completed, hash-verified extraction
receipt binds the crop files to one immutable Hugging Face git revision.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from prot_loc_benchmark.config import ALL_PUBLIC_BATCHES, SUBCELL_CHANNEL_FILES
from prot_loc_benchmark.cell_crops import release_inventory as crop_inventory
from prot_loc_benchmark.provenance import save_json, sha256, read_json_with_hash

PROTOCOL = 'subcell-allele-rybg-v2'
IDENTITY = ['Metadata_CellID', 'Metadata_Plate', 'Metadata_Well',
            'Metadata_Site', 'Metadata_ImageNumber', 'Metadata_ObjectNumber']


def split_for_plate(plate):
    match = re.search(r'T([1-4])$', plate)
    if match is None:
        raise ValueError(f'Unknown physical technical-replicate plate: {plate!r}')
    return {'1': 'train', '2': 'train', '3': 'val', '4': 'test'}[match[1]]


def release_inventory(root):
    """SubCell additionally requires six fixed batches and downstream plate annotations."""
    plate_metadata = [f'representations/cellprofiler/{b}/features.parquet' for b in ALL_PUBLIC_BATCHES]
    inventory = crop_inventory(root, extra_paths=plate_metadata)
    batches = {p.split('/')[1] for p in inventory['files'] if p.startswith('single_cell_crops/')}
    if batches != set(ALL_PUBLIC_BATCHES):
        raise ValueError('Missing or extra SubCell protocol batches')
    return inventory


def release_tables(root):
    frames = []
    for batch in ALL_PUBLIC_BATCHES:
        path = Path(root) / f'manifest/manifest_Batch_{batch.rsplit("_", 1)[1]}.parquet'
        frame = pd.read_parquet(path)
        if frame[IDENTITY + ['Metadata_gene_allele']].isna().any().any():
            raise ValueError(f'Null identity/label in {path}')
        labels = frame.Metadata_gene_allele
        if not labels.map(lambda s: isinstance(s, str) and bool(s.strip()) and s == s.strip()
                                         and Path(s).name == s and s not in ('.', '..')).all():
            raise ValueError(f'Blank/invalid canonical allele in {path}')
        frame['batch_id'] = batch
        frame['cell_id'] = batch + '/' + frame.Metadata_CellID
        if frame.cell_id.duplicated().any():
            raise ValueError(f'Duplicate batch-qualified cell in {path}')
        frame['split'] = frame.Metadata_Plate.map(split_for_plate)
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    classes = sorted(frame.loc[frame.split == 'train', 'Metadata_gene_allele'].unique())
    class_index = {allele: idx for idx, allele in enumerate(classes)}
    unknown = set(frame.Metadata_gene_allele) - set(classes)
    if unknown:
        raise ValueError(f'Evaluation alleles absent from training: {sorted(unknown)}')
    sizes = frame.loc[frame.split == 'train'].groupby('Metadata_gene_allele').size()
    if (sizes < 8).any():
        raise ValueError(f'Fewer than eight training cells: {sizes[sizes < 8].to_dict()}')
    frame['class_index'] = frame.Metadata_gene_allele.map(class_index)
    return frame, class_index


def add_plate_maps(frame, root, inventory):
    # Platemaps are downstream CV metadata, NOT technical-replicate split labels.
    # Read only plate annotations from released CP tables, never CP feature tensors
    # or their filtered cell cohort. A well filtered by CP must not drop a crop.
    tables = []
    for batch in ALL_PUBLIC_BATCHES:
        relative = f'representations/cellprofiler/{batch}/features.parquet'
        path = Path(root) / relative
        if sha256(path) != inventory['files'][relative]['sha256']:
            raise ValueError(f'Released plate metadata hash mismatch: {path}')
        mapping = pd.read_parquet(path, columns=['Metadata_Plate', 'Metadata_plate_map_name']).drop_duplicates()
        if mapping.isna().any().any() or mapping.Metadata_Plate.duplicated().any():
            raise ValueError(f'Ambiguous physical plate to platemap mapping: {path}')
        mapping['batch_id'] = batch
        tables.append(mapping)
    result = frame.merge(pd.concat(tables), on=['batch_id', 'Metadata_Plate'], how='left',
                         validate='many_to_one', suffixes=('_existing', ''))
    if result.Metadata_plate_map_name.isna().any():
        raise ValueError('Missing canonical plate-map annotation; refusing to drop cells')
    if 'Metadata_plate_map_name_existing' in result:
        existing = result.pop('Metadata_plate_map_name_existing')
        if (existing.notna() & existing.ne(result.Metadata_plate_map_name)).any():
            raise ValueError('Conflicting canonical plate-map annotation')
    return result


def align_crop_rows(released, base_path):
    """The stored metadata order, NOT release-manifest order, is the NPY row index."""
    base_path = Path(base_path)
    metadata = pd.read_parquet(base_path / 'metadata.parquet')
    if metadata.empty or metadata[IDENTITY].isna().any().any() or metadata.Metadata_CellID.duplicated().any():
        raise ValueError(f'Empty/null/duplicate crop identities: {base_path}')
    if set(metadata.Metadata_CellID) != set(released.Metadata_CellID):
        raise ValueError(f'Crop/release coverage differs: {base_path}')
    ordered = released.set_index('Metadata_CellID').loc[metadata.Metadata_CellID].reset_index()
    for column in IDENTITY:
        if not np.array_equal(metadata[column].to_numpy(), ordered[column].to_numpy()):
            raise ValueError(f'Crop identity disagreement ({column}): {base_path}')
    if set(ordered.Metadata_gene_allele) != {base_path.name}:
        raise ValueError(f'Canonical allele/directory disagreement: {base_path}')
    if 'Metadata_gene_allele' in metadata and not np.array_equal(
            metadata.Metadata_gene_allele.to_numpy(), ordered.Metadata_gene_allele.to_numpy()):
        raise ValueError(f'Crop/release label disagreement: {base_path}')
    for channel in SUBCELL_CHANNEL_FILES:
        path = base_path / channel
        array = np.load(path, mmap_mode='r', allow_pickle=False)
        if array.dtype != np.uint16 or array.shape != (len(metadata), 128, 128):
            raise ValueError(f'Wrong channel header/metadata length: {path}: {array.shape}, {array.dtype}')
        if path.stat().st_size != array.offset + array.nbytes:
            raise ValueError(f'Truncated or trailing crop payload: {path}')
    ordered['cell_idx'] = np.arange(len(metadata), dtype=np.int64)
    ordered['base_path'] = str(base_path.resolve())
    return ordered


def build_manifest(root, crops, output):
    """Validate a completed extraction and save immutable cohort + fixed validation IDs."""
    from .subcell_allele_data import fixed_validation
    from prot_loc_benchmark.provenance import capture_source, invocation, code_fingerprint, verify_source

    root, crops, output = (Path(p).resolve() for p in (root, crops, output))
    if any(output.is_relative_to(p) for p in (root, crops)):
        raise ValueError('Preflight must not write inside the Hugging Face mirror or crop inputs')
    source_fingerprint = code_fingerprint()
    inventory = release_inventory(root)
    receipt_path = crops / 'extraction.json'
    if not receipt_path.exists():
        raise ValueError('No completed extraction.json; tar shards are not mmap training data. '
                         'Use 01_prepare_cell_crops.py extract into a NEW directory first.')
    receipt, receipt_digest = read_json_with_hash(receipt_path)
    recorded = receipt['release']
    required = {p for p in inventory['files'] if p.startswith(('manifest/', 'single_cell_crops/'))}
    # Accept both shared crop-only receipts and earlier receipts that also pinned
    # CP metadata. Never rewrite the old receipt or require a second extraction.
    if (recorded['revision'] != inventory['revision'] or recorded['remote'] != inventory['remote']
            or not required.issubset(recorded['files'])
            or any(inventory['files'].get(p) != info for p, info in recorded['files'].items())):
        raise ValueError('Extraction belongs to a different/mixed release')
    for relative, info in receipt['files'].items():
        path = crops / relative
        stat = path.stat()
        if stat.st_size != info['size'] or stat.st_mtime_ns != info['mtime_ns']:
            raise ValueError(f'Extracted file changed since verified extraction: {path}')
    released, classes = release_tables(root)
    released = add_plate_maps(released, root, inventory)
    frames = []
    expected_files = set()
    for (batch, allele), group in released.groupby(['batch_id', 'Metadata_gene_allele'], sort=True):
        base = crops / batch / allele
        for name in [*SUBCELL_CHANNEL_FILES, 'metadata.parquet']:
            expected_files.add(str((base / name).relative_to(crops)))
        frames.append(align_crop_rows(group, base))
    if set(receipt['files']) != expected_files:
        raise ValueError('Extraction file coverage does not equal canonical cohort')
    frame = pd.concat(frames, ignore_index=True).sort_values('cell_id').reset_index(drop=True)
    validation = fixed_validation(frame)
    # Create only after ALL checks pass; never overwrite an earlier preflight.
    output.mkdir(parents=True, exist_ok=False)
    capture_source(output)
    verify_source(output, source_fingerprint)
    frame.to_parquet(output / 'manifest.parquet', index=False)
    save_json(output / 'class_index.json', classes)
    save_json(output / 'validation_ids.json', validation)
    frame.groupby(['batch_id', 'Metadata_Plate', 'split', 'Metadata_gene_allele']).size().rename('cells').to_csv(
        output / 'counts.csv')
    if sha256(receipt_path) != receipt_digest:
        raise ValueError('Extraction receipt changed during preflight')
    save_json(output / 'preflight.json', {
        'protocol': PROTOCOL, 'release': inventory, 'release_root': str(root.resolve()),
        'crops_root': str(crops.resolve()),
        'invocation': invocation(output), 'source_archive_sha256': sha256(output / 'source.tar.gz'),
        'extraction_sha256': receipt_digest,
        'manifest_sha256': sha256(output / 'manifest.parquet'),
        'class_index_sha256': sha256(output / 'class_index.json'),
        'validation_ids_sha256': sha256(output / 'validation_ids.json'),
        'counts': frame.groupby('split').size().to_dict(), 'classes': len(classes),
        'missing_validation_classes': sorted(set(classes) - set(frame.loc[frame.split == 'val', 'Metadata_gene_allele'])),
    })


def load_preflight(path):
    path = Path(path)
    evidence = json.loads((path / 'preflight.json').read_text())
    if evidence['protocol'] != PROTOCOL:
        raise ValueError('Not an allele-v2 preflight')
    if sha256(path / 'source.tar.gz') != evidence['source_archive_sha256']:
        raise ValueError('Preflight source archive changed')
    for name in ('manifest', 'class_index', 'validation_ids'):
        filename = name + ('.parquet' if name == 'manifest' else '.json')
        if sha256(path / filename) != evidence[name + '_sha256']:
            raise ValueError(f'Preflight artifact changed: {filename}')
    crops = Path(evidence['crops_root'])
    receipt_path = crops / 'extraction.json'
    receipt, receipt_digest = read_json_with_hash(receipt_path)
    if receipt_digest != evidence['extraction_sha256']:
        raise ValueError('Extraction receipt changed after preflight')
    for name, info in receipt['files'].items():
        stat = (crops / name).stat()
        if stat.st_size != info['size'] or stat.st_mtime_ns != info['mtime_ns']:
            raise ValueError(f'Crop file changed after preflight: {name}')
    return (pd.read_parquet(path / 'manifest.parquet'),
            json.loads((path / 'class_index.json').read_text()),
            json.loads((path / 'validation_ids.json').read_text()), evidence)
