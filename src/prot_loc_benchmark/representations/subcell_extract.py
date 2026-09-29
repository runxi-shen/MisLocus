#!/usr/bin/env python3
"""Exact-coverage FP32 extraction from explicit v2-selected OR matching frozen weights.

Writes canonical metadata directly, never an inner join to historical CP features.
Output must be new. Both modes use the same six-batch preflight and image pipeline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import polars as pl
import torch
import yaml
from models.get_models import get_model_dict

from prot_loc_benchmark import provenance
from prot_loc_benchmark.config import ALL_PUBLIC_BATCHES, SUBCELL_EMBED_DIM, BATCH_CONTROLS, CPC_GENE_ALLELES
from prot_loc_benchmark.preprocessing.annotate import annotate_controls
from prot_loc_benchmark.representations.subcell_io import SUBCELL_REQUIRED_META_COLS
from prot_loc_benchmark.preprocessing.subcell import SubCellPreprocessor
from prot_loc_benchmark.representations.subcell_finetune import MisLocusSubCellDataset
from prot_loc_benchmark.representations.subcell_manifest import load_preflight, save_json, sha256
from prot_loc_benchmark.representations.subcell_protocol import model_config, PROTOCOL
from prot_loc_benchmark.representations.subcell_training import load_pretrained_weights
from prot_loc_benchmark.representations.subcell_run import capture_source, invocation, runtime_info, verify_selection


class InferenceWrapper:
    def __init__(self, family, state, device):
        config = model_config(family)
        config = {k: v for k, v in config.items() if k in ('mae_model', 'vit_model', 'pool_model', 'pl_args')}
        components = get_model_dict(config)
        self.encoder = components.get('encoder', components.get('vit_model')).eval().to(device)
        self.pool_model = components['pool_model'].eval().to(device)
        self.family = family
        if state is not None:
            for name in ('encoder', 'pool_model'):
                getattr(self, name).load_state_dict({k[len(name)+1:]: v for k, v in state.items()
                                                    if k.startswith(name + '.')}, strict=True)

    @torch.inference_mode()
    def extract(self, images):
        kwargs = {'mask_ratio': 0., 'object_mask': None} if self.family == 'mae' else {}
        encoded = self.encoder(images, output_attentions=False, **kwargs)
        pooled, _ = self.pool_model(encoded.last_hidden_state)
        result = pooled.cpu().numpy()
        if result.dtype != np.float32 or result.shape != (len(images), SUBCELL_EMBED_DIM) or not np.isfinite(result).all():
            raise ValueError('Invalid 1536-D FP32 embedding')
        return result


def main(*, frozen=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', required=True, type=Path)
    parser.set_defaults(checkpoint=None, frozen_weights=None, weights_sha256=None, selection=None)
    if frozen:
        parser.add_argument('--frozen-weights', type=Path, required=True, help='Matched four-channel HPA encoder/pool')
        parser.add_argument('--weights-sha256', required=True)
    else:
        parser.add_argument('--checkpoint', type=Path, required=True, help='Explicit trusted local best_model_ap.ckpt')
        parser.add_argument('--selection', type=Path, help='Archived selection.json if the original run was relocated')
    parser.add_argument('--family', choices=['mae', 'vit'], required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--split', choices=['train', 'val', 'test', 'all'], default='test')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--provenance-log', type=Path, help='External shared ledger; never inside an export/input')
    parser.add_argument('--crop-verification', type=Path, help='Saved successful 01 verify JSON, bound to this cohort')
    parser.add_argument('--pilot-cells-per-split', type=int,
                        help='Diagnostic only: first N cells per batch/split; writes pilot.json, NOT extraction.json')
    args = parser.parse_args()
    if args.batch_size < 1 or (args.pilot_cells_per_split is not None and args.pilot_cells_per_split < 1):
        parser.error('Batch size and pilot cell limit must be positive')
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError(f'Export output must be new: {args.output}')
    started = time.perf_counter()
    frame, classes, _, evidence = load_preflight(args.preflight)
    protected = [Path(evidence[k]).resolve() for k in ('release_root', 'crops_root')] + [args.preflight.resolve()]
    if args.checkpoint:
        protected.append(args.checkpoint.resolve().parent)
    if args.selection:
        protected.append(args.selection.resolve().parent)
    binding = {'protocol': PROTOCOL, 'family': args.family, 'data': evidence, 'precision': 'float32',
               'artifact_kind': 'diagnostic_embeddings' if args.pilot_cells_per_split else 'raw_embeddings',
               'pilot_cells_per_split': args.pilot_cells_per_split, 'batch_size': args.batch_size,
               'split': args.split, 'preprocessing': 'AGP,Mito,DNA,GFP;128->955->[253:701];joint-minmax-1e-6'}
    if args.crop_verification:
        verified = json.loads(args.crop_verification.read_text())
        if (verified['extraction_sha256'] != evidence['extraction_sha256']
                or Path(verified['crops_root']).resolve() != Path(evidence['crops_root']).resolve()
                or verified['revision'] != evidence['release']['revision'] or verified['files_verified'] <= 0):
            raise ValueError('Crop verification does not match the cohort')
        binding['crop_verification_sha256'] = sha256(args.crop_verification)
    state = None
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
        identity = checkpoint.get('allele_v2', {}).get('identity', {})
        if (identity.get('data') != evidence or identity.get('config', {}).get('family') != args.family
                or identity.get('kind') != 'production'):
            raise ValueError('Checkpoint is not a matching production allele-v2 run')
        selection = verify_selection(args.checkpoint, checkpoint, args.selection)
        protected.append(Path(identity['config']['output']).resolve())
        state = checkpoint['state_dict']
        binding.update(checkpoint_sha256=selection['sha256'], training=identity, selection=selection,
                       selected_pass=selection['pass'])
    elif not args.weights_sha256:
        parser.error('--weights-sha256 is required for frozen extraction')
    if args.frozen_weights:
        recipe = provenance.REPO_ROOT / f'configs/subcell_finetune_{args.family}_s42.yaml'
        if args.weights_sha256 != yaml.safe_load(recipe.read_text())['pretrained_sha256']:
            raise ValueError('Frozen weights must match the selected family\'s pinned pretrained identity')
    if any(args.output.is_relative_to(p) for p in protected):
        raise ValueError('Embedding output is inside a protected input directory')
    if args.provenance_log:
        ledger = args.provenance_log.resolve()
        if any(ledger.is_relative_to(p) for p in [*protected, args.output]):
            raise ValueError('Provenance log is inside a protected input/export directory')
        provenance.PROVENANCE_LOG = ledger
    binding['provenance_log'] = str(provenance.PROVENANCE_LOG)
    torch.set_float32_matmul_precision('high')
    device = torch.device(args.device)
    wrapper = InferenceWrapper(args.family, state, device)
    if args.frozen_weights:
        load_pretrained_weights(SimpleNamespace(encoder=wrapper.encoder, pool_model=wrapper.pool_model),
                                args.frozen_weights, args.weights_sha256)
        binding['frozen_sha256'] = args.weights_sha256
    del state
    if args.checkpoint:
        del checkpoint
    if args.split != 'all':
        frame = frame.loc[frame.split == args.split]
    if args.pilot_cells_per_split:
        frame = frame.groupby(['batch_id', 'split'], sort=False).head(args.pilot_cells_per_split)
    inputs = [args.preflight / name for name in ('preflight.json', 'manifest.parquet', 'class_index.json',
                                                'validation_ids.json')] + [args.checkpoint or args.frozen_weights]
    if args.checkpoint:
        inputs.append(args.selection or Path(identity['config']['output']) / 'selection.json')
    if args.crop_verification:
        inputs.append(args.crop_verification)
    binding['input_sha256'] = {str(p.resolve()): sha256(p) for p in inputs}
    args.output.mkdir(parents=True, exist_ok=False)
    capture_source(args.output)
    binding.update(invocation=invocation(args.output), runtime=runtime_info(),
                   source_archive_sha256=sha256(args.output / 'source.tar.gz'))
    preprocess = SubCellPreprocessor()
    outputs, constant_cells = {}, []
    with torch.inference_mode():
        for batch in ALL_PUBLIC_BATCHES:
            cells = frame.loc[frame.batch_id == batch]
            if cells.empty:
                raise ValueError(f'No eligible cells for required batch {batch}')
            dataset = MisLocusSubCellDataset(cells, classes)
            features = np.empty((len(cells), SUBCELL_EMBED_DIM), dtype=np.float32)
            indices = cells.index.to_numpy()
            for start in range(0, len(cells), args.batch_size):
                chunk = indices[start:start + args.batch_size]
                images = torch.stack([dataset[int(i)]['image'] for i in chunk]).to(device)
                prepared = preprocess(images)
                constant = (prepared.amax((1, 2, 3)) == prepared.amin((1, 2, 3))).cpu().numpy()
                constant_cells.extend(cells.loc[chunk[constant], 'cell_id'].tolist())
                features[start:start + len(chunk)] = wrapper.extract(prepared)
                done = start + len(chunk)
                if start // 4096 != done // 4096 or done == len(cells):
                    print(f'{batch}: {done}/{len(cells)} cells inferred', flush=True)
            # Already joined and coverage-checked at preflight. Do not join/filter again.
            metadata = cells[[c for c in cells if c.startswith('Metadata_')] + ['cell_id', 'batch_id']].reset_index(drop=True)
            metadata['Metadata_Split'] = cells.split.to_numpy()
            metadata['Metadata_well_position'] = metadata.Metadata_Well
            metadata = metadata.rename(columns={'cell_id': 'Metadata_BatchQualifiedCellID', 'batch_id': 'Metadata_Batch'})
            controls = BATCH_CONTROLS[batch]
            metadata = annotate_controls(pl.from_pandas(metadata).lazy(), tc=controls['TC'], nc=controls['NC'],
                                         pc=controls['PC'], cpc_gene_alleles=CPC_GENE_ALLELES).collect().to_pandas()
            if set(SUBCELL_REQUIRED_META_COLS) - set(metadata):
                raise ValueError('Missing required downstream metadata')
            embedding = pd.DataFrame(features, columns=[f'SubCell_{i}' for i in range(SUBCELL_EMBED_DIM)])
            result = pd.concat([metadata, embedding], axis=1)
            if (len(result) != len(cells) or result.Metadata_BatchQualifiedCellID.duplicated().any()
                    or set(result.Metadata_BatchQualifiedCellID) != set(cells.cell_id)):
                raise ValueError('Export cell coverage mismatch')
            destination = args.output / batch / 'embeddings.parquet'
            destination.parent.mkdir()
            result.to_parquet(destination, index=False)
            outputs[batch] = {'cells': len(result), 'sha256': sha256(destination),
                              'split_counts': cells.groupby('split').size().to_dict(),
                              'ordered_cell_ids_sha256': hashlib.sha256(
                                  ''.join(cell + '\n' for cell in cells.cell_id).encode()).hexdigest()}
            print(f'{batch}: {len(result)} embeddings', flush=True)
    binding.update(outputs=outputs, constant_cell_ids=constant_cells, duration_seconds=time.perf_counter() - started,
                   feature_columns=[f'SubCell_{i}' for i in range(SUBCELL_EMBED_DIM)], status='complete')
    if (binding['input_sha256'] != {str(p.resolve()): sha256(p) for p in inputs}
            or binding['invocation']['code_sha256'] != provenance.code_fingerprint()):
        raise ValueError('Extraction inputs/source changed during the run')
    if device.type == 'cuda':
        binding['gpu'] = {'name': torch.cuda.get_device_name(device),
                          'capability': torch.cuda.get_device_capability(device),
                          'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                          'peak_allocated_bytes': torch.cuda.max_memory_allocated(device)}
    provenance.record([args.output], input_paths=inputs, duration_seconds=binding['duration_seconds'])
    # Only this last atomic publication makes an export eligible for consumers.
    save_json(args.output / ('pilot.json' if args.pilot_cells_per_split else 'extraction.json'), binding)
