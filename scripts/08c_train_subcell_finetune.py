#!/usr/bin/env python3
"""Canonical allele-v2 fine-tuning. Explicit --fit; no automatic historical resume.

Run preflight first (08a). Multi-GPU runs require torchrun --nproc_per_node=N;
N must equal config.devices and divide sixteen. No T4 tensors are read here.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'vendor/subcell_embed')]

import lightning as L
import torch
import yaml
from lightning.pytorch.callbacks import Callback, EarlyStopping
from models.get_models import get_model_dict

from prot_loc_benchmark.representations.subcell_manifest import load_preflight, save_json, sha256
from prot_loc_benchmark.representations.subcell_protocol import model_config, resolved_protocol, validate_config
from prot_loc_benchmark.representations.subcell_training import (
    AlleleDataModule, SubCellAlleleModule, load_pretrained_weights, enable_checkpointing,
)
from prot_loc_benchmark.representations.subcell_run import (
    AlleleCheckpoint, capture_source, code_fingerprint, require_resumable, runtime_info, verify_source,
)


class StopAfterUpdates(Callback):
    def __init__(self, updates):
        self.updates = updates

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        if trainer.global_step >= self.updates:
            trainer.should_stop = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-c', '--config', type=Path, required=True)
    parser.add_argument('--resume', type=Path,
                        help='Compatible last.ckpt from an interrupted run at a pass boundary; finished runs are rejected')
    parser.add_argument('--fit', action='store_true', help='Start training AFTER review and readiness gates')
    parser.add_argument('--smoke-updates', type=int, help='Bounded diagnostic only; no selectable checkpoints')
    args = parser.parse_args()
    if args.smoke_updates is not None and (args.smoke_updates < 1 or args.resume):
        parser.error('Smoke checks require a positive update count and cannot resume')
    config = yaml.safe_load(args.config.read_text())
    validate_config(config)
    for key in ('preflight', 'pretrained_weights', 'output'):
        config[key] = str((ROOT / config[key]).resolve())
    frame, classes, validation, evidence = load_preflight(config['preflight'])
    if Path(config['output']).is_relative_to(Path(evidence['release_root'])):
        raise ValueError('Training output must stay outside the Hugging Face mirror')
    if sha256(config['pretrained_weights']) != config['pretrained_sha256']:
        raise ValueError('Pretrained SHA256 mismatch')
    world = int(os.environ.get('WORLD_SIZE', '1'))
    if world != config['devices']:
        parser.error('Launch with torchrun --nproc_per_node=config.devices (do not infer global batch from visible GPUs)')
    identity = {'config': config, 'resolved': resolved_protocol(config['family']), 'data': evidence,
                'code_sha256': code_fingerprint(),
                'git_head': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
                'world_size': world, 'kind': 'smoke' if args.smoke_updates else 'production',
                'smoke_updates': args.smoke_updates}
    print(json.dumps(identity, indent=2), flush=True)
    if not args.fit:
        print('Validated configuration/data bindings only. No training; use --fit after readiness review.')
        return
    if not torch.cuda.is_available():
        raise RuntimeError('Production runs require the pinned GPU stack')
    torch.set_float32_matmul_precision('high')
    L.seed_everything(config['seed'], workers=True)
    identity['runtime'] = runtime_info()
    output = Path(config['output'])
    if args.resume:
        # Trusted local Lightning checkpoint; never unpickle downloaded arbitrary checkpoints.
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        if checkpoint.get('allele_v2', {}).get('identity') != identity:
            raise ValueError('Resume protocol/config/data/seed/code/environment binding mismatch')
        if json.loads((output / 'run.json').read_text()) != identity:
            raise ValueError('Run directory identity mismatch')
        verify_source(output, identity['code_sha256'])
        # Only the latest completed checkpoint may resume into this output directory.
        if args.resume.resolve() != (output / 'models/last.ckpt').resolve():
            raise ValueError('Resume must explicitly name this run\'s last.ckpt; do not overwrite later results')
        require_resumable(output, checkpoint)
        del checkpoint
    elif int(os.environ.get('RANK', '0')) == 0:
        output.mkdir(parents=True, exist_ok=False)
        save_json(output / 'run.json', identity)
        save_json(output / 'class_index.json', classes)
        save_json(output / 'validation_ids.json', validation)
        capture_source(output)
        verify_source(output, identity['code_sha256'])
    model = SubCellAlleleModule(get_model_dict(model_config(config['family'])),
                               sorted(classes), identity, output)
    if args.resume is None:
        load_pretrained_weights(model, config['pretrained_weights'], config['pretrained_sha256'])
    enable_checkpointing(model)
    data = AlleleDataModule(frame, classes, validation[:128] if args.smoke_updates else validation,
                           config['sampling_seed'], config['workers'])
    callbacks = [StopAfterUpdates(args.smoke_updates)] if args.smoke_updates else [
        AlleleCheckpoint(output),
        EarlyStopping(monitor='val/macro_ap', mode='max', patience=5, min_delta=.001,
                      check_on_train_epoch_end=False),
    ]
    trainer = L.Trainer(default_root_dir=output, accelerator='gpu', devices=world,
                        strategy='ddp' if world > 1 else 'auto', max_epochs=100,
                        accumulate_grad_batches=1, precision='32-true', sync_batchnorm=True,
                        use_distributed_sampler=False, check_val_every_n_epoch=1 if args.smoke_updates else 10,
                        enable_checkpointing=not bool(args.smoke_updates),
                        gradient_clip_val=1., gradient_clip_algorithm='norm', num_sanity_val_steps=0,
                        callbacks=callbacks, log_every_n_steps=1)
    trainer.fit(model, datamodule=data, ckpt_path=args.resume)


if __name__ == '__main__':
    main()
