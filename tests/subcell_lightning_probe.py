"""Synthetic real-Lightning check (no released images; never a scientific run).

python tests/subcell_lightning_probe.py --prepare /tmp/subcell-probe
python tests/subcell_lightning_probe.py --root /tmp/subcell-probe --family mae
LD_LIBRARY_PATH=/run/opengl-driver/lib torchrun --standalone --nproc_per_node=2 \
    tests/subcell_lightning_probe.py --root /tmp/subcell-probe --family vit --gpu

Uses the production data module, losses, LambdaLR, metrics and checkpoint hooks.
Small test-only architecture/geometry; tests stochastic resumed trajectories.
"""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from lightning.pytorch.callbacks import Callback, EarlyStopping
from lightning.pytorch.strategies import DDPStrategy

from test_subcell_allele_v2 import make_cohort, tiny_components
from prot_loc_benchmark.preprocessing.subcell import SubCellPreprocessor
from prot_loc_benchmark.representations.subcell_allele_data import fixed_validation
from prot_loc_benchmark.representations.subcell_manifest import save_json
from prot_loc_benchmark.representations.subcell_training import (
    AlleleDataModule, SubCellAlleleModule, lr_factor, enable_checkpointing,
)
from prot_loc_benchmark.representations.subcell_run import AlleleCheckpoint, verify_selection


class StopAndTrace(Callback):
    def __init__(self, stop):
        self.stop, self.trace = stop, []

    def on_train_batch_start(self, trainer, module, batch, batch_idx):
        if module.device.type == 'cuda' and trainer.world_size > 1:
            assert any(isinstance(m, torch.nn.SyncBatchNorm) for m in module.modules())
        sampler = trainer.datamodule.train_sampler
        assert sampler.epoch == trainer.current_epoch
        lr = trainer.optimizers[0].param_groups[0]['lr']
        assert np.isclose(lr, 5e-5 * lr_factor(trainer.global_step, 100 * len(sampler), 5 * len(sampler)), atol=1e-15)
        ids = module.all_gather(batch['cell_index']).reshape(-1).cpu().tolist()
        labels = module.all_gather(batch['allele']).reshape(-1).cpu().tolist()
        assert len(ids) == len(set(ids)) == 128
        assert len(set(labels)) == 16 and all(labels.count(label) == 8 for label in set(labels))
        self.trace.append({'step': trainer.global_step, 'lr': lr, 'ids': ids, 'labels': labels})

    def on_train_epoch_end(self, trainer, module):
        if trainer.current_epoch + 1 == self.stop:
            trainer.should_stop = True


def fit(root, frame, classes, family, gpu, stop, name, resume=None):
    L.seed_everything(42, workers=True)
    world = int(os.environ.get('WORLD_SIZE', '1'))
    out = root / name
    out.mkdir(exist_ok=True)
    module = SubCellAlleleModule(tiny_components(family), sorted(classes), {'test': family, 'world': world}, out)
    module.preprocess = SubCellPreprocessor(scale_factor=1, input_size=128, output_size=32)
    enable_checkpointing(module)
    trace = StopAndTrace(stop)
    checkpoint = AlleleCheckpoint(out)
    stopping = EarlyStopping(monitor='val/macro_ap', mode='max', patience=5, min_delta=.001,
                             check_on_train_epoch_end=False)
    data = AlleleDataModule(frame, classes, fixed_validation(frame), 42, workers=2)
    trainer = L.Trainer(accelerator='gpu' if gpu else 'cpu', devices=world,
                        strategy=DDPStrategy(process_group_backend='nccl' if gpu else 'gloo') if world > 1 else 'auto',
                        max_epochs=100, check_val_every_n_epoch=10, use_distributed_sampler=False,
                        sync_batchnorm=gpu, gradient_clip_val=1., precision='32-true',
                        num_sanity_val_steps=0, logger=False, enable_progress_bar=False, enable_model_summary=False,
                        callbacks=[trace, stopping, checkpoint])
    trainer.fit(module, datamodule=data, ckpt_path=resume)
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}, trace.trace


def fresh_output(root):
    """Claim a new result directory collectively, before any fit can overwrite it."""
    if int(os.environ.get('WORLD_SIZE', '1')) == 1:
        root.mkdir(exist_ok=False)
        return
    # Gloo avoids choosing a CUDA device before Lightning owns device placement.
    dist.init_process_group('gloo', timeout=timedelta(seconds=60))
    errors = [None]
    try:
        if dist.get_rank() == 0:
            try:
                root.mkdir(exist_ok=False)
            except OSError as error:
                errors[0] = error
        dist.broadcast_object_list(errors, src=0)
    finally:
        dist.destroy_process_group()
    if errors[0] is not None:
        raise errors[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', type=Path)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--family', choices=['mae', 'vit'], default='vit')
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('high')
    if args.prepare:
        args.prepare.mkdir(parents=True, exist_ok=False)
        frame, classes = make_cohort(args.prepare, count=33)
        frame.to_parquet(args.prepare / 'cohort.parquet', index=False)
        save_json(args.prepare / 'classes.json', classes)
        return
    frame = pd.read_parquet(args.root / 'cohort.parquet')
    classes = json.loads((args.root / 'classes.json').read_text())
    root = args.root / f'{args.family}-{"gpu" if args.gpu else "cpu"}-{os.environ.get("WORLD_SIZE", "1")}'
    fresh_output(root)
    import prot_loc_benchmark.provenance as provenance
    provenance.PROVENANCE_LOG = root / 'provenance_log.json'
    full, trace_full = fit(root, frame, classes, args.family, args.gpu, 20, 'full')
    _, trace_first = fit(root, frame, classes, args.family, args.gpu, 10, 'first')
    resumed, trace_last = fit(root, frame, classes, args.family, args.gpu, 20, 'first', root / 'first/models/last.ckpt')
    assert trace_full == trace_first + trace_last, 'Resume changed LR/data schedule'
    errors = {key: (full[key].float() - resumed[key].float()).abs().max().item() for key in full}
    assert max(errors.values()) < 1e-6, f'Resume changed model/RNG trajectory: {sorted(errors.items(), key=lambda x: -x[1])[:5]}'
    if int(os.environ.get('RANK', '0')) == 0:
        best = root / 'first/models/best_model_ap.ckpt'
        saved = torch.load(best, map_location='cpu', weights_only=False)
        selected = verify_selection(best, saved, root / 'first/selection.json')
        assert selected['metric_dtype'] == 'float64'
        assert all(state['best_model_score'].dtype == torch.float64 for state in saved['callbacks'].values()
                   if 'best_model_score' in state)
        diagnostics = sorted((root / 'first/attempts').glob('*/pass-*.json'))
        assert len(diagnostics) == 20
        for path in diagnostics:
            report = json.loads(path.read_text())
            assert report['global_cell_presentations'] == 256
            assert all(r['gradient_norm_max_after_clip'] <= 1.00001 for r in report['ranks'])
            assert all('encoder' in r['sampled_parameter_max_abs_drift'] for r in report['ranks'])
        save_json(root / 'passed.json', {'world_size': int(os.environ.get('WORLD_SIZE', '1')),
                                        'family': args.family, 'gpu': args.gpu, 'trace': trace_full,
                                        'max_resume_parameter_error': max(errors.values())})
        print('PASS: real Lightning LR, rank/cell alignment, global AP, SyncBN (GPU), checkpoint and RNG/data resume')
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
