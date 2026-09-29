"""Protocol-v2 Lightning loop; vendored architectures, projectors and losses unchanged.

The HPA localization training modules are intentionally not reused: MisLocus uses
integer canonical allele CE targets, a fixed selector, and native step scheduling.
"""
from __future__ import annotations

import json
import random
import time
import uuid
from functools import partial
from pathlib import Path

import lightning as L
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from timm.optim.optim_factory import param_groups_weight_decay
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler, Subset
from torch.utils.checkpoint import checkpoint

from prot_loc_benchmark.config import SUBCELL_SCALE_FACTOR
from prot_loc_benchmark.preprocessing.subcell import SubCellPreprocessor
from .subcell_finetune import AlleleBatchSampler, MisLocusSubCellDataset, collate_cells
from .subcell_manifest import save_json, sha256


def setup_transforms():
    import torchvision.transforms.v2 as v2
    from torchvision.transforms import InterpolationMode
    from utils.augmentations import (GaussianNoise, PerBatchCompose, PerChannelAdjustSharpness,
        PerChannelColorJitter, PerChannelGaussianBlur, PerChannelRandomErasing, RemoveChannel, RescaleProtein)

    geometry = PerBatchCompose([
        v2.RandomHorizontalFlip(.5), v2.RandomVerticalFlip(.5),
        v2.RandomChoice([
            v2.RandomAffine(90, translate=(.2, .2), scale=(.8, 1.2), interpolation=InterpolationMode.BILINEAR, fill=0),
            v2.RandomPerspective(.25, p=.5, interpolation=InterpolationMode.BILINEAR, fill=0),
        ]),
    ])
    intensity = PerBatchCompose([
        RemoveChannel(p=.25), RescaleProtein(p=.25),
        PerChannelColorJitter(brightness=.5, contrast=.5, p=1.),
        v2.RandomChoice([PerChannelGaussianBlur(7, (.1, 2.), p=.5), PerChannelAdjustSharpness(2, p=.5)]),
        GaussianNoise((.01, .05), p=.5),
        PerChannelRandomErasing(scale=(.02, .1), ratio=(.3, 3.3), p=.5),
    ])
    return geometry, intensity


def enable_checkpointing(model):
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    if model.decoder is not None:
        # HF's decoder is an nn.Module, not a PreTrainedModel with an enable method.
        model.decoder.gradient_checkpointing = True
        model.decoder._gradient_checkpointing_func = partial(checkpoint, use_reentrant=False)


def parameter_samples(model):
    return {name: torch.cat([p.detach().flatten()[:16] for p in module.parameters() if p.requires_grad]).cpu()
            for name in ('encoder', 'decoder', 'pool_model', 'ssl_model', 'supcon_model', 'online_finetuner')
            if (module := getattr(model, name, None)) is not None}


def lr_factor(update, total, warmup):
    if not 0 <= update <= total or not 0 < warmup < total:
        raise ValueError('Invalid scheduler update/horizon')
    return update / warmup if update < warmup else .001 + .999 * (total - update) / (total - warmup)


def optimizer_groups(model):
    groups = []
    for name in ('encoder', 'decoder'):
        module = getattr(model, name, None)
        if module is not None:
            for group in param_groups_weight_decay(module, .05):
                groups.append({**group, 'name': name + ('/decay' if group['weight_decay'] else '/no_decay')})
    for name in ('pool_model', 'ssl_model', 'supcon_model', 'online_finetuner'):
        module = getattr(model, name, None)
        if module is not None:
            groups.append({'params': [p for p in module.parameters() if p.requires_grad],
                           'weight_decay': .01, 'name': name})
    ids = [id(p) for group in groups for p in group['params']]
    if len(ids) != len(set(ids)) or set(ids) != {id(p) for p in model.parameters() if p.requires_grad}:
        raise ValueError('Each trainable parameter must belong to exactly one optimizer group')
    return groups


def load_pretrained_weights(model, path, expected_sha256):
    if sha256(path) != expected_sha256:
        raise ValueError('Pretrained artifact SHA256 mismatch')
    state = torch.load(path, map_location='cpu', weights_only=True)
    expected = {f'{name}.{key}' for name in ('encoder', 'pool_model')
                for key in getattr(model, name).state_dict()}
    if set(state) != expected:
        raise ValueError(f'Pretrained keys differ: missing={sorted(expected - set(state))}, '
                         f'unexpected={sorted(set(state) - expected)}')
    for name in ('encoder', 'pool_model'):
        getattr(model, name).load_state_dict({k[len(name)+1:]: v for k, v in state.items()
                                             if k.startswith(name + '.')}, strict=True)


def allele_metrics(ids, targets, probabilities, losses, expected_ids):
    """Whole-set AP, deduplicating distributed padding by canonical manifest index."""
    ids, targets, probabilities, losses = map(np.asarray, (ids, targets, probabilities, losses))
    if not np.isfinite(probabilities).all() or not np.isfinite(losses).all():
        raise ValueError('Nonfinite validation predictions/loss')
    unique, first, inverse = np.unique(ids, return_index=True, return_inverse=True)
    if set(unique) != set(expected_ids):
        raise ValueError('Validation cell coverage mismatch')
    if not np.array_equal(targets, targets[first][inverse]) or not np.allclose(
            probabilities, probabilities[first][inverse], atol=1e-5, rtol=1e-5):
        raise ValueError('Distributed duplicate predictions/labels disagree')
    targets, probabilities, losses = targets[first], probabilities[first], losses[first]
    if targets.dtype.kind not in 'iu' or probabilities.ndim != 2 or not np.allclose(probabilities.sum(1), 1, atol=1e-5):
        raise ValueError('Expected integer CE targets and softmax probabilities')
    if np.any(targets < 0) or np.any(targets >= probabilities.shape[1]):
        raise ValueError('Targets outside training vocabulary')
    per_class = []
    for label in range(probabilities.shape[1]):
        truth = targets == label
        per_class.append(float(average_precision_score(truth, probabilities[:, label]))
                         if truth.any() and not truth.all() else None)
    supported = [ap for ap in per_class if ap is not None]
    if not supported:
        raise ValueError('No supported validation classes')
    top = np.argsort(probabilities, axis=1)[:, -min(5, probabilities.shape[1]):]
    return {'macro_ap': float(np.mean(supported)), 'top1': float((probabilities.argmax(1) == targets).mean()),
            'top5': float((top == targets[:, None]).any(1).mean()), 'probe_loss': float(losses.mean()),
            'per_allele_ap': per_class, 'omitted': [i for i, ap in enumerate(per_class) if ap is None]}


class AlleleDataModule(L.LightningDataModule):
    def __init__(self, frame, classes, validation_ids, sampling_seed, workers=2):
        super().__init__()
        self.frame, self.classes, self.validation_ids = frame, classes, validation_ids
        self.sampling_seed, self.workers = sampling_seed, workers

    def setup(self, stage=None):
        train = self.frame.loc[self.frame.split == 'train']
        val = self.frame.set_index('cell_id', drop=False).loc[self.validation_ids]
        if len(set(self.validation_ids)) != len(self.validation_ids) or set(val.split) != {'val'}:
            raise ValueError('Validation must be a unique, saved T3-only list')
        # Keep canonical integer manifest indices, in the saved validation order.
        val.index = self.frame.set_index('cell_id').loc[self.validation_ids].index.map(
            dict(zip(self.frame.cell_id, self.frame.index)))
        self.train_data = MisLocusSubCellDataset(train, self.classes)
        self.val_data = MisLocusSubCellDataset(val, self.classes)
        self.train_sampler = AlleleBatchSampler(train, self.sampling_seed, self.global_rank, self.world_size)

    @property
    def global_rank(self):
        return self.trainer.global_rank

    @property
    def world_size(self):
        return self.trainer.world_size

    def _loader_args(self):
        # A private generator prevents worker creation from consuming model/augmentation RNG.
        return dict(num_workers=self.workers, pin_memory=True, collate_fn=collate_cells,
                    generator=torch.Generator().manual_seed(self.sampling_seed),
                    **({'prefetch_factor': 1, 'persistent_workers': True} if self.workers else {}))

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_sampler=self.train_sampler, **self._loader_args())

    def val_dataloader(self):
        dataset = Subset(self.val_data, self.val_data.positions)
        sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=self.global_rank,
                                     shuffle=False, drop_last=False)
        return DataLoader(dataset, batch_size=16, sampler=sampler, **self._loader_args())


class SubCellAlleleModule(L.LightningModule):
    def __init__(self, components, categories, identity, output_dir, *, augment=True):
        super().__init__()
        self.encoder = components.get('encoder', components.get('vit_model'))
        self.decoder = components.get('decoder')
        self.pool_model = components['pool_model']
        self.ssl_model = components.get('ssl_model')
        self.supcon_model = components['supcon_model']
        dim = self.pool_model.out_dim
        self.online_finetuner = nn.Sequential(nn.Dropout(.5), nn.Linear(dim, dim // 2), nn.ReLU(),
                                             nn.Dropout(.5), nn.Linear(dim // 2, len(categories)))
        self.categories, self.identity, self.output_dir = categories, identity, Path(output_dir)
        self.preprocess = SubCellPreprocessor(scale_factor=SUBCELL_SCALE_FACTOR)
        self.geometry, self.intensity = setup_transforms() if augment else (nn.Identity(), nn.Identity())
        self.validation_outputs = []
        self.constant_cells = set()
        self.pending_rng = None
        self.epoch_cells, self.loss_sums = [], {}
        self.pass_started = None
        self.validation_seconds = 0.

    def forward(self, images, mask_ratio=0.):
        kwargs = {'mask_ratio': mask_ratio, 'object_mask': None} if self.decoder is not None else {}
        encoded = self.encoder(images, output_attentions=False, **kwargs)
        pooled, _ = self.pool_model(encoded.last_hidden_state)
        return encoded, pooled, self.online_finetuner(pooled.detach())

    def on_after_batch_transfer(self, batch, dataloader_idx):
        if batch['mask'] is not None:
            raise ValueError('Cell masks are forbidden')
        images = self.preprocess(batch['image'])
        constant = images.amax((1, 2, 3)) == images.amin((1, 2, 3))
        self.constant_cells.update(batch['cell_index'][constant].cpu().tolist())
        if self.training:
            first, second = self.geometry(images.clone()), self.geometry(images.clone())
            if self.decoder is None:
                first = self.intensity(first)
            second = self.intensity(second)
        else:
            first, second = images, None
        return {**batch, 'image': first, 'view2': second}

    def training_step(self, batch, batch_idx):
        images, second, labels = batch['image'], batch['view2'], batch['allele']
        if labels.dtype != torch.long or labels.ndim != 1:
            raise ValueError('Allele CE requires integer class targets')
        encoded, z1, logits1 = self(images, .25 if self.decoder is not None else 0.)
        _, z2, logits2 = self(second, 0.)
        gathered1 = self.all_gather(z1, sync_grads=True).reshape(-1, z1.shape[-1])
        gathered2 = self.all_gather(z2, sync_grads=True).reshape(-1, z2.shape[-1])
        gathered_labels = self.all_gather(labels).reshape(-1)
        global_ids = self.all_gather(batch['cell_index']).reshape(-1)
        groups, counts = torch.unique(gathered_labels, return_counts=True)
        if (gathered1.shape[0] != 128 or gathered2.shape[0] != 128 or len(global_ids.unique()) != 128
                or len(groups) != 16 or not torch.all(counts == 8)):
            raise ValueError('Expected exactly 16 distinct alleles × 8 distinct cells globally')
        self.epoch_cells.extend(batch['cell_index'].cpu().tolist())
        allele_loss = self.supcon_model(gathered1, gathered2, gathered_labels)
        probe_loss = (F.cross_entropy(logits1, labels) + F.cross_entropy(logits2, labels)) / 2
        loss = allele_loss + probe_loss
        losses = {'allele_supcon': allele_loss, 'probe_ce': probe_loss}
        if self.decoder is not None:
            cell_loss = self.ssl_model(gathered1, gathered2)
            prediction = self.decoder(encoded.last_hidden_state, encoded.ids_restore).logits
            p = self.encoder.config.patch_size
            b, c, h, w = images.shape
            target = images.reshape(b, c, h // p, p, w // p, p)
            target = torch.einsum('nchpwq->nhwpqc', target).reshape(b, -1, p * p * c)
            target = (target - target.mean(-1, keepdim=True)) / (target.var(-1, keepdim=True) + 1e-6).sqrt()
            reconstruction = ((prediction - target).square().mean(-1) * encoded.mask).sum() / encoded.mask.sum()
            loss = reconstruction + cell_loss + .1 * allele_loss + probe_loss
            losses.update(reconstruction=reconstruction, cell_contrastive=cell_loss)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite training loss')
        losses['total'] = loss
        for name, value in losses.items():
            self.log('train/' + name, value, sync_dist=True)
            self.loss_sums[name] = self.loss_sums.get(name, 0.) + float(value.detach())
        return loss

    def configure_optimizers(self):
        steps = len(self.trainer.datamodule.train_sampler)
        total = int(self.trainer.estimated_stepping_batches)
        if total != 100 * steps or self.trainer.accumulate_grad_batches != 1:
            raise ValueError(f'Post-sharding schedule mismatch: {total} != 100 * {steps}')
        self.steps_per_pass = steps
        groups = optimizer_groups(self)
        optimizer = torch.optim.AdamW(groups, lr=1e-4 * 128 / 256, betas=(.9, .95), eps=1e-8)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda u: lr_factor(u, total, 5 * steps))
        if self.trainer.is_global_zero:
            save_json(self.output_dir / 'optimizer.json', {
                'steps_per_pass': steps, 'total_updates': total, 'warmup_updates': 5 * steps,
                'groups': [{'name': g['name'], 'weight_decay': g['weight_decay'],
                            'parameters': sum(p.numel() for p in g['params'])} for g in groups]})
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}

    def on_before_optimizer_step(self, optimizer):
        norm = torch.nn.utils.clip_grad_norm_(self.parameters(), float('inf'), error_if_nonfinite=True)
        self.max_pre_clip = max(self.max_pre_clip, float(norm))
        self.log('train/grad_norm_before_clip', norm)
        self.log('train/lr_used', optimizer.param_groups[0]['lr'])

    def configure_gradient_clipping(self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None):
        self.clip_gradients(optimizer, gradient_clip_val=gradient_clip_val,
                            gradient_clip_algorithm=gradient_clip_algorithm)
        norm = torch.nn.utils.clip_grad_norm_(self.parameters(), float('inf'), error_if_nonfinite=True)
        if norm > 1.00001:
            raise FloatingPointError('Global gradient norm exceeds the protocol clip')
        self.max_post_clip = max(self.max_post_clip, float(norm))
        self.log('train/grad_norm_after_clip', norm)

    def on_fit_start(self):
        from .subcell_run import invocation, runtime_info
        attempt = self.trainer.strategy.broadcast(uuid.uuid4().hex if self.global_rank == 0 else None)
        self.attempt_dir = self.output_dir / 'attempts' / attempt
        if self.trainer.is_global_zero:
            self.attempt_dir.mkdir(parents=True, exist_ok=False)
            save_json(self.attempt_dir / 'start.json', {
                **invocation(), 'identity': self.identity, 'runtime': runtime_info(),
                'resume_checkpoint': str(self.trainer.ckpt_path) if self.trainer.ckpt_path else None,
                'resume_sha256': sha256(self.trainer.ckpt_path) if self.trainer.ckpt_path else None,
                'initialization': {'pretrained': ['encoder', 'pool_model'],
                                   'new': ['online_finetuner', 'supcon_model'] +
                                          (['decoder', 'ssl_model'] if self.decoder is not None else [])}})
        self.trainer.strategy.barrier()

    def on_train_epoch_start(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
            if self.trainer.world_size > 1 and not any(isinstance(m, nn.SyncBatchNorm) for m in self.modules()):
                raise RuntimeError('Distributed projectors require SyncBatchNorm during fitting')
        self.pass_started = time.perf_counter()
        self.validation_seconds = 0.
        self.epoch_cells, self.loss_sums = [], {}
        self.max_pre_clip = self.max_post_clip = 0.
        self.pass_start_step = self.global_step
        self.start_samples = parameter_samples(self)

    def on_train_epoch_end(self):
        # A validation-end checkpoint resumes by closing the saved epoch first;
        # no new cells were processed and on_train_epoch_start was not called.
        if self.pass_started is None:
            return
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        wall = time.perf_counter() - self.pass_started
        samples = parameter_samples(self)
        updates = self.global_step - self.pass_start_step
        local = {'rank': self.global_rank, 'pass': self.current_epoch + 1, 'updates': updates,
                 'completed_updates': self.global_step, 'cell_indices': self.epoch_cells,
                 'constant_cell_indices': sorted(self.constant_cells),
                 'loss_means': {k: v / updates for k, v in self.loss_sums.items()},
                 'sampled_parameter_max_abs_drift': {k: float((v - self.start_samples[k]).abs().max()) for k, v in samples.items()},
                 'gradient_norm_max_before_clip': self.max_pre_clip, 'gradient_norm_max_after_clip': self.max_post_clip,
                 'wall_seconds_including_validation': wall, 'validation_seconds': self.validation_seconds,
                 'training_seconds': wall - self.validation_seconds,
                 'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(self.device) if self.device.type == 'cuda' else 0,
                 'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved(self.device) if self.device.type == 'cuda' else 0}
        ranks = [None] * self.trainer.world_size
        if dist.is_initialized():
            dist.all_gather_object(ranks, local)
        else:
            ranks = [local]
        if self.trainer.is_global_zero:
            save_json(self.attempt_dir / f'pass-{self.current_epoch + 1:03}.json', {
                'ranks': ranks, 'global_cell_presentations': sum(len(r['cell_indices']) for r in ranks),
                'global_cells_per_training_second': sum(len(r['cell_indices']) for r in ranks) /
                                                   max(r['training_seconds'] for r in ranks)})
        self.pass_started = None

    def on_fit_end(self):
        if self.trainer.is_global_zero:
            from prot_loc_benchmark.provenance import record
            selection = self.output_dir / 'selection.json'
            summary = {'identity': self.identity, 'global_step': self.global_step,
                       'selection': json.loads(selection.read_text()) if selection.exists() else None,
                       'status': 'fit_completed', 'attempt': str(self.attempt_dir.relative_to(self.output_dir))}
            inputs = [self.output_dir / 'run.json', self.output_dir / 'source.json']
            if self.identity.get('config'):
                inputs += [Path(self.identity['config']['preflight']) / 'preflight.json',
                           Path(self.identity['config']['pretrained_weights'])]
            record([self.attempt_dir], input_paths=[p for p in inputs if p.exists()])
            save_json(self.attempt_dir / 'completed.json', summary)

    def on_validation_epoch_start(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        self.validation_started = time.perf_counter()

    def validation_step(self, batch, batch_idx):
        _, _, logits = self(batch['image'], 0.)
        labels = batch['allele']
        self.validation_outputs.append((batch['cell_index'].cpu(), labels.cpu(),
                                        logits.softmax(-1).cpu(), F.cross_entropy(logits, labels, reduction='none').cpu()))

    def on_validation_epoch_end(self):
        local = tuple(torch.cat([row[i] for row in self.validation_outputs]).numpy() for i in range(4))
        self.validation_outputs.clear()
        gathered = [None] * self.trainer.world_size
        if dist.is_initialized():
            dist.all_gather_object(gathered, local)
        else:
            gathered = [local]
        columns = [np.concatenate([rank[i] for rank in gathered]) for i in range(4)]
        metrics = allele_metrics(*columns, self.trainer.datamodule.val_data.positions)
        if self.trainer.sanity_checking:
            return
        for name in ('macro_ap', 'top1', 'top5', 'probe_loss'):
            # Models/probabilities remain FP32; preserve the AP statistic in FP64
            # so checkpoint comparisons and serialized JSON use identical values.
            self.log('val/' + name, torch.tensor(metrics[name], device=self.device, dtype=torch.float64),
                     sync_dist=False)
        if self.trainer.is_global_zero:
            metrics['categories'] = self.categories
            save_json(self.attempt_dir / f'validation-pass-{self.current_epoch + 1:03}.json', metrics)
        if hasattr(self, 'validation_started'):
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            self.validation_seconds += time.perf_counter() - self.validation_started

    def on_save_checkpoint(self, checkpoint):
        # Production checkpoints are pass-boundary only: no unverified mid-pass replay.
        local = {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
                 'cuda': torch.cuda.get_rng_state() if self.device.type == 'cuda' else None,
                 'constant_cells': sorted(self.constant_cells)}
        states = [None] * self.trainer.world_size
        if dist.is_initialized():
            dist.all_gather_object(states, local)
        else:
            states = [local]
        checkpoint['allele_v2'] = {'identity': self.identity, 'rng_states': states,
                                   'steps_per_pass': self.steps_per_pass,
                                   'next_pass': self.current_epoch + 1}

    def on_load_checkpoint(self, checkpoint):
        saved = checkpoint.get('allele_v2', {})
        if saved.get('identity') != self.identity:
            raise ValueError('Incompatible checkpoint: protocol/config/data/seed/environment differs')
        if checkpoint['global_step'] != saved['next_pass'] * saved['steps_per_pass']:
            raise ValueError('Only completed-pass resumes are supported')
        from .subcell_run import require_resumable
        require_resumable(self.output_dir, checkpoint)
        self.pending_rng = saved['rng_states']

    def on_train_start(self):
        if self.pending_rng is not None:
            state = self.pending_rng[self.global_rank]
            random.setstate(state['python'])
            np.random.set_state(state['numpy'])
            torch.set_rng_state(state['torch'].cpu())
            if state['cuda'] is not None:
                torch.cuda.set_rng_state(state['cuda'].cpu())
            self.constant_cells = set(state['constant_cells'])
            self.pending_rng = None
