"""Protocol-v2 training helpers; vendored architectures, projectors and losses unchanged.

The HPA localization training modules are intentionally not reused: MisLocus uses
integer canonical allele CE targets, a fixed selector, and native step scheduling.
"""
from __future__ import annotations

from functools import partial

import lightning as L
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from timm.optim.optim_factory import param_groups_weight_decay
from torch.utils.data import DataLoader, DistributedSampler, Subset
from torch.utils.checkpoint import checkpoint

from .subcell_allele_data import AlleleBatchSampler, MisLocusSubCellDataset, collate_cells
from .subcell_manifest import sha256


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
        val.index = val.index.map(dict(zip(self.frame.cell_id, self.frame.index)))
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
