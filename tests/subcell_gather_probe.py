"""Check actual v2 DDP/gather gradients against a full-global-batch reference.

PYTHONPATH=src:vendor/subcell_embed torchrun --standalone --nproc_per_node=4 \
    tests/subcell_gather_probe.py --output /tmp/gather.json
Synthetic tensors only; backward checks, no optimizer steps. Dropout is disabled
and MAE mask positions are fixed only in this numerical transport test.
"""
import argparse
import copy
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from lightning.pytorch.strategies import DDPStrategy
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from test_subcell_allele_v2 import tiny_components
from prot_loc_benchmark.representations.subcell_training import SubCellAlleleModule
from prot_loc_benchmark.representations.subcell_manifest import save_json


class Loss(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, batch):
        return self.model.training_step(batch, 0)


def fixed_sampling(sequence, noise, batch_size, seq_length, len_keep):
    ids = torch.arange(seq_length, device=sequence.device).expand(batch_size, -1)
    return ids, ids[:, :len_keep]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    torch.set_num_threads(1)
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    torch.set_float32_matmul_precision('highest')  # isolate gradient transport from TF32 rounding
    torch.backends.cudnn.allow_tf32 = False
    dist.init_process_group('nccl')
    torch.manual_seed(42)
    image = torch.randn(128, 4, 32, 32, generator=torch.Generator().manual_seed(7)).cuda()
    batch = {'image': image, 'view2': image * .9, 'allele': torch.arange(16).repeat_interleave(8).cuda(),
             'cell_index': torch.arange(128).cuda()}
    reports = []
    for family in ('mae', 'vit'):
        reference = SubCellAlleleModule(tiny_components(family), [str(i) for i in range(16)], {}, args.output.parent, augment=False).cuda()
        reference.log = lambda *a, **k: None
        for module in reference.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0
        if family == 'mae':
            reference.encoder.embeddings.random_sampling = fixed_sampling
        model = copy.deepcopy(reference)
        reference.all_gather = lambda tensor, **kwargs: tensor
        reference.training_step(batch, 0).backward()
        expected = {name: p.grad.clone() for name, p in reference.named_parameters() if p.requires_grad}
        del reference
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model._trainer = SimpleNamespace(strategy=DDPStrategy())
        distributed = DistributedDataParallel(Loss(model), device_ids=[int(os.environ['LOCAL_RANK'])])
        size = 128 // world
        local = {name: tensor[rank * size:(rank + 1) * size] for name, tensor in batch.items()}
        distributed(local).backward()
        errors = {}
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                torch.testing.assert_close(parameter.grad, expected[name], rtol=2e-4, atol=2e-5,
                                           msg=lambda message: f'{family}/{name}: {message}')
                errors[name] = (parameter.grad - expected[name]).abs().max().item()
        reports.append({'family': family, 'world_size': world, 'max_absolute_gradient_error': max(errors.values())})
        del distributed, model
    if rank == 0:
        save_json(args.output, reports)
        print('PASS: v2 MAE and ViT DDP gradients match full-global-batch reference', reports)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
