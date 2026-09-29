"""Allele-only mmap input and deterministic, rank-aware cell draws for protocol v2."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler
from torch.utils.data._utils.collate import default_collate

from prot_loc_benchmark.config import SUBCELL_CHANNEL_FILES


def stratified_draw(frame, count, rng):
    """Round-robin randomized nonempty (batch, physical plate, well) lists."""
    strata = [rng.permutation(group.index).tolist() for _, group in frame.groupby(
        ['batch_id', 'Metadata_Plate', 'Metadata_Well'], sort=True)]
    rng.shuffle(strata)
    result = []
    while strata and len(result) < count:
        for cells in strata:
            result.append(cells.pop())
            if len(result) == count:
                break
        strata = [cells for cells in strata if cells]
    return result


def fixed_validation(frame):
    selected = []
    for label, group in frame.loc[frame.split == 'val'].groupby('class_index', sort=True):
        # Canonical identity ordering makes draws independent of manifest serialization.
        group = group.sort_values('cell_id')
        selected.extend(stratified_draw(group, 32, np.random.default_rng([2026, 0, int(label)])))
    return frame.loc[selected, 'cell_id'].tolist()


class AlleleBatchSampler(Sampler):
    """One global 16-allele batch, sharded WITHOUT padding; eight cells per allele.

    The sampling pass is part of the key, not mutable worker RNG. Lightning calls
    set_epoch on this sampler. Resuming at a pass boundary needs only that epoch.
    """
    def __init__(self, frame, seed, rank=0, world_size=1):
        if world_size < 1 or 16 % world_size or not 0 <= rank < world_size:
            raise ValueError('World size must divide 16; rank must belong to that world')
        if set(frame.split) != {'train'} or frame.cell_id.duplicated().any():
            raise ValueError('Sampler requires unique T1/T2 cells only')
        self.groups = [group.sort_values('cell_id') for _, group in frame.groupby('class_index', sort=True)]
        if len(self.groups) < 16 or any(len(group) < 8 for group in self.groups):
            raise ValueError('Need >=16 alleles and >=8 distinct training cells per allele')
        self.seed, self.rank, self.world_size = seed, rank, world_size
        self.epoch = 0

    @property
    def sampler(self):
        # Lightning's epoch hook looks at dataloader.batch_sampler.sampler.
        return self

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.groups) // 16

    def __iter__(self):
        order = np.random.default_rng([self.seed, self.epoch]).permutation(len(self.groups))
        per_rank = 16 // self.world_size
        for start in range(0, len(self) * 16, 16):
            indices = []
            for idx in order[start + self.rank * per_rank:start + (self.rank + 1) * per_rank]:
                group = self.groups[idx]
                label = int(group.class_index.iloc[0])
                rng = np.random.default_rng([self.seed, self.epoch, label])
                indices.extend(stratified_draw(group, 8, rng))
            yield indices


class MisLocusSubCellDataset(Dataset):
    """Cell-indexed canonical cohort. Sampling belongs to AlleleBatchSampler.

    The DataFrame is a validated preflight manifest (or its fixed T3 subset).
    Its index is retained for distributed validation deduplication and coverage.
    """
    def __init__(self, frame: pd.DataFrame, class_index: dict, *, mask_prob=0,
                 return_cell_mask=False, object_mask_ratio=0):
        if mask_prob != 0 or return_cell_mask or object_mask_ratio != 0:
            raise ValueError('Protocol v2 forbids cell/background/object masks')
        if not isinstance(frame, pd.DataFrame):
            raise ValueError('Historical CSVs are unsupported; load a canonical v2 preflight')
        if not frame.index.is_unique or frame.cell_id.duplicated().any():
            raise ValueError('Duplicate cell indices or identities')
        if class_index != {allele: i for i, allele in enumerate(sorted(class_index))}:
            raise ValueError('Class vocabulary must be sorted and contiguous')
        expected = frame.Metadata_gene_allele.map(class_index)
        if expected.isna().any() or not np.array_equal(expected, frame.class_index):
            raise ValueError('Unknown allele or inconsistent canonical class index')
        self.rows = {int(i): (base, int(local), int(label)) for i, base, local, label in
                     frame[['base_path', 'cell_idx', 'class_index']].itertuples(index=True, name=None)}
        self.positions = frame.index.to_numpy()

    def __len__(self):
        return len(self.rows)

    @lru_cache(maxsize=32)
    def _arrays(self, base):
        arrays = [np.load(Path(base) / name, mmap_mode='r', allow_pickle=False) for name in SUBCELL_CHANNEL_FILES]
        if any(a.dtype != np.uint16 or a.ndim != 3 or a.shape[1:] != (128, 128) or
               a.shape != arrays[0].shape for a in arrays):
            raise ValueError(f'Invalid four-channel crop headers: {base}')
        return arrays

    def __getitem__(self, index):
        # Batch sampler emits canonical indices; evaluation uses a separate positional view.
        base, local, allele = self.rows[int(index)]
        arrays = self._arrays(base)
        if not 0 <= local < len(arrays[0]):
            raise ValueError(f'Out-of-bounds crop row: {base}[{local}]')
        image = torch.from_numpy(np.stack([a[local] for a in arrays]).astype(np.float32))
        return {'image': image, 'allele': allele, 'cell_index': int(index), 'mask': None}


def collate_cells(cells):
    if any(cell['mask'] is not None for cell in cells):
        raise ValueError('Cell masks are forbidden')
    return {**default_collate([{k: v for k, v in c.items() if k != 'mask'} for c in cells]), 'mask': None}
