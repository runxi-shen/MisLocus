"""Run: PYTHONPATH=src:vendor/subcell_embed python -m unittest discover -s tests."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from models.ntxent import get_contrastive_loss
from prot_loc_benchmark.config import ALL_PUBLIC_BATCHES, SUBCELL_CHANNEL_FILES
from prot_loc_benchmark.representations.subcell_allele_data import (
    AlleleBatchSampler, MisLocusSubCellDataset, collate_cells, fixed_validation, stratified_draw,
)
from prot_loc_benchmark.representations.subcell_manifest import add_plate_maps, align_crop_rows, sha256, split_for_plate


def make_cohort(root, count=17):
    alleles = sorted(['ALK', 'ALK_Thr1151Met'] + [f'GENE{i}' for i in range(count - 2)])
    classes = {a: i for i, a in enumerate(alleles)}
    rows = []
    for allele, label in classes.items():
        base = root / allele
        base.mkdir()
        metadata = []
        for idx in range(12):
            t = (idx % 2 + 1) if idx < 9 else 3
            plate = f'2024_01_17_B7A1R1_P1T{t}'
            cell = f'{plate}_A01_{label}_{idx}'
            metadata.append({'Metadata_CellID': cell, 'Metadata_Plate': plate, 'Metadata_Well': 'A01',
                             'Metadata_Site': 1, 'Metadata_ImageNumber': label, 'Metadata_ObjectNumber': idx})
            rows.append({**metadata[-1], 'Metadata_gene_allele': allele, 'Metadata_symbol': allele.split('_')[0],
                         'cell_id': 'B7/' + cell, 'batch_id': 'B7', 'split': split_for_plate(plate),
                         'class_index': label, 'cell_idx': idx, 'base_path': str(base)})
        pd.DataFrame(metadata).to_parquet(base / 'metadata.parquet', index=False)
        for c, channel in enumerate(SUBCELL_CHANNEL_FILES):
            image = np.zeros((12, 128, 128), dtype=np.uint16)
            image[:] = np.arange(12, dtype=np.uint16)[:, None, None] + c * 100
            np.save(base / channel, image)
    return pd.DataFrame(rows), classes


class AlleleRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.frame, cls.classes = make_cohort(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_same_gene_alleles_are_not_merged(self):
        frame = self.frame.loc[self.frame.Metadata_symbol == 'ALK']
        dataset = MisLocusSubCellDataset(frame, self.classes)
        reference = dataset[int(frame.loc[frame.Metadata_gene_allele == 'ALK'].index[0])]
        variant = dataset[int(frame.loc[frame.Metadata_gene_allele == 'ALK_Thr1151Met'].index[0])]
        self.assertNotEqual(reference['allele'], variant['allele'])
        batch = collate_cells([reference, variant])
        self.assertEqual(batch['allele'].dtype, torch.long)
        self.assertEqual(batch['allele'].shape, (2,))
        self.assertIsNone(batch['mask'])
        features = torch.nn.functional.normalize(torch.randn(2, 2, 8), dim=-1)
        self.assertNotEqual(get_contrastive_loss(features, .1, batch['allele']).item(),
                            get_contrastive_loss(features, .1, torch.zeros(2)).item())

    def test_manifest_row_alignment_and_channel_sentinels(self):
        released = self.frame.loc[self.frame.Metadata_gene_allele == 'ALK'].iloc[::-1]
        aligned = align_crop_rows(released, self.root / 'ALK')
        self.assertEqual(aligned.cell_idx.tolist(), list(range(12)))
        dataset = MisLocusSubCellDataset(aligned, self.classes)
        self.assertEqual(dataset[7]['image'][:, 0, 0].tolist(), [7, 107, 207, 307])
        with self.assertRaisesRegex(ValueError, 'coverage'):
            align_crop_rows(released.iloc[:-1], self.root / 'ALK')
        wrong = released.copy()
        wrong['Metadata_gene_allele'] = 'ALK_Thr1151Met'
        with self.assertRaisesRegex(ValueError, 'allele'):
            align_crop_rows(wrong, self.root / 'ALK')

    def test_plate_map_annotations_are_reconciled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory, rows = {'files': {}}, []
            for batch in ALL_PUBLIC_BATCHES:
                plate = f'{batch}_P1T1'
                relative = f'representations/cellprofiler/{batch}/features.parquet'
                path = root / relative
                path.parent.mkdir(parents=True)
                pd.DataFrame({'Metadata_Plate': [plate], 'Metadata_plate_map_name': ['P1']}).to_parquet(path)
                inventory['files'][relative] = {'sha256': sha256(path)}
                rows.extend({'batch_id': batch, 'Metadata_Plate': plate, 'cell_id': f'{batch}/{i}'}
                            for i in range(2))
            frame = pd.DataFrame(rows)
            expected = frame.assign(Metadata_plate_map_name='P1')
            for values in (None, ['P1'] * len(frame), [pd.NA] + ['P1'] * (len(frame) - 1)):
                with self.subTest(values=values):
                    supplied = frame.copy() if values is None else frame.assign(Metadata_plate_map_name=values)
                    original = supplied.copy(deep=True)
                    result = add_plate_maps(supplied, root, inventory)
                    pd.testing.assert_frame_equal(result, expected)
                    pd.testing.assert_frame_equal(supplied, original)
            conflicting = expected.copy()
            conflicting.loc[0, 'Metadata_plate_map_name'] = 'P2'
            with self.assertRaisesRegex(ValueError, 'Conflicting.*plate-map'):
                add_plate_maps(conflicting, root, inventory)
            missing = frame.copy()
            missing.loc[0, 'Metadata_Plate'] = 'unknown'
            with self.assertRaisesRegex(ValueError, 'Missing canonical plate-map'):
                add_plate_maps(missing, root, inventory)
            # Existing annotations must not bypass the released-table checksum.
            inventory['files'][relative]['sha256'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                add_plate_maps(expected, root, inventory)

    def test_strict_splits_labels_masks(self):
        for t, expected in [(1, 'train'), (2, 'train'), (3, 'val'), (4, 'test')]:
            self.assertEqual(split_for_plate(f'2024_01_17_B7A1R1_P1T{t}'), expected)
        for plate in ('B7A1R1_P1', 'P4', 'P1T5', 'P1T1_extra'):
            with self.assertRaises(ValueError):
                split_for_plate(plate)
        for kwargs in ({'mask_prob': .5}, {'return_cell_mask': True}, {'object_mask_ratio': .1}):
            with self.assertRaisesRegex(ValueError, 'masks'):
                MisLocusSubCellDataset(self.frame, self.classes, **kwargs)
        with self.assertRaisesRegex(ValueError, 'Unknown allele'):
            MisLocusSubCellDataset(self.frame, {'ALK': 0})
        with self.assertRaisesRegex(ValueError, 'Historical'):
            MisLocusSubCellDataset('old_gene_manifest.csv', self.classes)

    def test_sampling_sharding_workers_and_fixed_validation(self):
        train = self.frame.loc[self.frame.split == 'train']
        single = AlleleBatchSampler(train, 42)
        ranks = [AlleleBatchSampler(train, 42, rank, 2) for rank in range(2)]
        for epoch in (0, 1, 5):
            single.set_epoch(epoch)
            for rank in ranks:
                rank.set_epoch(epoch)
            expected = list(single)
            actual = [a + b for a, b in zip(*map(list, ranks))]
            self.assertEqual(expected, actual)
            self.assertEqual(len(expected), len(self.classes) // 16)
            self.assertEqual(len(set(expected[0])), 128)
            self.assertTrue((train.loc[expected[0]].groupby('class_index').size() == 8).all())
        with self.assertRaises(ValueError):
            AlleleBatchSampler(train, 42, world_size=3)
        with self.assertRaises(ValueError):
            AlleleBatchSampler(train.groupby('class_index').head(7), 42)
        dataset = MisLocusSubCellDataset(train, self.classes)
        for workers in (0, 2):
            batch = next(iter(DataLoader(dataset, batch_sampler=single, collate_fn=collate_cells, num_workers=workers)))
            self.assertEqual(batch['cell_index'].tolist(), list(single)[0])
        val_data = MisLocusSubCellDataset(self.frame.loc[self.frame.split == 'val'], self.classes)
        view = Subset(val_data, val_data.positions)
        self.assertEqual([view[i]['cell_index'] for i in range(len(view))], val_data.positions.tolist())
        validation = fixed_validation(self.frame)
        self.assertEqual(validation, fixed_validation(self.frame.sample(frac=1, random_state=9)))
        self.assertEqual(set(validation), set(self.frame.loc[self.frame.split == 'val', 'cell_id']))
        group = train.loc[train.class_index == 0]
        draw = stratified_draw(group, 8, np.random.default_rng(8))
        self.assertEqual(len(set(draw)), 8)
        self.assertEqual(train.loc[draw].groupby('Metadata_Plate').size().tolist(), [4, 4])


if __name__ == '__main__':
    unittest.main()
