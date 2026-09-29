"""Full-size pretrained parity/repeatability check; inference only, no crop cohort.

PYTHONPATH=src:vendor/subcell_embed:vendor/subcellportable python \
  tests/subcell_frozen_parity.py --weights-root /path/to/weights/rybg \
  --device cuda:0 --output /tmp/frozen-parity.json
"""
import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from transformers import ViTConfig
from vit_model import ViTInferenceModel, GatedAttentionPooler

from prot_loc_benchmark.config import REPO_ROOT, SUBCELL_MODEL_TYPES
from prot_loc_benchmark.preprocessing.subcell import SubCellPreprocessor
from prot_loc_benchmark.representations.subcell_extract import InferenceWrapper
from prot_loc_benchmark.representations.subcell_manifest import load_preflight, save_json
from prot_loc_benchmark.representations.subcell_finetune import MisLocusSubCellDataset
from prot_loc_benchmark.representations.subcell_protocol import model_config
from prot_loc_benchmark.representations.subcell_run import code_fingerprint, runtime_info
from prot_loc_benchmark.representations.subcell_training import load_pretrained_weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-root', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preflight', type=Path, help='Optional real T1/T2/T3 channel/row/parity check; never T4')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('high')
    input_evidence = {'kind': 'synthetic uint16', 'seed': 2026}
    if args.preflight:
        frame, classes, _, evidence = load_preflight(args.preflight)
        eligible = frame.loc[frame.split.isin(['train', 'val'])].copy()
        eligible['replicate'] = eligible.Metadata_Plate.str[-2:]
        selected = eligible.groupby(['batch_id', 'replicate'], sort=True).head(2)
        assert set(selected.replicate) == {'T1', 'T2', 'T3'}
        dataset = MisLocusSubCellDataset(selected, classes)
        images = []
        for index, row in selected.iterrows():
            image = dataset[index]['image']
            for channel, name in enumerate(('agp.npy', 'mito.npy', 'dna.npy', 'gfp.npy')):
                raw = np.load(Path(row.base_path) / name, mmap_mode='r', allow_pickle=False)
                np.testing.assert_array_equal(image[channel].numpy(), raw[int(row.cell_idx)].astype(np.float32))
            images.append(image)
        image = torch.stack(images).to(args.device)
        input_evidence = {'kind': 'real canonical T1/T2/T3 only', 'cell_ids': selected.cell_id.tolist(),
                          'manifest_sha256': evidence['manifest_sha256'], 'release': evidence['release']['revision']}
    else:
        image = torch.randint(0, 65536, (2, 4, 128, 128), generator=torch.Generator().manual_seed(2026)).float().to(args.device)
    prepared = SubCellPreprocessor()(image)
    reports = []
    for family in ('mae', 'vit'):
        config = yaml.safe_load((REPO_ROOT / f'configs/subcell_finetune_{family}.yaml').read_text())
        wrapper = InferenceWrapper(family, None, torch.device(args.device))
        load_pretrained_weights(SimpleNamespace(encoder=wrapper.encoder, pool_model=wrapper.pool_model),
                                args.weights_root / SUBCELL_MODEL_TYPES[family] / 'encoder.pth', config['pretrained_sha256'])
        backbone = model_config(family)['mae_model' if family == 'mae' else 'vit_model']['args']
        portable = ViTInferenceModel(ViTConfig(**backbone), add_pooling_layer=False).eval().to(args.device)
        portable.load_state_dict(wrapper.encoder.state_dict(), strict=True)
        pool = GatedAttentionPooler(768, 512, 2).eval().to(args.device)
        pool.load_state_dict({k.replace('attention_v.1.', 'attention_v.0.').replace('attention_u.1.', 'attention_u.0.'): v
                             for k, v in wrapper.pool_model.state_dict().items()}, strict=True)
        with torch.inference_mode():
            actual = np.concatenate([wrapper.extract(chunk) for chunk in prepared.split(2)])
            repeated = np.concatenate([wrapper.extract(chunk) for chunk in prepared.split(2)])
            expected = np.concatenate([pool(portable(chunk, output_attentions=False).last_hidden_state)[0].cpu().numpy()
                                       for chunk in prepared.split(2)])
        np.testing.assert_array_equal(actual, repeated)
        np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)
        reports.append({'family': family, 'shape': list(actual.shape), 'dtype': str(actual.dtype),
                        'max_absolute_portable_difference': float(np.max(np.abs(actual - expected))),
                        'repeat_difference': float(np.max(np.abs(actual - repeated))),
                        'pretrained_sha256': config['pretrained_sha256']})
        del wrapper, portable, pool
    save_json(args.output, {'checks': reports, 'runtime': runtime_info(), 'code_sha256': code_fingerprint(),
                           'input': input_evidence, 'atol': 1e-4, 'rtol': 1e-4})
    print('PASS: both full-size frozen encoders/pools match portable outputs and repeat exactly', reports)


if __name__ == '__main__':
    main()
