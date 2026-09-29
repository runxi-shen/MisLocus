"""Real full-size release gate, NOT production: two passes, full saved T3, reload.

Launch with torchrun --standalone --nproc_per_node=2 and --config PATH.
Only diagnostic validation cadence/stop differ; the production 100-pass optimizer
horizon, data, models, losses, augmentation and native checkpoint hooks are used.
Diagnostic checkpoints cannot be exported as selected production models.
"""
import argparse
import gc
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'vendor/subcell_embed')]

import lightning as L
import torch
import torch.distributed as dist
import yaml
from lightning.pytorch.callbacks import Callback, EarlyStopping
from models.get_models import get_model_dict

from prot_loc_benchmark.provenance import capture_source, code_fingerprint, save_json, sha256, verify_source
from prot_loc_benchmark.representations.subcell_manifest import load_preflight
from prot_loc_benchmark.representations.subcell_protocol import model_config, resolved_protocol, validate_config
from prot_loc_benchmark.representations.subcell_run import AlleleCheckpoint, runtime_info, verify_selection
from prot_loc_benchmark.representations.subcell_training import (
    AlleleDataModule, SubCellAlleleModule, enable_checkpointing, load_pretrained_weights,
)


class StopAtPass(Callback):
    def __init__(self, stop):
        self.stop = stop

    def on_train_epoch_end(self, trainer, module):
        if trainer.current_epoch + 1 >= self.stop:
            trainer.should_stop = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    validate_config(config)
    assert all(Path(config[k]).is_absolute() for k in ('preflight', 'pretrained_weights', 'output'))
    world = int(os.environ['WORLD_SIZE'])
    assert world == config['devices'] == 2
    torch.set_float32_matmul_precision('high')
    frame, classes, validation, evidence = load_preflight(config['preflight'])
    output = Path(config['output'])
    assert not output.resolve().is_relative_to(Path(evidence['release_root']))
    identity = {'kind': 'release_validation', 'config': config, 'data': evidence,
                'resolved': resolved_protocol(config['family']), 'runtime': runtime_info(),
                'world_size': world, 'code_sha256': code_fingerprint(),
                'git_head': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
                'diagnostic_overrides': {'validation_every': 1, 'stop_passes': [1, 2]}}
    if int(os.environ['RANK']) == 0:
        output.mkdir(parents=True, exist_ok=False)
        capture_source(output)
        verify_source(output, identity['code_sha256'])
        save_json(output / 'run.json', identity)
        save_json(output / 'class_index.json', classes)
        save_json(output / 'validation_ids.json', validation)
    steps = len(classes) // 16
    for stop in (1, 2):
        L.seed_everything(config['seed'], workers=True)
        module = SubCellAlleleModule(get_model_dict(model_config(config['family'])), sorted(classes), identity, output)
        if stop == 1:
            load_pretrained_weights(module, config['pretrained_weights'], config['pretrained_sha256'])
        enable_checkpointing(module)
        data = AlleleDataModule(frame, classes, validation, config['sampling_seed'], config['workers'])
        trainer = L.Trainer(accelerator='gpu', devices=world, strategy='ddp', max_epochs=100,
                            check_val_every_n_epoch=1, sync_batchnorm=True, use_distributed_sampler=False,
                            precision='32-true', accumulate_grad_batches=1, gradient_clip_val=1.,
                            gradient_clip_algorithm='norm', num_sanity_val_steps=0,
                            enable_progress_bar=False, enable_model_summary=False, logger=False,
                            callbacks=[StopAtPass(stop), AlleleCheckpoint(output),
                                       EarlyStopping('val/macro_ap', mode='max', patience=5, min_delta=.001,
                                                     check_on_train_epoch_end=False)])
        last = output / 'models/last.ckpt'
        trainer.fit(module, datamodule=data, ckpt_path=last if stop == 2 else None)
        assert trainer.global_step == stop * steps
        assert len(data.val_data) == len(validation)
        # Every rank checks the complete serialized model, not just sampled weights.
        saved = torch.load(last, map_location='cpu', weights_only=False)
        assert saved['global_step'] == saved['allele_v2']['next_pass'] * steps == stop * steps
        assert saved['lr_schedulers'][0]['last_epoch'] == stop * steps
        assert len(saved['allele_v2']['rng_states']) == world
        for name, tensor in module.state_dict().items():
            assert torch.equal(tensor.cpu(), saved['state_dict'][name]), name
        del saved, trainer, module, data
        gc.collect()
        torch.cuda.empty_cache()
    if int(os.environ['RANK']) == 0:
        best = output / 'models/best_model_ap.ckpt'
        saved = torch.load(best, map_location='cpu', weights_only=False)
        selection = verify_selection(best, saved)
        assert selection['identity']['kind'] == 'release_validation'
        assert selection['metric_dtype'] == 'float64'
        verify_source(output, identity['code_sha256'])
        reports = sorted(output.glob('attempts/*/pass-*.json'))
        assert len(reports) == 2
        for path in reports:
            report = json.loads(path.read_text())
            assert report['global_cell_presentations'] == steps * 128
            assert all(r['gradient_norm_max_after_clip'] <= 1.00001 for r in report['ranks'])
            assert all(all(v > 0 for v in r['sampled_parameter_max_abs_drift'].values()) for r in report['ranks'])
        for path in output.glob('attempts/*/validation-pass-*.json'):
            metrics = json.loads(path.read_text())
            assert sorted(metrics['categories'][i] for i in metrics['omitted']) == evidence['missing_validation_classes']
        save_json(output / 'passed.json', {'identity': identity, 'validation_cells': len(validation),
                  'completed_updates': 2 * steps, 'checkpoint_sha256': sha256(best),
                  'full_state_reload_equal': True, 'native_resume_passed': True,
                  'production_selectable': False})
        print('PASS: full saved T3, native checkpoint, exact state reload and resumed pass', flush=True)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
