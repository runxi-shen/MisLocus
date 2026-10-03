"""Immutable code snapshots and explicit runtime/selection evidence for SubCell v2."""
from __future__ import annotations

import importlib.metadata
import json
import os
import platform
from pathlib import Path

import torch
from lightning.pytorch.callbacks import ModelCheckpoint

from prot_loc_benchmark.provenance import save_json, sha256


def runtime_info():
    return {'python': platform.python_version(),
            'versions': {p: importlib.metadata.version(p) for p in
                         ('torch', 'torchvision', 'transformers', 'lightning', 'timm', 'numpy', 'pandas', 'scikit-learn')},
            'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
            'matmul_precision': torch.get_float32_matmul_precision(),
            'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
            'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'deterministic_warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
            'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            'flash_sdp': torch.backends.cuda.flash_sdp_enabled(),
            'mem_efficient_sdp': torch.backends.cuda.mem_efficient_sdp_enabled(),
            'math_sdp': torch.backends.cuda.math_sdp_enabled()}


class AlleleCheckpoint(ModelCheckpoint):
    """Native AP/tie selection, plus a hash-bound authoritative selection receipt.

    The pinned Lightning save hook is used so the receipt immediately follows the
    best checkpoint write, before last.ckpt is advanced. No second AP selector.
    """
    def __init__(self, output):
        self.output = Path(output).resolve()
        super().__init__(dirpath=self.output / 'models', filename='best_model_ap', monitor='val/macro_ap',
                         mode='max', save_top_k=1, save_last=True, enable_version_counter=False,
                         save_on_train_epoch_end=False)

    def on_validation_end(self, trainer, pl_module):
        super().on_validation_end(trainer, pl_module)
        # Lightning 2.6.1 otherwise advances last only when top-k improves.
        # Persist current optimizer/RNG/patience even on ties or regressions.
        if not self._should_skip_saving_checkpoint(trainer):
            self._save_last_checkpoint(trainer, self._monitor_candidates(trainer))

    def _save_checkpoint(self, trainer, filepath):
        super()._save_checkpoint(trainer, filepath)
        if trainer.is_global_zero and filepath == self.best_model_path:
            save_json(self.output / 'selection.json', {
                'checkpoint': str(Path(filepath).relative_to(self.output)), 'sha256': sha256(filepath),
                'pass': trainer.current_epoch + 1, 'global_step': trainer.global_step,
                'macro_ap': float(self.best_model_score), 'metric_dtype': 'float64',
                'identity': trainer.lightning_module.identity})


def verify_selection(checkpoint_path, checkpoint, selection_path=None):
    identity = checkpoint.get('allele_v2', {}).get('identity', {})
    # A supplied receipt permits archival relocation without changing its content.
    receipt_path = Path(selection_path) if selection_path else Path(identity['config']['output']) / 'selection.json'
    receipt = json.loads(receipt_path.read_text())
    if (receipt['identity'] != identity or receipt['sha256'] != sha256(checkpoint_path)
            or receipt['global_step'] != checkpoint['global_step'] or receipt['pass'] != checkpoint['epoch'] + 1
            or checkpoint['global_step'] <= 0):
        raise ValueError('Checkpoint does not match the authoritative selection receipt')
    state = checkpoint.get('callbacks', {}).get(AlleleCheckpoint(receipt_path.parent).state_key, {})
    score = state.get('best_model_score')
    if (not isinstance(score, torch.Tensor) or score.numel() != 1 or score.dtype != torch.float64
            or not torch.isfinite(score).item() or not 0 <= score.item() <= 1
            or receipt.get('metric_dtype') != 'float64'
            or type(receipt.get('macro_ap')) not in (int, float) or receipt['macro_ap'] != score.item()):
        raise ValueError('Selection score/dtype does not match the native checkpoint selector')
    return receipt


def require_resumable(output, checkpoint):
    """Reject terminal production runs; diagnostic probes may deliberately resume."""
    saved = checkpoint.get('allele_v2', {})
    if saved.get('identity', {}).get('kind') != 'production':
        return
    if any(Path(output).glob('attempts/*/completed.json')):
        raise ValueError('Production run already completed; refusing to resume into its outputs')
    if saved['next_pass'] >= 100:
        raise ValueError('Production run reached the 100-pass horizon; refusing to resume')
    for state in checkpoint.get('callbacks', {}).values():
        if 'stopped_epoch' in state and (state['stopped_epoch'] > 0 or
                state['wait_count'] >= state['patience']):
            raise ValueError('Production run already early-stopped; refusing to resume')
