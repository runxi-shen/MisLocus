"""Immutable code snapshots and explicit runtime/selection evidence for SubCell v2."""
from __future__ import annotations

import importlib.metadata
import os
import platform

import torch

from prot_loc_benchmark.provenance import (
    capture_source, code_fingerprint, invocation, save_json, sha256, verify_source,
)


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
