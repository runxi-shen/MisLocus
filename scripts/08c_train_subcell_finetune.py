#!/usr/bin/env python3
"""Fine-tune SubCell ViT-B/16 on MisLocus crops (HPA-scale preprocessing).

Trains SubCell models on B13-B16 crops using the same physical-scale
preprocessing as frozen extraction (7.47× upsample → center-crop 448).
Uses mandatory technical-replicate splits (T1+T2=train, T3=val, T4=test).

Two model variants:
  - MAE-CellS-ProtS-Pool (ContrastMAE): MAE recon + cell + protein contrastive
  - ViT-ProtS-Pool (BaseSSL): Protein-supervised contrastive only

Pretrained weights loaded from data/interim/subcell_portable/weights/rybg/.

Usage:
    # MAE multitask model (2× H100)
    CUDA_VISIBLE_DEVICES=0,1 pixi run -e subcell torchrun \\
        --nproc_per_node=2 --master_port=29506 \\
        scripts/08c_train_subcell_finetune.py \\
        -c configs/subcell_finetune_mae.yaml

    # ViT contrastive-only model (2× H100)
    CUDA_VISIBLE_DEVICES=2,3 pixi run -e subcell torchrun \\
        --nproc_per_node=2 --master_port=29507 \\
        scripts/08c_train_subcell_finetune.py \\
        -c configs/subcell_finetune_vit.yaml
"""
from __future__ import annotations

import argparse
import importlib
import os
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import yaml

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path setup: add vendored subcell-embed to sys.path
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
VENDOR_DIR = REPO_ROOT / "vendor" / "subcell_embed"
sys.path.insert(0, str(VENDOR_DIR))
sys.path.insert(0, str(REPO_ROOT / "src"))

import lightning as L  # noqa: E402
from lightning import LightningModule, Trainer  # noqa: E402
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint  # noqa: E402
from lightning.pytorch.strategies import DDPStrategy  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    label_ranking_average_precision_score,
)
from torch.utils.data import DataLoader  # noqa: E402
from torcheval.metrics.functional import (  # noqa: E402
    multilabel_auprc,
    topk_multilabel_accuracy,
)

from data.collate_fn import collate_fn_train  # noqa: E402  (vendor/subcell_embed/data/)
from models.get_models import get_model_dict  # noqa: E402
from models.lightning.callbacks.gc_callback import ScheduledGarbageCollector  # noqa: E402

from prot_loc_benchmark.representations.subcell_finetune import (  # noqa: E402
    MisLocusSubCellDataset,
)


class ValMetricsCallback(Callback):
    """Logs val_metrics/total_{ml_auprc,ml_topk_acc,mlrap,map} for ModelCheckpoint.

    Faithful minimal reproduction of the vendored ResultSaveCallback's
    metric logging — identical computations, just without matplotlib/seaborn
    plotting or UMAP feature dumping (the `plot_metrics=True, plot_feats=False`
    path from `vendor/subcell_embed/models/lightning/callbacks/result_callback.py`).

    Exposes the two monitors used by the original `main_lightning.py` so both
    `best_model_ap.ckpt` (on macro AUPRC) and `best_model_mlrap.ckpt`
    (on sample-ranking AP) are selected by the same criteria as the paper.
    """

    def __init__(self, num_labels: int) -> None:
        super().__init__()
        self.num_labels = num_labels
        self._outputs: list[torch.Tensor] = []
        self._targets: list[torch.Tensor] = []

    def on_validation_batch_end(
        self,
        trainer: "Trainer",
        pl_module: "LightningModule",
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if not isinstance(outputs, dict) or "output" not in outputs:
            return
        self._outputs.append(outputs["output"].detach())
        self._targets.append(outputs["target"].detach())

    def on_validation_epoch_end(
        self, trainer: "Trainer", pl_module: "LightningModule"
    ) -> None:
        if not self._outputs:
            return
        outputs = torch.cat(self._outputs, dim=0).float()
        targets = torch.cat(self._targets, dim=0).int()

        # DDP all-gather so val metric is computed on the FULL val set,
        # not per-rank shards. Matches vendor ResultSaveCallback lines 64-68.
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            if world_size > 1:
                gathered_out = [torch.empty_like(outputs) for _ in range(world_size)]
                gathered_tgt = [torch.empty_like(targets) for _ in range(world_size)]
                torch.distributed.all_gather(gathered_out, outputs)
                torch.distributed.all_gather(gathered_tgt, targets)
                outputs = torch.cat(gathered_out, dim=0)
                targets = torch.cat(gathered_tgt, dim=0)

        # torcheval (GPU-native): macro AUPRC + top-k accuracy
        ml_auprc = multilabel_auprc(
            outputs, targets, num_labels=self.num_labels, average="macro"
        )
        pl_module.log("val_metrics/total_ml_auprc", ml_auprc, logger=True)
        topk = topk_multilabel_accuracy(outputs, targets, criteria="hamming", k=5)
        pl_module.log("val_metrics/total_ml_topk_acc", topk, logger=True)

        # sklearn (CPU): LRAP (sample-level ranking) + macro AP cross-check
        outputs_np = outputs.cpu().numpy()
        targets_np = targets.cpu().numpy()
        mlrap = label_ranking_average_precision_score(targets_np, outputs_np)
        pl_module.log("val_metrics/total_mlrap", mlrap, logger=True)
        mean_ap = average_precision_score(targets_np, outputs_np, average="macro")
        pl_module.log("val_metrics/total_map", mean_ap, logger=True)

        self._outputs.clear()
        self._targets.clear()


def set_random_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")


def setup_transforms(config: dict):
    """Setup data augmentation transforms from subcell-embed convention."""
    import torchvision.transforms.v2 as v2
    from torchvision.transforms.v2 import InterpolationMode
    from utils.augmentations import (
        GaussianNoise,
        PerBatchCompose,
        PerChannelAdjustSharpness,
        PerChannelColorJitter,
        PerChannelGaussianBlur,
        PerChannelRandomErasing,
        RemoveChannel,
        RescaleProtein,
    )

    # Geometric transforms (applied to both views)
    transform_list = [
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomChoice([
            v2.RandomAffine(
                degrees=90,
                translate=(0.2, 0.2),
                scale=(0.8, 1.2),
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            ),
            v2.RandomPerspective(
                distortion_scale=0.25,
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            ),
        ]),
    ]

    # Intensity transforms for second SSL view
    transforms2_list = [
        RemoveChannel(p=0.25),
        RescaleProtein(p=0.25),
        PerChannelColorJitter(brightness=0.5, contrast=0.5, p=1.0),
        v2.RandomChoice([
            PerChannelGaussianBlur(kernel_size=7, sigma=(0.1, 2.0), p=0.5),
            PerChannelAdjustSharpness(sharpness_factor=2, p=0.5),
        ]),
        GaussianNoise(sigma_range=(0.01, 0.05), p=0.5),
        PerChannelRandomErasing(scale=(0.02, 0.1), ratio=(0.3, 3.3), p=0.5),
    ]

    transforms = PerBatchCompose(transform_list)
    transforms2 = (
        PerBatchCompose(transforms2_list)
        if config["data"]["args"].get("ssl_transform", True)
        else None
    )

    return transforms, transforms2


def load_pretrained_weights(model, weights_path: str) -> None:
    """Load pretrained encoder + pool_model weights from .pth file.

    The .pth checkpoint contains keys prefixed with 'encoder.' and 'pool_model.'.
    We load them into the corresponding submodules of the Lightning model.
    """
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)

    # Separate encoder and pool_model weights
    encoder_state = {}
    pool_state = {}
    for k, v in state_dict.items():
        if k.startswith("encoder."):
            encoder_state[k[len("encoder."):]] = v
        elif k.startswith("pool_model."):
            pool_state[k[len("pool_model."):]] = v

    # Both BaseSSL and ContrastMAE (BaseMAE) store encoder as model.encoder
    msg = model.encoder.load_state_dict(encoder_state, strict=False)
    print(f"  Encoder: loaded {len(encoder_state)} keys "
          f"(missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)})")

    if hasattr(model, "pool_model") and pool_state:
        msg = model.pool_model.load_state_dict(pool_state, strict=False)
        print(f"  Pool model: loaded {len(pool_state)} keys "
              f"(missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune SubCell on MisLocus (HPA-scale preprocessing)"
    )
    parser.add_argument("-c", "--config", required=True, help="YAML config file")
    parser.add_argument("-r", "--random-seed", type=int, default=42)
    parser.add_argument("--resume", help="Resume from Lightning checkpoint")
    args = parser.parse_args()

    set_random_seed(args.random_seed)

    # Load config
    config = OmegaConf.to_container(OmegaConf.load(args.config))
    print("=" * 60)
    print("SubCell Fine-Tuning (HPA-Scale Preprocessing)")
    print("=" * 60)
    print(yaml.dump(config, default_flow_style=False))

    # Setup experiment folder
    exp_folder = Path(config["exp_folder"]) / config["exp_name"] / config["exp_mode"]
    exp_folder.mkdir(parents=True, exist_ok=True)
    model_folder = exp_folder / "models"
    model_folder.mkdir(exist_ok=True)

    # Save config
    with open(exp_folder / "config.yaml", "w") as f:
        yaml.dump(config, f)

    # GPU setup
    n_gpus = torch.cuda.device_count()
    print(f"GPUs available: {n_gpus}")

    # Attention backend: FP32 uses memory-efficient, FP16/BF16 uses FlashAttention
    precision = str(config["trainer"]["precision"])
    if precision in ("32", "32-true"):
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_math_sdp(False)
        print("Memory-efficient attention enabled (FP32)")
    else:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(False)
        print("FlashAttention enabled (FP16/BF16)")

    # Datasets — reuse the same manifest as Cytoself (same cell set, same splits)
    manifest_csv = config["data"].get("manifest_csv")
    if not manifest_csv:
        manifest_csv = str(
            REPO_ROOT / "data" / "interim" / "cytoself" / "manifests" / "manifest_b7b8b13b16_siteqc.csv"
        )
    manifest_csv = str(REPO_ROOT / manifest_csv) if not Path(manifest_csv).is_absolute() else manifest_csv

    dataset_kwargs = config["data"]["args"]
    train_dataset = MisLocusSubCellDataset(
        manifest_csv=manifest_csv, split="train", **dataset_kwargs
    )
    val_dataset = MisLocusSubCellDataset(
        manifest_csv=manifest_csv, split="val", **dataset_kwargs
    )

    # Batch sizes
    train_batch_size = config["train"]["train_batch_size"]
    val_batch_size = config["train"].get("test_batch_size", train_batch_size)
    device_train_bs = max(1, train_batch_size // n_gpus)
    device_val_bs = max(1, val_batch_size // n_gpus)

    # Data loaders
    num_workers = config.get("num_workers", 4)
    train_loader = DataLoader(
        train_dataset,
        batch_size=device_train_bs,
        shuffle=True,
        collate_fn=collate_fn_train,
        num_workers=num_workers,
        pin_memory=config.get("pin_memory", True),
        persistent_workers=num_workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=device_val_bs,
        shuffle=False,
        collate_fn=collate_fn_train,
        num_workers=num_workers,
        pin_memory=config.get("pin_memory", True),
        persistent_workers=num_workers > 0,
    )

    # Setup transforms (augmentation)
    transforms, transforms2 = setup_transforms(config)

    # Prepend HPA-scale GPU preprocessing to the transform pipeline.
    # Dataset returns raw 128×128 crops; resize happens on GPU here.
    # Import SubCellPreprocessor directly to avoid preprocessing/__init__.py (needs pycytominer)
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location(
        "subcell_preproc", str(REPO_ROOT / "src" / "prot_loc_benchmark" / "preprocessing" / "subcell.py")
    )
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    SubCellPreprocessor = _mod.SubCellPreprocessor

    from prot_loc_benchmark.config import SUBCELL_SCALE_FACTOR, SUBCELL_INFERENCE_CROP
    gpu_preprocess = SubCellPreprocessor(
        scale_factor=SUBCELL_SCALE_FACTOR,
        input_size=128,
        output_size=SUBCELL_INFERENCE_CROP,
    )

    # Wrap transforms to apply GPU preprocess first
    from utils.augmentations import PerBatchCompose
    if transforms is not None:
        orig_transforms = transforms
        class _PreprocessThenTransform:
            def __call__(self, x, mask=None):
                x = gpu_preprocess(x)
                if mask is not None:
                    return orig_transforms(x, mask)
                return orig_transforms(x)
        transforms = _PreprocessThenTransform()
    else:
        transforms = gpu_preprocess

    # Validation also needs the resize (no augmentation)
    valid_transforms = gpu_preprocess

    # Build model
    model_dict = get_model_dict(config["model"])
    n_cells = config["data"]["args"]["n_cells"]
    microbatch_size = device_train_bs * n_cells

    model_dict.update({
        "color_channels": train_dataset.color_channels,
        "save_folder": str(exp_folder),
        "num_classes": train_dataset.num_classes,
        "categories": train_dataset.unique_cats,
        "class_weights": torch.ones(train_dataset.num_classes),
        "batches_per_epoch": len(train_loader),
        "transforms": transforms,
        "transforms2": transforms2,
        "valid_transforms": valid_transforms,
    })

    # Scale learning rate by effective batch size
    model_dict["init_lr"] = (
        model_dict["init_lr"] * microbatch_size * n_gpus
    ) / 256

    # Create Lightning module
    pl_module_name = config["train"]["pl_module"]
    pl_model_module = importlib.import_module("models.lightning")
    model = getattr(pl_model_module, pl_module_name)(**model_dict)

    # Load pretrained weights
    pretrained_path = config["train"].get("pretrained_weights")
    if pretrained_path:
        pretrained_path = str(REPO_ROOT / pretrained_path)
        print(f"\nLoading pretrained weights: {pretrained_path}")
        load_pretrained_weights(model, pretrained_path)

    # Gradient checkpointing on HF transformer blocks.
    # Same math as non-checkpointed training — activations recomputed in backward.
    # Needed to fit FP32 n_cells=8 + batch=16 on 93 GiB H100 NVL.
    # use_reentrant=False required for DDP compatibility (static graph, find_unused_parameters).
    for attr in ("encoder", "decoder"):
        sub = getattr(model, attr, None)
        if sub is not None and hasattr(sub, "gradient_checkpointing_enable"):
            sub.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            print(f"  Gradient checkpointing enabled on {attr} (use_reentrant=False)")

    # Training strategy
    strategy_name = config["trainer"].get("strategy", "auto")
    if strategy_name == "ddp":
        strategy = DDPStrategy(process_group_backend="nccl")
    else:
        strategy = "auto"

    # Callbacks — matches vendor/subcell_embed/main_lightning.py lines 172-203:
    # ValMetricsCallback logs val_metrics/* before the two ModelCheckpoints
    # look up their monitored keys.
    callbacks = [
        ScheduledGarbageCollector(
            gen_1_batch_interval=config["trainer"]["gc_interval"]
        ),
        ValMetricsCallback(num_labels=train_dataset.num_classes),
        ModelCheckpoint(
            dirpath=str(model_folder),
            filename="best_model_ap",
            monitor="val_metrics/total_ml_auprc",
            verbose=True,
            save_last=True,
            save_top_k=1,
            mode="max",
            enable_version_counter=False,
        ),
        ModelCheckpoint(
            dirpath=str(model_folder),
            filename="best_model_mlrap",
            monitor="val_metrics/total_mlrap",
            verbose=True,
            save_last=False,  # save_last already handled by the ap checkpoint
            save_top_k=1,
            mode="max",
            enable_version_counter=False,
        ),
        # Fine-tuning converges well before max_epochs; stop when val AUPRC
        # plateaus. patience counts validation checks (not epochs), so with
        # valid_every=10 a patience of 5 = 50 epochs of no improvement.
        EarlyStopping(
            monitor="val_metrics/total_ml_auprc",
            mode="max",
            patience=5,
            min_delta=0.001,
            verbose=True,
        ),
    ]

    # Trainer
    # devices: honor torchrun's WORLD_SIZE so each rank binds to a distinct GPU.
    # Previous `devices=1` caused both DDP ranks to collide on cuda:0.
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    trainer = L.Trainer(
        default_root_dir=str(exp_folder),
        accelerator="gpu",
        devices=world_size,
        strategy=strategy,
        check_val_every_n_epoch=config["trainer"]["valid_every"],
        max_epochs=model.max_epochs,
        log_every_n_steps=config["trainer"]["logging_interval"],
        sync_batchnorm=True,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        callbacks=callbacks,
        num_sanity_val_steps=0,
        precision=config["trainer"]["precision"],
    )

    # Resume or fresh start
    ckpt_path = args.resume
    if not ckpt_path:
        potential = model_folder / "last.ckpt"
        if potential.exists():
            ckpt_path = str(potential)
            print(f"Resuming from: {ckpt_path}")

    # Train
    print(f"\nStarting training: {pl_module_name}, {model.max_epochs} epochs")
    print(f"  Train: {len(train_dataset)} groups, Val: {len(val_dataset)} groups")
    print(f"  Batch size: {train_batch_size} (per-device: {device_train_bs})")
    print(f"  Effective LR: {model_dict['init_lr']:.2e}")
    print()

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=ckpt_path,
    )

    print(f"\nTraining complete! Results: {exp_folder}")


if __name__ == "__main__":
    # Transitional dispatch: keep legacy recipes usable until both families migrate.
    route = argparse.ArgumentParser(add_help=False)
    route.add_argument('-c', '--config')
    known, _ = route.parse_known_args()
    if (known.config and not {'-h', '--help'}.intersection(sys.argv[1:])
            and yaml.safe_load(Path(known.config).read_text()).get('protocol') == 'subcell-allele-rybg-v2'):
        import runpy
        runpy.run_path(str(REPO_ROOT / 'scripts/08c_train_subcell_allele_v2.py'), run_name='__main__')
    else:
        main()
