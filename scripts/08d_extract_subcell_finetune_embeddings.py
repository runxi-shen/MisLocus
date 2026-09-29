#!/usr/bin/env python3
"""Extract embeddings from fine-tuned SubCell checkpoints.

Uses the same HPA-scale preprocessing as frozen extraction (08a):
  128×128 → 7.47× bilinear → center-crop 448 → min-max normalize

Loads Lightning .ckpt files from 08c training output, extracts 1536-dim
GatedAttentionPooler embeddings for all cells in all 8 batches.

Usage:
    pixi run -e subcell python scripts/08d_extract_subcell_finetune_embeddings.py \\
        --model mae --gpu 0

    pixi run -e subcell python scripts/08d_extract_subcell_finetune_embeddings.py \\
        --model vit --gpu 1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "vendor" / "subcell_embed"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from prot_loc_benchmark.config import (  # noqa: E402
    FOCUS_BATCHES,
    INTERIM_DIR,
    SINGLE_CELL_CROPS_DIR,
    SUBCELL_CHANNEL_FILES,
    SUBCELL_EMBED_DIM,
    SUBCELL_INFERENCE_CROP,
    SUBCELL_INPUT_CROP_SIZE,
    SUBCELL_PIXEL_SIZE_HPA,
    SUBCELL_PIXEL_SIZE_MISLOCUS,
    SUBCELL_SCALE_FACTOR,
)
from prot_loc_benchmark.preprocessing.subcell import SubCellPreprocessor  # noqa: E402
from prot_loc_benchmark.representations.subcell_io import export_subcell_batch  # noqa: E402




def load_finetuned_model(model_type: str, device: torch.device, seed: int | None = None):
    """Load fine-tuned model from Lightning checkpoint.

    Extracts the encoder + pool_model from the full Lightning module,
    wraps them in an inference container matching SubCellPortable's interface.

    When ``seed`` is provided, loads the seed-suffixed variant
    (``models/{model_type}_s{seed}/``); otherwise loads the base variant.
    Always use that historical run's archived config, not a mutable recipe.
    """
    from omegaconf import OmegaConf

    suffix = f"_s{seed}" if seed is not None else ""
    model_subdir = f"{model_type}{suffix}"

    # Determine checkpoint and config paths
    model_dir = INTERIM_DIR / "subcell_finetune" / "models" / model_subdir
    ckpt_path = model_dir / "models" / "best_model_ap.ckpt"
    if not ckpt_path.exists():
        # Try alternate location
        ckpt_path = model_dir / "best_model_ap.ckpt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}. Run 08c training "
            f"with -c configs/subcell_finetune_{model_subdir}.yaml first."
        )

    config_path = model_dir / "config.yaml"
    config = OmegaConf.to_container(OmegaConf.load(config_path))

    # Load checkpoint to get hyperparameters (num_classes from training)
    import importlib
    from models.get_models import get_model_dict

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    hyper_params = ckpt.get("hyper_parameters", {})
    num_classes = hyper_params.get("num_classes", 68)

    model_dict = get_model_dict(config["model"])
    model_dict.update({
        "color_channels": ["red", "yellow", "blue", "green"],
        "save_folder": str(model_dir),
        "num_classes": num_classes,
        "categories": [f"class_{i}" for i in range(num_classes)],
        "class_weights": torch.ones(num_classes),
        "batches_per_epoch": 100,
        "transforms": None,
        "transforms2": None,
        "valid_transforms": None,
    })

    pl_module_name = config["train"]["pl_module"]
    pl_model_module = importlib.import_module("models.lightning")
    model = getattr(pl_model_module, pl_module_name)(**model_dict)

    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)

    return model


class InferenceWrapper:
    """Wraps a Lightning module for inference (encoder + pool_model).

    Encoder type differs between the two fine-tuned variants:
      - BaseSSL (ViT)      → HuggingFace ViTModel: accepts interpolate_pos_encoding
      - ContrastMAE (MAE)  → HuggingFace ViTMAEModel: accepts mask_ratio; no pos interp

    Both were trained at the same image_size (448) we use for inference, so no
    position interpolation is needed — we only include the kwarg for the ViT
    variant for parity with 08a's frozen-extraction call site.
    """

    def __init__(self, model, device: torch.device):
        self.device = device
        self.encoder = model.encoder
        self.pool_model = model.pool_model
        self.encoder.eval()
        self.pool_model.eval()
        # Detect encoder variant by class name to avoid importing ViTMAEModel
        self._is_mae = type(self.encoder).__name__ == "ViTMAEModel"

    @torch.no_grad()
    def extract(self, batch: torch.Tensor) -> np.ndarray:
        """Extract 1536-dim pooled embeddings from a batch of images."""
        if self._is_mae:
            encoder_out = self.encoder(
                batch, output_attentions=False, mask_ratio=0.0
            )
        else:
            encoder_out = self.encoder(
                batch, output_attentions=False, interpolate_pos_encoding=True
            )
        pool_op, _ = self.pool_model(encoder_out.last_hidden_state)
        return pool_op.float().cpu().numpy()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract embeddings from fine-tuned SubCell checkpoints."
    )
    p.add_argument(
        "--model",
        choices=["mae", "vit"],
        required=True,
        help="Fine-tuned model variant: mae or vit",
    )
    p.add_argument(
        "--batch",
        nargs="+",
        default=[b for b in FOCUS_BATCHES],
        help="Batch ID(s) to process (default: all focus batches)",
    )
    p.add_argument("--gpu", type=int, default=0, help="GPU device index")
    p.add_argument("--batch-size", type=int, default=128, help="Inference batch size")
    p.add_argument("--fp32", action="store_true", help="Use FP32 instead of FP16")
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Multi-seed retrain selector. When set, loads the model from "
             "{model}_s{seed}/ and writes embeddings to "
             "subcell_finetuned_{model}_s{seed}/. Default uses the unsuffixed model.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    model_type = args.model
    seed_suffix = f"_s{args.seed}" if args.seed is not None else ""
    rep_name = f"subcell_finetuned_{model_type}{seed_suffix}"
    use_fp16 = not args.fp32

    print(f"SubCell fine-tuned embedding extraction")
    print(f"  Representation: {rep_name}")
    print(f"  Model: {model_type}")
    print(f"  Scale factor: {SUBCELL_SCALE_FACTOR:.3f}× "
          f"({SUBCELL_PIXEL_SIZE_MISLOCUS} µm/px → {SUBCELL_PIXEL_SIZE_HPA} µm/px)")
    print(f"  Precision: {'FP16' if use_fp16 else 'FP32'}")
    print()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Load fine-tuned model
    print(f"  Loading fine-tuned {model_type} checkpoint...")
    model = load_finetuned_model(model_type, device, seed=args.seed)
    wrapper = InferenceWrapper(model, device)
    print(f"  Model loaded")
    print()

    preprocessor = SubCellPreprocessor(
        scale_factor=SUBCELL_SCALE_FACTOR,
        input_size=SUBCELL_INPUT_CROP_SIZE,
        output_size=SUBCELL_INFERENCE_CROP,
    )

    # Process each batch
    output_dirs = []
    for batch_id in args.batch:
        print(f"=== {batch_id} ===")
        crops_dir = SINGLE_CELL_CROPS_DIR / batch_id
        if not crops_dir.exists():
            print(f"  SKIP: {crops_dir} not found")
            continue

        # Discover alleles with all 4 channels + metadata
        allele_dirs = sorted([
            d for d in crops_dir.iterdir()
            if d.is_dir()
            and all((d / ch).exists() for ch in SUBCELL_CHANNEL_FILES)
            and (d / "metadata.parquet").exists()
        ])
        print(f"  Found {len(allele_dirs)} alleles with all channels")

        # Stream per-allele to avoid loading entire batch into RAM (~65 GB)
        print("  Extracting embeddings...")
        torch.set_grad_enabled(False)
        all_emb_parts = []
        all_meta_dfs = []

        for allele_dir in tqdm(allele_dirs, desc="  Alleles"):
            meta = pl.read_parquet(str(allele_dir / "metadata.parquet"))
            n = meta.height
            if n == 0:
                continue
            all_meta_dfs.append(meta)

            # Load channels for this allele only
            stacked = np.stack(
                [np.load(str(allele_dir / ch), mmap_mode="r")[:n].astype(np.float32)
                 for ch in SUBCELL_CHANNEL_FILES],
                axis=1,
            )  # [n, 4, 128, 128]

            # Process in GPU chunks
            for start in range(0, n, args.batch_size):
                end = min(start + args.batch_size, n)
                tensor = torch.from_numpy(stacked[start:end]).to(
                    device=device, dtype=torch.float32
                )
                tensor = preprocessor(tensor)
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_fp16):
                    emb = wrapper.extract(tensor)
                all_emb_parts.append(emb)

        all_metadata = pl.concat(all_meta_dfs)
        all_emb = np.concatenate(all_emb_parts, axis=0)
        n_total = all_metadata.height
        assert all_emb.shape == (n_total, SUBCELL_EMBED_DIM)
        print(f"  Total: {n_total:,} cells × {SUBCELL_EMBED_DIM} dims")

        # Export
        output_dir = INTERIM_DIR / rep_name / batch_id
        export_subcell_batch(batch_id, all_emb, all_metadata, output_dir)
        output_dirs.append(output_dir)
        print()

    if output_dirs:
        from prot_loc_benchmark.provenance import record
        record(output_dirs=output_dirs)


if __name__ == "__main__":
    main()
