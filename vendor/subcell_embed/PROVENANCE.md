# SubCell-embed Vendored Code

## Source

- **Repository**: https://github.com/CellProfiling/subcell-embed
- **Commit**: `a71236f35d75b849cb7435b38f4f0e0b28c97a62` (2026-04-20)
- **Vendored date**: 2026-04-20

## Purpose

Training infrastructure for fine-tuning SubCell ViT-B/16 models on MisLocus
single-cell crops. Two model variants:

1. **MAE-CellS-ProtS-Pool** (`ContrastMAE`): Masked Autoencoder reconstruction
   + cell-level contrastive + protein-level contrastive
2. **ViT-ProtS-Pool** (`BaseSSL`): Protein-supervised contrastive only

## Modifications

This is a selected, locally adapted subset, not an unmodified upstream checkout.
Against the pinned commit above, the retained files have these deviations:

| File | Local deviation |
|---|---|
| `models/object_aware_mae.py` | Four-line eval-only zero-mask fast path: retain token order, return zero mask and identity restore indices, and skip the masking RNG draw. Training-mode and nonzero-mask behavior remain upstream behavior. |
| `data/__init__.py` | Removes the eager dataset import to avoid loading MosaicML `streaming` for this workflow. |
| `models/__init__.py` | Trims eager imports of torch/transformers, centroid utilities and object-aware MAE; retains the selected model helpers. |
| `models/lightning/__init__.py` | Imports only `base_mae`, `base_ssl` and `contrast_mae`, rather than all upstream Lightning variants. |
| `models/lightning/save_utils.py` | Replaces upstream visualization utilities with no-op `save_feat_nmf`, `save_overlay_attn` and `save_recon` stubs; this workflow does not use those visualizations. |

Local-only files (not present at the pinned upstream paths):
- `PROVENANCE.md`: this provenance description.
- `models/lightning/callbacks/__init__.py`: empty package initializer.
- `utils/__init__.py`: empty package initializer.

The import trims, visualization stubs and local-only files predate PR #2.
PR #2 adds the zero-mask evaluation patch; it does not introduce those older
adaptations. Other retained upstream files are unchanged. Upstream files not
needed by this vendored subset are not included; this is not a full-repository mirror.
Custom MisLocus dataset/training code lives in `src/prot_loc_benchmark/` and `scripts/`.
The historical vendor localization training modules are not the v2 MisLocus entry point.

### Scientific rationale for zero-mask evaluation

We use deterministic, all-patch evaluation to measure representations without
stochastic masking augmentation. Upstream still permutes tokens at zero masking;
we bypass this redundant permutation to eliminate its numerical variability and
RNG consumption, consistently across frozen and adapted models. Training-mode
masking remains unchanged.

Positional embeddings are attached before masking/shuffling; retaining all tokens
in their original order removes an unnecessary evaluation permutation, not spatial
information. This is a measurement/reproducibility choice, not a claim of improved
biological accuracy, an unchanged whole-training RNG trajectory versus upstream,
or universal bitwise determinism. The fast path applies only to the MAE encoder.

### Production correspondence

The archived `object_aware_mae.py` in all six completed MAE/ViT runs (seeds 42/43/44,
producer `03b19615508ebfbec767bf67d4d784ee1c7ad9c1`) matches the PR #2 file
byte-for-byte. Its SHA256 is
`4a9f28b57db0aee7d5daf6cac398dcccb1bbf6bf429159f4c2da0ce0f3f9e795`.
The MAE path is used for MAE evaluation, not by the ViT encoder. This correspondence
is provenance evidence, separate from the scientific rationale. Correcting this
text does not rewrite historical source archives, runs, checkpoints or exports.

## Key Architecture

- Encoder: ViT-B/16 (768 hidden, 12 layers, 12 heads, patch_size=16, image_size=448)
- Pooler: GatedAttentionPooler (768 → 1536, 2 heads, 512 int_dim)
- Projection: 1536 → 8192 → 8192 → 512 (for contrastive loss)
- Decoder (MAE only): 512 hidden, 8 layers, 16 heads

## Dependencies

The validated companion stack pins PyTorch2.4.1, Transformers4.45.2,
Torchvision0.19.1, Lightning2.6.1 and timm1.0.26; use the `subcell` environment
and committed `pixi.lock`, not an unconstrained upgrade. Transformer internals
such as `ViTSdpaAttention` are version-sensitive. Other dependencies include
omegaconf and scipy.

## Execution scope

Completed allele-v2 training used companion commit `03b1961`; seed42 adapted
extraction used `e306037`; the completed matched frozen four-channel exports used
`d0c61f0`. The latter fixes checkpoint recovery on non-improving validation, not
model mathematics, input preprocessing or the six original fits. Their immutable
source archives are authoritative, not this later documentation revision. See
[protocol and limitations](../../docs/subcell_allele_v2.md) and [completion evidence](../../docs/evidence/README.md).

The local loop retains same-run cross-rank RNG correlation. Different run seeds
are not identical experiments, but no claim of independent per-rank streams or
single-seed downstream robustness is made. Later runtime-metadata hardening does
not reconstruct unrecorded historical determinism flags.

## Paper

Gupta et al., "SubCell: Vision foundation models for microscopy capture
single-cell biology", bioRxiv 2024. DOI: 10.1101/2024.12.06.627299
