# MisLocus — companion repo

Reproduce the protein-localization representation benchmark from the published
**MisLocus single-cell crop dataset**. This repo ships the code needed
*downstream* of feature preprocessing — `features.parquet` files come
pre-computed in the dataset bundle, ready for classification and benchmarking.

## What you get

The dataset bundle (download separately, see below) ships into `data/`:

- **Per-rep features** — `data/interim/{cellprofiler,cytoself,subcell_portable_*,vit}/{batch}/features.parquet`.
- **Crop manifest** — `data/interim/crop_manifest/{batch}/manifest.parquet` (one row per cell).
- **QC'd single-cell crops** (optional) — `data/interim/single_cell_crops/{batch}/{allele}/*.npy` (128×128 uint16, 4 channels: DNA / GFP / AGP / Mito).

Reference annotations (HPA gene-localization + ClinVar / dbNSFP / pLDDT
allele collection) live in `annotations/` and are tracked in this repo
(~9.7 MB total). They're consumed by `10_benchmark_clinvar.py` and
`10b_benchmark_hpa.py` and don't need to be downloaded.

See [`docs/dataset_bundle.md`](docs/dataset_bundle.md) for the full layout
and the download script's CLI options.

## Quickstart

The five recipes you'll actually use (run `just --list` for the full set):

```bash
# 0. Install pixi: https://pixi.sh/install
just install

# 1. Browseable sample (~1.2 GB, no GPU needed) — 8 alleles × 2 batches
#    of single-cell crops, plus a sample manifest.
just download-sample
just inspect-sample

# 2. One batch of CellProfiler features (~3 GB), then classify on it.
just download-batch         # default: cellprofiler / 2025_01_27_Batch_13
just classify-batch         # XGBoost AUROC + copairs PA mAP

# 3. Or pull everything (every rep × every batch + crop tarballs).
just download-all
```

`just download-batch` and `just classify-batch` both accept `BATCH=…` and
`REP=…` overrides (`just classify-batch BATCH=2025_01_28_Batch_14
REP=cytoself`). `just classify-batch` runs `09_classify.py --gpu`
(GPU-required) followed by `09c_classify_PA.py` (CPU); the recipe sets the
two env vars (`CONDA_OVERRIDE_CUDA=12.0`, `CUDA_VERSION=12.0`) the
lab-server's pixi env needs to keep XGBoost on GPU.

If you're going to run SubCell extraction/training (`08b` / `08c` / `08d`), also fetch
the encoder weights:

```bash
just download-subcell       # ~2 GB, CZI's public S3 bucket
```

## Shared raw-crop preparation (optional, any model)

With an existing **git/LFS Hugging Face release checkout**, use one model-independent
preparer. It preserves the source and native128×128 uint16 arrays; no resizing,
normalization, model weights, CUDA or torch are needed (Python3.12+ and git only).

```bash
python scripts/01_prepare_cell_crops.py inspect --release /path/to/HF-checkout
python scripts/01_prepare_cell_crops.py extract \
  --release /path/to/HF-checkout --crops /separate/storage/crops
# Explicit read-only payload rehash; save stdout outside the crops directory.
python scripts/01_prepare_cell_crops.py verify --crops /separate/storage/crops
```

The destination must be new. Receipts record the HF commit/LFS hashes, extracted
file hashes and producer source. Every model can read the same staged crops and
choose its own channel order, geometry and normalization. This unpacks published
cell crops; it does not segment/crop raw microscopy images. Existing model scripts
may still need their crop-input paths configured; this does not launch other models.
The legacy `00_download_dataset.py` remap workflow above is unchanged, and its
remapped output is not the git/LFS source accepted by01.

SubCell uses `01 → 08a preflight → 08b frozen extraction`, or
`01 → 08a preflight → 08c training → 08d adapted extraction`. Its physical-scale
preprocessing remains in `preprocessing/subcell.py`, not in the common preparer.

## Pipeline

```
data/interim/{rep}/{batch}/features.parquet      ← shipped in the bundle
            │
   ┌────────┴────────┐
   ▼                 ▼
09_classify.py    09c_classify_PA.py
(XGBoost AUROC)   (copairs mAP + p95 hit-call)
            │
            ▼
data/processed/classification{,_PA}/{rep}/{batch}/
            │
   ┌────────┴────────┐
   ▼                 ▼
10_benchmark_clinvar.py    10b_benchmark_hpa.py
            │
            ▼
11_summarize_across_reps.py
            │
            ▼
data/processed/benchmark/clinvar/full_dataset/summary_across_reps/
```

## Pixi environments

| Env       | Used for                                                 |
|-----------|----------------------------------------------------------|
| `default` | PA mAP (`09c`), benchmarks (`10` / `10b` / `11`). CPU only. |
| `gpu`     | XGBoost classification (`09 --gpu`). Inherits `default` + adds the CUDA-12 system requirement so conda-forge resolves to GPU-enabled xgboost. |

The `cytoself` / `subcell` / `vit` envs only matter if you re-extract or
retrain a representation from raw crops; they're not needed for the
classification + benchmark workflow.

Optional, corrected allele-level SubCell retraining is documented in
[`docs/subcell_allele_v2.md`](docs/subcell_allele_v2.md). It keeps an existing
Hugging Face mirror unchanged and stages crops/run outputs separately.
Six fresh MAE/ViT × seed42/43/44 fits and the two seed42 all-T1–T4 raw exports
are complete and verified. Each export contains 3,332,309 cells × 1,536 FP32
features. See the [completion evidence and limitations](docs/evidence/README.md)
and [current review follow-up](docs/reviews/subcell-v2-readiness-followup.md).
Matched frozen four-channel MAE/ViT bulk exports are also complete: the same
3,332,309 cells per model, with full readback and exact ordered metadata matching.
See their [additive evidence inventory](docs/evidence/subcell-v2-frozen-completion.json).
These outputs are in external versioned storage, **not uploaded into the frozen HF
bundle**. Historical features are not relabeled as v2. Matched downstream processing
and scientific/evaluation policies remain separate gates; single-seed adapted
reporting does not establish seed robustness.

## Adding a new representation

The benchmark is a contract — anything that writes
`data/interim/<myrep>/{batch}/features.parquet` with the right schema
plugs in without touching `09 / 09c / 10 / 10b / 11`. See
[`docs/dataset_bundle.md#adding-a-new-representation`](docs/dataset_bundle.md#adding-a-new-representation)
for the integration recipe.
