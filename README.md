# MisLocus

MisLocus is a toolkit for benchmarking protein-localization representations
using single-cell images and variant annotations. It brings together feature
preprocessing, model training and retraining, feature extraction, and downstream
analysis of variant effects and protein localization.

## Workflows

The repository is designed around three workflows:

| Starting point | Workflow |
|----------------|----------|
| **Published features (default)** | Download cleaned features, validate cell identities and comparison cohorts, then score and benchmark representations. |
| **New raw embeddings** | Apply quality control, representation-specific feature normalization and selection, then use the benchmark workflow. |
| **Single-cell crops** | Prepare model inputs, optionally train or fine-tune an encoder, and extract embeddings for feature preprocessing and benchmarking. |

Published `features.parquet` files are already cleaned. **Do not run
`06_preprocess_profiles.py` or normalize them again.** Encoder training and
extraction are optional; fitting XGBoost classifiers is part of downstream
scoring, not encoder training. Image preparation and feature normalization are
separate steps.

## What you get

### Included in this repository

- **Code:** preprocessing, training, extraction, and benchmark scripts in
  `scripts/`, with shared analysis functions in `src/`.
- **Reference annotations:** HPA gene localizations and the ClinVar / dbNSFP /
  pLDDT allele collection in `annotations/`. No separate download is needed.

### Available from the [Hugging Face dataset](https://huggingface.co/datasets/anonymous-xyz96/MisLocus)

| Component | Contents |
|-----------|----------|
| Precomputed features | Seven representations: CellProfiler, Cytoself, MorphEm, and portable RBG / fine-tuned SubCell MAE and ViT. |
| Crop manifests | Per-cell metadata for the image crops. |
| Single-cell images | QC'd 128×128 crops with DNA, GFP, AGP, and Mito channels; optional when using published features. |

Feature files are stored at `data/interim/{rep}/{batch}/features.parquet`.
MorphEm is named `morphem` on Hugging Face and stored locally as `vit`; the
downloader and scoring tools accept either name.

See the [dataset guide](docs/dataset_bundle.md) for the complete layout and
input formats.

## Download published features

From the repository root on **Linux x86-64**, with [Pixi](https://pixi.sh/install)
installed:

```bash
# Install only the default dependencies, not the project package.
pixi install --frozen -e default --skip prot-loc-benchmark

# Start with one representation and batch; no crop archives are downloaded.
.pixi/envs/default/bin/python scripts/00_download_dataset.py \
    --rep morphem --batch 2024_02_06_Batch_8
```

The feature file will be at
`data/interim/vit/2024_02_06_Batch_8/features.parquet`.

To download all published representations and batches instead (a large transfer):

```bash
.pixi/envs/default/bin/python scripts/00_download_dataset.py --no-include-crops
```

Downloads use a pinned HF commit, not moving main. Use a fresh `data/` directory
when changing repositories or revisions. See the
[download and revision options](docs/dataset_bundle.md#reproducibility) for
version selection. These commands prepare inputs; they do not run benchmarks.
Continue with [CPU scoring and script navigation](docs/dataset_bundle.md#run-cpu-scoring-on-published-features)
for commands, output locations, and input requirements.

## Benchmark workflow

The intended analysis flow is:

```text
Cleaned features + annotations → identity, schema, and cohort checks
    ├─ XGBoost / phenotypic-activity scores + control calibration
    │    ├─ Mislocalization hit calls
    │    └─ Scores and hit calls → ClinVar / variant-effect predictor comparisons
    └─ Reference-cell profiles + HPA labels → localization retrieval
```

Comparisons distinguish continuous scores from hit calls and account for
biological replicates, missing observations, and multiple testing. HPA retrieval
uses reference proteins rather than the variant-comparison cohort.

Task outputs are designed to include numerical tables, coverage and exclusion
summaries, and provenance linking results to inputs and settings. Control
calibration must match the scoring backend and settings.

## Environments

| Environment | Purpose |
|-------------|---------|
| `default` | Feature downloads, preprocessing, CPU XGBoost, and benchmark analyses. |
| `gpu` | GPU XGBoost scoring with a compatible CUDA setup. |
| `cytoself`, `subcell`, `vit` | Model-specific image preparation, training, and extraction. |

The default feature-based workflow does not require an encoder environment.
Install model-specific environments only when working from images or retraining
models.
