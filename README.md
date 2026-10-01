# MisLocus

MisLocus benchmarks protein-localization representations using single-cell
images and variant annotations. This repository includes feature preprocessing,
model training and retraining, feature extraction, and downstream classification
and benchmarking tools.

You can start from **published features** or **single-cell images**:

- **Published features:** download the precomputed `features.parquet` files and
  use them directly for scoring. They are already cleaned—do not run
  `06_preprocess_profiles.py` or normalize them again.
- **Single-cell images:** use the training and extraction scripts to generate
  features, then preprocess those raw features before scoring. These workflows
  need their model-specific environments; they are not required to use the
  published features.

The quickstart below follows the published-feature route.

## What you get

### Included in this repository

- **Code:** scripts for preprocessing, training, extraction, classification,
  and representation comparisons, with shared analysis functions in `src/`.
- **Reference annotations:** HPA gene localizations and the ClinVar / dbNSFP /
  pLDDT allele collection in `annotations/`. These do not need a separate download.

### Available from the Hugging Face dataset

| Component | Contents |
|-----------|----------|
| Precomputed features | Seven representations: CellProfiler, Cytoself, MorphEm, and portable RBG / fine-tuned SubCell MAE and ViT. |
| Crop manifests | Per-cell metadata for the image crops. |
| Single-cell images | QC'd 128×128 crops with DNA, GFP, AGP, and Mito channels; optional when using published features. |

Downloads are stored under `data/`. Feature files go to
`data/interim/{rep}/{batch}/features.parquet`. MorphEm is named `morphem` on
Hugging Face and stored locally as `vit`; the downloader and scoring tools
accept either name.

See the [dataset guide](docs/dataset_bundle.md) for download options and the
complete directory layout.

## Quickstart

From the repository root, with [Pixi](https://pixi.sh/install) installed:

```bash
# Dependencies only: no project-package installation or optional environments.
pixi install --frozen -e default --skip prot-loc-benchmark
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

# All published features, manifests, and metadata at the pinned HF commit.
# This can still be a large transfer; crop archives are excluded.
.pixi/envs/default/bin/python scripts/00_download_dataset.py --no-include-crops

# Alternatively, start with one representation and batch.
.pixi/envs/default/bin/python scripts/00_download_dataset.py \
    --rep morphem --batch 2024_02_06_Batch_8
```

Downloads default to the tested immutable revision documented in
[`docs/dataset_bundle.md`](docs/dataset_bundle.md#reproducibility), not HF main.
Override it with `--revision <commit-sha>`; alternate repositories require an
explicit revision. Use a fresh `data/` directory when switching revisions.
Identical features can be re-imported; different destination bytes are rejected.
This prepares inputs, not comparison results or an exact synchronized mirror.

Use the direct interpreter above to avoid Pixi automatically installing the
project package. Existing Justfile recipes retain older environment/default
choices and are not this dependency-only quickstart. `--sample` downloads only
browseable crops, not scoring features; encoder weights are unnecessary here.

## Pipeline

Bounded real-data CPU checks cover all seven representations through native
XGBoost, PA, and HPA scoring APIs. They do not certify a complete comparison
workflow, production calibration/hit calls, GPU parity, or paper reproduction.
Further downstream integration remains in progress. HPA reads reference-cell
features directly; ClinVar and cross-representation summaries consume scores.

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
| `default` | Downloads, CPU XGBoost (`09`), PA (`09c`), benchmarks (`10` / `10b` / `11`). |
| `gpu`     | XGBoost classification (`09 --gpu`). Inherits `default` + adds the CUDA-12 system requirement so conda-forge resolves to GPU-enabled xgboost. |

The `cytoself` / `subcell` / `vit` envs only matter if you re-extract or
retrain a representation from raw crops; they're not needed for the
classification + benchmark workflow.
