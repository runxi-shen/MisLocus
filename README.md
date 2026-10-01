# MisLocus — companion repo

Consume the published **MisLocus single-cell crop dataset** for
protein-localization representation scoring. The shipped `features.parquet`
files are already cleaned: **skip `06_preprocess_profiles.py` and do not
normalize them again**. Crops, encoder weights, and training are optional.

## What you get

The dataset bundle (download separately, see below) ships into `data/`:

- **Seven representations** — CellProfiler, Cytoself, MorphEm, and the portable RBG / fine-tuned SubCell MAE and ViT models, under `data/interim/{rep}/{batch}/features.parquet`. Public `morphem` maps to local `vit`; both names are accepted by the downloader and scoring consumers.
- **Crop manifest** — `data/interim/crop_manifest/{batch}/manifest.parquet` (one row per cell).
- **QC'd single-cell crops** (optional) — `data/interim/single_cell_crops/{batch}/{allele}/*.npy` (128×128 uint16, 4 channels: DNA / GFP / AGP / Mito).

Reference annotations (HPA gene-localization + ClinVar / dbNSFP / pLDDT
allele collection) live in `annotations/` and are tracked in this repo
(~9.7 MB total). They're consumed by `10_benchmark_clinvar.py` and
`10b_benchmark_hpa.py` and don't need to be downloaded.

See [`docs/dataset_bundle.md`](docs/dataset_bundle.md) for the full layout
and the download script's CLI options.

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

## Adding a new representation

The benchmark is a contract — anything that writes
`data/interim/<myrep>/{batch}/features.parquet` with the right schema
plugs in without touching `09 / 09c / 10 / 10b / 11`. See
[`docs/dataset_bundle.md#adding-a-new-representation`](docs/dataset_bundle.md#adding-a-new-representation)
for the integration recipe.
