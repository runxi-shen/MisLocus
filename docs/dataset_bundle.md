# Dataset bundle

The published `features.parquet` files are already cleaned and ready for
scoring. **Skip `scripts/06_preprocess_profiles.py`: do not normalize them
again.** Crops and encoder weights are optional. The bundle is published
alongside the parent paper / preprint on Hugging Face Hub at
[`anonymous-xyz96/MisLocus`](https://huggingface.co/datasets/anonymous-xyz96/MisLocus).

## Downloading

Two scripts, run independently:

| Script                                      | Source                                                             | When you need it                                       |
|---------------------------------------------|--------------------------------------------------------------------|--------------------------------------------------------|
| `scripts/00_download_dataset.py`            | HF dataset repo `anonymous-xyz96/MisLocus`                         | Always (unless you already have features locally).     |
| `scripts/00b_download_subcell_weights.py`   | CZI public S3 (`czi-subcell-public.s3.amazonaws.com/models/`)      | Only if you'll run `08a` / `08c` (SubCell extract / fine-tune). |

Use the dependency-only setup in the [README](../README.md#download-published-features), from
the repository root. The commands below use that environment directly.

### Dataset bundle — all features at the pinned revision

```bash
# Every representation and batch, plus manifests/metadata; no crop archives.
.pixi/envs/default/bin/python scripts/00_download_dataset.py --no-include-crops

# Optional full bundle, including large crop archives and browseable sample.
.pixi/envs/default/bin/python scripts/00_download_dataset.py
```

Identical feature re-imports are accepted; conflicting destination bytes stop
feature remapping before any feature moves. `--force` re-fetches but does not
override this guard. Because downloaded files are moved, repeated commands may
transfer them again. This is input materialization, not an exact sync: stale
files are not pruned and revision changes are not transactional. Use a fresh
`data/` directory (preserve the previous one) when changing repo or revision.

### Dataset bundle — smoke-test subset (one rep × one batch)

For a quick sanity check or partial materialization, restrict by
representation and/or batch. Crop shards are excluded by default in
subset mode:

```bash
# One real-data scoring input; this does not run downstream comparisons.
.pixi/envs/default/bin/python scripts/00_download_dataset.py \
    --rep morphem --batch 2024_02_06_Batch_8

# Multiple reps / batches (comma-separated or repeat the flag)
.pixi/envs/default/bin/python scripts/00_download_dataset.py \
    --rep cytoself,cellprofiler --batch 2025_01_27_Batch_13

# Optional crop archives (large)
.pixi/envs/default/bin/python scripts/00_download_dataset.py \
    --batch 2025_01_27_Batch_13 --include-crops
```

### Dataset bundle — override the HF repo

Alternate repositories, including `PROT_LOC_BENCHMARK_HF_REPO` overrides, require
an explicit full commit SHA belonging to that repository. Branches and tags
are rejected; `repo@sha` is not supported syntax.

```bash
.pixi/envs/default/bin/python scripts/00_download_dataset.py \
    --hf-repo myorg/my-dataset --revision <40-character-commit-sha> --no-include-crops
```

### SubCell pretrained weights

```bash
# Six encoder.pth checkpoints (~2 GB) → data/interim/subcell_portable/weights/
.pixi/envs/default/bin/python scripts/00b_download_subcell_weights.py
```

Skip this step if you're only consuming the shipped per-rep
`features.parquet` files. You only need the encoder weights if you'll
run `scripts/08a_extract_subcell_embeddings.py` (frozen ViT extraction)
or `scripts/08c_train_subcell_finetune.py` (fine-tuning).

### What the dataset-download script does

After `snapshot_download`, paths are remapped:
- `representations/{rep}/{batch}/features.parquet` → `data/interim/{rep}/{batch}/features.parquet`, except public `morphem` → local `vit` (feature bytes unchanged)
- `manifest/manifest_Batch_X.parquet` → `data/interim/crop_manifest/{full_batch}/manifest.parquet`
- `single_cell_crops/{batch}/shard-NN.tar.gz` → extracted into `data/interim/single_cell_crops/{batch}/`

HF-only repo metadata (`LICENSE`, `README.md`, `MisLocus_croissant.json`,
`.gitattributes`) is removed at the end of the remap; the
`.gitattributes` in particular would otherwise silently activate LFS
smudge for every parquet in the working tree.

Anything not under `data/` (notably the curated reference parquets in
`annotations/`) is in-repo and does not come from either downloader.

## Layout after download

The seven published names are `cellprofiler`, `cytoself`, `morphem`,
`subcell_finetuned_mae`, `subcell_finetuned_vit`, `subcell_portable_rbg_mae`, and
`subcell_portable_rbg_vit`. `--rep vit` and `--rep morphem` both select HF
`morphem` and admit it to `interim/vit/`; outputs retain the internal `vit` label.
MorphEm uses `ViT_{gfp,dna,agp,mito}_0..383`: GFP=384, Morph=1152, ALL=1536
features. Cleaned feature counts for other representations can vary by batch.
Representation names alone do not establish fine-tuned model/export lineage.

The download script populates `data/` (everything under `data/` is
gitignored — pure dataset payload):

```
data/
├── interim/
│   ├── single_cell_crops/             # only present if --include-crops
│   │   └── {batch}/{allele}/
│   │       ├── dna.npy        # uint16, (N, 128, 128)
│   │       ├── gfp.npy
│   │       ├── agp.npy
│   │       └── mito.npy
│   ├── crop_manifest/
│   │   └── {batch}/manifest.parquet
│   ├── cellprofiler/
│   │   └── {batch}/features.parquet
│   ├── cytoself/
│   │   └── {batch}/features.parquet
│   ├── vit/                          # public HF name: morphem
│   │   └── {batch}/features.parquet
│   ├── subcell_portable_rbg_vit/
│   │   └── {batch}/features.parquet
│   ├── subcell_portable_rbg_mae/
│   ├── subcell_finetuned_vit/
│   └── subcell_finetuned_mae/
└── processed/                         # empty until 09 / 10 / 10b / 11 run
```

In-repo (NOT downloaded — tracked in git, ~9.7 MB total):

```
annotations/
├── full_allele_collection.parquet         # ClinVar / dbNSFP / pLDDT / RSA / G2P / OMIM
└── hpa_gene_localization_table.parquet    # 262 genes × 50 organelles
```

These are read via `prot_loc_benchmark.config.{ALLELE_COLLECTION_PATH,
HPA_GENE_LOCALIZATION_PATH}`.

## Batch identifiers

The bundle is keyed by MisLocus batch IDs:

| Batch | ID                  | Notes      |
|-------|---------------------|------------|
| 7     | 2024_01_23_Batch_7  | A1R1 only  |
| 8     | 2024_02_06_Batch_8  | A1R2 only  |
| 13    | 2025_01_27_Batch_13 |            |
| 14    | 2025_01_28_Batch_14 |            |
| 15    | 2025_03_17_Batch_15 |            |
| 16    | 2025_03_17_Batch_16 |            |

Biological-replicate pairs used for benchmark averaging:
B7+B8, B13+B14, B15+B16 (`config.BIOREP_PAIRS`).

## Manifest schema

`data/interim/crop_manifest/{batch}/manifest.parquet` is the canonical
cell ordering. Each row is one cell; the corresponding crop sits at index
`i` in every `*.npy` channel file under
`data/interim/single_cell_crops/{batch}/{allele}/`.

Required `Metadata_*` columns (also produced by every embeddings extractor —
see `scripts/08b_extract_vit_embeddings.py:52`):

```
Metadata_Plate              str
Metadata_well_position      str
Metadata_Site               int
Metadata_ImageNumber        int
Metadata_ObjectNumber       int
Metadata_gene_allele        str
Metadata_symbol             str
Metadata_node_type          str   # "disease_wt" | "allele" | "TC"
Metadata_Control            str   # "Exp" | "cPC" | "NC" | "PC" | "TC"
Metadata_plate_map_name     str
```

## Adding a new representation

The pipeline is contract-based: anything that writes a parquet matching the
schema below at the right path can be plugged in.

### 1. Write your extraction script

Create `scripts/08x_extract_<myrep>_embeddings.py`. The cleanest template is
`scripts/08b_extract_vit_embeddings.py` (single-pass DataLoader → frozen ViT
→ parquet writer). Your script must:

- Read crops from `data/interim/single_cell_crops/{batch}/{allele}/{channel}.npy`.
- Read cell ordering from `data/interim/crop_manifest/{batch}/manifest.parquet`.
- Run your model and produce one feature vector per cell.
- Write `data/interim/<myrep>/{batch}/embeddings.parquet` containing:
  - **All 10 `Metadata_*` columns** listed above.
  - **Your feature columns** — any names. If you want per-imaging-channel
    splits to fall out automatically (see step 3), include the channel name
    in the column name (e.g. `<myrep>_GFP_0042`).

### 2. Register the rep in `config.py`

In `src/prot_loc_benchmark/config.py`, add three entries:

```python
REP_FEATURE_FILES["<myrep>"]   = "features.parquet"
REP_RAW_FILES["<myrep>"]       = "embeddings.parquet"
BENCHMARK_CHANNELS["<myrep>"]  = ["EMBED"]   # or ["GFP", "DNA", "AGP", "Mito", "Morph", "ALL"]
```

The `BENCHMARK_CHANNELS` value picks which channel splits the classifier
will train on. `EMBED` (single channel containing all features) is the
default for opaque embeddings.

### 3. (Optional) Extend channel splitting

If your features have internal structure beyond per-imaging-channel
(e.g. Cytoself's `global` / `spectrum` / `combined` split), add a branch to
`get_feature_channels` in
`src/prot_loc_benchmark/classification/channels.py`.

### 4. (Optional) Add a pixi env

If your model needs an isolated PyTorch / CUDA stack, mirror the
`[tool.pixi.feature.vit]` block in `pyproject.toml`. Otherwise reuse
`default`.

### 5. Run the pipeline (optional raw-feature producer route)

These older producer commands require the relevant model environments; they
are not the dependency-only published-feature quickstart. Skip extraction and
step 06 entirely when using the cleaned HF bundle.

```bash
pixi run -e <env> python scripts/08x_extract_<myrep>_embeddings.py --batch <batch>
pixi run python scripts/06_preprocess_profiles.py --representation <myrep> --batch <batch>
pixi run -e gpu python scripts/09_classify.py     --batch <batch> --representation <myrep> --gpu
pixi run python scripts/09c_classify_PA.py        --batch <batch> --representation <myrep>
# After running across all batches in BIOREP_PAIRS:
pixi run python scripts/10_benchmark_clinvar.py
pixi run python scripts/10b_benchmark_hpa.py
pixi run python scripts/11_summarize_across_reps.py   # picks up <myrep> automatically
```

### Notes

- **If you only have pre-computed embeddings (no extraction needed):** skip
  step 1 and write the parquet directly. Steps 2–5 still apply.
- **If your model needs training from crops:** mirror `07a_preprocess_cytoself.py`
  (manifest builder) + `07b_train_cytoself.py` (training loop). These are
  decoupled from extraction — only step 1 (the extraction script) interfaces
  with the rest of the pipeline.
- **Pre-computed `features.parquet`?** If you've already done plate
  normalization and feature selection externally, write your output as
  `data/interim/<myrep>/{batch}/features.parquet` and skip step 06. Make
  sure the columns match what the classifier expects (`Metadata_*` + numeric
  features).

## Reproducibility

- Both bundle and `--sample` downloads default to commit
  `74f63113a76b4a832285da308f3df2932266e456` of `anonymous-xyz96/MisLocus`.
  `--revision <40-character-commit-sha>` overrides it; the downloader logs and
  passes the SHA to `snapshot_download(revision=...)`. No fallback to main.
- Validation inspected 42 Parquet schemas and hash-verified seven Batch 8
  payloads against this revision. Small unchanged real-row subsets passed
  CPU XGBoost, PA and HPA API checks. This is not a fresh-network full-pipeline
  test, production calibration, GPU equivalence, or paper-result reproduction.
- A dataset pin fixes the source selection, not the entire experiment. Retain
  code/environment/annotation versions and backend/settings-matched controls.
  It does not verify historical producer lineage or prevent mixing old local
  files if a nonempty data directory is reused across revisions.
